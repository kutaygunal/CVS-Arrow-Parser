#include "gpu_csv_parser.hpp"
#include "tuneable_params.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <limits>

namespace gpu_csv {

// ---------------------------------------------------------------------------
// Host utilities
// ---------------------------------------------------------------------------
static inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " -> "
                  << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

static size_t type_size(ColumnType t) {
    switch (t) {
        case ColumnType::INT64:   return sizeof(int64_t);
        case ColumnType::FLOAT64: return sizeof(double);
    }
    return 0;
}

std::vector<uint8_t> DeviceColumn::copy_rows_to_host(std::size_t start,
                                                      std::size_t count) const {
    size_t type_sz = type_size(type);
    std::vector<uint8_t> out(count * type_sz);
    if (count == 0) return out;
    uint8_t* d_ptr = static_cast<uint8_t*>(data.get());
    CUDA_CHECK(cudaMemcpy(out.data(), d_ptr + start * type_sz,
                          count * type_sz, cudaMemcpyDeviceToHost));
    return out;
}

// ---------------------------------------------------------------------------
// Fast device parsing primitives -- assumes well-formed simple data, no
// locale, no exponent, no NaN/Inf.  Empty fields are handled by the caller.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int64_t fast_atol(const char* s, const char* e) {
    bool neg = (*s == '-');
    s += neg;
    int64_t val = 0;
    #pragma unroll 8
    for (; s < e; ++s) {
        val = val * 10 + (*s - '0');
    }
    return neg ? -val : val;
}

__device__ __forceinline__ double fast_atof(const char* s, const char* e) {
    bool neg = (*s == '-');
    s += neg;

    int64_t int_part = 0;
    const char* dot = nullptr;
    for (const char* p = s; p < e; ++p) {
        if (*p == '.') { dot = p; break; }
        int_part = int_part * 10 + (*p - '0');
    }

    double val = double(int_part);
    if (dot) {
        int64_t frac = 0;
        int64_t div  = 1;
        #pragma unroll 8
        for (const char* p = dot + 1; p < e; ++p) {
            frac = frac * 10 + (*p - '0');
            div *= 10;
        }
        val += double(frac) / double(div);
    }
    return neg ? -val : val;
}

// ---------------------------------------------------------------------------
// Safe / general parsers (kept for fallback / non-fast paths)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int64_t dev_atol(const char* s, const char* e,
                                            bool* ok) {
    if (s >= e) { *ok = false; return 0; }
    bool neg = false;
    if (*s == '-') { neg = true; ++s; }
    int64_t val = 0;
    for (; s < e; ++s) {
        char c = *s;
        if (c < '0' || c > '9') { *ok = false; return 0; }
        val = val * 10 + (c - '0');
    }
    *ok = true;
    return neg ? -val : val;
}

__device__ __forceinline__ double dev_atof(const char* s, const char* e,
                                            bool* ok) {
    if (s >= e) { *ok = false; return 0.0; }
    bool neg = false;
    if (*s == '-') { neg = true; ++s; }

    const char* dot = nullptr;
    for (const char* p = s; p < e; ++p) {
        if (*p == '.') { dot = p; break; }
    }

    double integral = 0.0;
    const char* int_end = dot ? dot : e;
    for (const char* p = s; p < int_end; ++p) {
        char c = *p;
        if (c < '0' || c > '9') { *ok = false; return 0.0; }
        integral = integral * 10.0 + (c - '0');
    }

    double fractional = 0.0;
    if (dot) {
        double div = 1.0;
        for (const char* p = dot + 1; p < e; ++p) {
            char c = *p;
            if (c < '0' || c > '9') { *ok = false; return 0.0; }
            fractional = fractional * 10.0 + (c - '0');
            div *= 10.0;
        }
        integral += fractional / div;
    }

    *ok = true;
    return neg ? -integral : integral;
}

__device__ __forceinline__ bool eval_predicate_int64(int64_t val,
                                                      PredicateOp op,
                                                      int64_t rhs) {
    switch (op) {
        case PredicateOp::GT: return val > rhs;
        case PredicateOp::GE: return val >= rhs;
        case PredicateOp::LT: return val < rhs;
        case PredicateOp::LE: return val <= rhs;
        case PredicateOp::EQ: return val == rhs;
    }
    return false;
}

__device__ __forceinline__ bool eval_predicate_float64(double val,
                                                        PredicateOp op,
                                                        double rhs) {
    switch (op) {
        case PredicateOp::GT: return val > rhs;
        case PredicateOp::GE: return val >= rhs;
        case PredicateOp::LT: return val < rhs;
        case PredicateOp::LE: return val <= rhs;
        case PredicateOp::EQ: return val == rhs;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Warp-level helpers: one warp collaboratively scans a single CSV row.
// ---------------------------------------------------------------------------

// Find the begin/end offsets (relative to `data`) of ONE field.
// Every thread in the calling warp participates.  Only lane 0 returns valid
// *out_b / *out_e, and the return value is broadcast.
__device__ __forceinline__ bool warp_find_field(
    const char* data,
    int64_t row_start,
    int64_t row_end,
    char delimiter,
    int target_field,
    int* out_b,
    int* out_e)
{
    int lane = threadIdx.x & 31;
    int fb = static_cast<int>(row_start);   // current field begin
    int field_idx = 0;
    int64_t chunk = row_start;
    bool found = false;
    int ob = 0, oe = 0;

    while (chunk < row_end) {
        int64_t pos = chunk + lane;
        char c = (pos < row_end) ? data[pos] : 0;
        unsigned int delim_mask = __ballot_sync(0xFFFFFFFF, c == delimiter);

        if (lane == 0) {
            unsigned int m = delim_mask;
            while (m) {
                int d = __ffs(static_cast<int>(m)) - 1;
                if (field_idx == target_field) {
                    ob = fb;
                    oe = static_cast<int>(chunk + d);
                    found = true;
                    break;
                }
                fb = static_cast<int>(chunk + d) + 1;
                ++field_idx;
                m &= m - 1;
            }
        }
        if (found) break;
        chunk += 32;
    }

    if (lane == 0 && !found && field_idx == target_field && fb < row_end) {
        ob = fb;
        oe = static_cast<int>(row_end);
        found = true;
    }

    // Broadcast result to all lanes (so callers can branch uniformly)
    int found_i = __shfl_sync(0xFFFFFFFF, found ? 1 : 0, 0);
    if (found_i) {
        *out_b = __shfl_sync(0xFFFFFFFF, ob, 0);
        *out_e = __shfl_sync(0xFFFFFFFF, oe, 0);
    }
    return found_i != 0;
}

// Scan all fields of a row and store their begin/end offsets.
// `f_begin` / `f_end` must live in shared memory or per-thread arrays with
// at least `max_fields` entries.  Returns field count (broadcast to all lanes).
__device__ __forceinline__ int warp_scan_fields(
    const char* data,
    int64_t row_start,
    int64_t row_end,
    char delimiter,
    int max_fields,
    int* f_begin,
    int* f_end)
{
    int lane = threadIdx.x & 31;
    int fb = static_cast<int>(row_start);
    int field_idx = 0;
    int64_t chunk = row_start;

    while (chunk < row_end && field_idx < max_fields) {
        int64_t pos = chunk + lane;
        char c = (pos < row_end) ? data[pos] : 0;
        unsigned int delim_mask = __ballot_sync(0xFFFFFFFF, c == delimiter);

        if (lane == 0) {
            unsigned int m = delim_mask;
            while (m && field_idx < max_fields) {
                int d = __ffs(static_cast<int>(m)) - 1;
                f_begin[field_idx] = fb;
                f_end[field_idx]   = static_cast<int>(chunk + d);
                fb = static_cast<int>(chunk + d) + 1;
                ++field_idx;
                m &= m - 1;
            }
        }
        chunk += 32;
    }

    if (lane == 0 && field_idx < max_fields && fb < row_end) {
        f_begin[field_idx] = fb;
        f_end[field_idx]   = static_cast<int>(row_end);
        ++field_idx;
    }

    return __shfl_sync(0xFFFFFFFF, field_idx, 0);
}

// ---------------------------------------------------------------------------
// Row discovery kernels (unchanged, already coalesced and fast)
// ---------------------------------------------------------------------------
__global__ void mark_newlines(const char* data, size_t n, uint8_t* marks) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    marks[i] = (data[i] == '\n') ? 1 : 0;
}

__global__ void compact_newlines(const char* data, size_t n,
                                const uint32_t* cumsum, int64_t* newlines) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (data[i] == '\n') {
        newlines[cumsum[i] - 1] = static_cast<int64_t>(i);
    }
}

// ---------------------------------------------------------------------------
// Full parse kernel -- one warp per row, warp-level field scan
// ---------------------------------------------------------------------------
__global__ void parse_rows_kernel_warp(const char* data,
                                        const int64_t* newlines,
                                        int64_t num_rows,
                                        bool ends_with_newline,
                                        int64_t file_size,
                                        char delimiter,
                                        int num_cols,
                                        const ColumnType* types,
                                        void** col_data,
                                        uint8_t** col_nulls) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id >= num_rows) return;
    int64_t row = warp_id;

    int64_t start = (row == 0) ? 0 : (newlines[row - 1] + 1);
    int64_t end   = newlines[row];
    if (!ends_with_newline && row == num_rows - 1) end = file_size;
    if (end > start && data[end - 1] == '\r') --end;

    // Shared memory scratch for this warp's field boundaries
    __shared__ int s_f_b[32 * 32]; // max 32 warps * 32 fields
    __shared__ int s_f_e[32 * 32];
    int warp_slot = (threadIdx.x >> 5) * 32;
    int* f_b = s_f_b + warp_slot;
    int* f_e = s_f_e + warp_slot;

    int nf = warp_scan_fields(data, start, end, delimiter, num_cols, f_b, f_e);

    if (lane == 0) {
        for (int c = 0; c < num_cols; ++c) {
            if (c >= nf) {
                col_nulls[c][row] = 0;
                continue;
            }
            bool has = (f_b[c] < f_e[c]);
            if (has) {
                switch (types[c]) {
                    case ColumnType::INT64: {
                        reinterpret_cast<int64_t*>(col_data[c])[row] =
                            fast_atol(data + f_b[c], data + f_e[c]);
                        break;
                    }
                    case ColumnType::FLOAT64: {
                        reinterpret_cast<double*>(col_data[c])[row] =
                            fast_atof(data + f_b[c], data + f_e[c]);
                        break;
                    }
                }
            }
            col_nulls[c][row] = has ? 1 : 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused filter: Stage 1 -- evaluate predicate per row (warp per row)
// ---------------------------------------------------------------------------
__global__ void filter_mask_kernel_warp(const char* data,
                                         const int64_t* newlines,
                                         int64_t num_rows,
                                         bool ends_with_newline,
                                         int64_t file_size,
                                         char delimiter,
                                         int predicate_col,
                                         ColumnType predicate_type,
                                         PredicateOp op,
                                         int64_t pred_i64,
                                         double pred_f64,
                                         uint8_t* mask) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id >= num_rows) return;
    int64_t row = warp_id;

    int64_t start = (row == 0) ? 0 : (newlines[row - 1] + 1);
    int64_t end   = newlines[row];
    if (!ends_with_newline && row == num_rows - 1) end = file_size;
    if (end > start && data[end - 1] == '\r') --end;

    int fb = 0, fe = 0;
    bool found = warp_find_field(data, start, end, delimiter,
                                  predicate_col, &fb, &fe);

    bool pass = false;
    if (lane == 0) {
        if (found && fb < fe) {
            if (predicate_type == ColumnType::INT64) {
                int64_t v = fast_atol(data + fb, data + fe);
                pass = eval_predicate_int64(v, op, pred_i64);
            } else {
                double v = fast_atof(data + fb, data + fe);
                pass = eval_predicate_float64(v, op, pred_f64);
            }
        }
    }

    unsigned int pass_mask = __ballot_sync(0xFFFFFFFF, lane == 0 ? pass : false);
    if (lane == 0) mask[row] = (pass_mask != 0) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Fused filter: Stage 2 -- parse projected columns for surviving rows
// Uses warp-level field scan and writes output using precomputed d_scanned.
// ---------------------------------------------------------------------------
__global__ void parse_filtered_rows_kernel_warp(const char* data,
                                                const int64_t* newlines,
                                                int64_t num_rows,
                                                bool ends_with_newline,
                                                int64_t file_size,
                                                char delimiter,
                                                int num_proj_cols,
                                                const int* proj_indices,
                                                const ColumnType* types,
                                                const uint8_t* mask,
                                                const int* scanned,
                                                void** col_data,
                                                uint8_t** col_nulls) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id >= num_rows) return;
    int64_t row = warp_id;

    if (!mask[row]) return;

    // Coalesce the scanned offset read through lane 0 + shfl
    int out_row = 0;
    if (lane == 0) out_row = scanned[row];
    out_row = __shfl_sync(0xFFFFFFFF, out_row, 0);

    int64_t start = (row == 0) ? 0 : (newlines[row - 1] + 1);
    int64_t end   = newlines[row];
    if (!ends_with_newline && row == num_rows - 1) end = file_size;
    if (end > start && data[end - 1] == '\r') --end;

    __shared__ int s_f_b[32 * 32];
    __shared__ int s_f_e[32 * 32];
    int warp_slot = (threadIdx.x >> 5) * 32;
    int* f_b = s_f_b + warp_slot;
    int* f_e = s_f_e + warp_slot;

    int nf = warp_scan_fields(data, start, end, delimiter,
                              32, f_b, f_e);

    if (lane == 0) {
        for (int pj = 0; pj < num_proj_cols; ++pj) {
            int col = proj_indices[pj];
            if (col >= nf) {
                col_nulls[pj][out_row] = 0;
                continue;
            }
            bool has = (f_b[col] < f_e[col]);
            if (has) {
                switch (types[pj]) {
                    case ColumnType::INT64: {
                        reinterpret_cast<int64_t*>(col_data[pj])[out_row] =
                            fast_atol(data + f_b[col], data + f_e[col]);
                        break;
                    }
                    case ColumnType::FLOAT64: {
                        reinterpret_cast<double*>(col_data[pj])[out_row] =
                            fast_atof(data + f_b[col], data + f_e[col]);
                        break;
                    }
                }
            }
            col_nulls[pj][out_row] = has ? 1 : 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Reusable buffer helper
// ---------------------------------------------------------------------------
static void ensure_buffer(void** d_ptr, size_t* current_cap, size_t required) {
    if (*current_cap >= required) return;
    if (*d_ptr) CUDA_CHECK(cudaFree(*d_ptr));
    CUDA_CHECK(cudaMalloc(d_ptr, required));
    *current_cap = required;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------
class CSVParser::Impl {
    cudaStream_t stream_ = nullptr;
    cudaStream_t stream2_ = nullptr;

    // Reusable device buffers (resized on demand)
    char* d_data_ = nullptr;       size_t d_data_cap_ = 0;
    uint8_t* d_marks_ = nullptr;    size_t d_marks_cap_ = 0;
    uint32_t* d_cumsum_ = nullptr;  size_t d_cumsum_cap_ = 0;
    void* d_temp_ = nullptr;        size_t d_temp_cap_ = 0;
    void* d_temp2_ = nullptr;       size_t d_temp2_cap_ = 0;
    int64_t* d_newlines_ = nullptr; size_t d_newlines_cap_ = 0;
    uint8_t* d_mask_ = nullptr;     size_t d_mask_cap_ = 0;
    int* d_scanned_ = nullptr;      size_t d_scanned_cap_ = 0;

    // Reusable pinned host buffer
    char* h_pinned_buf_ = nullptr; size_t h_pinned_cap_ = 0;

    void ensure_buffer(void** d_ptr, size_t* current_cap, size_t required) {
        if (*current_cap >= required) return;
        if (*d_ptr) CUDA_CHECK(cudaFree(*d_ptr));
        CUDA_CHECK(cudaMalloc(d_ptr, required));
        *current_cap = required;
    }

    void ensure_data_(size_t n)     { ensure_buffer((void**)&d_data_,     &d_data_cap_,     n); }
    void ensure_marks_(size_t n)    { ensure_buffer((void**)&d_marks_,    &d_marks_cap_,    n); }
    void ensure_cumsum_(size_t n)   { ensure_buffer((void**)&d_cumsum_,   &d_cumsum_cap_,   n * sizeof(uint32_t)); }
    void ensure_temp_(size_t n)   { ensure_buffer((void**)&d_temp_,     &d_temp_cap_,     n); }
    void ensure_temp2_(size_t n)  { ensure_buffer((void**)&d_temp2_,    &d_temp2_cap_,    n); }
    void ensure_newlines_(size_t n) { ensure_buffer((void**)&d_newlines_, &d_newlines_cap_, n * sizeof(int64_t)); }
    void ensure_mask_(size_t n)     { ensure_buffer((void**)&d_mask_,     &d_mask_cap_,     n * sizeof(uint8_t)); }
    void ensure_scanned_(size_t n)  { ensure_buffer((void**)&d_scanned_,  &d_scanned_cap_,  n * sizeof(int)); }
    void ensure_h_pinned_(size_t n) {
        if (h_pinned_cap_ >= n) return;
        if (h_pinned_buf_) CUDA_CHECK(cudaFreeHost(h_pinned_buf_));
        CUDA_CHECK(cudaMallocHost(&h_pinned_buf_, n));
        h_pinned_cap_ = n;
    }

public:
    Impl() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaStreamCreate(&stream2_));
    }
    ~Impl() {
        if (d_data_)     cudaFree(d_data_);
        if (d_marks_)    cudaFree(d_marks_);
        if (d_cumsum_)   cudaFree(d_cumsum_);
        if (d_temp_)     cudaFree(d_temp_);
        if (d_temp2_)    cudaFree(d_temp2_);
        if (d_newlines_) cudaFree(d_newlines_);
        if (d_mask_)     cudaFree(d_mask_);
        if (d_scanned_)  cudaFree(d_scanned_);
        if (h_pinned_buf_) cudaFreeHost(h_pinned_buf_);
        if (stream_)     cudaStreamDestroy(stream_);
        if (stream2_)    cudaStreamDestroy(stream2_);
    }

    ParsedCSV parse_multistream(const char* h_pinned,
                                 size_t data_offset,
                                 size_t payload_size,
                                 const ParseOptions& opts) {
        using clock = std::chrono::high_resolution_clock;
        ParsedCSV result;
        result.num_rows = 0;
        result.num_filtered_rows = 0;
        result.parse_time_ms = 0.0;
        result.h2d_time_ms = 0.0;

        const int num_streams = TUNE_NUM_STREAMS;
        if (num_streams < 2 || payload_size < static_cast<size_t>(TUNE_MIN_CHUNK_SIZE_BYTES) * 2) {
            return result; // signal caller to fall back
        }

        // Ensure buffers before multistream path
        ensure_data_(payload_size);
        ensure_marks_(payload_size);
        ensure_cumsum_(payload_size);

        // Split payload into chunks at newline boundaries
        std::vector<size_t> chunk_off(num_streams);
        std::vector<size_t> chunk_sz(num_streams);
        size_t cursor = data_offset;
        size_t remaining = payload_size;
        for (int i = 0; i < num_streams; ++i) {
            if (i == num_streams - 1) {
                chunk_off[i] = cursor - data_offset;
                chunk_sz[i] = remaining;
            } else {
                size_t target = remaining / (num_streams - i);
                const char* base = h_pinned + cursor;
                size_t pos = target;
                while (pos < remaining && base[pos] != '\n') ++pos;
                if (pos < remaining) pos += 1;
                chunk_off[i] = cursor - data_offset;
                chunk_sz[i] = pos;
                cursor += pos;
                remaining -= pos;
            }
        }

        size_t max_chunk = 0;
        for (size_t s : chunk_sz) if (s > max_chunk) max_chunk = s;

        size_t temp_bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_marks_, d_cumsum_,
                                        static_cast<int>(max_chunk));
        ensure_temp_(temp_bytes);
        ensure_temp2_(temp_bytes);

        cudaStream_t streams[4] = {stream_, stream2_, nullptr, nullptr};
        void* temps[4] = {d_temp_, d_temp2_, nullptr, nullptr};

        // Phase 1: H2D + mark + scan concurrently
        std::vector<int64_t> chunk_rows(num_streams, 0);
        std::vector<uint32_t> h_last_cumsum(num_streams, 0);
        std::vector<uint32_t*> d_chunk_cumsum(num_streams);
        std::vector<int64_t*> d_chunk_nl(num_streams);
        std::vector<bool> chunk_ends_nl(num_streams);

        for (int s = 0; s < num_streams; ++s) {
            size_t csz = chunk_sz[s];
            size_t coff = chunk_off[s];
            cudaStream_t st = streams[s];

            CUDA_CHECK(cudaMemcpyAsync(d_data_ + coff, h_pinned + data_offset + coff,
                                       csz, cudaMemcpyHostToDevice, st));

            int threads = TUNE_MARK_THREADS_PER_BLOCK;
            int blocks = static_cast<int>((csz + threads - 1) / threads);
            mark_newlines<<<blocks, threads, 0, st>>>(d_data_ + coff, csz, d_marks_ + coff);
            CUDA_CHECK(cudaGetLastError());

            d_chunk_cumsum[s] = d_cumsum_ + coff;
            cub::DeviceScan::InclusiveSum(temps[s], temp_bytes,
                                          d_marks_ + coff, d_chunk_cumsum[s],
                                          static_cast<int>(csz), st);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(&h_last_cumsum[s], d_chunk_cumsum[s] + csz - 1,
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost, st));
        }

        for (int s = 0; s < num_streams; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }

        // Compact newlines per chunk
        for (int s = 0; s < num_streams; ++s) {
            size_t csz = chunk_sz[s];
            size_t coff = chunk_off[s];
            cudaStream_t st = streams[s];
            uint32_t nl = h_last_cumsum[s];
            if (nl == 0) {
                d_chunk_nl[s] = nullptr;
                chunk_rows[s] = 0;
                chunk_ends_nl[s] = false;
                continue;
            }
            bool ends = false;
            if (csz > 0) {
                char lc = h_pinned[data_offset + coff + csz - 1];
                ends = (lc == '\n');
            }
            chunk_ends_nl[s] = ends;
            chunk_rows[s] = static_cast<int64_t>(nl);
            if (!ends) chunk_rows[s] += 1;

            CUDA_CHECK(cudaMalloc(&d_chunk_nl[s], (chunk_rows[s] + 1) * sizeof(int64_t)));
            int threads = TUNE_COMPACT_THREADS_PER_BLOCK;
            int blocks = static_cast<int>((csz + threads - 1) / threads);
            compact_newlines<<<blocks, threads, 0, st>>>(d_data_ + coff, csz,
                                                                d_chunk_cumsum[s], d_chunk_nl[s]);
            CUDA_CHECK(cudaGetLastError());

            if (!ends) {
                int64_t end_off = static_cast<int64_t>(csz);
                CUDA_CHECK(cudaMemcpyAsync(d_chunk_nl[s] + nl, &end_off,
                                           sizeof(int64_t), cudaMemcpyHostToDevice, st));
            }
        }

        int num_cols = static_cast<int>(opts.schema.size());
        int64_t total_rows = 0;
        for (auto r : chunk_rows) total_rows += r;
        if (total_rows == 0) {
            for (int s = 0; s < num_streams; ++s) if (d_chunk_nl[s]) cudaFree(d_chunk_nl[s]);
            return result;
        }

        // Allocate combined output columns
        std::vector<void*> h_col_data(num_cols);
        std::vector<uint8_t*> h_col_nulls(num_cols);
        for (int c = 0; c < num_cols; ++c) {
            size_t bsize = static_cast<size_t>(total_rows) * type_size(opts.schema[c].type);
            void* dp = nullptr;
            uint8_t* np = nullptr;
            CUDA_CHECK(cudaMalloc(&dp, bsize));
            CUDA_CHECK(cudaMalloc(&np, total_rows * sizeof(uint8_t)));
            h_col_data[c] = dp;
            h_col_nulls[c] = np;
        }

        std::vector<ColumnType> host_types;
        host_types.reserve(num_cols);
        for (const auto& spec : opts.schema) host_types.push_back(spec.type);

        // Phase 2: Parse each chunk into combined buffers concurrently
        for (int s = 0; s < num_streams; ++s) {
            if (chunk_rows[s] == 0) continue;
            size_t csz = chunk_sz[s];
            size_t coff = chunk_off[s];
            cudaStream_t st = streams[s];
            int64_t row_offset = 0;
            for (int j = 0; j < s; ++j) row_offset += chunk_rows[j];

            std::vector<void*> h_chunk_data(num_cols);
            std::vector<uint8_t*> h_chunk_nulls(num_cols);
            for (int c = 0; c < num_cols; ++c) {
                size_t ts = type_size(opts.schema[c].type);
                h_chunk_data[c] = static_cast<uint8_t*>(h_col_data[c]) + row_offset * ts;
                h_chunk_nulls[c] = h_col_nulls[c] + row_offset;
            }

            void** d_chunk_data = nullptr;
            uint8_t** d_chunk_nulls = nullptr;
            ColumnType* d_types = nullptr;
            CUDA_CHECK(cudaMalloc(&d_chunk_data, num_cols * sizeof(void*)));
            CUDA_CHECK(cudaMalloc(&d_chunk_nulls, num_cols * sizeof(uint8_t*)));
            CUDA_CHECK(cudaMalloc(&d_types, num_cols * sizeof(ColumnType)));
            CUDA_CHECK(cudaMemcpyAsync(d_chunk_data, h_chunk_data.data(),
                                       num_cols * sizeof(void*), cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_chunk_nulls, h_chunk_nulls.data(),
                                       num_cols * sizeof(uint8_t*), cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_types, host_types.data(),
                                       num_cols * sizeof(ColumnType), cudaMemcpyHostToDevice, st));

            constexpr int WARP_THREADS = 32;
            constexpr int THREADS_PER_BLOCK = TUNE_THREADS_PER_BLOCK;
            int warps_per_block = THREADS_PER_BLOCK / WARP_THREADS;
            int blocks = static_cast<int>((chunk_rows[s] + warps_per_block - 1) / warps_per_block);

            parse_rows_kernel_warp<<<blocks, THREADS_PER_BLOCK, 0, st>>>(
                d_data_ + coff, d_chunk_nl[s], chunk_rows[s],
                chunk_ends_nl[s], static_cast<int64_t>(csz), opts.delimiter,
                num_cols, d_types, d_chunk_data, d_chunk_nulls);
            CUDA_CHECK(cudaGetLastError());

            cudaFree(d_chunk_data);
            cudaFree(d_chunk_nulls);
            cudaFree(d_types);
        }

        for (int s = 0; s < num_streams; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }

        // Assemble result
        result.num_rows = static_cast<size_t>(total_rows);
        result.columns.reserve(num_cols);
        for (int c = 0; c < num_cols; ++c) {
            DeviceColumn col;
            col.type = opts.schema[c].type;
            col.num_rows = static_cast<size_t>(total_rows);
            col.data_bytes = col.num_rows * type_size(col.type);
            col.data = std::unique_ptr<void, void (*)(void*)>(
                h_col_data[c], +[](void* p) { cudaFree(p); });
            col.null_mask = std::unique_ptr<uint8_t, void (*)(void*)>(
                h_col_nulls[c], +[](void* p) { cudaFree(p); });
            result.columns.push_back(std::move(col));
        }

        for (int s = 0; s < num_streams; ++s) {
            if (d_chunk_nl[s]) cudaFree(d_chunk_nl[s]);
        }
        return result;
    }

    ParsedCSV parse(const std::string& filepath, const ParseOptions& opts) {
        using clock = std::chrono::high_resolution_clock;
        auto t0 = clock::now();

        // 1. Fast file read into reusable pinned host buffer
        FILE* fp = fopen(filepath.c_str(), "rb");
        if (!fp) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        fseek(fp, 0, SEEK_END);
        auto file_size_host = static_cast<size_t>(ftell(fp));
        fseek(fp, 0, SEEK_SET);

        ensure_h_pinned_(file_size_host);
        char* h_pinned = h_pinned_buf_;
        size_t bytes_read = fread(h_pinned, 1, file_size_host, fp);
        fclose(fp);
        if (bytes_read != file_size_host) {
            throw std::runtime_error("Failed to read file: " + filepath);
        }

        // Skip header: advance pointer/size after first newline
        size_t data_offset = 0;
        if (opts.skip_header) {
            for (size_t i = 0; i < file_size_host; ++i) {
                if (h_pinned[i] == '\n') { data_offset = i + 1; break; }
            }
        }
        size_t payload_size = file_size_host - data_offset;
        if (payload_size == 0) {
            return {};
        }
        if (payload_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "File too large for CUB scan limit (INT_MAX bytes). "
                "Chunked streaming mode will be added in a future update.");
        }

        // Try multi-stream path first for large payloads
        if (!opts.fused_filter && TUNE_NUM_STREAMS >= 2) {
            auto ms_result = parse_multistream(h_pinned, data_offset, payload_size, opts);
            if (ms_result.num_rows > 0) {
                ms_result.parse_time_ms = std::chrono::duration<double, std::milli>(
                    clock::now() - t0).count();
                return ms_result;
            }
        }

        ParsedCSV result;
        result.num_rows = 0;
        result.num_filtered_rows = 0;
        result.parse_time_ms = 0.0;
        result.h2d_time_ms = 0.0;

        // Ensure reusable buffers are large enough
        ensure_data_(payload_size);
        ensure_marks_(payload_size);
        ensure_cumsum_(payload_size);

        // 2. Async H2D copy on stream
        auto t1 = clock::now();
        CUDA_CHECK(cudaMemcpyAsync(d_data_, h_pinned + data_offset, payload_size,
                                   cudaMemcpyHostToDevice, stream_));

        // 3. Mark newlines on stream
        {
            int threads = TUNE_MARK_THREADS_PER_BLOCK;
            int blocks = static_cast<int>((payload_size + threads - 1) / threads);
            mark_newlines<<<blocks, threads, 0, stream_>>>(d_data_, payload_size, d_marks_);
            CUDA_CHECK(cudaGetLastError());
        }

        // 4. Inclusive scan to get newline indices on stream
        size_t temp_bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_marks_, d_cumsum_,
                                      static_cast<int>(payload_size));
        ensure_temp_(temp_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_, temp_bytes, d_marks_, d_cumsum_,
                                      static_cast<int>(payload_size), stream_);
        CUDA_CHECK(cudaGetLastError());

        // 5. Get total newlines (async D2H, then sync stream)
        uint32_t h_total_nl = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_total_nl, d_cumsum_ + payload_size - 1,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        auto t2 = clock::now(); // H2D + row discovery complete
        result.h2d_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (h_total_nl == 0) {
            return {};
        }

        ensure_newlines_(h_total_nl + 1);
        int64_t* d_newlines = d_newlines_;
        {
            int threads = TUNE_COMPACT_THREADS_PER_BLOCK;
            int blocks = static_cast<int>((payload_size + threads - 1) / threads);
            compact_newlines<<<blocks, threads, 0, stream_>>>(d_data_, payload_size,
                                                                d_cumsum_, d_newlines);
            CUDA_CHECK(cudaGetLastError());
        }

        // If last char is not newline, append file_size as final boundary
        bool ends_with_newline = false;
        if (payload_size > 0) {
            char lastc = h_pinned[data_offset + payload_size - 1];
            ends_with_newline = (lastc == '\n');
        }
        int64_t num_rows = static_cast<int64_t>(h_total_nl);
        if (!ends_with_newline) {
            int64_t h_file_size = static_cast<int64_t>(payload_size);
            CUDA_CHECK(cudaMemcpyAsync(d_newlines + h_total_nl, &h_file_size,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, stream_));
            num_rows += 1;
        }

        // 6. Prepare column outputs
        int num_cols = static_cast<int>(opts.schema.size());
        int num_proj = opts.projection.empty()
                           ? num_cols
                           : static_cast<int>(opts.projection.size());

        std::vector<int> proj_indices;
        if (opts.projection.empty()) {
            for (int i = 0; i < num_cols; ++i) proj_indices.push_back(i);
        } else {
            proj_indices = opts.projection;
        }

        std::vector<ColumnType> proj_types;
        for (int idx : proj_indices) proj_types.push_back(opts.schema[idx].type);

        result.num_rows = static_cast<size_t>(num_rows);

        // Warp kernel config: 128 threads = 4 warps/block gives good occupancy
        constexpr int WARP_THREADS = 32;
        constexpr int THREADS_PER_BLOCK = TUNE_THREADS_PER_BLOCK;
        int warps_per_block = THREADS_PER_BLOCK / WARP_THREADS;

        if (opts.fused_filter) {
            // ---- FUSED PATH ----
            ensure_mask_(num_rows);
            ensure_scanned_(num_rows);
            uint8_t* d_mask = d_mask_;
            int* d_scanned = d_scanned_;

            // 6a. evaluate predicate (warp per row)
            {
                int blocks = static_cast<int>((num_rows + warps_per_block - 1) / warps_per_block);
                filter_mask_kernel_warp<<<blocks, THREADS_PER_BLOCK, 0, stream_>>>(
                    d_data_, d_newlines, num_rows, ends_with_newline,
                    static_cast<int64_t>(payload_size), opts.delimiter,
                    opts.predicate_col, opts.predicate_col_type, opts.predicate_op,
                    opts.predicate_int64, opts.predicate_float64, d_mask);
                CUDA_CHECK(cudaGetLastError());
            }

            // 6b. exclusive scan mask on stream
            cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_mask, d_scanned,
                                          static_cast<int>(num_rows));
            ensure_temp_(temp_bytes);
            cub::DeviceScan::ExclusiveSum(d_temp_, temp_bytes, d_mask, d_scanned,
                                          static_cast<int>(num_rows), stream_);
            CUDA_CHECK(cudaGetLastError());

            uint8_t h_last_mask = 0;
            CUDA_CHECK(cudaMemcpyAsync(&h_last_mask, d_mask + num_rows - 1,
                                       sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_));
            int h_last_scanned = 0;
            CUDA_CHECK(cudaMemcpyAsync(&h_last_scanned, d_scanned + num_rows - 1,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            int64_t survivors = static_cast<int64_t>(h_last_scanned + h_last_mask);
            result.num_filtered_rows = static_cast<size_t>(survivors);

            if (survivors > 0) {
                std::vector<void*> h_proj_data(num_proj);
                std::vector<uint8_t*> h_proj_nulls(num_proj);
                for (int pj = 0; pj < num_proj; ++pj) {
                    size_t bsize = survivors * type_size(proj_types[pj]);
                    void* dp = nullptr;
                    uint8_t* np = nullptr;
                    CUDA_CHECK(cudaMalloc(&dp, bsize));
                    CUDA_CHECK(cudaMalloc(&np, survivors * sizeof(uint8_t)));
                    h_proj_data[pj] = dp;
                    h_proj_nulls[pj] = np;
                }

                void** d_proj_data = nullptr;
                uint8_t** d_proj_nulls = nullptr;
                int* d_proj_indices = nullptr;
                ColumnType* d_proj_types = nullptr;
                CUDA_CHECK(cudaMalloc(&d_proj_data, num_proj * sizeof(void*)));
                CUDA_CHECK(cudaMalloc(&d_proj_nulls, num_proj * sizeof(uint8_t*)));
                CUDA_CHECK(cudaMalloc(&d_proj_indices, num_proj * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_proj_types, num_proj * sizeof(ColumnType)));
                CUDA_CHECK(cudaMemcpyAsync(d_proj_data, h_proj_data.data(),
                                           num_proj * sizeof(void*), cudaMemcpyHostToDevice, stream_));
                CUDA_CHECK(cudaMemcpyAsync(d_proj_nulls, h_proj_nulls.data(),
                                           num_proj * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream_));
                CUDA_CHECK(cudaMemcpyAsync(d_proj_indices, proj_indices.data(),
                                           num_proj * sizeof(int), cudaMemcpyHostToDevice, stream_));
                CUDA_CHECK(cudaMemcpyAsync(d_proj_types, proj_types.data(),
                                           num_proj * sizeof(ColumnType), cudaMemcpyHostToDevice, stream_));

                {
                    int blocks = static_cast<int>((num_rows + warps_per_block - 1) / warps_per_block);
                    parse_filtered_rows_kernel_warp<<<blocks, THREADS_PER_BLOCK, 0, stream_>>>(
                        d_data_, d_newlines, num_rows, ends_with_newline,
                        static_cast<int64_t>(payload_size), opts.delimiter,
                        num_proj, d_proj_indices, d_proj_types, d_mask,
                        d_scanned, d_proj_data, d_proj_nulls);
                    CUDA_CHECK(cudaGetLastError());
                }

                result.columns.reserve(num_proj);
                for (int pj = 0; pj < num_proj; ++pj) {
                    DeviceColumn col;
                    col.type = proj_types[pj];
                    col.num_rows = static_cast<size_t>(survivors);
                    col.data_bytes = survivors * type_size(col.type);
                    col.data = std::unique_ptr<void, void (*)(void*)>(
                        h_proj_data[pj], +[](void* p) { cudaFree(p); });
                    col.null_mask = std::unique_ptr<uint8_t, void (*)(void*)>(
                        h_proj_nulls[pj], +[](void* p) { cudaFree(p); });
                    result.columns.push_back(std::move(col));
                }

                cudaFree(d_proj_data);
                cudaFree(d_proj_nulls);
                cudaFree(d_proj_indices);
                cudaFree(d_proj_types);
            }

        } else {
            // ---- NON-FUSED PATH ----
            std::vector<void*> h_col_data(num_cols);
            std::vector<uint8_t*> h_col_nulls(num_cols);
            for (int c = 0; c < num_cols; ++c) {
                size_t bsize = static_cast<size_t>(num_rows) * type_size(opts.schema[c].type);
                void* dp = nullptr;
                uint8_t* np = nullptr;
                CUDA_CHECK(cudaMalloc(&dp, bsize));
                CUDA_CHECK(cudaMalloc(&np, num_rows * sizeof(uint8_t)));
                h_col_data[c] = dp;
                h_col_nulls[c] = np;
            }

            void** d_col_data = nullptr;
            uint8_t** d_col_nulls = nullptr;
            ColumnType* d_types = nullptr;
            CUDA_CHECK(cudaMalloc(&d_col_data, num_cols * sizeof(void*)));
            CUDA_CHECK(cudaMalloc(&d_col_nulls, num_cols * sizeof(uint8_t*)));
            CUDA_CHECK(cudaMalloc(&d_types, num_cols * sizeof(ColumnType)));
            CUDA_CHECK(cudaMemcpyAsync(d_col_data, h_col_data.data(),
                                       num_cols * sizeof(void*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_col_nulls, h_col_nulls.data(),
                                       num_cols * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream_));
            std::vector<ColumnType> host_types;
            host_types.reserve(num_cols);
            for (const auto& spec : opts.schema) host_types.push_back(spec.type);
            CUDA_CHECK(cudaMemcpyAsync(d_types, host_types.data(),
                                       num_cols * sizeof(ColumnType), cudaMemcpyHostToDevice, stream_));

            {
                int blocks = static_cast<int>((num_rows + warps_per_block - 1) / warps_per_block);
                parse_rows_kernel_warp<<<blocks, THREADS_PER_BLOCK, 0, stream_>>>(
                    d_data_, d_newlines, num_rows, ends_with_newline,
                    static_cast<int64_t>(payload_size), opts.delimiter,
                    num_cols, d_types, d_col_data, d_col_nulls);
                CUDA_CHECK(cudaGetLastError());
            }

            result.columns.reserve(num_cols);
            for (int c = 0; c < num_cols; ++c) {
                DeviceColumn col;
                col.type = opts.schema[c].type;
                col.num_rows = static_cast<size_t>(num_rows);
                col.data_bytes = col.num_rows * type_size(col.type);
                col.data = std::unique_ptr<void, void (*)(void*)>(
                    h_col_data[c], +[](void* p) { cudaFree(p); });
                col.null_mask = std::unique_ptr<uint8_t, void (*)(void*)>(
                    h_col_nulls[c], +[](void* p) { cudaFree(p); });
                result.columns.push_back(std::move(col));
            }

            cudaFree(d_col_data);
            cudaFree(d_col_nulls);
            cudaFree(d_types);
        }

        // Cleanup per-call allocations

        // Final sync to ensure all stream work is done
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        auto t3 = clock::now();
        result.parse_time_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();
        return result;
    }
};

CSVParser::CSVParser() : pimpl_(std::make_unique<Impl>()) {}
CSVParser::~CSVParser() = default;

ParsedCSV CSVParser::parse(const std::string& filepath,
                           const ParseOptions& opts) {
    return pimpl_->parse(filepath, opts);
}

} // namespace gpu_csv
