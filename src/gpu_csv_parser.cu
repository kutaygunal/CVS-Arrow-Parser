#include "gpu_csv_parser.hpp"

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
// Device parsing primitives (no locale, no exponent, simple but fast)
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
// Row discovery kernels
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
// Full parse kernel (no filter)
// ---------------------------------------------------------------------------
__global__ void parse_rows_kernel(const char* data,
                                  const int64_t* newlines,
                                  int64_t num_rows,
                                  bool ends_with_newline,
                                  int64_t file_size,
                                  char delimiter,
                                  int num_cols,
                                  const ColumnType* types,
                                  void** col_data,
                                  uint8_t** col_nulls) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int64_t start = (row == 0) ? 0 : (newlines[row - 1] + 1);
    int64_t end   = newlines[row];
    if (!ends_with_newline && row == num_rows - 1) end = file_size;

    // trim trailing \r if present
    if (end > start && data[end - 1] == '\r') --end;

    const char* fb = data + start;
    int field = 0;

    for (int64_t p = start; p < end && field < num_cols; ++p) {
        if (data[p] == delimiter) {
            const char* fe = data + p;
            bool ok = false;
            if (fb < fe) {
                switch (types[field]) {
                    case ColumnType::INT64: {
                        int64_t v = dev_atol(fb, fe, &ok);
                        if (ok) reinterpret_cast<int64_t*>(col_data[field])[row] = v;
                        break;
                    }
                    case ColumnType::FLOAT64: {
                        double v = dev_atof(fb, fe, &ok);
                        if (ok) reinterpret_cast<double*>(col_data[field])[row] = v;
                        break;
                    }
                }
            }
            col_nulls[field][row] = ok ? 1 : 0;
            ++field;
            fb = data + p + 1;
        }
    }

    // Last field
    if (field < num_cols) {
        const char* fe = data + end;
        bool ok = false;
        if (fb < fe) {
            switch (types[field]) {
                case ColumnType::INT64: {
                    int64_t v = dev_atol(fb, fe, &ok);
                    if (ok) reinterpret_cast<int64_t*>(col_data[field])[row] = v;
                    break;
                }
                case ColumnType::FLOAT64: {
                    double v = dev_atof(fb, fe, &ok);
                    if (ok) reinterpret_cast<double*>(col_data[field])[row] = v;
                    break;
                }
            }
        }
        col_nulls[field][row] = ok ? 1 : 0;
    }

    // Null out any missing columns (malformed row shorter than schema)
    for (int f = field + 1; f < num_cols; ++f) {
        col_nulls[f][row] = 0;
    }
}

// ---------------------------------------------------------------------------
// Fused filter: Stage 1 — evaluate predicate per row
// ---------------------------------------------------------------------------
__global__ void filter_mask_kernel(const char* data,
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
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int64_t start = (row == 0) ? 0 : (newlines[row - 1] + 1);
    int64_t end   = newlines[row];
    if (!ends_with_newline && row == num_rows - 1) end = file_size;
    if (end > start && data[end - 1] == '\r') --end;

    // Scan to predicate column
    const char* fb = data + start;
    int field = 0;
    bool found = false;
    bool ok = false;
    int64_t vi = 0;
    double vd = 0.0;

    for (int64_t p = start; p < end && field <= predicate_col; ++p) {
        if (data[p] == delimiter || p == end - 1) {
            const char* fe = (data[p] == delimiter) ? (data + p) : (data + p + 1);
            if (field == predicate_col) {
                if (fb < fe) {
                    if (predicate_type == ColumnType::INT64) {
                        vi = dev_atol(fb, fe, &ok);
                    } else {
                        vd = dev_atof(fb, fe, &ok);
                    }
                }
                found = true; break;
            }
            ++field;
            fb = data + p + 1;
        }
    }

    bool pass = false;
    if (found && ok) {
        if (predicate_type == ColumnType::INT64)
            pass = eval_predicate_int64(vi, op, pred_i64);
        else
            pass = eval_predicate_float64(vd, op, pred_f64);
    }
    mask[row] = pass ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Fused filter: Stage 2 — parse projected columns for surviving rows only
// ---------------------------------------------------------------------------
__global__ void parse_filtered_rows_kernel(const char* data,
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
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    if (!mask[row]) return;

    int out_row = scanned[row];
    int64_t start = (row == 0) ? 0 : (newlines[row - 1] + 1);
    int64_t end   = newlines[row];
    if (!ends_with_newline && row == num_rows - 1) end = file_size;
    if (end > start && data[end - 1] == '\r') --end;

    // Pre-compute all field boundaries for this row (small, stack alloc)
    const int MAX_COLS = 32;
    const char* f_begin[MAX_COLS];
    const char* f_end[MAX_COLS];
    int fcount = 0;
    const char* fb = data + start;
    for (int64_t p = start; p < end && fcount < MAX_COLS; ++p) {
        if (data[p] == delimiter) {
            f_begin[fcount] = fb;
            f_end[fcount]   = data + p;
            ++fcount;
            fb = data + p + 1;
        }
    }
    if (fcount < MAX_COLS) {
        f_begin[fcount] = fb;
        f_end[fcount]   = data + end;
        ++fcount;
    }

    for (int pj = 0; pj < num_proj_cols; ++pj) {
        int col = proj_indices[pj];
        if (col >= fcount) {
            col_nulls[pj][out_row] = 0;
            continue;
        }
        bool ok = false;
        const char* b = f_begin[col];
        const char* e = f_end[col];
        if (b < e) {
            switch (types[pj]) {
                case ColumnType::INT64: {
                    int64_t v = dev_atol(b, e, &ok);
                    if (ok) reinterpret_cast<int64_t*>(col_data[pj])[out_row] = v;
                    break;
                }
                case ColumnType::FLOAT64: {
                    double v = dev_atof(b, e, &ok);
                    if (ok) reinterpret_cast<double*>(col_data[pj])[out_row] = v;
                    break;
                }
            }
        }
        col_nulls[pj][out_row] = ok ? 1 : 0;
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

    // Reusable device buffers (resized on demand)
    char* d_data_ = nullptr;       size_t d_data_cap_ = 0;
    uint8_t* d_marks_ = nullptr;  size_t d_marks_cap_ = 0;
    uint32_t* d_cumsum_ = nullptr; size_t d_cumsum_cap_ = 0;
    void* d_temp_ = nullptr;      size_t d_temp_cap_ = 0;

    void ensure_data_(size_t n)   { ensure_buffer((void**)&d_data_,   &d_data_cap_,   n); }
    void ensure_marks_(size_t n)  { ensure_buffer((void**)&d_marks_,  &d_marks_cap_,  n); }
    void ensure_cumsum_(size_t n) { ensure_buffer((void**)&d_cumsum_, &d_cumsum_cap_, n * sizeof(uint32_t)); }
    void ensure_temp_(size_t n)   { ensure_buffer((void**)&d_temp_,  &d_temp_cap_,   n); }

public:
    Impl() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    ~Impl() {
        if (d_data_)   cudaFree(d_data_);
        if (d_marks_)  cudaFree(d_marks_);
        if (d_cumsum_) cudaFree(d_cumsum_);
        if (d_temp_)   cudaFree(d_temp_);
        if (stream_)   cudaStreamDestroy(stream_);
    }

    ParsedCSV parse(const std::string& filepath, const ParseOptions& opts) {
        using clock = std::chrono::high_resolution_clock;
        auto t0 = clock::now();

        // 1. Read file into pinned host memory
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        if (!ifs) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        auto file_size_host = static_cast<size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);

        char* h_pinned = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_pinned, file_size_host));
        ifs.read(h_pinned, file_size_host);
        ifs.close();

        // Skip header: advance pointer/size after first newline
        size_t data_offset = 0;
        if (opts.skip_header) {
            for (size_t i = 0; i < file_size_host; ++i) {
                if (h_pinned[i] == '\n') { data_offset = i + 1; break; }
            }
        }
        size_t payload_size = file_size_host - data_offset;
        if (payload_size == 0) {
            cudaFreeHost(h_pinned);
            return {};
        }
        if (payload_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            cudaFreeHost(h_pinned);
            throw std::runtime_error(
                "File too large for CUB scan limit (INT_MAX bytes). "
                "Chunked streaming mode will be added in a future update.");
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
            int threads = 256;
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
            cudaFreeHost(h_pinned);
            return {};
        }

        int64_t* d_newlines = nullptr;
        CUDA_CHECK(cudaMalloc(&d_newlines, (h_total_nl + 1) * sizeof(int64_t)));
        {
            int threads = 256;
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

        if (opts.fused_filter) {
            // ---- FUSED PATH ----
            uint8_t* d_mask = nullptr;
            int* d_scanned = nullptr;
            CUDA_CHECK(cudaMalloc(&d_mask, num_rows * sizeof(uint8_t)));
            CUDA_CHECK(cudaMalloc(&d_scanned, num_rows * sizeof(int)));

            // 6a. evaluate predicate
            {
                int threads = 256;
                int blocks = static_cast<int>((num_rows + threads - 1) / threads);
                filter_mask_kernel<<<blocks, threads, 0, stream_>>>(
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
                    int threads = 256;
                    int blocks = static_cast<int>((num_rows + threads - 1) / threads);
                    parse_filtered_rows_kernel<<<blocks, threads, 0, stream_>>>(
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

            cudaFree(d_mask);
            cudaFree(d_scanned);
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
                int threads = 256;
                int blocks = static_cast<int>((num_rows + threads - 1) / threads);
                parse_rows_kernel<<<blocks, threads, 0, stream_>>>(
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
        cudaFree(d_newlines);
        cudaFreeHost(h_pinned);

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
