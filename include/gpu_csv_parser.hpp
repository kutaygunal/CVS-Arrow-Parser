#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace gpu_csv {

enum class ColumnType { INT64, FLOAT64 };

enum class PredicateOp { GT, GE, LT, LE, EQ };

struct ColumnSpec {
    ColumnType type;
    std::string name; // informational only
};

// Device-side column buffer. Memory lives on the GPU.
struct DeviceColumn {
    ColumnType type;
    std::size_t num_rows = 0;

    // Fixed-width data buffer
    std::unique_ptr<void, void (*)(void*)> data{
        nullptr, +[](void* p) { cudaFree(p); }};

    // Null mask: one bit per row. null_count cached for Arrow compatibility.
    std::unique_ptr<uint8_t, void (*)(void*)> null_mask{
        nullptr, +[](void* p) { cudaFree(p); }};
    std::size_t null_count = 0;

    // Bytes allocated in data buffer
    std::size_t data_bytes = 0;

    // Copy a window of rows back to host for verification (used in CLI/tests)
    std::vector<uint8_t> copy_rows_to_host(std::size_t start,
                                           std::size_t count) const;
};

struct ParseOptions {
    std::vector<ColumnSpec> schema;
    char delimiter = ',';
    char newline = '\n';
    bool skip_header = true;

    // Fused filter configuration
    bool fused_filter = false;
    int predicate_col = -1;          // column index to filter on
    ColumnType predicate_col_type = ColumnType::INT64;
    PredicateOp predicate_op = PredicateOp::GT;
    int64_t predicate_int64 = 0;
    double predicate_float64 = 0.0;
    // Projection: if non-empty, only these columns are materialized.
    // Empty means all schema columns.
    std::vector<int> projection;
};

struct ParsedCSV {
    std::vector<DeviceColumn> columns;
    std::size_t num_rows = 0;
    std::size_t num_filtered_rows = 0; // if fused_filter was used
    double parse_time_ms = 0.0;
    double h2d_time_ms = 0.0;
};

class CSVParser {
public:
    CSVParser();
    ~CSVParser();

    ParsedCSV parse(const std::string& filepath, const ParseOptions& opts);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace gpu_csv
