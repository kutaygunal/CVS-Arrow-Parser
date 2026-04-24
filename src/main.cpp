#include "gpu_csv_parser.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace gpu_csv;

int main(int argc, char** argv) {
    std::string filepath = (argc > 1) ? argv[1] : "data/sample.csv";

    ParseOptions opts;
    opts.schema = {
        {ColumnType::INT64,   "id"},
        {ColumnType::FLOAT64, "value"},
        {ColumnType::INT64,   "score"}
    };
    opts.skip_header = true;

    std::cout << "=== GPU CSV Parser Demo ===\n";
    std::cout << "File: " << filepath << "\n\n";

    CSVParser parser;

    // ------------------------------------------------------------------
    // 1) Full parse (all columns, no filter)
    // ------------------------------------------------------------------
    std::cout << "[1] Full parse (all columns)\n";
    auto full = parser.parse(filepath, opts);
    std::cout << "  Rows parsed:      " << full.num_rows << "\n";
    std::cout << "  Columns:          " << full.columns.size() << "\n";
    std::cout << "  H2D time:         " << std::fixed << std::setprecision(3)
              << full.h2d_time_ms << " ms\n";
    std::cout << "  Parse time:       " << full.parse_time_ms << " ms\n";

    if (!full.columns.empty() && full.num_rows > 0) {
        size_t n = std::min<size_t>(5, full.num_rows);
        auto buf = full.columns[0].copy_rows_to_host(0, n);
        int64_t* vals = reinterpret_cast<int64_t*>(buf.data());
        std::cout << "  First 5 ids:      ";
        for (size_t i = 0; i < n; ++i) std::cout << vals[i] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";

    // ------------------------------------------------------------------
    // 2) Fused filter parse (score > 50, project only id & value)
    // ------------------------------------------------------------------
    std::cout << "[2] Fused filter parse (score > 50, project id & value)\n";
    ParseOptions fused = opts;
    fused.fused_filter = true;
    fused.predicate_col = 2;               // score
    fused.predicate_col_type = ColumnType::INT64;
    fused.predicate_op = PredicateOp::GT;
    fused.predicate_int64 = 50;
    fused.projection = {0, 1};

    auto filtered = parser.parse(filepath, fused);
    std::cout << "  Total scanned:    " << filtered.num_rows << "\n";
    std::cout << "  Survivors:        " << filtered.num_filtered_rows << "\n";
    std::cout << "  H2D time:         " << filtered.h2d_time_ms << " ms\n";
    std::cout << "  Fused parse time: " << filtered.parse_time_ms << " ms\n";

    if (!filtered.columns.empty() && filtered.num_filtered_rows > 0) {
        size_t n = std::min<size_t>(5, filtered.num_filtered_rows);
        auto buf = filtered.columns[0].copy_rows_to_host(0, n);
        int64_t* vals = reinterpret_cast<int64_t*>(buf.data());
        std::cout << "  First 5 filtered: ";
        for (size_t i = 0; i < n; ++i) std::cout << vals[i] << " ";
        std::cout << "\n";
    }

    return 0;
}
