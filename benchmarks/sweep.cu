#include "gpu_csv_parser.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>

using namespace gpu_csv;

struct BenchResult {
    std::string file;
    double size_mb = 0.0;
    size_t rows = 0;
    double full_ms = 0.0;
    double full_mib_s = 0.0;
    double fused_ms = 0.0;
    double fused_mib_s = 0.0;
    size_t survivors = 0;
    double selectivity = 0.0;
};

static double file_size_mb(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f ? static_cast<double>(f.tellg()) / (1024.0 * 1024.0) : 0.0;
}

static BenchResult benchmark_file(const std::string& filepath,
                                   const ParseOptions& base_opts,
                                   const ParseOptions& fused_opts,
                                   int warmup, int runs) {
    CSVParser parser;

    BenchResult r;
    r.file = filepath;
    r.size_mb = file_size_mb(filepath);

    // Full parse warmup + runs
    for (int i = 0; i < warmup; ++i) {
        volatile auto tmp = parser.parse(filepath, base_opts);
        (void)tmp;
    }
    double total_ms = 0.0;
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = parser.parse(filepath, base_opts);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (i == 0) r.rows = result.num_rows;
    }
    r.full_ms = total_ms / runs;
    r.full_mib_s = r.size_mb / (r.full_ms / 1000.0);

    // Fused parse warmup + runs
    for (int i = 0; i < warmup; ++i) {
        volatile auto tmp = parser.parse(filepath, fused_opts);
        (void)tmp;
    }
    total_ms = 0.0;
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = parser.parse(filepath, fused_opts);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (i == 0) {
            r.survivors = result.num_filtered_rows;
            r.selectivity = result.num_rows > 0
                              ? static_cast<double>(result.num_filtered_rows) / result.num_rows
                              : 0.0;
        }
    }
    r.fused_ms = total_ms / runs;
    r.fused_mib_s = r.size_mb / (r.fused_ms / 1000.0);

    return r;
}

int main(int argc, char** argv) {
    std::vector<std::string> files = {
        "data/sample.csv",
        "data/1M.csv",
        "data/5M.csv",
        "data/10M.csv"
    };

    // Allow override from CLI
    if (argc > 1) {
        files.clear();
        for (int i = 1; i < argc; ++i) files.push_back(argv[i]);
    }

    ParseOptions base;
    base.schema = {
        {ColumnType::INT64,   "id"},
        {ColumnType::FLOAT64, "value"},
        {ColumnType::INT64,   "score"}
    };
    base.skip_header = true;

    ParseOptions fused = base;
    fused.fused_filter = true;
    fused.predicate_col = 2;
    fused.predicate_col_type = ColumnType::INT64;
    fused.predicate_op = PredicateOp::GT;
    fused.predicate_int64 = 50;
    fused.projection = {0, 1};

    const int warmup = 2;
    const int runs   = 10;

    std::cout << "=== GPU CSV Parser: Multi-Size Benchmark ===\n";
    std::cout << "Warmup: " << warmup << " | Runs: " << runs << "\n\n";

    std::cout << std::left
              << std::setw(18) << "File"
              << std::setw(10) << "Size(MiB)"
              << std::setw(12) << "Rows"
              << std::setw(10) << "Full(ms)"
              << std::setw(12) << "Full(MiB/s)"
              << std::setw(10) << "Fused(ms)"
              << std::setw(14) << "Fused(MiB/s)"
              << std::setw(12) << "Survivors"
              << std::setw(10) << "Selectivity"
              << "\n";
    std::cout << std::string(110, '-') << "\n";

    std::vector<BenchResult> results;
    for (const auto& f : files) {
        results.push_back(benchmark_file(f, base, fused, warmup, runs));
    }

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(18) << r.file
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.size_mb
                  << std::setw(12) << r.rows
                  << std::setw(10) << std::setprecision(2) << r.full_ms
                  << std::setw(12) << std::setprecision(1) << r.full_mib_s
                  << std::setw(10) << std::setprecision(2) << r.fused_ms
                  << std::setw(14) << std::setprecision(1) << r.fused_mib_s
                  << std::setw(12) << r.survivors
                  << std::setw(10) << std::setprecision(2) << r.selectivity
                  << "\n";
    }

    std::cout << "\nNotes:\n"
              << "- Throughput measures full end-to-end time (file read + H2D + parse + sync).\n"
              << "- Fused filter includes predicate evaluation + projection; fewer bytes materialized.\n"
              << "- MiB/s = 1,048,576 bytes per second.\n";
    return 0;
}
