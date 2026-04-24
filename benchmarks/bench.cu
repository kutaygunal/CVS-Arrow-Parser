#include "gpu_csv_parser.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>

using namespace gpu_csv;

static double file_size_mb(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f ? static_cast<double>(f.tellg()) / (1024.0 * 1024.0) : 0.0;
}

int main(int argc, char** argv) {
    std::string filepath = (argc > 1) ? argv[1] : "data/sample.csv";
    int warmup = 2;
    int runs   = 10;

    ParseOptions opts;
    opts.schema = {
        {"id", ColumnType::INT64},
        {"value", ColumnType::FLOAT64},
        {"score", ColumnType::INT64}
    };
    opts.skip_header = true;

    CSVParser parser;
    double file_mb = file_size_mb(filepath);

    std::cout << "=== Benchmark: Full Parse ===\n";
    std::cout << "File size: " << std::fixed << std::setprecision(2) << file_mb << " MiB\n";

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        volatile auto tmp = parser.parse(filepath, opts);
        (void)tmp;
    }

    double total_ms = 0.0;
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = parser.parse(filepath, opts);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        std::cout << "  Run " << (i+1) << ": " << ms << " ms | throughput: "
                  << (file_mb / (ms / 1000.0)) << " MiB/s\n";
    }
    std::cout << "  Avg:   " << (total_ms / runs) << " ms | throughput: "
              << (file_mb / ((total_ms / runs) / 1000.0)) << " MiB/s\n\n";

    // Fused filter benchmark
    std::cout << "=== Benchmark: Fused Filter (score > 50) ===\n";
    ParseOptions fused = opts;
    fused.fused_filter = true;
    fused.predicate_col = 2;
    fused.predicate_col_type = ColumnType::INT64;
    fused.predicate_op = PredicateOp::GT;
    fused.predicate_int64 = 50;
    fused.projection = {0, 1};

    for (int i = 0; i < warmup; ++i) {
        volatile auto tmp = parser.parse(filepath, fused);
        (void)tmp;
    }

    total_ms = 0.0;
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = parser.parse(filepath, fused);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        std::cout << "  Run " << (i+1) << ": " << ms << " ms | survivors: " << result.num_filtered_rows << "\n";
    }
    std::cout << "  Avg:   " << (total_ms / runs) << " ms\n";
    return 0;
}
