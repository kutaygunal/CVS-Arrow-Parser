cmake_minimum_required(VERSION 3.20)
project(csv_arrow_parser LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Main parser library
add_library(gpu_csv_parser STATIC
    src/gpu_csv_parser.cu
)
target_include_directories(gpu_csv_parser PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(gpu_csv_parser PUBLIC CUDA::cudart)
target_compile_options(gpu_csv_parser PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# CLI executable
add_executable(csv_parser_cli src/main.cpp)
target_link_libraries(csv_parser_cli PRIVATE gpu_csv_parser)

# Benchmark executable
add_executable(csv_parser_bench benchmarks/bench.cu)
target_link_libraries(csv_parser_bench PRIVATE gpu_csv_parser)

# Sweep benchmark executable
add_executable(csv_parser_sweep benchmarks/sweep.cu)
target_link_libraries(csv_parser_sweep PRIVATE gpu_csv_parser)
