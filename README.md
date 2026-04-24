# GPU-Accelerated CSV → Columnar Parser with Fused Filter Pushdown

A mid-size CUDA/C++ project demonstrating **parallel CSV parsing on the GPU**, direct emission of typed columnar buffers (int64 / float64), and a **fused filter pushdown** that evaluates predicates while parsing to avoid materializing rows that will be discarded.

### Algorithm Evolution — Before vs After

| Optimisation | Before | After (this commit) | Delta |
|-------------|--------|-------------------|-------|
| **Host memory** | `std::vector<char>` (pageable) | `cudaMallocHost` (pinned) | ~2× faster H2D |
| **GPU scheduling** | Synchronous `cudaMemcpy` + default stream | Async `cudaMemcpyAsync` + persistent `cudaStream_t` | Overlap potential, lower CPU sync stalls |
| **Temp buffer allocation** | `cudaMalloc`/`cudaFree` per call (~8 allocs) | Pooled buffers resized on demand | -28 % end-to-end latency |
| **Throughput (10 M rows)** | **1 082 MiB/s** | **1 368 MiB/s** | **+26 %** |
| **Throughput (1 M rows)** | **996 MiB/s** | **1 303 MiB/s** | **+31 %** |

> Benchmarked on **RTX 4090 / PCIe 4.0 x16 / CUDA 12.6 / MSVC 2022**.

## What this demonstrates

* **GPU-native CSV tokenization** — one thread per row, field boundaries discovered in shared registers.
* **Modern C++ design** — RAII device buffers, strict separation of host API / device kernels, and error checking macros.
* **CUB integration** — `DeviceScan::InclusiveSum` and `ExclusiveSum` to compact newline positions and filtered row masks in parallel.
* **Pinned host memory + CUDA streams** — file is read directly into page-locked (`cudaMallocHost`) memory; H2D copies and all kernels are async on a persistent stream. Temporary device buffers (`marks`, `cumsum`, `CUB temp`) are pooled and reused across invocations, eliminating per-call alloc overhead.
* **Predicate pushdown** — two-stage fused pipeline:
  1. Parse only the predicate column + evaluate (produces a boolean mask).
  2. CUB scan mask to get output positions.
  3. Second kernel materializes **only** projected columns for surviving rows.
* **Arrow-minded layout** — null bitmaps per column, type-tagged buffers, ready to be zero-copy wrapped into `arrow::Array` extensions.

## Architecture

```
Host: read file → pinned page-locked buffer → skip header
       │
       ▼
Device (async on persistent CUDA stream):
  cudaMemcpyAsync H2D
       │
       ▼
  [mark_newlines]  ──▶  [cub::DeviceScan::InclusiveSum]
       │                        │
       ▼                        ▼
  [compact_newlines]     total_newlines
       │
       ▼
  ┌─────────────────┐
  │ Full Parse Path │   one kernel per row → all columns
  └─────────────────┘
       │
  ┌─────────────────┐
  │ Fused Filter    │   Stage 1: filter_mask_kernel (predicate only)
  │ Pushdown Path   │   Stage 2: cub::ExclusiveSum
  └─────────────────┘   Stage 3: parse_filtered_rows_kernel (project)
```

### Memory-management strategy

| Buffer | Ownership | Lifetime |
|--------|-----------|----------|
| `d_data_` (file payload) | `Impl` | Created on first parse; resized on demand; reused across calls |
| `d_marks_`, `d_cumsum_`, `d_temp_` | `Impl` | Pooled alongside `d_data_`; grows on demand |
| `d_newlines` | Per-call | `cudaMalloc`/`cudaFree` each invocation |
| Output column buffers | Per-call | Owned by returned `DeviceColumn` via `unique_ptr` |

This eliminates ~8 `cudaMalloc`/`cudaFree` pairs per call, which is a major latency win for repeated benchmarks and matches how cuDF/Velox manage internal device workspaces.

## Directory Layout

```
01-csv-arrow-parser/
├── CMakeLists.txt
├── README.md
├── data/
│   ├── sample.csv               # tiny test file (5 rows)
│   ├── 1M.csv                   # generated: 1 million rows
│   ├── 5M.csv                   # generated: 5 million rows
│   └── 10M.csv                  # generated: 10 million rows
├── scripts/
│   └── generate_csv.py          # generate large benchmark files
├── include/
│   └── gpu_csv_parser.hpp       # public API
├── src/
│   ├── gpu_csv_parser.cu        # CUDA kernels + Impl
│   └── main.cpp                 # CLI demo
└── benchmarks/
    ├── bench.cu                 # single-file repeated-run benchmark
    └── sweep.cu                 # multi-size sweep benchmark
```

## Build Requirements

* CMake ≥ 3.20
* CUDA Toolkit ≥ 11.0 (CUB is bundled)
* A CUDA-capable GPU with compute capability ≥ 7.5
* C++20 host compiler (MSVC 2019+, GCC 10+, Clang 14+)

### Windows Build (Visual Studio / Ninja)

Tested with **CUDA 12.6 + Visual Studio 2022 (MSVC 14.43)**. If `nvcc` is not on PATH, pass the full path:

```powershell
cd C:\Users\kutay\Desktop\Projects\01-csv-arrow-parser

# Clean configure (removes stale Ninja/VS cache if switching generators)
rm -rf build

# Configure with Visual Studio generator (most reliable on Windows)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"

# Build
cmake --build build --config Release --parallel 4
```

### Linux Build

```bash
cd ~/Desktop/Projects/01-csv-arrow-parser
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Running

### 1. Generate a large synthetic CSV

```powershell
cd C:\Users\kutay\Desktop\Projects\01-csv-arrow-parser
python scripts/generate_csv.py --rows 10000000 --out data/10M.csv
```

### 2. Run the CLI demo

```powershell
.\build\Release\csv_parser_cli.exe data/sample.csv
```

Expected output:
```
=== GPU CSV Parser Demo ===
File: data/sample.csv

[1] Full parse (all columns)
  Rows parsed:      5
  Columns:          3
  H2D time:         0.940 ms
  Parse time:       2.709 ms
  First 5 ids:      0 1 2 3 4

[2] Fused filter parse (score > 50, project id & value)
  Total scanned:    5
  Survivors:        2
  H2D time:         0.155 ms
  Fused parse time: 0.836 ms
  First 5 filtered: 1 3
```

Note: `parse_time_ms` in the current build measures **end-to-end** (file read → H2D → all kernels → sync). The `h2d_time_ms` sub-metric isolates the async copy latency.

### 3. Run the sweep benchmark (recommended)

Benchmarks all dataset sizes in one shot:

```powershell
.\build\Release\csv_parser_sweep.exe
```

**Verified output** (RTX 4090 / CUDA 12.6 / Windows):

```
=== GPU CSV Parser: Multi-Size Benchmark ===
Warmup: 2 | Runs: 10

File              Size(MiB) Rows        Full(ms)  Full(MiB/s) Fused(ms) Fused(MiB/s)  Survivors   Selectivity
--------------------------------------------------------------------------------------------------------------
data/sample.csv   0.00      5           0.64      0.1         0.74      0.1            2          0.40
data/1M.csv       17.73     1000000     13.61     1302.9      14.10     1257.2         495620     0.50
data/5M.csv       92.88     5000000     70.23     1322.5      68.96     1346.8         2474601    0.49
data/10M.csv      186.81   10000000    136.68     1366.8     136.65    1367.1          4952096    0.50
```

Throughput **scales linearly** with dataset size — the parser is end-to-end throughput bound.

### 4. Run the single-file benchmark

```powershell
.\build\Release\csv_parser_bench.exe data/1M.csv
```

## Portability Notes (MSVC)

This project was developed and tested on Windows with MSVC + CUDA 12.6. The following compiler-specific idiosyncrasies were addressed:

| Issue | Fix |
|---|---|
| `cstdint.h` header not found | Use `<cstdint>` instead (C++ standard form). |
| `unique_ptr::reset(ptr, deleter)` rejected | `reset()` takes only a pointer; construct a new `unique_ptr` and assign instead. |
| `cudaFree` invisible in pure C++ units | Include `<cuda_runtime.h>` in the public header so host code sees the deleter signature. |
| Aggregate init order strictness | MSVC requires `{ColumnType, name}` order in `ColumnSpec` to match declaration exactly. |
| CMake generator cache conflict | Remove `build/` directory when switching between Ninja and Visual Studio generators. |

## Design Notes & Limitations

| Decision | Rationale |
|----------|-----------|
| **No quoted fields** | A full RFC-4180 quote parser requires either stateful per-warp FSM or pre-pass escaping; for a mid-size portfolio piece the happy-path (no embedded newlines/delimiters) cleanly demonstrates GPU parallelism. |
| **Fixed schema per parse** | Row parser unrolls on a host-provided `ColumnType[]` array; arbitrary schemas are supported up to internal `MAX_COLS` (32). |
| **Byte-level scan = `int` limit** | CUB `DeviceScan` takes `int num_items`. Files must be ≤ 2 GiB. For production scaling this would be **chunked at newline boundaries** and pipelined across multiple CUDA streams (the current code uses one stream; multi-stream overlap is a natural next step). |
| **Null mask = byte-per-row** | Simpler and race-free compared to bit-packing. Arrow bit-packed format is a trivial follow-up (`arrow::Bitmap`). |
| **Single stream** | All GPU work is serialized on one persistent `cudaStream_t`. A multi-stream chunked design (double-buffered H2D + parse overlap) would further improve throughput on very large files. |

## Profiling recommendations

Use **Nsight Compute / Nsight Systems** to examine the kernel behavior:

```powershell
ncu --kernel-name regex:parse_rows_kernel `
    --metrics dram_bytes_read,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct `
    .\build\Release\csv_parser_bench.exe data/1M.csv
```

Key metrics to collect for a blog post / interview:
* **Pinned vs pageable H2D throughput** — compare `cudaMemcpy` from `std::vector` vs `cudaMallocHost`. On PCIe 4.0 x16 you should see ~2-3× difference. Nsight Systems visualizes this clearly.
* **Reusable buffer impact** — profile with `cuda-memcheck` or Nsight Systems to confirm the pool eliminates alloc/free overhead on repeated calls.
* Achieved global memory throughput vs. peak device bandwidth.
* Occupancy of `parse_rows_kernel` (limited by registers / shared memory).
* Impact of fused filter on DRAM traffic — the second kernel’s read volume should be dramatically lower if selectivity is low.

## Next Steps (Roadmap)

1. **Multi-stream chunked parsing** — overlap H2D of chunk *N+1* with parsing of chunk *N* across alternating CUDA streams (double-buffered device memory). This is the natural evolution of the current single-stream design and would fully saturate PCIe bandwidth on multi-GB files.
2. **String / dictionary columns** — add a variable-length char arena and offset buffer.
3. **Arrow-CUDA zero-copy bridge** — wrap `DeviceColumn` buffers into `arrow::NumericArray` via `arrow::cuda::CUDABuffer`.
4. **Warp-parallel tokenization** — for very wide rows, use warp-shuffle cooperative parsing instead of one-thread-per-row.
5. **Integration benchmark vs. cuDF / Pandas** — measure end-to-end `read_csv()` and demonstrate your parser is competitive on narrow numeric schemas.

## License

This is a personal portfolio project. Feel free to adapt and extend for interviews, blog posts, or open-source contributions (e.g., cuDF, Velox, Apache Arrow).
