# GPU-Accelerated CSV → Columnar Parser with Fused Filter Pushdown

A mid-size CUDA/C++ project demonstrating **parallel CSV parsing on the GPU**, direct emission of typed columnar buffers (int64 / float64), and a **fused filter pushdown** that evaluates predicates while parsing to avoid materializing rows that will be discarded.

## What this demonstrates

* **GPU-native CSV tokenization** — one thread per row, field boundaries discovered in shared registers.
* **Modern C++ design** — RAII device buffers, strict separation of host API / device kernels, and error checking macros.
* **CUB integration** — `DeviceScan::InclusiveSum` and `ExclusiveSum` to compact newline positions and filtered row masks in parallel.
* **Predicate pushdown** — two-stage fused pipeline:
  1. Parse only the predicate column + evaluate (produces a boolean mask).
  2. CUB scan mask to get output positions.
  3. Second kernel materializes **only** projected columns for surviving rows.
* **Arrow-minded layout** — null bitmaps per column, type-tagged buffers, ready to be zero-copy wrapped into `arrow::Array` extensions.

## Architecture

```
Host: read file → skip header → pinned H2D copy
       │
       ▼
Device:
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

## Directory Layout

```
01-csv-arrow-parser/
├── CMakeLists.txt
├── README.md
├── data/
│   └── sample.csv               # tiny test file
├── scripts/
│   └── generate_csv.py          # generate large benchmark files
├── include/
│   └── gpu_csv_parser.hpp       # public API
├── src/
│   ├── gpu_csv_parser.cu        # CUDA kernels + Impl
│   └── main.cpp                 # CLI demo
└── benchmarks/
    └── bench.cu                  # throughput benchmark
```

## Build Requirements

* CMake ≥ 3.20
* CUDA Toolkit ≥ 11.0 (CUB is bundled)
* A CUDA-capable GPU with compute capability ≥ 7.5
* C++20 host compiler (MSVC 2019+, GCC 10+, Clang 14+)

### Windows Build (Visual Studio / Ninja)

```powershell
cd C:\Users\kutay\Desktop\Projects\01-csv-arrow-parser
# Generate
#  - If using Visual Studio generator:
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#  - If using Ninja (recommended with CUDA):
cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe"

# Build
cmake --build build --config Release
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
.\build\csv_parser_cli.exe data/10M.csv
```

Expected output:
```
=== GPU CSV Parser Demo ===
File: data/10M.csv

[1] Full parse (all columns)
  Rows parsed:      10000000
  Columns:          3
  H2D time:         ... ms
  Parse time:       ... ms
  First 5 ids:      0 1 2 3 4

[2] Fused filter parse (score > 50, project id & value)
  Total scanned:    10000000
  Survivors:        ~5000000
  H2D time:         ... ms
  Fused parse time: ... ms
  First 5 filtered: ...
```

### 3. Run benchmarks

```powershell
.\build\csv_parser_bench.exe data/10M.csv
```

## Design Notes & Limitations

| Decision | Rationale |
|----------|-----------|
| **No quoted fields** | A full RFC-4180 quote parser requires either stateful per-warp FSM or pre-pass escaping; for a mid-size portfolio piece the happy-path (no embedded newlines/delimiters) cleanly demonstrates GPU parallelism. |
| **Fixed schema per parse** | Row parser unrolls on a host-provided `ColumnType[]` array; arbitrary schemas are supported up to internal `MAX_COLS` (32). |
| **Byte-level scan = `int` limit** | CUB `DeviceScan` takes `int num_items`. Files must be ≤ 2 GiB. For production scaling this would be chunked by 2 GiB windows and pipelined with the host. |
| **Null mask = byte-per-row** | Simpler and race-free compared to bit-packing. Arrow bit-packed format is a trivial follow-up (`arrow::Bitmap`). |
| **Strings deferred** | The same row-tokenization kernel can be extended to emit `(offset, length)` pairs into a global char pool; adding that is a recommended “stretch goal.” |

## Profiling recommendations

Use **Nsight Compute / Nsight Systems** to examine the kernel behavior:

```powershell
ncu --kernel-name regex:parse_rows_kernel --metrics dram_bytes_read,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct .\build\csv_parser_bench.exe data/10M.csv
```

Key metrics to collect for a blog post / interview:
* Achieved global memory throughput vs. peak device bandwidth.
* Occupancy of `parse_rows_kernel` (limited by registers / shared memory).
* Impact of fused filter on DRAM traffic — the second kernel’s read volume should be dramatically lower if selectivity is low.

## Next Steps (Roadmap)

1. **String / dictionary columns** — add a variable-length char arena and offset buffer.
2. **Arrow-CUDA zero-copy bridge** — wrap `DeviceColumn` buffers into `arrow::NumericArray` via `arrow::cuda::CUDABuffer`.
3. **Multi-file / chunked streaming** — overlap H2D copies of chunk *N* with parsing of chunk *N-1* using CUDA streams.
4. **Warp-parallel tokenization** — for very wide rows, use warp-shuffle cooperative parsing instead of one-thread-per-row.
5. **Integration benchmark vs. cuDF / Pandas** — measure end-to-end `read_csv()` and demonstrate your parser is competitive on narrow numeric schemas.

## License

This is a personal portfolio project. Feel free to adapt and extend for interviews, blog posts, or open-source contributions (e.g., cuDF, Velox, Apache Arrow).
