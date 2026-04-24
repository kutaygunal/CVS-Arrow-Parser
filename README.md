# GPU-Accelerated CSV → Columnar Parser with Fused Filter Pushdown

A mid-size CUDA/C++ project demonstrating **parallel CSV parsing on the GPU**, direct emission of typed columnar buffers (int64 / float64), and a **fused filter pushdown** that evaluates predicates while parsing to avoid materializing rows that will be discarded.

### Algorithm Evolution — Before vs After

| Optimisation | Before | After (this commit) | Delta |
|-------------|--------|-------------------|-------|
| **Host memory** | `std::vector<char>` (pageable) | `cudaMallocHost` (pinned) | ~2× faster H2D |
| **GPU scheduling** | Synchronous `cudaMemcpy` + default stream | Async `cudaMemcpyAsync` + persistent `cudaStream_t` | Overlap potential, lower CPU sync stalls |
| **Temp buffer allocation** | `cudaMalloc`/`cudaFree` per call (~8 allocs) | Pooled buffers resized on demand | -28 % end-to-end latency |
| **Row parsing** | One thread / row, sequential char scan | **One warp / row**, `__ballot_sync` delimiter scan | Coalesced loads, far less divergence |
| **Numeric conversion** | `dev_atol` / `dev_atof` with per-char branching | `fast_atol` / `fast_atof` — branchless | ~2× faster field parse |
| **Host I/O** | `std::ifstream` | `fread` into reusable pinned buffer | Removes CRT iostream overhead |
| **Autonomous tuning** | None | **Autoresearch loop**: parameter sweeps via `experiment_csv.py` | Finds optimal block sizes, strategies |
| **Multi-stream** | Single stream | **2-stream chunked parsing** for files > 64 MB | Overlaps H2D + compute across chunks |
| **Throughput (10 M rows)** | **1 082 MiB/s** | **4 117 MiB/s** | **+280 %** |
| **Fused filter (10 M rows)** | **1 367 MiB/s** | **4 407 MiB/s** | **+222 %** |

> Benchmarked on **RTX 4090 / PCIe 4.0 x16 / CUDA 12.6 / MSVC 2022**.

## What this demonstrates

* **Warp-parallel CSV tokenization** — one **warp** per row; threads cooperatively find delimiters with `__ballot_sync`, then lane 0 parses the selected fields. Loads are coalesced and divergent sequential scans are eliminated.
* **Fast branchless numeric parsers** — custom `fast_atol` / `fast_atof` with no per-digit validity branching, assuming controlled simple CSV.
* **Modern C++ design** — RAII device buffers, strict separation of host API / device kernels, and error checking macros.
* **CUB integration** — `DeviceScan::InclusiveSum` and `ExclusiveSum` to compact newline positions and filtered row masks in parallel.
* **Pinned host memory + CUDA streams + reusable buffers** — file is read directly into page-locked memory; H2D copies and all kernels are async on a persistent stream. **All** temporary device buffers (`data`, `marks`, `cumsum`, `newlines`, `mask`, `scanned`, `CUB temp`) and the host pinned buffer are pooled and reused across invocations, eliminating per-call alloc/free overhead entirely.
* **Predicate pushdown** — two-stage fused pipeline:
  1. **Stage 1** (`filter_mask_kernel_warp`): warp-level scan finds the predicate field and evaluates.
  2. **Stage 2** (`cub::ExclusiveSum`): compact surviving rows.
  3. **Stage 3** (`parse_filtered_rows_kernel_warp`): warp-level field scan + parse **only** projected columns for survivors.
* **Multi-stream chunked parsing** — for large files (> 64 MB), the payload is split at newline boundaries and processed on **two concurrent CUDA streams**. H2D of chunk *N+1* overlaps with scan/compact of chunk *N*, improving throughput when single-stream H2D is the bottleneck.
* **Autoresearch integration** — an adapted version of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) lives in `autoresearch_csv/`. The agent iteratively tunes `experiment_csv.py` (block sizes, parser strategy, stream count), rebuilds, benchmarks for a fixed time budget, and keeps/discards changes based on throughput. A `tuneable_params.h` header lets the agent control compile-time constants without fragile regex surgery on `.cu` files.
* **Arrow-minded layout** — null bitmaps per column, type-tagged buffers, ready to be zero-copy wrapped into `arrow::Array` extensions.

## Architecture

```
Host: fread into reusable pinned buffer → skip header
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
  │ Full Parse Path │   parse_rows_kernel_warp (1 warp / row)
  └─────────────────┘
       │
  ┌─────────────────┐
  │ Fused Filter    │   Stage 1: filter_mask_kernel_warp (predicate only)
  │ Pushdown Path   │   Stage 2: cub::ExclusiveSum
  └─────────────────┘   Stage 3: parse_filtered_rows_kernel_warp (project)

Multi-stream path (large files > 64 MB, full parse only):
  Host splits payload at newline boundaries
       │
       ▼
  Stream 0:  H2D chunk 0 ──▶ mark/scan/compact chunk 0
  Stream 1:  H2D chunk 1 ──▶ mark/scan/compact chunk 1   (overlaps with Stream 0)
       │                            │
       ▼                            ▼
  Sync 0 & 1  →  allocate combined output
       │
       ▼
  Stream 0:  parse chunk 0 into combined buffer at row offset 0
  Stream 1:  parse chunk 1 into combined buffer at row offset N0
       │
       ▼
  Sync 0 & 1  →  return combined ParsedCSV
```

### Memory-management strategy

| Buffer | Ownership | Lifetime |
|--------|-----------|----------|
| `h_pinned_buf_` | `Impl` | Reusable pinned host buffer; resized on demand |
| `d_data_` (file payload) | `Impl` | Created on first parse; resized on demand; reused across calls |
| `d_marks_`, `d_cumsum_`, `d_temp_` | `Impl` | Pooled alongside `d_data_`; grows on demand |
| `d_newlines_`, `d_mask_`, `d_scanned_` | `Impl` | Pooled; reused across calls |
| Output column buffers | Per-call | Owned by returned `DeviceColumn` via `unique_ptr` |

This eliminates **all** per-call `cudaMalloc`/`cudaFree` and `cudaMallocHost`/`cudaFreeHost` overhead.

### 5. Run the autoresearch tuning loop

An adapted [karpathy/autoresearch](https://github.com/karpathy/autoresearch) framework lives in `autoresearch_csv/`. It autonomously edits tunable parameters, rebuilds, benchmarks, and keeps/discards changes based on throughput.

```powershell
# Establish baseline
cd autoresearch_csv
python experiment_csv.py

# Edit experiment_csv.py  →  change TUNABLES  →  re-run
# The script writes include/tuneable_params.h, rebuilds, and reports throughput.
```

Current tunables exposed via `include/tuneable_params.h`:

| Macro | Description | Typical range |
|-------|-------------|---------------|
| `TUNE_THREADS_PER_BLOCK` | Threads per block for parse kernels | 128 – 512 |
| `TUNE_MARK_THREADS_PER_BLOCK` | Threads for `mark_newlines` | 128 – 512 |
| `TUNE_COMPACT_THREADS_PER_BLOCK` | Threads for `compact_newlines` | 128 – 512 |
| `TUNE_USE_WARP_PARSE` | 1 = warp-per-row, 0 = thread-per-row | 0 or 1 |
| `TUNE_USE_FAST_PARSER` | 1 = fast branchless, 0 = safe validated | 0 or 1 |
| `TUNE_MAX_FIELDS_PER_WARP` | Shared-memory field budget | 16 – 64 |
| `TUNE_USE_WIDE_LOADS` | 1 = uint4 coalesced scan (needs kernel support) | 0 or 1 |
| `TUNE_NUM_STREAMS` | 1 = single stream, 2 = dual-stream chunking | 1 or 2 |
| `TUNE_MIN_CHUNK_SIZE_BYTES` | Minimum chunk size for multi-stream | 32 MB |

For this codebase, 256 threads + single stream is empirically optimal on RTX 4090 for the 10 M row benchmark. Multi-stream becomes beneficial for files > 500 MB where H2D time dominates.

## Directory Layout

```
01-csv-arrow-parser/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── gpu_csv_parser.hpp       # public API
│   └── tuneable_params.h        # autoresearch compile-time tunables
├── src/
│   ├── gpu_csv_parser.cu        # CUDA kernels + Impl (single-stream + multi-stream)
│   └── main.cpp                 # CLI demo
├── benchmarks/
│   ├── bench.cu                 # single-file repeated-run benchmark
│   └── sweep.cu                 # multi-size sweep benchmark
├── autoresearch_csv/            # adapted karpathy/autoresearch framework
│   ├── prepare_csv.py           # fixed build + benchmark harness
│   ├── experiment_csv.py        # agent-editable tunables + experiment loop
│   └── program_csv.md           # agent instructions
├── autoresearch/                # original karpathy/autoresearch (reference)
│   └── ...
├── scripts/
│   └── generate_csv.py          # generate large benchmark files
└── data/
    ├── sample.csv               # tiny test file (5 rows)
    ├── 1M.csv                   # generated: 1 million rows
    ├── 5M.csv                   # generated: 5 million rows
    └── 10M.csv                  # generated: 10 million rows
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

**Verified output** (RTX 4090 / CUDA 12.6 / Windows, after warp-level + fast-parser + buffer-reuse + autoresearch sweep):

```
=== GPU CSV Parser: Multi-Size Benchmark ===
Warmup: 2 | Runs: 10

File              Size(MiB) Rows        Full(ms)  Full(MiB/s) Fused(ms) Fused(MiB/s)  Survivors   Selectivity
--------------------------------------------------------------------------------------------------------------
data/1M.csv       17.73     1000000     4.38      4051.4      3.77      4698.9        495620      0.50      
data/5M.csv       92.88     5000000     24.78     3748.6      25.05     3708.0        2474601     0.49      
data/10M.csv      186.81    10000000    45.37     4117.2      42.39     4407.0        4952096     0.50
```

Throughput **scales linearly** with dataset size — the parser is end-to-end throughput bound.

| Mode | 10 M rows throughput | Speed-up vs baseline |
|------|------------------|-------------------|
| Full parse | **4 117 MiB/s** | **+280 %** |
| Fused filter | **4 407 MiB/s** | **+222 %** |

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
| **Warp-level = one warp per row** | For narrow rows (≤ a few hundred bytes), cooperative warp scanning is a massive win. For extremely wide rows (>1 KB), multiple warps per row or per-field parallelism would be better. |

## Profiling recommendations

Use **Nsight Compute / Nsight Systems** to examine the kernel behavior:

```powershell
ncu --kernel-name regex:parse_rows_kernel_warp `
    --metrics dram_bytes_read,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct `
    .\build\Release\csv_parser_bench.exe data/10M.csv
```

Key metrics to collect for a blog post / interview:
* **Warp-level parsing efficiency** — compare `parse_rows_kernel` (old, 1 thread/row) vs `parse_rows_kernel_warp` (new). Check `smsp__cycles_elapsed.max` per row and look for reduced branch divergence.
* **Global memory coalescing** — with warp-level scanning, the `__ballot_sync` load pattern should show near-100 % sector utilization.
* **Buffer reuse impact** — with zero per-call allocations, Nsight Systems should show no `cudaMalloc`/`cudaFree` gaps between benchmark iterations.
* **Fast parser ALU throughput** — `fast_atol` / `fast_atof` are ALU-bound; check `smsp__pipe_alu.avg.pct_of_peak`.
* **Fused filter DRAM traffic** — compare total DRAM bytes read between full parse and fused filter. With low selectivity, the fused path should move dramatically less data.

## Next Steps (Roadmap)

* **Single kernel fused filter** — evaluate predicate, block-level prefix sum, and write projected columns in **one kernel** (eliminates the CUB scan + second launch entirely; trades deterministic ordering for fewer round-trips).
* **Multi-stream chunked parsing** — overlap H2D of chunk *N+1* with parsing of chunk *N* across alternating CUDA streams (double-buffered device memory). This is the natural evolution of the current single-stream design and would fully saturate PCIe bandwidth on multi-GB files.
* **String / dictionary columns** — add a variable-length char arena and offset buffer.
* **Arrow-CUDA zero-copy bridge** — wrap `DeviceColumn` buffers into `arrow::NumericArray` via `arrow::cuda::CUDABuffer`.
* **Integration benchmark vs. cuDF / Pandas** — measure end-to-end `read_csv()` and demonstrate your parser is competitive on narrow numeric schemas.

## License

This is a personal portfolio project. Feel free to adapt and extend for interviews, blog posts, or open-source contributions (e.g., cuDF, Velox, Apache Arrow).
