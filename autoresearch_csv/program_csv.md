# Autoresearch: GPU CSV Parser Optimization

This is an autonomous research experiment to maximize the end-to-end throughput (MiB/s) of a CUDA-accelerated CSV parser.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr23`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** (do NOT modify prepare_csv.py):
   - `README.md` — repository context.
   - `prepare_csv.py` — fixed build harness, benchmark runner, constants. Read-only.
   - `experiment_csv.py` — the file you modify. Tunable parameters + experiment logic.
   - `../src/gpu_csv_parser.cu` — the CUDA parser kernel source (read for context).
   - `../include/tuneable_params.h` — auto-generated from experiment_csv.py.
4. **Verify data exists**: Check that `../data/10M.csv` exists. If not, tell the user to generate it.
5. **Initialize results.tsv**: Create `results.tsv` with header row.
6. **Confirm and go**.

## Experimentation

Each experiment:
1. **You modify `experiment_csv.py`** — tune the `TUNABLES` dict, or add logic to modify `../src/gpu_csv_parser.cu` directly.
2. **Run** `python experiment_csv.py` from the `autoresearch_csv/` directory.
3. The script writes `tuneable_params.h`, builds, benchmarks repeatedly for ~45 seconds, and prints a summary.

**What you CAN do:**
- Edit `experiment_csv.py` — add new tunables, modify existing ones, add source rewriting logic for `.cu` files.
- Everything is fair game: kernel block sizes, parser strategy, memory management, launch bounds, etc.

**What you CANNOT do:**
- Modify `prepare_csv.py`. It is read-only.
- Install new packages. Only the standard library + what's in `prepare_csv.py`.
- Change the benchmark harness or evaluation metric.

**The goal is simple: maximize `throughput_mib_s` (mean).** Higher is better.

**VRAM** is a soft constraint. Don't add changes that dramatically increase memory usage for marginal gains.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

## Output format

After `experiment_csv.py` finishes it prints:

```
---
throughput_mib_s: 4091.630000
stdev_mib_s:      120.500000
n_runs:           12
build_seconds:    18.50
total_seconds:    63.2
---
```

## Logging results

Log each experiment to `results.tsv` (tab-separated). Columns:

```
commit	throughput_mib_s	stdev_mib_s	status	description
```

- `commit`: short git hash (7 chars)
- `throughput_mib_s`: the mean throughput from the summary
- `stdev_mib_s`: standard deviation
- `status`: `keep`, `discard`, or `crash`
- `description`: short text of what this experiment tried

Example:

```
commit	throughput_mib_s	stdev_mib_s	status	description
a1b2c3d	4091.63	120.50	keep	baseline
b2c3d4e	4250.10	95.20	keep	increase THREADS_PER_BLOCK to 512
c3d4e5f	3800.00	200.00	discard	switch to thread-per-row
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state and the current `results.tsv`.
2. Propose an experimental idea based on prior results.
3. Edit `experiment_csv.py` (only this file).
4. `git add experiment_csv.py` && `git commit -m "..."`
5. Run: `python autoresearch_csv/experiment_csv.py > run.log 2>&1`
6. Read out results: `grep "^throughput_mib_s:" run.log`
7. If empty, the run crashed. Read `tail -n 50 run.log` and attempt a fix. If you can't fix it after a few attempts, log "crash" and move on.
8. Record results in `results.tsv` (do NOT commit the TSV).
9. **If throughput improved**, keep the commit — advance the branch.
10. **If throughput is equal or worse**, `git reset --hard` back to the last good commit.

**Timeout**: Each experiment should take ~1 minute (build + benchmark). If a run exceeds 10 minutes, kill it and treat as failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Continue autonomously until manually interrupted.

## Ideas to try

Here are good starting experiments, roughly ordered by expected impact:

1. **Baseline**: Run as-is to establish the throughput number.
2. **Block size sweep**: Try `THREADS_PER_BLOCK` = 128, 192, 256, 384, 512. Larger blocks reduce launch overhead but may hurt occupancy.
3. **Mark/Compact block sizes**: Independently tune `MARK_THREADS_PER_BLOCK` and `COMPACT_THREADS_PER_BLOCK`.
4. **Shared memory size**: `MAX_FIELDS_PER_WARP` trades shared memory for wide-row support. Try 16, 24, 32, 48, 64.
5. **Wide loads**: Enable `USE_WIDE_LOADS` and see if uint4 coalesced reads help delimiter scanning.
6. **Multi-stream chunking**: Add logic in `experiment_csv.py` to rewrite `gpu_csv_parser.cu` to use multiple CUDA streams with chunked H2D overlap. This is likely the single biggest remaining win.
7. **Launch bounds**: Add `__launch_bounds__` hints to kernels to increase occupancy.
8. **Alternative numeric accumulation**: For float parsing, try accumulating in `float` then promoting to `double`.
9. **Skip header optimization**: Pre-compute header offset on host to avoid the per-row header skip logic.
10. **Async file read**: Use `fread` in a background thread while previous chunk is parsing.

Focus on ideas with high expected impact. Multi-streaming and wide loads are the most promising.
