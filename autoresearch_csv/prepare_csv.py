"""
prepare_csv.py  — fixed constants, build harness, benchmark runner.
Do NOT modify this file; it is read-only infrastructure.
"""
import subprocess
import os
import sys
import re
import time
import statistics

# ---------------------------------------------------------------------------
# Paths (relative to autoresearch_csv/)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_DIR = os.path.join(ROOT_DIR, "build")
BENCHMARK_EXE = os.path.join(BUILD_DIR, "Release", "csv_parser_bench.exe")
BENCHMARK_SWEEP_EXE = os.path.join(BUILD_DIR, "Release", "csv_parser_sweep.exe")
DATA_FILE = os.path.join(ROOT_DIR, "data", "10M.csv")

# Benchmarks are run repeatedly for a fixed wall-clock budget per experiment.
TIME_BUDGET_SECONDS = 45.0
WARMUP_RUNS = 2

# ---------------------------------------------------------------------------
# Build harness
# ---------------------------------------------------------------------------
def build_project():
    """Run CMake build for Release config. Returns (ok, elapsed_ms)."""
    t0 = time.time()
    cmd = [
        "cmake", "--build", BUILD_DIR,
        "--config", "Release",
        "--parallel", "4"
    ]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=True, timeout=300
        )
    except subprocess.CalledProcessError as e:
        print("BUILD FAILED")
        print(e.stdout)
        return False, 0.0
    except subprocess.TimeoutExpired:
        print("BUILD TIMEOUT")
        return False, 0.0

    elapsed = (time.time() - t0) * 1000.0
    return True, elapsed

# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------
def run_bench_once(exe=BENCHMARK_EXE, data=DATA_FILE):
    """Run a single benchmark invocation. Returns dict with metrics, or None."""
    cmd = [exe, data]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=True, timeout=180
        )
    except subprocess.CalledProcessError as e:
        print("BENCH FAILED")
        print(e.stdout[-2000:] if len(e.stdout) > 2000 else e.stdout)
        return None
    except subprocess.TimeoutExpired:
        print("BENCH TIMEOUT")
        return None

    out = result.stdout
    metrics = {}

    # Parse full parse: e.g. "Avg:   45.66 ms | throughput: 4091.63 MiB/s"
    m = re.search(r"Avg:\s+([\d.]+)\s+ms\s+\|\s+throughput:\s+([\d.]+)", out)
    if m:
        metrics["full_ms"] = float(m.group(1))
        metrics["full_mib_s"] = float(m.group(2))

    # Parse fused filter section (no throughput line, just ms)
    m = re.search(r"===\s+Benchmark:\s+Fused.*?Avg:\s+([\d.]+)\s+ms", out, re.DOTALL)
    if m:
        metrics["fused_ms"] = float(m.group(1))

    return metrics


def evaluate_current(exe=BENCHMARK_EXE, data=DATA_FILE):
    """
    Repeatedly benchmark until TIME_BUDGET_SECONDS is exhausted.
    Returns dict with aggregated stats.
    """
    if not os.path.exists(exe):
        print(f"Benchmark executable not found: {exe}")
        return None

    full_tp = []
    fused_ms = []
    elapsed = 0.0
    total_runs = 0

    # Warmup
    for _ in range(WARMUP_RUNS):
        run_bench_once(exe, data)

    t0 = time.time()
    while elapsed < TIME_BUDGET_SECONDS:
        m = run_bench_once(exe, data)
        if m is None:
            break
        if "full_mib_s" in m:
            full_tp.append(m["full_mib_s"])
        if "fused_ms" in m:
            fused_ms.append(m["fused_ms"])
        total_runs += 1
        elapsed = time.time() - t0

    if not full_tp:
        return None

    return {
        "mean_mib_s": statistics.mean(full_tp),
        "stdev_mib_s": statistics.stdev(full_tp) if len(full_tp) > 1 else 0.0,
        "min_mib_s": min(full_tp),
        "max_mib_s": max(full_tp),
        "n_runs": total_runs,
        "mean_fused_ms": statistics.mean(fused_ms) if fused_ms else 0.0,
    }


if __name__ == "__main__":
    print("Preparing CSV autoresearch environment...")
    ok, ms = build_project()
    if ok:
        print(f"Build succeeded in {ms:.0f} ms")
        res = evaluate_current()
        print(res)
    else:
        print("Build failed!")
        sys.exit(1)
