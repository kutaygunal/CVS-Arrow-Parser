#!/usr/bin/env python3
"""Generate a synthetic CSV for benchmarking the GPU parser."""
import argparse
import random
import os


def generate(path: str, rows: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("id,value,score\n")
        for i in range(rows):
            v = round(random.uniform(0.0, 100.0), 4)
            s = random.randint(0, 100)
            f.write(f"{i},{v},{s}\n")
    print(f"Generated {path} with {rows:,} rows.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate synthetic CSV")
    p.add_argument("--rows", type=int, default=1_000_000, help="Number of data rows")
    p.add_argument("--out", default="data/sample.csv", help="Output file path")
    args = p.parse_args()
    generate(args.out, args.rows)
