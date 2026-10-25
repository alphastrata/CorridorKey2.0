"""nsys one ck run and print the top kernels by GPU time: ``results/NAME.nsys-rep``.

uv run python bench/kprof.py NAME [ck args…]
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

from fps import ROOT, ck_binary

RESULTS = ROOT / "bench" / "results"


def top(report: Path, count: int = 25) -> list[str]:
    out = subprocess.run(
        [
            "nsys",
            "stats",
            "-q",
            "--force-export=true",
            "-r",
            "cuda_gpu_kern_sum",
            "-f",
            "csv",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = list(csv.reader(io.StringIO(out)))
    head = next(i for i, r in enumerate(rows) if r and r[0] == "Time (%)")
    kernels = [r for r in rows[head + 1 :] if len(r) > 1]
    total = sum(int(r[1]) for r in kernels) / 1e6
    return [f"total {total:.1f} ms"] + [
        f"{int(r[1]) / 1e6:9.1f} ms {r[2]:>6} {r[-1][:70]}" for r in kernels[:count]
    ]


def main() -> None:
    name, args = sys.argv[1], sys.argv[2:]
    ck = ck_binary()
    RESULTS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "nsys",
            "profile",
            "-f",
            "true",
            "-o",
            str(RESULTS / name),
            str(ck),
            "--ptx",
            "corridorkey/ptx/bin",
        ]
        + [
            "--weights",
            "weights/corridorkey_v2.safetensors",
            "--vfms",
            "weights/vfms.safetensors",
        ]
        + ["-o", str(RESULTS / "out" / name), *args],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    print("\n".join(top(RESULTS / f"{name}.nsys-rep")))


if __name__ == "__main__":
    main()
