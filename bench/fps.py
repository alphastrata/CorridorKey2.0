"""Python (stock kernels) and ck on the sample clip per tier and GPU (and all GPUs together): one csv, one table.

uv run python bench/fps.py [--frames 50] [--tiers 720 1080 2160]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import psutil
from clips import fetch, tier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "results" / "bench"
COLUMNS = [
    "gpu",
    "tier",
    "launcher",
    "frames",
    "exit",
    "wall_s",
    "fps",
    "ms_frame",
    "x_python",
]
COLUMNS += ["vram_gib", "util_pct", "power_w", "j_frame", "rss_gib"]
SMI = ["nvidia-smi", "--format=csv,noheader,nounits"]


def smi(query: str, gpus: str | None = None) -> list[list[str]]:
    out = subprocess.run(
        [*SMI, f"--query-gpu={query}", *(["-i", gpus] if gpus else [])],
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        [c.strip() for c in line.split(",")]
        for line in out.stdout.splitlines()
        if line.strip()
    ]


@dataclass
class Meter:
    gpus: str
    idle: dict[str, float]
    samples: list[tuple[float, str, float, float, float]] = field(default_factory=list)
    rss: int = 0

    def gpu(self) -> subprocess.Popen[str]:
        q = "index,memory.used,utilization.gpu,power.draw"
        proc = subprocess.Popen(
            [*SMI, f"--query-gpu={q}", "-i", self.gpus, "-lms", "100"],
            stdout=subprocess.PIPE,
            text=True,
        )
        threading.Thread(target=self.read, args=(proc,), daemon=True).start()
        return proc

    def read(self, proc: subprocess.Popen[str]) -> None:
        for line in proc.stdout or []:
            g, mem, util, watts = (c.strip() for c in line.split(","))
            try:
                self.samples.append(
                    (time.monotonic(), g, float(mem), float(util), float(watts))
                )
            except ValueError:
                continue

    def track(self, proc: psutil.Process) -> None:
        while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            try:
                tree = [proc, *proc.children(recursive=True)]
                self.rss = max(self.rss, sum(p.memory_info().rss for p in tree))
            except psutil.Error:
                pass
            time.sleep(0.1)

    def stats(self) -> tuple[float, float, float, float]:
        per: dict[str, list[tuple[float, float, float, float]]] = {}
        for t, g, mem, util, watts in self.samples:
            per.setdefault(g, []).append((t, mem, util, watts))
        if not per:
            return 0.0, 0.0, 0.0, 0.0
        vram = (
            sum(
                max(m - self.idle.get(g, 0.0) for _, m, _, _ in s)
                for g, s in per.items()
            )
            / 1024
        )
        util = sum(sum(u for _, _, u, _ in s) / len(s) for s in per.values()) / len(per)
        watts = sum(sum(w for *_, w in s) / len(s) for s in per.values())
        joules = sum((b[0] - a[0]) * b[3] for s in per.values() for a, b in pairwise(s))
        return vram, util, watts, joules


def run(
    gpus: str, label: str, cmd: list[str], log: Path, env: dict[str, str]
) -> tuple[int, float, Meter]:
    meter = Meter(gpus, {g: float(m) for g, m in smi("index,memory.used", gpus)})
    sampler = meter.gpu()
    start = time.monotonic()
    with log.open("w") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env, "CUDA_VISIBLE_DEVICES": gpus},
        )
        watch = threading.Thread(
            target=meter.track, args=(psutil.Process(proc.pid),), daemon=True
        )
        watch.start()
        rc = proc.wait()
    wall = time.monotonic() - start
    sampler.terminate()
    sampler.wait()
    watch.join()
    print(f"{label} → {log.name}: exit {rc}, {wall:.1f} s")
    return rc, wall, meter


def ck_binary() -> Path:
    manifest = ROOT / "launcher" / "Cargo.toml"
    subprocess.run(
        [
            "cargo",
            "build",
            "-q",
            "--release",
            "--manifest-path",
            str(manifest),
            "-p",
            "ck",
        ],
        check=True,
    )
    meta = subprocess.run(
        [
            "cargo",
            "metadata",
            "-q",
            "--format-version",
            "1",
            "--no-deps",
            "--manifest-path",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(json.loads(meta.stdout)["target_directory"]) / "release" / "ck"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--tiers", type=int, nargs="+", default=[720, 1080, 2160])
    a = p.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    fetch(a.frames)
    ck = ck_binary()
    for sub in ("logs", "out"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    names = {i: n.removeprefix("NVIDIA ") for i, n in smi("index,name")}
    rows: list[dict[str, str]] = []
    base: dict[str, float] = {}

    def record(
        gpus: str,
        t: int,
        launcher: str,
        cmd: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        tag = f"{gpus.replace(',', '+')}_{t}p_{launcher}"
        label = " + ".join(names[g] for g in gpus.split(","))
        rc, wall, meter = run(gpus, label, cmd, OUT / "logs" / f"{tag}.log", env or {})
        if launcher == "python":
            base[gpus + str(t)] = wall
        ok, ref = rc == 0, base.get(gpus + str(t))
        vram, util, watts, joules = meter.stats()
        rows.append(
            {
                "gpu": label,
                "tier": f"{t}p",
                "launcher": launcher,
                "frames": str(a.frames),
                "exit": str(rc),
                "wall_s": f"{wall:.1f}",
                "fps": f"{a.frames / wall:.2f}" if ok else "",
                "ms_frame": f"{1000 * wall / a.frames:.0f}" if ok else "",
                "x_python": f"{ref / wall:.2f}" if ok and ref else "",
                "vram_gib": f"{vram:.2f}",
                "util_pct": f"{util:.0f}",
                "power_w": f"{watts:.0f}",
                "j_frame": f"{joules / a.frames:.0f}" if ok else "",
                "rss_gib": f"{meter.rss / 2**30:.1f}",
            }
        )

    def ck_run(gpus: str, t: int) -> None:
        out = OUT / "out" / f"{gpus.replace(',', '+')}_{t}p_ck"
        record(
            gpus,
            t,
            "ck",
            [
                str(ck),
                "--ptx",
                "corridorkey/ptx/bin",
                "--weights",
                "weights/corridorkey_v2.safetensors",
                "--vfms",
                "weights/vfms.safetensors",
                "--frames",
                str(a.frames),
                "--chunk",
                "auto",
                "-o",
                str(out),
                str(tier(t)),
            ],
        )

    for t in a.tiers:
        for g in sorted(names, key=int):
            out = OUT / "out" / f"{g}_{t}p_python"
            record(
                g,
                t,
                "python",
                [
                    sys.executable,
                    "infer.py",
                    "--input",
                    str(tier(t)),
                    "--output_dir",
                    str(out),
                    "--num_frames",
                    str(a.frames),
                ],
                {"CORRIDORKEY_PTX": "0"},
            )
            ck_run(g, t)
        if len(names) > 1:
            ck_run(",".join(sorted(names, key=int)), t)
    with (OUT / "bench.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in COLUMNS}
    print()
    for r in [dict(zip(COLUMNS, COLUMNS)), *rows]:
        print("  ".join(r[c].ljust(widths[c]) for c in COLUMNS))


if __name__ == "__main__":
    main()
