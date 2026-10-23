"""Markdown report from ``scripts/bench.sh`` output: end to end, ck stages, ck kernels, and the Python suite.

python kernels/bench/report.py RESULTS_DIR FRAMES
"""

from __future__ import annotations

import csv
import io
import itertools
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
CATEGORIES = (
    ("GEMM", ("gemm", "cutlass", "cublas")),
    ("attention", ("attn",)),
    ("conv", ("im2col", "col2im", "dw_")),
)


@dataclass(frozen=True)
class Sampled:
    peak_gib: float
    util: float
    joules: float


def gpus(b: Path) -> dict[str, str]:
    rows = csv.reader((b / "gpus.csv").read_text().splitlines())
    return {r[0].strip(): r[1].strip().removeprefix("NVIDIA ") for r in rows}


def label(ids: str, names: dict[str, str]) -> str:
    return " + ".join(names.get(i, i) for i in ids.split("+"))


def sampled(b: Path, tag: str) -> Sampled | None:
    base_file, log = b / "samples" / f"{tag}.base", b / "samples" / f"{tag}.csv"
    if not log.exists():
        return None
    base = {
        r[0].strip(): float(r[1])
        for r in csv.reader(base_file.read_text().splitlines())
    }
    series: dict[str, list[tuple[datetime, float, float, float]]] = defaultdict(list)
    for r in csv.reader(log.read_text().splitlines()):
        if len(r) == 5 and "N/A" not in "".join(r[2:]):
            t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            series[r[1].strip()].append(
                (t, float(r[2]) - base.get(r[1].strip(), 0.0), float(r[3]), float(r[4]))
            )
    if not series:
        return None
    peak = sum(max(s[1] for s in v) for v in series.values()) / 1024
    util = sum(sum(s[2] for s in v) / len(v) for v in series.values()) / len(series)
    joules = sum(
        sum((b[0] - a[0]).total_seconds() * a[3] for a, b in itertools.pairwise(v))
        for v in series.values()
    )
    return Sampled(peak, util, joules)


def e2e(b: Path, frames: int, names: dict[str, str]) -> list[str]:
    path = b / "e2e.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.read_text().splitlines()))
    stock = {
        (r["gpus"], r["tier"]): float(r["wall_s"])
        for r in rows
        if r["config"] == "stock" and r["exit"] == "0"
    }
    out = [
        "## End to end",
        "",
        (
            f"{frames} frames, wall time including load, decode and encode. VRAM, utilisation and energy sampled by "
            "`nvidia-smi` every 100 ms (VRAM above the idle baseline, summed over GPUs)."
        ),
        "",
        "| GPU | tier | runner | wall s | fps | × stock | peak VRAM GiB | GPU util % | J / frame | host RSS GiB |",
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        wall, ok = float(r["wall_s"]), r["exit"] == "0"
        s = sampled(b, f"{r['gpus']}_{r['tier']}p_{r['config']}")
        ref = stock.get((r["gpus"], r["tier"]))
        cells = (
            [
                f"{wall:.1f}",
                f"{frames / wall:.2f}",
                f"{ref / wall:.2f}×" if ref else "—",
            ]
            if ok
            else ["failed", "—", "—"]
        )
        cells += (
            [
                f"{s.peak_gib:.1f}",
                f"{s.util:.0f}",
                f"{s.joules / frames:.0f}" if ok else "—",
            ]
            if s
            else ["—"] * 3
        )
        cells.append(f"{int(r['rss_kib']) / 2**20:.1f}")
        out.append(
            f"| {label(r['gpus'], names)} | {r['tier']}p | {r['config']} | "
            + " | ".join(cells)
            + " |"
        )
    return [*out, ""]


def stages(b: Path, frames: int, names: dict[str, str]) -> list[str]:
    logs = sorted((b / "logs").glob("stages_*.log"))
    if not logs:
        return []
    parsed: list[tuple[str, str, dict[str, float], float]] = []
    order: list[str] = []
    for log in logs:
        _, gpu, tier = log.stem.split("_")
        ms: dict[str, float] = defaultdict(float)
        peak = 0.0
        for line in log.read_text().splitlines():
            if m := re.match(r"\s*time\s+(.+?):\s+([\d.]+) ms", line):
                ms[m[1]] += float(m[2])
                order += [m[1]] if m[1] not in order else []
            elif m := re.match(
                r"\s*mem\s+.+?: live\s+[\d.]+ GiB, peak\s+([\d.]+) GiB", line
            ):
                peak = max(peak, float(m[1]))
        parsed.append((gpu, tier, ms, peak))
    out = [
        "## ck stages",
        "",
        (
            "Milliseconds per frame per stage (`CK_TIME`, synchronised, so slightly above the free-running total), "
            "and the largest live-tensor peak (`CK_MEM`)."
        ),
        "",
        "| GPU | tier | " + " | ".join(order) + " | total | peak GiB |",
        "|---|---|" + "--:|" * (len(order) + 2),
    ]
    for gpu, tier, ms, peak in parsed:
        cells = [f"{ms.get(k, 0.0) / frames:.1f}" for k in order]
        total = sum(ms.values()) / frames
        out.append(
            f"| {label(gpu, names)} | {tier} | "
            + " | ".join(cells)
            + f" | {total:.1f} | {peak:.2f} |"
        )
    return [*out, ""]


def kernels(b: Path, names: dict[str, str]) -> list[str]:
    reports = sorted((b / "kernels").glob("*.csv"))
    if not reports:
        return []
    out = [
        "## ck kernels",
        "",
        (
            "One `nsys` run per GPU and tier. Busy % is kernel time over wall time (the rest is load, decode, encode "
            "and host gaps)."
        ),
        "",
        "| GPU | tier | wall s | kernel s | busy % | "
        + " | ".join(c for c, _ in CATEGORIES)
        + " | other | top kernels |",
        "|---|---|--:|--:|--:|" + "--:|" * (len(CATEGORIES) + 1) + "---|",
    ]
    for rep in reports:
        gpu, tier = rep.stem.split("_")
        text = rep.read_text()
        start = text.find("Time (%)")
        if start < 0:
            continue
        rows = [r for r in csv.reader(io.StringIO(text[start:])) if len(r) > 2][1:]
        times = [(r[-1], int(r[1]) / 1e9) for r in rows]
        total = sum(t for _, t in times)
        wall = float((b / "kernels" / f"{rep.stem}.wall").read_text())
        split = dict.fromkeys([c for c, _ in CATEGORIES] + ["other"], 0.0)
        for name, t in times:
            low = name.lower()
            split[
                next(
                    (c for c, keys in CATEGORIES if any(k in low for k in keys)),
                    "other",
                )
            ] += t
        top = ", ".join(f"{_short(n)} {t / total:.0%}" for n, t in times[:4])
        cells = [f"{wall:.1f}", f"{total:.1f}", f"{total / wall:.0%}"]
        cells += [f"{v / total:.0%}" for v in split.values()]
        out.append(
            f"| {label(gpu, names)} | {tier} | " + " | ".join(cells) + f" | {top} |"
        )
    return [*out, ""]


def _short(name: str) -> str:
    if m := re.search(r"cutlass_\w*?(\d+x\d+)_", name):
        return f"GEMM {m[1]}"
    return name.split("(")[0].split("<")[0].removeprefix("void ")[:24]


def suite() -> list[str]:
    readme = ROOT / "results" / "README.md"
    if not readme.exists():
        return []
    table = [line for line in readme.read_text().splitlines() if line.startswith("|")]
    return [
        "## Python suite (stock vs ours)",
        "",
        (
            "Per-part timings, SOL and precision against fp32 / float64 references: "
            "[summary](../README.md) and the per-GPU reports beside it."
        ),
        "",
        *table,
        "",
    ]


def main() -> None:
    b, frames = Path(sys.argv[1]), int(sys.argv[2])
    names = gpus(b)
    lines = [
        "# CorridorKey 2.0 benchmark",
        "",
        *e2e(b, frames, names),
        *stages(b, frames, names),
    ]
    lines += [*kernels(b, names), *suite()]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
