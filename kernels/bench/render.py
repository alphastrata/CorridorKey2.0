"""The report: ``results/<gpu>.md`` for every ``results/<gpu>.json`` that ``suite.py`` wrote, and ``results/README.md``
across GPUs. ``uv run python kernels/bench/render.py`` rebuilds them without measuring."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from categories import OURS, by_category, category

RESULTS = Path(__file__).parent / "results"
TIE, NEAR = 1.001, 1.02
COMPONENTS = ("attn", "swiglu", "qkv")
PARTS = ("VFMs", "stage 1", "stage 2", "end-to-end")
nan = math.nan
ARCH = {
    120: "`ck_attn` runs bf16 `m16n8k16` MMA with `cp.async` double-buffered K/V tiles; unmasked attention "
    "stays on PyTorch's flash kernel, which it does not beat",
    75: "Turing has no bf16 MMA: `ck_attn` widens bf16 to f16 as it stages K/V (synchronously, one stage) "
    "and runs two `m16n8k8`; all attention goes through it, as PyTorch has no bf16 flash or "
    "memory-efficient SDPA before sm_80",
}
KERNELS = (
    (
        "`ck_attn`: flash attention, bf16, head sizes 64/80/96, with Swin's shifted-window regions as one launch "
        "(key tiles no query can see are skipped)"
    ),
    (
        "`ck_dw`: the 7×7 depthwise SwiGLU / QKV mixers, NHWC, padding read in place, f32 arithmetic, residual "
        "fused"
    ),
    (
        "`ck_glue`: RMSNorm → bf16 with Swin's pad and roll folded in; roll back, crop and residual add as one "
        "pass; SwiGLU as one gate|up GEMM and a fused silu·mul; qk-norm and RoPE as one pass each"
    ),
)


@dataclass
class R:
    stage: str
    config: str
    ms: float = nan
    ideal: float = nan
    gib: float = nan
    mean: float = nan
    max: float = nan
    frames: list[list[float]] = field(default_factory=list)
    span: int = 0
    batch: int = 1
    kernels: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def frame_ms(self) -> float:
        return self.ms / self.batch

    @property
    def ran(self) -> bool:
        return not math.isnan(self.ms)


def short(kernel: str, width: int = 90) -> str:
    """Kernel name without ``void``, namespaces, argument list, or a CUTLASS ``Kernel2<…>`` wrapper."""
    k = re.sub(
        r"at::native::|\(anonymous namespace\)::", "", kernel.removeprefix("void ")
    )
    depth = 0
    for i, ch in enumerate(k):
        depth += (ch == "<") - (ch == ">")
        if ch == "(" and depth == 0 and i:
            k = k[:i].rstrip()
            break
    k = re.sub(r"^cutlass(__5x_cudnn)?::Kernel2?<(.*)>$", r"\2", k)
    return (k if len(k) <= width else k[: width - 1] + "…").replace("|", "\\|")


def num(x: float, fmt: str = ".1f", oom: str = "OOM") -> str:
    return oom if math.isnan(x) else format(x, fmt)


def geo(r: R, s: R, k: int) -> float:
    pairs = zip(r.frames, s.frames, strict=True)
    return math.exp(
        statistics.fmean(
            math.log(max(a[k], 1e-12) / max(b[k], 1e-12)) for a, b in pairs
        )
    )


def ratios(r: R, s: R) -> tuple[float, float]:
    if r.frames and s.frames:
        return geo(r, s, 0), geo(r, s, 1)
    return r.mean / s.mean if s.mean else nan, r.max / s.max if s.max else nan


def clip_logs(r: R, s: R, k: int) -> list[float]:
    """Per clip, the mean log error ratio: frames within a clip are correlated, so clips are the unit."""
    logs = [
        math.log(max(a[k], 1e-12) / max(b[k], 1e-12))
        for a, b in zip(r.frames, s.frames, strict=True)
    ]
    return [statistics.fmean(logs[i : i + r.span]) for i in range(0, len(logs), r.span)]


def precision(r: R, s: R, o: R | None) -> str:
    """✓ no worse than stock (0.1 % tie); ≈ worse, but no worse than exact ops in our slots (the oracle), within
    2 % (components) or not significantly (two standard errors over clips); ✗ a loss; — not judged."""
    if math.isnan(r.mean) or math.isnan(s.mean):
        return "—"
    if max(ratios(r, s)) <= TIE:
        return "✓"
    if o is not None and o.frames and max(ratios(r, o)) <= TIE:
        return "≈"
    if not (r.frames and s.frames):
        return "≈" if max(ratios(r, s)) <= NEAR else "✗"
    worse = any(
        len(v) > 1
        and statistics.fmean(v) - 2 * statistics.stdev(v) / math.sqrt(len(v))
        > math.log(TIE)
        for v in (clip_logs(r, s, k) for k in (0, 1))
    )
    return "✗" if worse else "≈"


class Gpu:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.meta = data["meta"]
        self.rows = [R(**r) for r in data["rows"]]
        self.at = {(r.stage, r.config): r for r in self.rows}
        self.name = self.meta["gpu"]
        self.file = path.with_suffix(".md").name

    def get(self, stage: str, config: str) -> R:
        return self.at.get((stage, config)) or R(stage, config)

    def verdict(self, stage: str) -> str:
        return precision(
            self.get(stage, "ours"),
            self.get(stage, "stock"),
            self.at.get((stage, "oracle")),
        )

    def sm(self) -> str:
        return ", ".join(f"`{c}` sm_{a}" for c, a in self.meta["kernels"].items())

    def headline(self, gpu_column: bool = False) -> list[str]:
        lead = "| GPU " if gpu_column else ""
        lines = [
            (
                f"{lead}| resolution | stock ms / frame | ours ms / frame | speed-up | stock fps | ours fps "
                "| stock peak GiB | ours peak GiB | memory saved | precision |"
            ),
            ("|---" if gpu_column else "")
            + "|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|",
        ]
        for t in self.meta["tiers"]:
            s, o = (
                self.get(f"end-to-end {t}p", "stock"),
                self.get(f"end-to-end {t}p", "ours"),
            )
            h, w = self.meta["shapes"][str(t)]
            up = s.ms / o.ms if s.ran and o.ran else nan
            weights = self.meta["weights_gib"]
            saved = (
                1 - (weights + o.gib) / (weights + s.gib) if s.ran and o.ran else nan
            )
            lines.append(
                (f"| {self.name} " if gpu_column else "")
                + f"| {t}p ({w}×{h}) | {num(s.frame_ms)} | {num(o.frame_ms)} "
                + f"| {num(up, '.2f', 'stock OOM')}{'×' if s.ran and o.ran else ''} "
                + f"| {num(1e3 / s.frame_ms, '.2f')} | {num(1e3 / o.frame_ms, '.2f')} | {num(weights + s.gib)} "
                + f"| {num(weights + o.gib)} "
                + f"| {num(100 * saved, '.0f', '—')}{'%' if s.ran and o.ran else ''} | {self.verdict(f'end-to-end {t}p')} |"
            )
        return lines

    def parts(self) -> list[str]:
        lines = [
            (
                "| resolution | part | stock ms / frame | ours ms / frame | speed-up | saved ms / frame | stock SOL % "
                "| ours SOL % | stock activations GiB | ours activations GiB | error ÷ stock (mean / p99.9) | precision |"
            ),
            "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|",
        ]
        for t in self.meta["tiers"]:
            for part in PARTS:
                stage = f"{part} {t}p"
                s, o = self.get(stage, "stock"), self.get(stage, "ours")
                both = s.ran and o.ran
                q = (
                    ratios(o, s)
                    if not math.isnan(o.mean) and not math.isnan(s.mean)
                    else (nan, nan)
                )
                ideal = s.ideal / s.batch if not math.isnan(s.ideal) else nan
                lines.append(
                    f"| {t}p | {part} | {num(s.frame_ms)} | {num(o.frame_ms)} "
                    f"| {num(s.ms / o.ms if both else nan, '.2f', '—')}{'×' if both else ''} "
                    f"| {num(s.frame_ms - o.frame_ms if both else nan, '.1f', '—')} "
                    f"| {num(100 * ideal / s.frame_ms, '.0f', '—')} | {num(100 * ideal / o.frame_ms, '.0f', '—')} "
                    f"| {num(s.gib)} | {num(o.gib)} | {num(q[0], '.3f', '—')} / {num(q[1], '.3f', '—')} "
                    f"| {self.verdict(stage) if part != 'VFMs' else 'not ours'} |"
                )
        return lines

    def time_split(self, t: int) -> list[str]:
        s, o = (
            self.get(f"end-to-end {t}p", "stock"),
            self.get(f"end-to-end {t}p", "ours"),
        )
        cat = lambda r: by_category(r.kernels)
        cs, co = cat(s), cat(o)
        busy_s, busy_o = sum(cs.values()), sum(co.values())
        rows = sorted(set(cs) | set(co), key=lambda k: -max(cs.get(k, 0), co.get(k, 0)))
        pct = lambda v, total: num(100 * v / total, ".1f", "—") if total else "—"
        lines = [
            "| kernels | stock ms / frame | stock % | ours ms / frame | ours % | saved ms / frame |",
            "|---|--:|--:|--:|--:|--:|",
        ]
        wall_s, wall_o = (
            (s.frame_ms if s.kernels else nan),
            (o.frame_ms if o.kernels else nan),
        )
        for k in rows:
            a, b = cs.get(k, 0.0), co.get(k, 0.0)
            name = f"**{k}**" if k.startswith(OURS) else k
            lines.append(
                f"| {name} | {a:.2f} | {pct(a, wall_s)} | {b:.2f} | {pct(b, wall_o)} | {a - b:+.2f} |"
            )
        idle_s, idle_o = wall_s - busy_s, wall_o - busy_o
        lines.append(
            f"| GPU idle (launches, host, sync) | {num(idle_s, '.2f', '—')} | {pct(idle_s, wall_s)} "
            f"| {num(idle_o, '.2f', '—')} | {pct(idle_o, wall_o)} | {num(idle_s - idle_o, '+.2f', '—')} |"
        )
        lines.append(
            f"| **total (wall)** | {num(wall_s, '.2f', '—')} | 100 | {num(wall_o, '.2f', '—')} | 100 "
            f"| {num(wall_s - wall_o, '+.2f', '—')} |"
        )
        return lines

    def top(self, t: int, n: int = 15) -> list[str]:
        o = self.get(f"end-to-end {t}p", "ours")
        flat: dict[str, float] = {}
        for ks in o.kernels.values():
            for k, v in ks.items():
                flat[k] = flat.get(k, 0.0) + v
        lines = [
            "| kernel (ours, end to end) | category | ms / frame | % of frame |",
            "|---|---|--:|--:|",
        ]
        for k, v in sorted(flat.items(), key=lambda kv: -kv[1])[:n]:
            c, name = category(k), short(k)
            lines.append(
                f"| {'**' + name + '**' if c.startswith(OURS) else '`' + name + '`'} | {c} | {v:.2f} | {100 * v / o.frame_ms:.1f} |"
            )
        return lines

    def components(self) -> list[str]:
        lines = [
            (
                "| op | stock ms | ours ms | speed-up | stock SOL % | ours SOL % | stock error vs f64 (mean / max) "
                "| ours error vs f64 | precision |"
            ),
            "|---|--:|--:|--:|--:|--:|--:|--:|:-:|",
        ]
        for r in (
            r
            for r in self.rows
            if r.config == "stock" and r.stage.startswith(COMPONENTS)
        ):
            o = self.get(r.stage, "ours")
            sol = [num(100 * r.ideal / x.ms, ".0f", "—") for x in (r, o)]
            lines.append(
                f"| {r.stage} | {num(r.ms, '.2f')} | {num(o.ms, '.2f')} | {num(r.ms / o.ms, '.2f', '—')}× "
                f"| {sol[0]} | {sol[1]} | {r.mean:.2e} / {r.max:.2e} | {o.mean:.2e} / {o.max:.2e} "
                f"| {precision(o, r, None)} |"
            )
        return lines

    def detail(self) -> list[str]:
        lines = [
            (
                "| stage | config | ms / call | ms / frame | speed-up | mean err | p99.9 err | err ÷ stock (mean / p99.9) "
                "| frames better | precision |"
            ),
            "|---|---|--:|--:|--:|--:|--:|--:|:-:|:-:|",
        ]
        for r in (
            r
            for r in self.rows
            if not r.stage.startswith(COMPONENTS) and not r.stage.startswith("VFMs")
        ):
            s = self.get(r.stage, "stock")
            judged = r.frames and s.frames
            q = ratios(r, s) if judged else (nan, nan)
            won = (
                f"{sum(a[0] < b[0] for a, b in zip(r.frames, s.frames, strict=True))}/{len(r.frames)}"
                if judged
                else ""
            )
            prec = self.verdict(r.stage) if r.config == "ours" else ""
            blank = "—" if r.config == "oracle" else "OOM"
            lines.append(
                f"| {r.stage} | {r.config} | {num(r.ms, '.2f', blank)} | {num(r.frame_ms, '.2f', blank)} "
                f"| {num(s.ms / r.ms, '.2f') + '×' if s.ran and r.ran else '—'} | {num(r.mean, '.2e', '—')} "
                f"| {num(r.max, '.2e', '—')} | {num(q[0], '.3f', '—')} / {num(q[1], '.3f', '—')} | {won} | {prec} |"
            )
        return lines

    def method(self) -> list[str]:
        m, p = self.meta, self.meta["peaks"]
        cap = m["capability"][0] * 10 + m["capability"][1]
        sm = next(iter(m["kernels"].values()))
        return [
            (
                f"- **GPU:** {self.name} (sm_{cap}), torch {m['torch']}. Measured peaks: {p['tensor'] / 1e12:.0f} TFLOPS "
                f"16-bit tensor, {p['fp32'] / 1e12:.0f} TFLOPS f32, {p['dram'] / 1e9:.0f} GB/s DRAM."
            ),
            f"- **Kernels:** {self.sm()}. {ARCH.get(sm, '')}.",
            (
                f"- **Data:** {m['clips']} clips × {m['frames']} consecutive frames per resolution from public green-screen "
                "footage (`just prep`), through the runtime's own preprocessing, VFMs (C-RADIO, MoGe, RVM), stage 1 over "
                "the window and stage 2 per frame, without a hint."
            ),
            (
                "- **Configs:** `stock` is upstream (bf16 autocast). `ours` patches the kernels in at load "
                "(`corridorkey.ptx.enable`). `oracle` is stock with our depthwise and MLP slots in strict f32: what exact "
                "ops there would score."
            ),
            (
                "- **Time:** CUDA events on each resolution's first clip, stock and ours alternating call by call on "
                "shared inputs so clock and thermal drift fall on both alike; median per config. Per frame = per call "
                "÷ frames in the call (stage 1 and end to end run the whole window); fps = 1000 ÷ ms per frame. The "
                "oracle is a precision bound only, not timed."
            ),
            (
                f"- **Memory:** per call, the peak torch allocates above what is already live (after a warm-up call); "
                f"peak = that plus the {self.meta['weights_gib']:.1f} GiB of weights (model and VFMs, f32)."
            ),
            (
                "- **Kernel split:** CUPTI device time per kernel (`torch.profiler`) over one call each of the VFMs, "
                "stage 1 and stage 2, per frame; idle = wall − kernel time."
            ),
            (
                "- **SOL %:** FLOPs counted by `FlopCounterMode` at the measured tensor peak (model rows), or each op's "
                "own FLOP / byte bound (`bench/sol.py`), ÷ measured time."
            ),
            "- **Precision:** stage rows take the fp32 reference's inputs, so judge that stage alone; end to end "
            "chains everything. Errors are per frame (mean, 99.9th percentile), compared with stock by geometric "
            "mean ratio. End-to-end error is dominated by bf16 VFM noise that stage 2 can amplify, so exact ops "
            "can score worse than stock there, hence the oracle bound. Components are scored against float64 on "
            "identical inputs. " + " ".join((precision.__doc__ or "").split()),
        ]


def report(g: Gpu) -> str:
    tiers = g.meta["tiers"]
    split = [
        f"### {t}p\n\n" + "\n".join(g.time_split(t)) + "\n\n" + "\n".join(g.top(t))
        for t in tiers
        if g.get(f"end-to-end {t}p", "ours").kernels
    ]
    first, rest = (split[0], split[1:]) if split else ("", [])
    folded = "".join(
        f"\n\n<details><summary>{s.split(chr(10))[0][4:]}</summary>\n\n{s}\n\n</details>"
        for s in rest
    )
    return "\n".join(
        [
            f"# CorridorKey 2.0 on {g.name}: stock vs ours",
            "",
            f"Time per frame end to end (VFMs, stage 1, stage 2) at every resolution; kernels {g.sm()}.",
            "",
            "## Time per frame",
            "",
            *g.headline(),
            "",
            "## Per part",
            "",
            *g.parts(),
            "",
            "## Where a frame's time goes",
            "",
            "GPU time by kernel category, stock and ours; **bold** rows are our kernels.",
            "",
            first + folded,
            "",
            "## Our kernels vs the ops they replace",
            "",
            "Each distinct call of the first window stock fits, replayed on identical inputs.",
            "",
            *g.components(),
            "",
            "## Precision detail",
            "",
            *g.detail(),
            "",
            "## What ours is",
            "",
            *(f"- {k}" for k in KERNELS),
            "",
            "## Method",
            "",
            *g.method(),
            "",
        ]
    )


def main() -> str:
    gpus = [Gpu(p) for p in sorted(RESULTS.glob("*.json"))]
    for g in gpus:
        (RESULTS / g.file).write_text(report(g))
    rows = [
        line
        for i, g in enumerate(gpus)
        for line in g.headline(gpu_column=True)[0 if i == 0 else 2 :]
    ]
    readme = "\n".join(
        [
            "# CorridorKey 2.0: stock vs ours",
            "",
            "End-to-end time per frame (VFMs, stage 1, stage 2) on real footage, peak memory and precision. "
            "Per-GPU reports: "
            + ", ".join(f"[{g.name}]({g.file})" for g in gpus)
            + ".",
            "",
            *rows,
            "",
            "## What ours is",
            "",
            *(f"- {k}" for k in KERNELS),
            *(
                f"- {g.name}: {g.sm()}. {ARCH.get(next(iter(g.meta['kernels'].values())), '')}."
                for g in gpus
            ),
            "",
            "Rebuild with `just bench` (measure every GPU, then render) or `just render` (render only).",
            "",
        ]
    )
    (RESULTS / "README.md").write_text(readme)
    return readme


if __name__ == "__main__":
    print(main())
