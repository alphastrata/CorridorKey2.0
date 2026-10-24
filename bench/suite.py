"""Stock vs ours on real footage at every resolution: time per frame of each part (VFMs, stage 1, stage 2, end to
end), peak memory, GPU time by kernel, our kernels against the ops they replace, and precision against fp32 /
float64 references. Writes ``results/<gpu>.json``, then ``render.py`` builds the report from every GPU's JSON.

uv run python bench/suite.py [--tiers 2160 1080 720] [--clips 4] [--frames 8] [--device cuda:0] [--frames-dir DIR]

Each tier runs ``--clips`` clips of ``--frames`` consecutive frames (``prep.py``) through the runtime's own
preprocessing, VFMs, stage 1 over the window and stage 2 per frame, without a hint. Timings, memory and the
kernel breakdown use each tier's first clip; precision uses every clip whose fp32 reference fits.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

import breakdown
import render
import sol
from components import component_rows
from harness import (
    Clip,
    Err,
    Feats,
    Row,
    config,
    e2e,
    frame2,
    keep_fp32,
    load,
    oom,
    per_frame,
    reference,
    stage1,
    stage2,
    vfm,
)

from corridorkey import ptx
from corridorkey.model import CorridorKeyV3
from corridorkey.runtime import RuntimePaths, load_model

CONFIGS = ("stock", "oracle", "ours")
TIMED = ("stock", "ours")
PARTS = ("VFMs", "stage 1", "stage 2", "end-to-end")
CRATES = ("ck_attn", "ck_dw", "ck_glue")


def peak(fn: Callable[[], object]) -> float:
    """GiB a call allocates at its peak above what is already allocated (weights, its inputs), after a warm-up
    call so one-off workspaces are not charged to whichever config runs first."""
    fn()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2**30


def event_ms(fn: Callable[[], object]) -> float:
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def inputs(
    model: CorridorKeyV3, c: Clip
) -> tuple[torch.Tensor, torch.Tensor, Feats, torch.Tensor] | None:
    """Shared timing inputs: the clip, its VFM features and a stage-1 baton (stock's, else ours if stock OOMs)."""
    x, display = c.x.cuda(), c.display.cuda()

    for name in TIMED:
        with config(model, name):
            f = oom(functools.partial(vfm, model, display), None)
            baton = (
                oom(lambda f=f: stage1(model, x, f)[0][:, :1], None)
                if f is not None
                else None
            )
        if baton is not None and f is not None:
            return x, display, f, baton
    return None


def timings(
    model: CorridorKeyV3, rows: dict[str, dict[str, Row]], c: Clip, p: sol.Peaks
) -> None:
    """Stock and ours alternate call by call on shared inputs, so clock and thermal drift fall on both alike;
    median per config. Each config is also profiled by kernel."""
    got = inputs(model, c)
    if got is None:
        return
    x, display, f, baton = got
    t = x.shape[1]

    calls: dict[str, tuple[Callable[[], object], int]] = {
        "VFMs": (lambda: vfm(model, display), t),
        "stage 1": (lambda: stage1(model, x, f), t),
        "stage 2": (lambda: frame2(model, x[:, :1], baton), 1),
        "end-to-end": (lambda: e2e(model, c), t),
    }

    for part, (fn, n) in calls.items():
        live = []
        for name in TIMED:
            with config(model, name):
                gib = oom(functools.partial(peak, fn), None)
            if gib is not None:
                rows[name][part].gib, rows[name][part].batch = gib, n
                live.append(name)

        times: dict[str, list[float]] = {name: [] for name in live}
        for i in range(10 if n == 1 or part == "VFMs" else 4):
            for name in live if i % 2 == 0 else live[::-1]:
                with config(model, name):
                    times[name].append(event_ms(fn))
        for name in live:
            rows[name][part].ms = statistics.median(times[name])

        if "stock" in live:
            with config(model, "stock"):
                rows["stock"][part].ideal = 1e3 * sol.flops(fn) / p.tensor

    for name in TIMED:
        with config(model, name):
            parts = {k: calls[k] for k in breakdown.PARTS}
            rows[name]["end-to-end"].kernels = oom(
                functools.partial(breakdown.per_frame, parts), {}
            )


def errors(model: CorridorKeyV3, rows: dict[str, Row], clips: list[Clip]) -> None:
    """Stage rows on the reference's inputs (that stage alone); end to end chained. Set only once all fit."""
    got: list[list[Err]] = [[], [], []]
    for c in clips:
        x, f = c.x.cuda(), {k: c.ref[k].cuda() for k in ("cradio", "moge", "rvm")}

        baton, aux = stage1(model, x, f)
        got[0] += per_frame([baton, aux], [c.ref["baton"], c.ref["aux"]])

        got[1] += per_frame(
            [stage2(model, x, c.ref["baton"].cuda())], [c.ref["logits"]]
        )
        got[2] += per_frame([e2e(model, c)], [c.ref["logits"]])
    for part, frames in zip(PARTS[1:], got, strict=True):
        r = rows[part]
        r.frames, r.span = frames, clips[0].x.shape[1]
        r.mean = statistics.fmean(e[0] for e in frames)
        r.max = max(e[1] for e in frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=int, nargs="+", default=[2160, 1080, 720])
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames-dir", type=Path, default=ROOT / "frames")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    dev = torch.cuda.current_device()

    model = load_model(RuntimePaths.from_package(REPO, None), bf16_weights=False).cuda()
    keep_fp32(model)
    p = sol.peaks()

    rows: list[Row] = []
    shapes: dict[int, list[int]] = {}

    for tier in a.tiers:
        clips = [
            load(d, a.frames)
            for d in sorted(a.frames_dir.glob(f"*/{tier}p"))[: a.clips]
        ]
        shapes[tier] = list(clips[0].x.shape[-2:])

        if not any(
            r.config == "stock" and r.stage.startswith(render.COMPONENTS) for r in rows
        ):
            rows += oom(functools.partial(component_rows, model, clips[0], p), [])

        reference(model, clips)

        tier_rows = {
            name: {part: Row(f"{part} {tier}p", name) for part in PARTS}
            for name in CONFIGS
        }
        oom(functools.partial(timings, model, tier_rows, clips[0], p), None)

        judged = [c for c in clips if c.ref]
        for name in CONFIGS:
            if judged:
                with config(model, name):
                    oom(functools.partial(errors, model, tier_rows[name], judged), None)
            rows += tier_rows[name].values()
        torch.cuda.empty_cache()

    gpu = torch.cuda.get_device_name(dev)
    out = ROOT / "results" / f"{gpu.replace(' ', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    meta = {
        "gpu": gpu,
        "capability": list(torch.cuda.get_device_capability(dev)),
        "torch": torch.__version__,
        "kernels": {c: ptx.arch(c, dev) for c in CRATES},
        "peaks": asdict(p),
        "tiers": a.tiers,
        "clips": a.clips,
        "frames": a.frames,
        "shapes": shapes,
        "weights_gib": sum(
            t.numel() * t.element_size()
            for t in [*model.parameters(), *model.buffers()]
        )
        / 2**30,
    }

    out.write_text(
        json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=1)
    )
    print(render.main())


if __name__ == "__main__":
    main()
