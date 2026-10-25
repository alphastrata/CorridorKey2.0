"""GPU time by kernel category: one profiled call each of the VFMs, stage 1 and stage 2, per frame."""

from __future__ import annotations

import collections
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity, profile

PARTS = ("VFMs", "stage 1", "stage 2")


def kernels(fn: Callable[[], object]) -> collections.Counter[str]:
    """Device ms per kernel name over one call of ``fn`` (after a warm-up call)."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    out: collections.Counter[str] = collections.Counter()
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            out[e.name] += e.device_time / 1e3
    return out


def per_frame(
    parts: dict[str, tuple[Callable[[], object], int]],
) -> dict[str, dict[str, float]]:
    """``{part: {kernel: ms per frame}}`` for ``{part: (call, frames per call)}``."""
    return {
        part: {k: v / n for k, v in kernels(fn).items()}
        for part, (fn, n) in parts.items()
    }
