"""Speed of light: measured device peaks, and ideal seconds of a kernel at them (its slowest resource).

``SOL %`` = ideal ÷ measured. Stage rows count every FLOP torch's ``FlopCounterMode`` sees at the tensor peak.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode


@dataclass
class Peaks:
    tensor: float
    fp32: float
    dram: float


def seconds(fn: Callable[[], object], reps: int = 10) -> float:
    for _ in range(3):
        fn()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(reps)
    ]
    for s, e in events:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in events) / 1e3


def peaks() -> Peaks:
    n = 8192
    a, b = (
        torch.randn(n, n, device="cuda").half(),
        torch.randn(n, n, device="cuda").half(),
    )
    tensor = 2 * n**3 / seconds(lambda: a @ b)

    prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    a, b = a[:4096, :4096].float(), b[:4096, :4096].float()
    fp32 = 2 * 4096**3 / seconds(lambda: a @ b)
    torch.set_float32_matmul_precision(prec)

    x = torch.empty(2**28, device="cuda")
    y = torch.empty_like(x)
    dram = 2 * x.numel() * 4 / seconds(lambda: y.copy_(x))
    return Peaks(tensor, fp32, dram)


def bound(p: Peaks, tensor: float = 0, fp32: float = 0, dram: float = 0) -> float:
    return max(tensor / p.tensor, fp32 / p.fp32, dram / p.dram)


def flops(fn: Callable[[], object]) -> float:
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def attention(p: Peaks, q: torch.Tensor, rid: torch.Tensor | None) -> float:
    """QKᵀ and PV over the key pairs the mask keeps; q, k, v and the output each cross DRAM once."""
    b, h, n, d = q.shape
    if rid is None:
        pairs = b * n * n
    else:
        counts = [
            (r[:, None] == r[None, :]).logical_and_(r[:, None] >= 0).sum().item()
            for r in rid
        ]
        pairs = b // rid.shape[0] * sum(counts)
    return bound(p, tensor=4 * h * d * pairs, dram=4 * q.numel() * q.element_size())


def depthwise(p: Peaks, x: torch.Tensor, filters: int) -> float:
    """``filters`` 7×7 taps per element in f32; read once, written once."""
    return bound(
        p, fp32=2 * 49 * filters * x.numel(), dram=2 * x.numel() * x.element_size()
    )
