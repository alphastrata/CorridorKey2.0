"""Turing has no bf16 tensor cores: there, bf16-autocast Linears run as f16 GEMMs into f32, then round to bf16."""

from __future__ import annotations

import functools
import types

import torch
import torch.nn.functional as F

from . import available

F16, BF16 = torch.float16, torch.bfloat16


@functools.cache
def turing(device: int) -> bool:
    return torch.cuda.get_device_capability(device) < (8, 0)


def _ok(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == BF16
        and turing(x.device.index)
    )


def _half(w: torch.Tensor) -> torch.Tensor:
    if (hit := getattr(w, "_ck_f16", None)) is None:
        hit = w.detach().to(F16).t()
        setattr(w, "_ck_f16", hit)
    return hit


def linear(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    if not _ok(x):
        return F.linear(x, w, b)
    k = x.shape[-1]
    with torch.autocast("cuda", enabled=False):
        a = x.reshape(-1, k).to(F16)
        y = (
            torch.mm(a, _half(w), out_dtype=torch.float32)
            if b is None
            else torch.addmm(b.float(), a, _half(w), out_dtype=torch.float32)
        )
    return y.to(BF16).view(*x.shape[:-1], w.shape[0])


def _forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return linear(x, self.weight, self.bias)


def patch(model: torch.nn.Module) -> int:
    if not (available() and turing(torch.cuda.current_device())):
        return 0
    mods = [m for m in model.modules() if type(m) is torch.nn.Linear]
    for m in mods:
        setattr(m, "forward", types.MethodType(_forward, m))
    return len(mods)


def unpatch(model: torch.nn.Module) -> None:
    for m in model.modules():
        if type(m) is torch.nn.Linear:
            m.__dict__.pop("forward", None)
