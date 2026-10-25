"""bf16-autocast GEMMs and convolutions where torch's defaults are slow. Thin-input convolutions (patchify) as
im2col and one GEMM."""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F

from . import available

BF16 = torch.bfloat16
THIN = 16


def _bf16(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == BF16
    )


def linear(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    return F.linear(x, w, b)


def _conv(self: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    if (
        _bf16(x)
        and self.groups == 1
        and self.in_channels < THIN
        and self.padding_mode == "zeros"
    ):
        return _unfold(self, x)
    return torch.nn.Conv2d.forward(self, x)


def _unfold(self: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Few input channels starve cuDNN; im2col and one GEMM instead."""
    (kh, kw), (sh, sw) = self.kernel_size, self.stride
    ph, pw = cast("tuple[int, int]", self.padding)
    (dh, dw), (h, w) = self.dilation, x.shape[-2:]
    oh, ow = (
        (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1,
        (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1,
    )
    cols = F.unfold(x.to(BF16), (kh, kw), (dh, dw), (ph, pw), (sh, sw))
    y = F.linear(cols.transpose(1, 2), self.weight.flatten(1), self.bias)
    return y.transpose(1, 2).view(x.shape[0], -1, oh, ow)


FORWARDS: dict[type, Callable] = {torch.nn.Conv2d: _conv}


def patch(model: torch.nn.Module) -> int:
    if not available():
        return 0
    mods = [m for m in model.modules() if type(m) in FORWARDS]
    for m in mods:
        setattr(m, "forward", types.MethodType(FORWARDS[type(m)], m))
    return len(mods)


def unpatch(model: torch.nn.Module) -> None:
    for m in model.modules():
        if type(m) in FORWARDS:
            m.__dict__.pop("forward", None)
