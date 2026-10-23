"""Turing has no bf16 tensor cores: there, bf16-autocast Linears run as f16 GEMMs into f32, dense convolutions
as cuDNN f16, each rounded back to bf16."""

from __future__ import annotations

import functools
import types
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F

from . import available

F16, BF16 = torch.float16, torch.bfloat16
THIN = 16


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


def _half(w: torch.Tensor, t: bool = True) -> torch.Tensor:
    if (hit := getattr(w, "_ck_f16", None)) is None:
        hit = w.detach().to(F16)
        hit = hit.t() if t else hit
        setattr(w, "_ck_f16", hit)
    return hit


def linear(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    if not _ok(x):
        return F.linear(x, w, b)
    with torch.autocast("cuda", enabled=False):
        return linear_f16(x, w, b)


def linear_f16(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
) -> torch.Tensor:
    a = x.reshape(-1, x.shape[-1]).to(F16)
    y = (
        torch.mm(a, _half(w), out_dtype=torch.float32)
        if b is None
        else torch.addmm(b.float(), a, _half(w), out_dtype=torch.float32)
    )
    return y.to(BF16).view(*x.shape[:-1], w.shape[0])


def _forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return linear(x, self.weight, self.bias)


def _conv(self: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    if not _ok(x) or self.groups != 1:
        return torch.nn.Conv2d.forward(self, x)
    if self.in_channels < THIN and self.padding_mode == "zeros":
        return _unfold(self, x)
    b = None if self.bias is None else _half(self.bias, False)
    with torch.autocast("cuda", enabled=False):
        return self._conv_forward(x.to(F16), _half(self.weight, False), b).to(BF16)


def _unfold(self: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Few input channels starve cuDNN's f16 kernels; im2col and one GEMM instead."""
    (kh, kw), (sh, sw) = self.kernel_size, self.stride
    ph, pw = cast("tuple[int, int]", self.padding)
    (dh, dw), (h, w) = self.dilation, x.shape[-2:]
    oh, ow = (
        (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1,
        (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1,
    )
    with torch.autocast("cuda", enabled=False):
        cols = F.unfold(x.to(F16), (kh, kw), (dh, dw), (ph, pw), (sh, sw))
        y = linear_f16(cols.transpose(1, 2), self.weight.flatten(1), self.bias)
    return y.transpose(1, 2).view(x.shape[0], -1, oh, ow)


def _deconv(self: torch.nn.ConvTranspose2d, x: torch.Tensor) -> torch.Tensor:
    if not _ok(x) or self.groups != 1:
        return torch.nn.ConvTranspose2d.forward(self, x)
    b = None if self.bias is None else _half(self.bias, False)
    with torch.autocast("cuda", enabled=False):
        y = F.conv_transpose2d(
            x.to(F16),
            _half(self.weight, False),
            b,
            self.stride,
            cast("tuple[int, int]", self.padding),
            self.output_padding,
            1,
            self.dilation,
        )
    return y.to(BF16)


FORWARDS: dict[type, Callable] = {
    torch.nn.Linear: _forward,
    torch.nn.Conv2d: _conv,
    torch.nn.ConvTranspose2d: _deconv,
}


def patch(model: torch.nn.Module) -> int:
    if not (available() and turing(torch.cuda.current_device())):
        return 0
    mods = [m for m in model.modules() if type(m) in FORWARDS]
    for m in mods:
        setattr(m, "forward", types.MethodType(FORWARDS[type(m)], m))
    return len(mods)


def unpatch(model: torch.nn.Module) -> None:
    for m in model.modules():
        if type(m) in FORWARDS:
            m.__dict__.pop("forward", None)
