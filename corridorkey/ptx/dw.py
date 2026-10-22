"""Route ``DepthwiseSwiGLU2D`` and ``QKVDepthwiseSpatial`` through ck_dw: NHWC read in place, padding fused."""

from __future__ import annotations

import types

import torch

from corridorkey.blocks import DepthwiseSwiGLU2D, QKVDepthwiseSpatial

from . import available, load

K = 7
CG, TH, TW = 32, 8, 16
TILE = (TH + K - 1) * (TW + K - 1) * CG * 4
SUFFIX = {torch.float32: "f32", torch.bfloat16: "bf16"}


def _taps(*convs: torch.nn.Conv2d) -> torch.Tensor:
    """``[F][49][C]`` f32 weights."""
    return torch.stack(
        [c.weight.detach().float().flatten(1).t() for c in convs]
    ).contiguous()


def _bias(*convs: torch.nn.Conv2d) -> torch.Tensor:
    return torch.cat([c.bias.detach().float() for c in convs]).contiguous()


def _shared(filters: int) -> int:
    return TILE + (K * K * CG * 4 if filters == 1 else 0)


def _grid(n: int, h: int, w: int, c: int) -> tuple[int, int, int]:
    return (-(-w // TW), -(-h // TH), n * c // CG)


def _reflect(h: int, w: int) -> int:
    return int(h > K // 2 and w > K // 2)


def swiglu(m: DepthwiseSwiGLU2D, x: torch.Tensor, add: bool = False) -> torch.Tensor:
    """``m(x)`` (``x + m(x)`` with ``add``) for NHWC ``x``, contiguous, in ``x.dtype``."""
    if not hasattr(m, "_ck"):
        m._ck = (
            m.norm.weight.detach().float(),
            _taps(m.value, m.gate),
            _bias(m.value, m.gate),
            m.gamma.detach().float(),
        )
    wn, taps, bias, gamma = m._ck
    n, h, w, c = x.shape
    x = x.contiguous()
    mod = load("ck_dw", x.device.index)
    r = x.new_empty(n * h * w, dtype=torch.float32)
    sfx = SUFFIX[x.dtype]
    mod.launch(
        f"rms_{sfx}",
        (-(-r.numel() // 8), 1, 1),
        (256, 1, 1),
        [x.view(-1), r, r.numel(), c, float(m.norm.eps)],
    )
    out = torch.empty_like(x)
    args = [x, r, wn, taps, bias, gamma, out, h, w, c, _reflect(h, w), int(add)]
    mod.launch(f"mix_{sfx}", _grid(n, h, w, c), (256, 1, 1), args, _shared(2))
    return out


def qkv(m: QKVDepthwiseSpatial, y: torch.Tensor) -> torch.Tensor:
    if not hasattr(m, "_ck"):
        m._ck = (_taps(m.dw), _bias(m.dw))
    taps, bias = m._ck
    h, w, c = y.shape[-3:]
    y = y.contiguous()
    out = torch.empty_like(y)
    n = y.numel() // (h * w * c)
    args = [y, taps, bias, out, h, w, c, _reflect(h, w)]
    load("ck_dw", y.device.index).launch(
        "dw_bf16", _grid(n, h, w, c), (256, 1, 1), args, _shared(1)
    )
    return out


def _dtype(x: torch.Tensor) -> torch.dtype:
    return (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else x.dtype
    )


def _fits(m: DepthwiseSwiGLU2D, x: torch.Tensor) -> bool:
    ok = x.is_cuda and x.dim() == 4 and x.dtype in SUFFIX and x.shape[-1] % CG == 0
    return ok and m.kernel_size == K and _dtype(x) == torch.bfloat16


def _swiglu_forward(self: DepthwiseSwiGLU2D, x: torch.Tensor) -> torch.Tensor:
    return swiglu(self, x) if _fits(self, x) else DepthwiseSwiGLU2D.forward(self, x)


def mix_add(m: DepthwiseSwiGLU2D, x: torch.Tensor) -> torch.Tensor:
    """``x + m(x)`` in one pass where ck_dw takes ``x``."""
    return swiglu(m, x, add=True) if _fits(m, x) else x + m(x)


def _qkv_forward(self: QKVDepthwiseSpatial, y: torch.Tensor) -> torch.Tensor:
    ok = (
        y.is_cuda
        and y.dim() in (4, 5)
        and y.shape[-1] % CG == 0
        and self.kernel_size == K
    )
    if not ok or _dtype(y) != torch.bfloat16:
        return QKVDepthwiseSpatial.forward(self, y)
    return qkv(self, y.to(torch.bfloat16))


def patch(model: torch.nn.Module) -> int:
    """Swap the forward of every 7×7 depthwise branch; returns how many were patched."""
    if not available():
        return 0
    fwd = {DepthwiseSwiGLU2D: _swiglu_forward, QKVDepthwiseSpatial: _qkv_forward}
    mods = [m for m in model.modules() if type(m) in fwd]
    for m in mods:
        m.forward = types.MethodType(fwd[type(m)], m)
    return len(mods)


def unpatch(model: torch.nn.Module) -> None:
    for m in model.modules():
        if type(m) in (DepthwiseSwiGLU2D, QKVDepthwiseSpatial):
            m.__dict__.pop("forward", None)
