"""ck_glue: ``SwiGLU`` as one gate|up GEMM and a fused ``silu · mul``; the pre-MLP ``RMSNorm`` writes bf16;
qk-norm and RoPE as one f32 pass each over bf16 heads."""

from __future__ import annotations

import types
from typing import cast

import torch
import torch.nn.functional as F

from corridorkey import blocks
from corridorkey.blocks import RMSNorm, SwiGLU
from corridorkey.rope import apply_rotary

from . import Arg, available, half, load

BF16 = torch.bfloat16
HEADS = (64, 80, 96)
NORM, ROPE = 1, 2


def _bf16() -> bool:
    return (
        torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == BF16
    )


Shift = tuple[int, int, int, int]


def norm(m: RMSNorm, x: torch.Tensor, shift: Shift | None = None) -> torch.Tensor:
    """``m(x)`` for f32 NHWC ``x``, rounded once to bf16; with ``shift = (hp, wp, sh, sw)``, written into the
    grid zero-padded to ``hp × wp`` and rolled by ``(-sh, -sw)``, as Swin's shifted windows want it."""
    h, w, c = x.shape[-3:]
    hp, wp, sh, sw = shift or (h, w, 0, 0)
    x = x.contiguous()
    out = x.new_empty(*x.shape[:-3], hp, wp, c, dtype=BF16)
    pixels = out.numel() // c

    args: list[Arg] = [
        x,
        m.weight.detach(),
        out.view(torch.int32),
        pixels,
        c,
        float(m.eps),
        h,
        w,
        hp,
        wp,
        sh,
        sw,
    ]

    load("ck_glue", x.device.index).launch(
        "norm_bf16", (-(-pixels // 8), 1, 1), (256, 1, 1), args
    )
    return out


def unshift_add(
    x: torch.Tensor, a: torch.Tensor, shift: Shift | None = None
) -> torch.Tensor:
    """``x + roll(a, (sh, sw))[..., :h, :w, :]``: f32 ``x``, bf16 ``a`` on the padded grid."""
    h, w, c = x.shape[-3:]
    hp, wp, sh, sw = shift or (h, w, 0, 0)
    x, a = x.contiguous(), a.contiguous()
    out = torch.empty_like(x)
    pixels = x.numel() // c

    args: list[Arg] = [x, a.view(torch.int32), out, pixels, c, h, w, hp, wp, sh, sw]

    load("ck_glue", x.device.index).launch(
        "unshift_add", (-(-pixels * c // 8 // 256), 1, 1), (256, 1, 1), args
    )
    return out


def norm_mod(
    m: RMSNorm, x: torch.Tensor, mods: torch.Tensor, per: int, shift: int, scale: int
) -> torch.Tensor:
    """``m(x) · (1 + mods[scale]) + mods[shift]`` in bf16, ``mods`` rows ``[frames, ld]`` one per ``per`` pixels."""
    c = x.shape[-1]
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=BF16, device=x.device)
    pixels = x.numel() // c
    half = x.dtype == BF16
    args: list[Arg] = [
        x.view(torch.int32) if half else x,
        m.weight.detach().float(),
        mods,
        out.view(torch.int32),
        pixels,
        c,
        float(m.eps),
        per,
        mods.shape[-1],
        shift,
        scale,
    ]
    load("ck_glue", x.device.index).launch(
        "norm_mod_bf16" + ("_bf16" if half else ""),
        (-(-pixels // 8), 1, 1),
        (256, 1, 1),
        args,
    )
    return out


def add_gate(
    x: torch.Tensor, a: torch.Tensor, mods: torch.Tensor, per: int, off: int
) -> torch.Tensor:
    """``x + a · mods[off]``: f32 or bf16 ``x``, bf16 ``a``."""
    c = x.shape[-1]
    x, a = x.contiguous(), a.contiguous()
    out = torch.empty_like(x)
    pixels = x.numel() // c
    half = x.dtype == BF16
    xs, outs = (t.view(torch.int32) if half else t for t in (x, out))
    args: list[Arg] = [
        xs,
        a.view(torch.int32),
        mods,
        outs,
        pixels,
        c,
        per,
        mods.shape[-1],
        off,
    ]
    load("ck_glue", x.device.index).launch(
        "add_gate" + ("_bf16" if half else ""),
        (-(-pixels * c // 8 // 256), 1, 1),
        (256, 1, 1),
        args,
    )
    return out


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """``silu(g) · u`` of bf16 rows ``[g | u]``."""
    h = gu.shape[-1] // 2
    out = gu.new_empty(*gu.shape[:-1], h)
    rows = gu.numel() // (2 * h)

    args: list[Arg] = [gu.view(torch.int32), out.view(torch.int32), rows, h, h]
    load("ck_glue", gu.device.index).launch(
        "swiglu_bf16", (-(-rows * h // 8 // 256), 1, 1), (256, 1, 1), args
    )
    return out


def head(
    x: torch.Tensor,
    mode: int,
    weight: torch.Tensor | None = None,
    eps: float = 0.0,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """qk-norm (``NORM``) and/or interleaved RoPE (``ROPE``, f32 tables ``[N, D / 2]``) of bf16 ``[..., N, D]``."""
    d = x.shape[-1]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = x.numel() // d
    inner, stride = _layout(x)
    span = (rows // inner - 1) * stride * 2 + inner * d

    one = x.new_zeros(4, dtype=torch.float32)
    args: list[Arg] = [
        x.as_strided((span,), (1,)).view(torch.int32),
        one if weight is None else weight,
        one if cos is None else cos,
        one if sin is None else sin,
        out.view(torch.int32),
        rows,
        x.shape[-2],
        inner,
        stride,
        eps,
        mode,
    ]

    load("ck_glue", x.device.index).launch(
        f"head{d}", (blocks_for(rows, d), 1, 1), (256, 1, 1), args
    )
    return out


def blocks_for(rows: int, d: int) -> int:
    """Blocks of eight warps, ``32 // (d / 8)`` rows to a warp."""
    return -(-rows // (8 * (32 // (d // 8))))


def _layout(x: torch.Tensor) -> tuple[int, int]:
    """``(rows per contiguous block, block stride in bf16 pairs)``: contiguous, or contiguous below dim 0."""
    rows = x.numel() // x.shape[-1]
    return (rows, 0) if x.is_contiguous() else (rows // x.shape[0], x.stride(0) // 2)


def _heads(x: torch.Tensor) -> bool:
    blocked = x.is_contiguous() or (
        x.dim() > 2 and x[0].is_contiguous() and x.stride(0) % 8 == 0
    )
    return (
        x.is_cuda
        and x.dtype == BF16
        and x.shape[-1] in HEADS
        and blocked
        and x.data_ptr() % 16 == 0
    )


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    n, d = x.shape[-2:]
    if not (_heads(x) and cos.shape == sin.shape == (n, d // 2)):
        return apply_rotary(x, cos, sin)
    tables = [t.float().contiguous() for t in (cos, sin)]
    return head(x, ROPE, cos=tables[0], sin=tables[1])


def _qk_forward(self: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    if not _heads(x):
        return RMSNorm.forward(self, x)
    return head(x, NORM, self.weight.detach(), float(self.eps))


def _fuse(m: SwiGLU) -> tuple[torch.nn.Parameter, torch.Tensor | None]:
    """One ``[gate; up]`` weight (a parameter, so autocast caches its cast); ``w_gate``/``w_up`` become views."""
    pair = m.__dict__.get("_ck")
    if pair is not None:
        return cast("tuple[torch.nn.Parameter, torch.Tensor | None]", pair)

    w = torch.nn.Parameter(torch.cat([m.w_gate.weight, m.w_up.weight]).detach())
    m.w_gate.weight = torch.nn.Parameter(w[: m.hidden])
    m.w_up.weight = torch.nn.Parameter(w[m.hidden :])
    b = (
        None
        if m.w_gate.bias is None
        else torch.cat([m.w_gate.bias, m.w_up.bias]).detach()
    )
    m.__dict__["_ck"] = (w, b)
    return w, b


def _norm_forward(self: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.float32 and x.shape[-1] % 8 == 0 and _bf16():
        return norm(self, x)
    return RMSNorm.forward(self, x)


def _swiglu_forward(self: SwiGLU, x: torch.Tensor) -> torch.Tensor:
    chunked = getattr(self, "inference_chunk_tokens", 0)
    if not (
        x.is_cuda
        and self.dw is None
        and not chunked
        and self.hidden % 8 == 0
        and _bf16()
    ):
        return SwiGLU.forward(self, x)
    w, b = _fuse(self)
    return self.w_down(silu_mul(half.linear(x.to(BF16), w, b)))


def patch(model: torch.nn.Module) -> int:
    """Swap every ``SwiGLU``, each ``norm2`` that feeds one directly (no AdaLN between), every head-sized
    ``q_norm``/``k_norm``, and ``blocks.apply_rotary``; returns the module count."""
    if not available():
        return 0

    setattr(blocks, "apply_rotary", rotary)

    qk = [
        getattr(b, n)
        for b in model.modules()
        for n in ("q_norm", "k_norm")
        if type(getattr(b, n, None)) is RMSNorm
        and getattr(b, n).weight.numel() in HEADS
    ]
    for m in qk:
        setattr(m, "forward", types.MethodType(_qk_forward, m))

    mlps = [m for m in model.modules() if type(m) is SwiGLU]
    norms = [
        b.norm2
        for b in model.modules()
        if type(getattr(b, "norm2", None)) is RMSNorm
        and type(getattr(b, "mlp", None)) is SwiGLU
        and getattr(b, "adaln", None) is None
    ]
    for m in mlps:
        setattr(m, "forward", types.MethodType(_swiglu_forward, m))
    for m in norms:
        setattr(m, "forward", types.MethodType(_norm_forward, m))
    return len(mlps) + len(norms) + len(qk)


def unpatch(model: torch.nn.Module) -> None:
    setattr(blocks, "apply_rotary", apply_rotary)
    for m in model.modules():
        if type(m) in (SwiGLU, RMSNorm):
            m.__dict__.pop("forward", None)
