"""Swin window attention on the grid: QKV and the gate as grid Linears; ck_glue ``gather`` takes windows,
qk-norm and RoPE in one pass, ``scatter`` gates heads back onto the grid; no partition copies."""

from __future__ import annotations

import types
from collections.abc import Callable

import torch

from corridorkey import blocks
from corridorkey.blocks import SwinAttention2D, SwinAttention3D

from . import Arg, available, half, load
from .glue import BF16, HEADS, NORM, ROPE, _bf16, blocks_for

Attn = SwinAttention2D | SwinAttention3D


def _ok(m: Attn, x: torch.Tensor, window: tuple[int, int]) -> bool:
    h, w = x.shape[-3:-1]
    return (
        x.is_cuda
        and x.dtype == BF16
        and _bf16()
        and m.head_dim in HEADS
        and h % window[0] == 0
        and w % window[1] == 0
    )


def gather(
    m: Attn,
    qkv: torch.Tensor,
    window: tuple[int, int],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[B, T, Hp, Wp, 3C]`` → q, k, v ``[windows, heads, N, D]``, q/k normed and rotated."""
    b, t, hp, wp, _ = qkv.shape
    heads, d = m.num_heads, m.head_dim
    nw = b * hp // window[0] * wp // window[1]
    n = t * window[0] * window[1]
    out = qkv.new_empty(3, nw, heads, n, d)
    rows = out.numel() // d

    one = qkv.new_zeros(4, dtype=torch.float32)
    mode = ROPE | (NORM if m.qk_norm else 0)
    wq, wk = (
        (m.q_norm.weight.detach(), m.k_norm.weight.detach())
        if m.qk_norm
        else (one, one)
    )
    eps = float(m.q_norm.eps) if m.qk_norm else 0.0

    args: list[Arg] = [
        qkv.contiguous().view(torch.int32),
        wq,
        wk,
        cos.float().contiguous(),
        sin.float().contiguous(),
        out.view(torch.int32),
        rows,
        heads,
        t,
        hp,
        wp,
        *window,
        eps,
        mode,
    ]
    load("ck_glue", qkv.device.index).launch(
        f"gather{d}", (blocks_for(rows, d), 1, 1), (256, 1, 1), args
    )
    q, k, v = out.unbind(0)
    return q, k, v


def scatter(
    a: torch.Tensor,
    logits: torch.Tensor,
    grid: tuple[int, ...],
    window: tuple[int, int],
) -> torch.Tensor:
    """``a [windows, heads, N, D] · sigmoid(logits [B, T, Hp, Wp, heads])`` → ``[B, T, Hp, Wp, heads · D]``."""
    _, heads, _, d = a.shape
    out = a.new_empty(*grid, heads * d)
    items = logits.numel()

    args: list[Arg] = [
        a.view(torch.int32),
        logits.contiguous().view(torch.int32),
        out.view(torch.int32),
        items,
        heads,
        *grid[1:],
        *window,
    ]
    load("ck_glue", a.device.index).launch(
        f"scatter{d}", (blocks_for(items, d), 1, 1), (256, 1, 1), args
    )
    return out


def attend(
    m: Attn,
    x: torch.Tensor,
    window: tuple[int, int],
    cos: torch.Tensor,
    sin: torch.Tensor,
    region_groups: tuple | None,
) -> torch.Tensor:
    grid = x.shape[:-1] if x.dim() == 5 else (x.shape[0], 1, *x.shape[1:-1])
    y = m.qkv.qkv(x)
    if m.qkv_dw is not None:
        y = m.qkv_dw(y)
    q, k, v = gather(m, y.view(*grid, -1), window, cos, sin)
    if region_groups is not None:
        a = blocks._sdpa_with_region_groups(q, k, v, *region_groups)
    else:
        a = blocks.F.scaled_dot_product_attention(q, k, v)
    logits = half.linear(x, m.attn_gate.proj.weight).view(*grid, -1)
    return m.proj(scatter(a.to(BF16), logits, grid, window).view(x.shape))


def _forward2d(
    self: SwinAttention2D,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    region_groups: tuple | None = None,
    region_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    w = self.window_size
    if attn_mask is not None or region_ids is not None or not _ok(self, x, (w, w)):
        return SwinAttention2D.forward(self, x, attn_mask, region_groups, region_ids)
    cos, sin = self.rope.get_cos_sin(w, w, device=x.device)
    return attend(self, x, (w, w), cos, sin, region_groups)


def _forward3d(
    self: SwinAttention3D,
    x: torch.Tensor,
    t_stride: float = 1.0,
    region_groups: tuple | None = None,
    region_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    window = (self.window_H, self.window_W)
    if region_ids is not None or not _ok(self, x, window):
        return SwinAttention3D.forward(self, x, t_stride, region_groups, region_ids)
    cos, sin = self.rope.get_cos_sin(
        x.shape[1], *window, device=x.device, t_stride=t_stride
    )
    return attend(self, x, window, cos, sin, region_groups)


FORWARDS: dict[type, Callable] = {
    SwinAttention2D: _forward2d,
    SwinAttention3D: _forward3d,
}


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
