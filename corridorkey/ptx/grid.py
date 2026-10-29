"""Attention on the grid through ck_ops ``split``/``merge``, as ck runs it: QKV and the gate as grid Linears;
``split`` takes each head group's windows or strips with qk-norm and RoPE, ``merge`` gates heads back onto the
grid. Swin, CSwin strips and full attention differ only in window shape."""

from __future__ import annotations

import functools
import math
import types
from collections.abc import Callable
from dataclasses import dataclass

import torch

from corridorkey import blocks
from corridorkey.blocks import (
    CSwinAttention2D,
    CSwinAttention3D,
    FullAttention3D,
    FullSpatialAttention2D,
    SwinAttention2D,
    SwinAttention3D,
)

from . import Arg, attn, available, half, load
from .glue import BF16, HEADS, NORM, ROPE, _bf16

Grid = tuple[int, int, int, int]
Attn = (
    SwinAttention2D
    | SwinAttention3D
    | CSwinAttention2D
    | CSwinAttention3D
    | FullSpatialAttention2D
    | FullAttention3D
)


@dataclass(frozen=True)
class Group:
    head0: int
    heads: int
    window: tuple[int, int]
    tables: tuple[torch.Tensor, torch.Tensor]
    regions: tuple | None = None


def _padded(grid: Grid, window: tuple[int, int]) -> tuple[int, int]:
    (h, w), (wh, ww) = grid[2:], window
    return math.ceil(h / wh) * wh, math.ceil(w / ww) * ww


@functools.cache
def _pad_rid(
    t: int, h: int, w: int, window: tuple[int, int], device: torch.device
) -> torch.Tensor | None:
    """``[windows, N]`` region ids: 0 for real tokens, −1 for padding; ``None`` when nothing pads."""
    hp, wp = _padded((1, t, h, w), window)
    if (hp, wp) == (h, w):
        return None
    wh, ww = window
    real = torch.zeros(hp, wp, dtype=torch.bool)
    real[:h, :w] = True
    real = (
        real.view(hp // wh, wh, wp // ww, ww)
        .permute(0, 2, 1, 3)
        .reshape(-1, 1, wh * ww)
    )
    rid = torch.where(real.expand(-1, t, -1), 0, -1).reshape(real.shape[0], -1)
    return rid.int().to(device)


@functools.cache
def _none(device: torch.device) -> torch.Tensor:
    return torch.zeros(2, dtype=torch.int32, device=device)


def _geo(grid: Grid, g: Group, m: Attn) -> list[Arg]:
    _, t, h, w = grid
    geo: list[Arg] = [
        t,
        h,
        w,
        *_padded(grid, g.window),
        *g.window,
        m.num_heads,
        m.head_dim,
        g.head0,
        g.heads,
    ]
    return geo


def _sorted(rid: torch.Tensor, pattern: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Region ids with each window's tokens sorted by region (padding last), and the split/merge
    ``order``: forward then inverse permutation over the first ``pattern`` windows."""
    if (hit := getattr(rid, "_ck_sorted", None)) is not None:
        return hit
    r = rid[:pattern].long()
    perm = torch.sort(torch.where(r < 0, r.max() + 1, r), dim=1, stable=True).indices
    order = torch.cat([perm.flatten(), perm.argsort(dim=1).flatten()]).int()
    hit = (torch.gather(r, 1, perm).int().contiguous(), order)
    setattr(rid, "_ck_sorted", hit)
    return hit


def _mask(
    grid: Grid, g: Group, n: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if g.regions is not None:
        windows, _, groups = g.regions
        rid = attn._group_rid(windows, groups, n, device)
    else:
        rid = _pad_rid(*grid[1:], g.window, device)
    if rid is None:
        return None
    hp, wp = _padded(grid, g.window)
    return _sorted(rid, hp // g.window[0] * wp // g.window[1])


def _split(
    m: Attn, qkv: torch.Tensor, grid: Grid, g: Group, order: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hp, wp = _padded(grid, g.window)
    windows = grid[0] * hp // g.window[0] * wp // g.window[1]
    d, n = m.head_dim, grid[1] * g.window[0] * g.window[1]
    q, k, v = (qkv.new_empty(windows, g.heads, n, d) for _ in range(3))
    rows = windows * g.heads * n

    norm = getattr(m, "qk_norm", False)
    one = qkv.new_zeros(4, dtype=torch.float32)
    wq, wk = (m.q_norm.weight.float(), m.k_norm.weight.float()) if norm else (one, one)
    args: list[Arg] = [
        qkv,
        wq,
        wk,
        *(t.float().contiguous() for t in g.tables),
        q,
        k,
        v,
        order,
        windows,
        *_geo(grid, g, m),
        float(m.q_norm.eps) if norm else 0.0,
        ROPE | (NORM if norm else 0),
    ]
    blocks_ = -(-rows // (8 * (32 // (d // 8))))
    load("ck_ops", qkv.device.index).launch("split", (blocks_, 1, 1), (256, 1, 1), args)
    return q, k, v


def attend(m: Attn, x: torch.Tensor, grid: Grid, groups: list[Group]) -> torch.Tensor:
    y = m.qkv.qkv(x)
    if m.qkv_dw is not None:
        y = m.qkv_dw(y)
    y = y.contiguous()
    gate = half.linear(x, m.attn_gate.proj.weight).float().contiguous()
    out = torch.empty(x.shape, dtype=BF16, device=x.device)
    tokens = math.prod(grid)

    for g in groups:
        n = grid[1] * g.window[0] * g.window[1]
        mask = _mask(grid, g, n, x.device)
        order = _none(x.device) if mask is None else mask[1]
        q, k, v = _split(m, y, grid, g, order)
        a = (
            blocks.F.scaled_dot_product_attention(q, k, v)
            if mask is None
            else attn.attention(q, k, v, mask[0])
        )
        args: list[Arg] = [
            a.to(BF16).contiguous(),
            gate,
            out,
            order,
            tokens,
            x.shape[-1],
            *_geo(grid, g, m),
        ]
        size = -(-tokens * g.heads * m.head_dim // 8 // 256)
        load("ck_ops", x.device.index).launch("merge", (size, 1, 1), (256, 1, 1), args)
    return m.proj(out)


def _grid2d(x: torch.Tensor) -> Grid:
    b, h, w, _ = x.shape
    return b, 1, h, w


def _grid3d(x: torch.Tensor) -> Grid:
    b, t, h, w, _ = x.shape
    return b, t, h, w


def _ok(m: Attn, x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype == BF16 and _bf16() and m.head_dim in HEADS


def _swin2d(
    self: SwinAttention2D,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    region_groups: tuple | None = None,
    region_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if attn_mask is not None or region_ids is not None or not _ok(self, x):
        return SwinAttention2D.forward(self, x, attn_mask, region_groups, region_ids)
    w = self.window_size
    tables = self.rope.get_cos_sin(w, w, device=x.device)
    grid = _grid2d(x)
    return attend(
        self, x, grid, [Group(0, self.num_heads, (w, w), tables, region_groups)]
    )


def _swin3d(
    self: SwinAttention3D,
    x: torch.Tensor,
    t_stride: float = 1.0,
    region_groups: tuple | None = None,
    region_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if region_ids is not None or not _ok(self, x):
        return SwinAttention3D.forward(self, x, t_stride, region_groups, region_ids)
    window = (self.window_H, self.window_W)
    tables = self.rope.get_cos_sin(
        x.shape[1], *window, device=x.device, t_stride=t_stride
    )
    grid = _grid3d(x)
    return attend(
        self, x, grid, [Group(0, self.num_heads, window, tables, region_groups)]
    )


def _cswin(
    m: CSwinAttention2D | CSwinAttention3D,
    x: torch.Tensor,
    grid: Grid,
    tables: Callable,
) -> torch.Tensor:
    _, t, h, w = grid
    groups = [
        Group(0, m.h_heads, (h, 1), tables(t, h, 1)),
        Group(m.h_heads, m.w_heads, (1, w), tables(t, 1, w)),
    ]
    return attend(m, x, grid, groups)


def _cswin2d(self: CSwinAttention2D, x: torch.Tensor) -> torch.Tensor:
    if not _ok(self, x):
        return CSwinAttention2D.forward(self, x)
    grid = _grid2d(x)
    return _cswin(
        self, x, grid, lambda _, h, w: self.rope.get_cos_sin(h, w, device=x.device)
    )


def _cswin3d(
    self: CSwinAttention3D, x: torch.Tensor, t_stride: float = 1.0
) -> torch.Tensor:
    if not _ok(self, x):
        return CSwinAttention3D.forward(self, x, t_stride)
    return _cswin(
        self,
        x,
        _grid3d(x),
        lambda t, h, w: self.rope.get_cos_sin(
            t, h, w, device=x.device, t_stride=t_stride
        ),
    )


def _full2d(self: FullSpatialAttention2D, x: torch.Tensor) -> torch.Tensor:
    if not _ok(self, x):
        return FullSpatialAttention2D.forward(self, x)
    b, t, h, w, _ = x.shape
    tables = self.rope.get_cos_sin(h, w, device=x.device)
    return attend(self, x, (b * t, 1, h, w), [Group(0, self.num_heads, (h, w), tables)])


def _full3d(
    self: FullAttention3D, x: torch.Tensor, t_stride: float = 1.0
) -> torch.Tensor:
    if not _ok(self, x):
        return FullAttention3D.forward(self, x, t_stride)
    b, t, h, w, _ = x.shape
    tables = self.rope.get_cos_sin(t, h, w, device=x.device, t_stride=t_stride)
    return attend(self, x, (b, t, h, w), [Group(0, self.num_heads, (h, w), tables)])


FORWARDS: dict[type, Callable] = {
    SwinAttention2D: _swin2d,
    SwinAttention3D: _swin3d,
    CSwinAttention2D: _cswin2d,
    CSwinAttention3D: _cswin3d,
    FullSpatialAttention2D: _full2d,
    FullAttention3D: _full3d,
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
