"""Transformer blocks with their glue fused: norm1 (→ Swin's pad → roll) as one bf16 write, (roll back → crop →)
residual add as one pass (ck_glue ``norm_bf16`` / ``unshift_add``), the local mix's residual inside ck_dw.
Attention inputs only feed Linears, so they round as stock's."""

from __future__ import annotations

import math
import types
from collections.abc import Callable

import torch

from corridorkey import blocks
from corridorkey.blocks import (
    CSwinBlock2D,
    CSwinBlock3D,
    FullResolutionRecurrentMemorySite,
    FullSpatialBlock2D,
    FullSpatialBlock3D,
    FullSpatioTemporalBlock,
    SwinBlock2D,
    SwinBlock3D,
)

from . import available
from .dw import mix_add
from .glue import BF16, norm, unshift_add


def _ok(block: torch.nn.Module, x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.float32
        and x.shape[-1] % 8 == 0
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == BF16
        and getattr(block, "adaln", None) is None
    )


def _attend(
    block: Block,
    x: torch.Tensor,
    window: tuple[int, int],
    run: Callable[[torch.Tensor, tuple | None], torch.Tensor],
    groups: Callable[[int, int, int, int], tuple],
) -> torch.Tensor:
    """``x + attn(norm1(x))`` with Swin's shift; ``groups(H, W, Hp, Wp)`` gives the shifted region groups."""
    sh, sw = getattr(block, "shift_h", 0), getattr(block, "shift_w", 0)
    h, w = x.shape[-3:-1]
    shift = None
    if sh or sw:
        hp, wp = (math.ceil(n / k) * k for n, k in zip((h, w), window, strict=True))
        shift = (hp, wp, sh, sw)

    a = run(norm(block.norm1, x, shift), groups(h, w, *shift[:2]) if shift else None)
    if a.dtype == BF16:
        return unshift_add(x, a, shift)

    back = a.roll(shift[2:], (-3, -2))[..., :h, :w, :] if shift else a
    return x + back


Block = (
    SwinBlock3D
    | SwinBlock2D
    | FullSpatioTemporalBlock
    | FullSpatialBlock2D
    | FullSpatialBlock3D
    | CSwinBlock2D
    | CSwinBlock3D
)


def _mlp(block: Block, x: torch.Tensor) -> torch.Tensor:
    """``x + mlp(norm2(x))``, the add in one vectorised pass when the MLP rounds to bf16."""
    y = block.mlp(block.norm2(x))
    return unshift_add(x, y) if y.dtype == BF16 else x + y


def _mix(block: Block, x: torch.Tensor) -> torch.Tensor:
    if block.local_mix is None:
        return x
    return mix_add(block.local_mix, x.flatten(0, x.dim() - 4)).view(x.shape)


def _forward3d(
    self: SwinBlock3D,
    x: torch.Tensor,
    t_stride: float = 1.0,
    adaln_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _ok(self, x):
        return SwinBlock3D.forward(self, x, t_stride, adaln_cond)
    window = (self.window_H, self.window_W)
    t = x.shape[1]

    x = _attend(
        self,
        x,
        window,
        lambda y, g: (
            self.attn(y, t_stride=t_stride, region_groups=g)
            if g
            else self.attn(y, t_stride=t_stride)
        ),
        lambda h, w, hp, wp: blocks._shifted_region_groups_3d_cached(
            t, h, w, hp, wp, *window, self.shift_h, self.shift_w
        ),
    )

    return _mlp(self, _mix(self, x))


def _forward2d(self: SwinBlock2D, x: torch.Tensor) -> torch.Tensor:
    if not _ok(self, x):
        return SwinBlock2D.forward(self, x)
    k = self.window_size

    x = _attend(
        self,
        x,
        (k, k),
        lambda y, g: self.attn(y, region_groups=g) if g else self.attn(y),
        lambda h, w, hp, wp: blocks._shifted_region_groups_2d_cached(
            h, w, hp, wp, k, self.shift_h, self.shift_w
        ),
    )

    return _mlp(self, _mix(self, x))


def _full3d(
    self: FullSpatioTemporalBlock,
    x: torch.Tensor,
    t_stride: float = 1.0,
    recurrent_site: FullResolutionRecurrentMemorySite | None = None,
    recurrent_state: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if recurrent_site is not None or not _ok(self, x):
        return FullSpatioTemporalBlock.forward(
            self, x, t_stride, recurrent_site, recurrent_state
        )

    x = _attend(self, x, (1, 1), lambda y, _: self.attn(y, t_stride=t_stride), _none)

    return _mlp(self, _mix(self, x))


def _full2d(self: FullSpatialBlock2D, x: torch.Tensor) -> torch.Tensor:
    if not _ok(self, x):
        return FullSpatialBlock2D.forward(self, x)

    x = _attend(self, x, (1, 1), lambda y, _: self.attn(y[:, None]).squeeze(1), _none)

    return _mlp(self, _mix(self, x))


def _cswin2d(self: CSwinBlock2D, x: torch.Tensor) -> torch.Tensor:
    if not _ok(self, x):
        return CSwinBlock2D.forward(self, x)
    x = _attend(self, x, (1, 1), lambda y, _: self.attn(y), _none)
    return _mlp(self, _mix(self, x))


def _cswin3d(
    self: CSwinBlock3D, x: torch.Tensor, t_stride: float = 1.0
) -> torch.Tensor:
    if not _ok(self, x):
        return CSwinBlock3D.forward(self, x, t_stride)
    x = _attend(self, x, (1, 1), lambda y, _: self.attn(y, t_stride=t_stride), _none)
    return _mlp(self, _mix(self, x))


def _spatial3d(
    self: FullSpatialBlock3D,
    x: torch.Tensor,
    t_stride: float = 1.0,
    adaln_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _ok(self, x):
        return FullSpatialBlock3D.forward(self, x, t_stride, adaln_cond)
    x = _attend(self, x, (1, 1), lambda y, _: self.attn(y), _none)
    return _mlp(self, _mix(self, x))


def _none(*_: int) -> tuple:
    return ()


FORWARDS: dict[type, Callable] = {
    SwinBlock3D: _forward3d,
    SwinBlock2D: _forward2d,
    FullSpatioTemporalBlock: _full3d,
    FullSpatialBlock2D: _full2d,
    CSwinBlock2D: _cswin2d,
    CSwinBlock3D: _cswin3d,
    FullSpatialBlock3D: _spatial3d,
}


def patch(model: torch.nn.Module) -> int:
    if not available():
        return 0
    fwd = FORWARDS
    mods = [m for m in model.modules() if type(m) in fwd]
    for m in mods:
        setattr(m, "forward", types.MethodType(fwd[type(m)], m))
    return len(mods)


def unpatch(model: torch.nn.Module) -> None:
    for m in model.modules():
        if type(m) in FORWARDS:
            m.__dict__.pop("forward", None)
