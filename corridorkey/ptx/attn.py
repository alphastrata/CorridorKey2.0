"""ck_attn: SDPA for bf16 ``[B, H, N, D]`` (D 64/80/96) with Swin region masks as one launch."""

from __future__ import annotations

import functools
import math
import types

import torch
import torch.nn.functional as F

from corridorkey import blocks

from . import arch, available, load

HEADS = (64, 80, 96)
BR, BC = 128, 64
STAGES = {75: 1, 120: 2}


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rid: torch.Tensor | None = None
) -> torch.Tensor:
    """``softmax(q kᵀ / √D) v``; with ``rid [windows, N]`` int32, a query sees only keys of its region."""
    b, h, n, d = q.shape
    out = torch.empty_like(q)
    dev = q.device.index
    shared = _stages(dev) * (2 * BC * (d + 8) * 2 + 4 * BC)
    mask = rid if rid is not None else q.new_zeros(2, dtype=torch.int32)
    windows = rid.shape[0] if rid is not None else 0
    pairs = [t.contiguous().view(torch.int32) for t in (q, k, v)]
    args = [
        *pairs,
        mask,
        mask if rid is None else spans(rid),
        out.view(torch.int32),
        n,
        h,
        windows,
        math.log2(math.e) / math.sqrt(d),
    ]
    name = f"attn{d}" + ("_masked" if rid is not None else "")
    load("ck_attn", dev).launch(
        name, (-(-n // BR), b * h, 1), (256, 1, 1), args, shared
    )
    return out


def spans(rid: torch.Tensor) -> torch.Tensor:
    """``[windows, ⌈N / 64⌉, 2]`` (min, max) region of each key tile, rows past N repeating the last token;
    kept on ``rid``, which ``_group_rid`` caches."""
    if (hit := getattr(rid, "_ck_spans", None)) is not None:
        return hit
    w, n = rid.shape
    tiles = -(-n // BC)
    padded = torch.cat([rid, rid[:, -1:].expand(w, tiles * BC - n)], 1).view(
        w, tiles, BC
    )
    rid._ck_spans = torch.stack([padded.amin(2), padded.amax(2)], 2).int().contiguous()
    return rid._ck_spans


@functools.cache
def _stages(device: int) -> int:
    return STAGES[arch("ck_attn", device)]


@functools.cache
def plain(device: int) -> bool:
    """Whether unmasked SDPA goes to ck too: only where no flash kernel exists (pre-sm_80)."""
    return torch.cuda.get_device_capability(device) < (8, 0)


def supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and q.dim() == 4
        and q.shape == k.shape == v.shape
        and q.shape[-1] in HEADS
        and all(t.dtype == torch.bfloat16 for t in (q, k, v))
    )


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    **kw: object,
) -> torch.Tensor:
    ok = dropout_p == 0.0 and not is_causal and not kw and supported(q, k, v)
    if ok and attn_mask is None and plain(q.device.index):
        return attention(q, k, v)
    key_pad = (
        attn_mask is not None
        and attn_mask.dtype == torch.bool
        and attn_mask.shape == (q.shape[0], 1, 1, q.shape[2])
    )
    if ok and key_pad:
        return attention(
            q, k, v, torch.where(attn_mask.view(q.shape[0], -1), 0, -1).int()
        )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw
    )


_RIDS: dict[tuple, torch.Tensor] = {}


def _group_rid(
    num_windows: int, groups: tuple, n: int, device: torch.device
) -> torch.Tensor:
    """``[num_windows, N]`` region ids: 0 for whole windows, one label per (token set, window) group, else −1.

    Keyed on the token tuples' identity: ``blocks`` slices fresh group tuples per call but reuses its cached
    token tuples, which stay alive in its ``lru_cache``."""
    key = (num_windows, n, device, tuple((id(tokens), wins) for tokens, wins in groups))
    if (hit := _RIDS.get(key)) is not None:
        return hit
    rid = torch.zeros(num_windows, n, dtype=torch.int32)
    partial = sorted({w for _, wins in groups for w in wins})
    rid[partial] = -1
    for label, (tokens, wins) in enumerate(groups, 1):
        rid[torch.tensor(wins)[:, None], torch.tensor(tokens)[None, :]] = label
    rid = _RIDS[key] = rid.to(device)
    return rid


def _region_groups(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_windows: int,
    full_idx: tuple[int, ...],
    groups: tuple,
) -> torch.Tensor:
    if num_windows <= 0 or not supported(q, k, v) or q.shape[0] % num_windows:
        return _stock_groups(q, k, v, num_windows, full_idx, groups)
    return attention(q, k, v, _group_rid(num_windows, groups, q.shape[2], q.device))


def _region_ids(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, region_ids: torch.Tensor | None
) -> torch.Tensor:
    if (
        region_ids is None
        or not supported(q, k, v)
        or region_ids.shape != (q.shape[0], q.shape[2])
    ):
        return _stock_ids(q, k, v, region_ids)
    return attention(q, k, v, region_ids.int())


_stock_groups = blocks._sdpa_with_region_groups
_stock_ids = blocks._sdpa_with_region_ids


class _Functional(types.ModuleType):
    def __getattr__(self, name: str) -> object:
        return _sdpa if name == "scaled_dot_product_attention" else getattr(F, name)


def patch() -> bool:
    """Route ``corridorkey.blocks``' attention through ck_attn (module-wide)."""
    if not available():
        return False
    blocks.F = _Functional("F")
    blocks._sdpa_with_region_groups = _region_groups
    blocks._sdpa_with_region_ids = _region_ids
    return True


def unpatch() -> None:
    blocks.F = F
    blocks._sdpa_with_region_groups = _stock_groups
    blocks._sdpa_with_region_ids = _stock_ids
