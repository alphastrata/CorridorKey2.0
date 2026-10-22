"""Each distinct depthwise and masked-attention call of a stock window, replayed stock and ours on identical
inputs and scored against float64."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import sol
import torch
import torch.nn.functional as F
from harness import Clip, Err, Row, config, err, frame2, ms, nan, stage1, vfm

from corridorkey import blocks
from corridorkey.blocks import DepthwiseSwiGLU2D, QKVDepthwiseSpatial
from corridorkey.model import CorridorKeyV3
from corridorkey.ptx import attn, dw

Call = tuple[
    str,
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], Err],
    Callable[[], Err],
    float,
]


def _slice(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rid: torch.Tensor
) -> torch.Tensor:
    """float64 attention on the first two batches and heads, rid-masked; queries without a region give 0."""
    q, k, v = (t[:2, :2].double() for t in (q, k, v))
    r = rid[torch.arange(q.shape[0], device=q.device) % rid.shape[0]]
    with torch.autocast("cuda", enabled=False):
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=r[:, None, :, None] == r[:, None, None, :]
        )
    return torch.where((r < 0)[:, None, :, None], 0.0, o)


def attn_call(
    kind: str,
    stock: Callable[[], torch.Tensor],
    ck: Callable[[], torch.Tensor],
    qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rid: torch.Tensor,
    p: sol.Peaks,
) -> Call:
    q = qkv[0]
    exact = _slice(*qkv, rid)

    def score(fn: Callable[[], torch.Tensor]) -> Callable[[], Err]:
        def run() -> Err:
            return err(fn()[:2, :2], exact)

        return run

    name = f"attn {kind} {tuple(q.shape)}"
    return name, stock, ck, score(stock), score(ck), 1e3 * sol.attention(p, q, rid)


def dw_call(m: torch.nn.Module, x: torch.Tensor, p: sol.Peaks) -> Call:
    swiglu = type(m) is DepthwiseSwiGLU2D
    fwd: Callable[..., torch.Tensor] = type(m).forward
    with torch.autocast("cuda", enabled=False):
        exact = fwd(copy.deepcopy(m).double(), x.double())
    fast = cast(
        Callable[..., torch.Tensor],
        dw._swiglu_forward if swiglu else dw._qkv_forward,
    )

    stock, ck = (lambda: fwd(m, x)), (lambda: fast(m, x))
    name = f"{'swiglu' if swiglu else 'qkv'} dw {tuple(x.shape)} {str(x.dtype)[6:]}"
    ideal = 1e3 * sol.depthwise(p, x, 2 if swiglu else 1)
    return name, stock, ck, lambda: err(stock(), exact), lambda: err(ck(), exact), ideal


@contextmanager
def recording(model: torch.nn.Module, p: sol.Peaks) -> Iterator[list[Call]]:
    """While active (stock), the first call of each distinct depthwise and masked-attention site is kept."""
    calls: list[Call] = []
    seen: set[tuple] = set()

    busy: list[bool] = []

    def first(key: tuple) -> bool:
        new = not busy and key not in seen
        if new:
            seen.add(key)
        return new

    @contextmanager
    def inside() -> Iterator[None]:
        busy.append(True)
        try:
            yield
        finally:
            busy.pop()

    def dw_hook(m: torch.nn.Module, a: tuple[torch.Tensor, ...]) -> None:
        if first((type(m), a[0].shape, a[0].dtype)):
            calls.append(dw_call(m, a[0].clone(), p))

    def groups(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_windows: int,
        full_idx: tuple[int, ...],
        groups_: tuple,
    ) -> torch.Tensor:
        with inside():
            out = attn._stock_groups(q, k, v, num_windows, full_idx, groups_)

        if attn.supported(q, k, v) and num_windows > 0 and first(("regions", q.shape)):
            args = (q.clone(), k.clone(), v.clone(), num_windows, full_idx, groups_)
            rid = attn._group_rid(num_windows, groups_, q.shape[2], q.device)
            calls.append(
                attn_call(
                    "regions",
                    lambda: attn._stock_groups(*args),
                    lambda: attn._region_groups(*args),
                    args[:3],
                    rid,
                    p,
                )
            )
        return out

    def ids(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        region_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        with inside():
            out = attn._stock_ids(q, k, v, region_ids)

        if (
            region_ids is not None
            and attn.supported(q, k, v)
            and first(("region ids", q.shape))
        ):

            args = (q.clone(), k.clone(), v.clone(), region_ids.clone())
            calls.append(
                attn_call(
                    "region ids",
                    lambda: attn._stock_ids(*args),
                    lambda: attn._region_ids(*args),
                    args[:3],
                    region_ids.int(),
                    p,
                )
            )
        return out

    def sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kw: Any,
    ) -> torch.Tensor:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, **kw)
        pad = (
            attn_mask is not None
            and attn_mask.dtype == torch.bool
            and attn_mask.shape == (q.shape[0], 1, 1, q.shape[2])
        )

        if pad and not kw and attn.supported(q, k, v) and first(("key-pad", q.shape)):
            qkv, mask = (q.clone(), k.clone(), v.clone()), attn_mask.clone()
            rid = torch.where(mask.view(q.shape[0], -1), 0, -1).int()

            calls.append(
                attn_call(
                    "key-pad",
                    lambda: F.scaled_dot_product_attention(*qkv, attn_mask=mask),
                    lambda: attn._sdpa(*qkv, attn_mask=mask),
                    qkv,
                    rid,
                    p,
                )
            )
        return out

    class Recorder(attn._Functional):
        def __getattr__(self, name: str) -> object:
            return sdpa if name == "scaled_dot_product_attention" else getattr(F, name)

    hooks = [
        m.register_forward_pre_hook(dw_hook)
        for m in model.modules()
        if type(m) in (DepthwiseSwiGLU2D, QKVDepthwiseSpatial) and m.kernel_size == dw.K
    ]
    setattr(blocks, "F", Recorder("F"))
    setattr(blocks, "_sdpa_with_region_groups", groups)
    setattr(blocks, "_sdpa_with_region_ids", ids)

    try:
        yield calls
    finally:
        attn.unpatch()
        for h in hooks:
            h.remove()


def timed(
    name: str,
    cfg: str,
    fn: Callable[[], object],
    score: Callable[[], Err],
    ideal: float,
) -> Row:
    try:
        return Row(name, cfg, ms(fn), ideal, nan, *score())
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return Row(name, cfg, ideal=ideal)


def component_rows(model: CorridorKeyV3, c: Clip, p: sol.Peaks) -> list[Row]:
    with config(model, "stock"), recording(model, p) as calls:
        x = c.x.cuda()
        baton = stage1(model, x, vfm(model, c.display.cuda()))[0]
        frame2(model, x[:, :1], baton[:, :1])
    with config(model, "stock"):
        return [
            timed(name, cfg, fn, score, ideal)
            for name, stock, ck, s_err, c_err, ideal in calls
            for cfg, fn, score in (("stock", stock, s_err), ("ours", ck, c_err))
        ]
