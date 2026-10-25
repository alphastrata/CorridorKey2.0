"""The runtime's pipeline as the bench drives it: clips, configs (``ref``, ``stock``, ``oracle``, ``ours``), VFMs,
stage 1 over a window, stage 2 per frame, and the fp32 reference."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import math
import types
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import cast

import sol
import torch
import torch.nn.functional as F

ptx: ModuleType | None = (
    importlib.import_module("corridorkey.ptx")
    if importlib.util.find_spec("corridorkey.ptx")
    else None
)
from corridorkey.blocks import DepthwiseSwiGLU2D, QKVDepthwiseSpatial, SwiGLU
from corridorkey.model import CorridorKeyV3
from corridorkey.runtime import (
    InferenceConfig,
    _build_foundation_inputs,
    bf16_storage,
    configure_low_vram_inference,
)
from corridorkey.video_io import preprocess_frame_dir_path, probe_frame_dir_meta

LOW_VRAM = (
    "low_vram_enc_frame_chunk",
    "inference_chunk_tokens",
    "inference_window_batch",
    "inference_strip_batch",
    "inference_query_chunk_tokens",
)
_FP32: weakref.WeakKeyDictionary[torch.nn.Module, list[tuple[torch.nn.Module, str, torch.Tensor]]] = (
    weakref.WeakKeyDictionary()
)
Err = tuple[float, float]
Feats = dict[str, torch.Tensor]
nan = math.nan


@dataclass
class Row:
    stage: str
    config: str
    ms: float = nan
    ideal: float = nan
    gib: float = nan
    mean: float = nan
    max: float = nan
    frames: list[Err] = field(default_factory=list)
    span: int = 0
    batch: int = 1
    kernels: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def frame_ms(self) -> float:
        return self.ms / self.batch


@dataclass
class Clip:
    x: torch.Tensor
    display: torch.Tensor
    ref: dict[str, torch.Tensor] = field(default_factory=dict)


def tail(got: torch.Tensor, ref: torch.Tensor) -> Err:
    """Mean and 99.9th-percentile absolute error."""
    d = (got.cuda().double() - ref.cuda().double()).abs().flatten()
    return d.mean().item(), d.kthvalue(max(1, int(0.999 * d.numel()))).values.item()


def per_frame(got: list[torch.Tensor], ref: list[torch.Tensor]) -> list[Err]:
    """Per frame, worst over outputs."""
    pairs = [
        [tail(g[0, i], r[0, i]) for g, r in zip(got, ref, strict=True)]
        for i in range(got[0].shape[1])
    ]
    return [(max(p[0] for p in f), max(p[1] for p in f)) for f in pairs]


def err(got: torch.Tensor, ref: torch.Tensor) -> Err:
    d = (got.double() - ref.double()).abs()
    return d.mean().item(), d.max().item()


def ms(fn: Callable[[], object], reps: int = 5) -> float:
    return 1e3 * sol.seconds(fn, reps)


def load(d: Path, t: int) -> Clip:
    meta = probe_frame_dir_meta(
        str(d),
        num_frames=t,
        shape_quantum=InferenceConfig.shape_quantum,
        fps=24.0,
        exr_linear=False,
        model_scale=1.0,
    )
    views = [
        preprocess_frame_dir_path(p, meta, need_model=True) for p in meta["frame_paths"]
    ]

    return Clip(
        x=torch.stack([v[0] for v in views])[None],
        display=torch.stack([v[1] for v in views])[None],
    )


def keep_fp32(model: torch.nn.Module) -> None:
    """Host copies of the fp32 weights the runtime stores in bf16 (``model`` loaded with ``bf16_weights=False``), so
    :func:`config` can give each run the weights it needs; installs the runtime's bf16 ones."""
    _FP32[model] = [
        (m, n, p.detach().cpu())
        for m in model.modules()
        if bf16_storage(m)
        for n, p in m.named_parameters(recurse=False)
    ]
    weights(model, torch.bfloat16)


def weights(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Installs ``dtype`` weights, dropping ck's fused gate|up copies so they are rebuilt from them."""
    stale = [
        (m, n, w) for m, n, w in _FP32.get(model, []) if getattr(m, n).dtype != dtype
    ]
    for m, n, w in stale:
        setattr(
            m,
            n,
            torch.nn.Parameter(w.to(getattr(m, n).device, dtype), requires_grad=False),
        )
    if stale:
        for m in model.modules():
            m.__dict__.pop("_ck", None)


@contextmanager
def config(model: torch.nn.Module, name: str) -> Iterator[None]:
    """``ref``: fp32 (no TF32), chunked as ``--low_vram`` so it fits; else bf16 autocast: ``ours`` patched,
    ``oracle`` with our depthwise and MLP slots computed in strict f32. The weight-cast cache is off: the runtime
    opens autocast per stage call, so casts never outlive one."""
    ref = name == "ref"
    weights(model, torch.float32 if name in ("ref", "oracle") else torch.bfloat16)
    if ptx is not None:
        ptx.enable(model, name == "ours")
    if ref:
        configure_low_vram_inference(model, InferenceConfig())
    if name == "oracle":
        exact_slots(model)

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = (
        name not in ("ref", "oracle")
    )

    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", torch.bfloat16, enabled=not ref, cache_enabled=False
            ),
        ):
            yield
    finally:
        for m in model.modules():
            for a in LOW_VRAM:
                vars(m).pop(a, None)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True


def _exact(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = x.dtype if type(m) is DepthwiseSwiGLU2D else torch.bfloat16
    with torch.autocast("cuda", enabled=False):
        return type(m).forward(m, x.float()).to(out)


def exact_slots(model: torch.nn.Module) -> None:
    for m in model.modules():
        if type(m) in (DepthwiseSwiGLU2D, QKVDepthwiseSpatial, SwiGLU):
            m.forward = types.MethodType(_exact, m)


def vfm(model: CorridorKeyV3, display: torch.Tensor) -> Feats:
    """The runtime's VFM features: C-RADIO and MoGe per frame, RVM over the window, normalised."""
    f = model.foundations
    b, t, _, h, w = display.shape
    radio, moge = _build_foundation_inputs(display)
    ff = f(rgb_cradio=radio.flatten(0, 1), rgb_moge=moge.flatten(0, 1))

    small = F.interpolate(
        display.flatten(0, 1).float(), size=(h // 4, w // 4), mode="bilinear"
    )

    rv = f.rvm
    assert rv is not None, "model does not load the RVM foundation"
    raw, _ = rv(small.view(b, t, 3, h // 4, w // 4).clamp(0, 1), state=None)

    def norm(s: torch.Tensor) -> torch.Tensor:
        return s.view(1, 1, -1, 1, 1).float()

    rvm = (
        (
            (raw.float() - norm(cast(torch.Tensor, f.rvm_mean)))
            / norm(cast(torch.Tensor, f.rvm_std)).clamp(min=1e-6)
        )
        .half()
        .float()
    )

    grid = (b, t, -1, h // 64, w // 64)
    return {
        "cradio": ff["cradio"].reshape(grid),
        "moge": ff["moge"].reshape(grid),
        "rvm": rvm,
    }


def stage1(
    model: CorridorKeyV3, x: torch.Tensor, f: Feats
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t = x.shape[:2]
    hint = x.new_zeros(b, t, 1, *x.shape[-2:])
    quality = torch.full((b, t), -1.0, device=x.device, dtype=x.dtype)

    s = model.stage1(
        x,
        hint,
        f["moge"],
        cradio_feat=f["cradio"],
        rvm_feat=f["rvm"],
        t_stride=1.0,
        hint_quality=quality,
    )
    return s["baton"].float(), s["aux"].float()


def frame2(model: CorridorKeyV3, x: torch.Tensor, baton: torch.Tensor) -> torch.Tensor:
    out = model.stage2(x, baton)
    return (out[0] if isinstance(out, tuple) else out).float()


def stage2(model: CorridorKeyV3, x: torch.Tensor, baton: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            frame2(model, x[:, i : i + 1], baton[:, i : i + 1])
            for i in range(x.shape[1])
        ],
        1,
    )


def e2e(model: CorridorKeyV3, c: Clip) -> torch.Tensor:
    x = c.x.cuda()
    return stage2(model, x, stage1(model, x, vfm(model, c.display.cuda()))[0])


def reference(model: CorridorKeyV3, clips: list[Clip]) -> None:
    """fp32 outputs per clip; a clip whose reference does not fit keeps ``ref`` empty (timed, not judged)."""
    with config(model, "ref"):
        for c in clips:
            oom(functools.partial(_reference, model, c), None)
            torch.cuda.empty_cache()


def _reference(model: CorridorKeyV3, c: Clip) -> None:
    x = c.x.cuda().float()
    f = vfm(model, c.display.cuda().float())
    baton, aux = stage1(model, x, f)
    c.ref = {k: v.cpu() for k, v in f.items()}
    c.ref |= {
        "baton": baton.cpu(),
        "aux": aux.cpu(),
        "logits": stage2(model, x, baton).cpu(),
    }


def oom[T](fn: Callable[[], T], fallback: T) -> T:
    try:
        return fn()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return fallback
