"""Kernel name → category, for the time split (torch-free, so ``render`` can use it)."""

from __future__ import annotations

import collections
import re

OURS = "ours: "
CATEGORIES = (
    (OURS + "attention (ck_attn)", r"^attn\d+"),
    (OURS + "depthwise SwiGLU / QKV (ck_dw)", r"^(mix_|dw_bf16|rms_)"),
    (OURS + "norm, roll, residual (ck_glue)", r"^(norm_bf16|unshift_add)"),
    (OURS + "SwiGLU silu·mul (ck_glue)", r"^swiglu_bf16"),
    (OURS + "qk-norm, RoPE (ck_glue)", r"^head\d+"),

    ("attention (flash / SDPA)", r"flash|fmha|attention|efficient|softmax"),
    ("depthwise conv", r"conv_depthwise"),
    ("convolution", r"fprop|dgrad|wgrad|implicit|cudnn|[Nn]hwc|[Nn]chw|conv"),
    ("GEMM", r"gemm|nvjet|wmma|cublas|sm\d+_xmma"),
    ("norms", r"layer_norm|rms_norm|group_norm|batch_norm"),
    (
        "copies, casts, layout",
        r"[Cc]opy|Memcpy|Memset|cat_|index|gather|scatter|roll|pad|permute|transpose",
    ),
    ("resampling", r"upsample|interp|pool|downsample"),
    ("elementwise", r"elementwise|Functor|silu|sigmoid|mul|add|reduce|exp"),
)


def category(name: str) -> str:
    return next((c for c, rx in CATEGORIES if re.search(rx, name)), "other")


def by_category(parts: dict[str, dict[str, float]]) -> dict[str, float]:
    out: collections.Counter[str] = collections.Counter()
    for ks in parts.values():
        for k, v in ks.items():
            out[category(k)] += v
    return dict(out)
