# CorridorKey 2.0: stock vs ours

End-to-end time per frame (VFMs, stage 1, stage 2) on real footage, peak memory and precision. Per-GPU reports: [NVIDIA GeForce RTX 5080](NVIDIA_GeForce_RTX_5080.md), [NVIDIA TITAN RTX](NVIDIA_TITAN_RTX.md).

| GPU | resolution | stock ms / frame | ours ms / frame | speed-up | stock fps | ours fps | stock peak GiB | ours peak GiB | memory saved | precision |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| NVIDIA GeForce RTX 5080 | 2160p (3840×2176) | OOM | OOM | stock OOM | OOM | OOM | OOM | OOM | — | — |
| NVIDIA GeForce RTX 5080 | 1080p (1920×1152) | 181.6 | 123.8 | 1.47× | 5.51 | 8.08 | 6.6 | 6.4 | 3% | ≈ |
| NVIDIA GeForce RTX 5080 | 720p (1280×768) | 75.4 | 52.0 | 1.45× | 13.26 | 19.23 | 5.3 | 5.3 | 1% | ✓ |
| NVIDIA TITAN RTX | 2160p (3840×2176) | OOM | 3763.4 | stock OOM | OOM | 0.27 | OOM | 10.7 | — | — |
| NVIDIA TITAN RTX | 1080p (1920×1152) | OOM | 908.3 | stock OOM | OOM | 1.10 | OOM | 6.4 | — | — |
| NVIDIA TITAN RTX | 720p (1280×768) | 536.0 | 389.2 | 1.38× | 1.87 | 2.57 | 13.1 | 5.3 | 60% | ≈ |

## What ours is

- `ck_attn`: flash attention, bf16, head sizes 64/80/96, with Swin's shifted-window regions as one launch (key tiles no query can see are skipped)
- `ck_dw`: the 7×7 depthwise SwiGLU / QKV mixers, NHWC, padding read in place, f32 arithmetic, residual fused
- `ck_glue`: RMSNorm → bf16 with Swin's pad and roll folded in; roll back, crop and residual add as one pass; SwiGLU as one gate|up GEMM and a fused silu·mul; qk-norm and RoPE as one pass each
- NVIDIA GeForce RTX 5080: `ck_attn` sm_120, `ck_dw` sm_120, `ck_glue` sm_120. `ck_attn` runs bf16 `m16n8k16` MMA with `cp.async` double-buffered K/V tiles; unmasked attention stays on PyTorch's flash kernel, which it does not beat.
- NVIDIA TITAN RTX: `ck_attn` sm_75, `ck_dw` sm_75, `ck_glue` sm_75. Turing has no bf16 MMA: `ck_attn` widens bf16 to f16 as it stages K/V (synchronously, one stage) and runs two `m16n8k8`; all attention goes through it, as PyTorch has no bf16 flash or memory-efficient SDPA before sm_80.

Rebuild with `just bench` (measure every GPU, then render) or `just render` (render only).
