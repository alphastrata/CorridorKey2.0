# CorridorKey 2.0 on NVIDIA GeForce RTX 5080: stock vs ours

Time per frame end to end (VFMs, stage 1, stage 2) at every resolution; kernels `ck_attn` sm_120, `ck_dw` sm_120, `ck_glue` sm_120.

## Time per frame

| resolution | stock ms / frame | ours ms / frame | speed-up | stock fps | ours fps | stock peak GiB | ours peak GiB | memory saved | precision |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| 2160p (3840×2176) | OOM | OOM | stock OOM | OOM | OOM | OOM | OOM | — | — |
| 1080p (1920×1152) | 181.6 | 123.8 | 1.47× | 5.51 | 8.08 | 6.6 | 6.4 | 3% | ≈ |
| 720p (1280×768) | 75.4 | 52.0 | 1.45× | 13.26 | 19.23 | 5.3 | 5.3 | 1% | ✓ |

## Per part

| resolution | part | stock ms / frame | ours ms / frame | speed-up | saved ms / frame | stock SOL % | ours SOL % | stock activations GiB | ours activations GiB | error ÷ stock (mean / p99.9) | precision |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| 2160p | VFMs | 39.8 | 39.7 | 1.00× | 0.1 | 61 | 61 | 1.0 | 1.0 | — / — | not ours |
| 2160p | stage 1 | 244.5 | 172.2 | 1.42× | 72.4 | — | — | 6.4 | 5.6 | — / — | — |
| 2160p | stage 2 | OOM | OOM | — | — | — | — | OOM | OOM | — / — | — |
| 2160p | end-to-end | OOM | OOM | — | — | — | — | OOM | OOM | — / — | — |
| 1080p | VFMs | 9.5 | 9.5 | 1.00× | 0.0 | 55 | 55 | 0.3 | 0.3 | — / — | not ours |
| 1080p | stage 1 | 55.5 | 36.0 | 1.54× | 19.5 | 29 | 45 | 1.8 | 1.6 | 0.940 / 0.926 | ✓ |
| 1080p | stage 2 | 113.0 | 75.5 | 1.50× | 37.6 | 32 | 48 | 0.6 | 0.5 | 0.990 / 0.989 | ✓ |
| 1080p | end-to-end | 181.6 | 123.8 | 1.47× | 57.9 | 32 | 46 | 2.0 | 1.8 | 1.008 / 1.026 | ≈ |
| 720p | VFMs | 5.8 | 5.8 | 1.00× | -0.0 | 39 | 39 | 0.1 | 0.1 | — / — | not ours |
| 720p | stage 1 | 21.8 | 14.4 | 1.51× | 7.4 | 27 | 41 | 0.7 | 0.6 | 0.946 / 0.873 | ✓ |
| 720p | stage 2 | 46.8 | 30.6 | 1.53× | 16.3 | 30 | 46 | 0.2 | 0.2 | 0.982 / 0.891 | ✓ |
| 720p | end-to-end | 75.4 | 52.0 | 1.45× | 23.4 | 30 | 43 | 0.7 | 0.7 | 0.994 / 0.995 | ✓ |

## Where a frame's time goes

GPU time by kernel category, stock and ours; **bold** rows are our kernels.

### 1080p

| kernels | stock ms / frame | stock % | ours ms / frame | ours % | saved ms / frame |
|---|--:|--:|--:|--:|--:|
| GEMM | 47.15 | 26.0 | 46.77 | 37.8 | +0.37 |
| copies, casts, layout | 40.78 | 22.5 | 13.36 | 10.8 | +27.42 |
| depthwise conv | 25.49 | 14.0 | 0.00 | 0.0 | +25.49 |
| attention (flash / SDPA) | 18.02 | 9.9 | 13.48 | 10.9 | +4.54 |
| elementwise | 16.90 | 9.3 | 3.55 | 2.9 | +13.34 |
| convolution | 14.76 | 8.1 | 14.69 | 11.9 | +0.08 |
| **ours: depthwise SwiGLU / QKV (ck_dw)** | 0.00 | 0.0 | 11.30 | 9.1 | -11.30 |
| norms | 9.48 | 5.2 | 0.59 | 0.5 | +8.89 |
| **ours: qk-norm, RoPE (ck_glue)** | 0.00 | 0.0 | 4.90 | 4.0 | -4.90 |
| **ours: attention (ck_attn)** | 0.00 | 0.0 | 4.86 | 3.9 | -4.86 |
| **ours: SwiGLU silu·mul (ck_glue)** | 0.00 | 0.0 | 2.66 | 2.1 | -2.66 |
| **ours: norm, roll, residual (ck_glue)** | 0.00 | 0.0 | 2.43 | 2.0 | -2.43 |
| resampling | 0.52 | 0.3 | 0.52 | 0.4 | -0.00 |
| GPU idle (launches, host, sync) | 8.55 | 4.7 | 4.67 | 3.8 | +3.88 |
| **total (wall)** | 181.64 | 100 | 123.78 | 100 | +57.86 |

| kernel (ours, end to end) | category | ms / frame | % of frame |
|---|---|--:|--:|
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` | GEMM | 43.66 | 35.3 |
| `cutlass_tensorop_bf16_s16816fprop_optimized_bf16_128x128_32x3_nhwc_align8` | convolution | 10.26 | 8.3 |
| `pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<96, 128, 64, 4, false, false, cut…` | attention (flash / SDPA) | 8.83 | 7.1 |
| `elementwise_kernel<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorItera…` | copies, casts, layout | 7.15 | 5.8 |
| **mix_f32** | ours: depthwise SwiGLU / QKV (ck_dw) | 6.13 | 4.9 |
| **dw_bf16** | ours: depthwise SwiGLU / QKV (ck_dw) | 4.00 | 3.2 |
| **attn64_masked** | ours: attention (ck_attn) | 3.80 | 3.1 |
| **head80** | ours: qk-norm, RoPE (ck_glue) | 3.21 | 2.6 |
| `pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cu…` | attention (flash / SDPA) | 3.12 | 2.5 |
| `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lam…` | copies, casts, layout | 2.87 | 2.3 |
| **swiglu_bf16** | ours: SwiGLU silu·mul (ck_glue) | 2.66 | 2.1 |
| `unrolled_elementwise_kernel<CUDAFunctor_add<float>, std::array<char*, 3ul>, 4, TrivialOff…` | copies, casts, layout | 2.19 | 1.8 |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8` | GEMM | 2.06 | 1.7 |
| `cutlass_tensorop_bf16_s16816dgrad_optimized_bf16_128x128_32x3_nhwc_unity_stride_align8` | convolution | 1.79 | 1.4 |
| `elementwise_kernel<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<c10::BFloat16, c10::BFloa…` | elementwise | 1.51 | 1.2 |

<details><summary>720p</summary>

### 720p

| kernels | stock ms / frame | stock % | ours ms / frame | ours % | saved ms / frame |
|---|--:|--:|--:|--:|--:|
| GEMM | 21.12 | 28.0 | 20.56 | 39.5 | +0.56 |
| copies, casts, layout | 15.18 | 20.1 | 5.66 | 10.9 | +9.52 |
| depthwise conv | 10.45 | 13.8 | 0.00 | 0.0 | +10.45 |
| convolution | 7.62 | 10.1 | 7.63 | 14.7 | -0.01 |
| elementwise | 5.53 | 7.3 | 1.36 | 2.6 | +4.17 |
| **ours: depthwise SwiGLU / QKV (ck_dw)** | 0.00 | 0.0 | 4.48 | 8.6 | -4.48 |
| attention (flash / SDPA) | 4.17 | 5.5 | 3.24 | 6.2 | +0.93 |
| norms | 3.35 | 4.4 | 0.25 | 0.5 | +3.09 |
| **ours: qk-norm, RoPE (ck_glue)** | 0.00 | 0.0 | 1.56 | 3.0 | -1.56 |
| **ours: attention (ck_attn)** | 0.00 | 0.0 | 1.36 | 2.6 | -1.36 |
| **ours: norm, roll, residual (ck_glue)** | 0.00 | 0.0 | 0.81 | 1.6 | -0.81 |
| **ours: SwiGLU silu·mul (ck_glue)** | 0.00 | 0.0 | 0.80 | 1.5 | -0.80 |
| resampling | 0.22 | 0.3 | 0.22 | 0.4 | -0.00 |
| GPU idle (launches, host, sync) | 7.81 | 10.3 | 4.07 | 7.8 | +3.73 |
| **total (wall)** | 75.44 | 100 | 51.99 | 100 | +23.44 |

| kernel (ours, end to end) | category | ms / frame | % of frame |
|---|---|--:|--:|
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` | GEMM | 18.19 | 35.0 |
| `cutlass_tensorop_bf16_s16816fprop_optimized_bf16_128x128_32x3_nhwc_align8` | convolution | 5.44 | 10.5 |
| **mix_f32** | ours: depthwise SwiGLU / QKV (ck_dw) | 2.60 | 5.0 |
| `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lam…` | copies, casts, layout | 2.23 | 4.3 |
| `elementwise_kernel<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorItera…` | copies, casts, layout | 2.03 | 3.9 |
| `pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<96, 128, 64, 4, false, false, cut…` | attention (flash / SDPA) | 1.97 | 3.8 |
| **dw_bf16** | ours: depthwise SwiGLU / QKV (ck_dw) | 1.39 | 2.7 |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8` | GEMM | 1.28 | 2.5 |
| **attn64_masked** | ours: attention (ck_attn) | 1.06 | 2.0 |
| `unrolled_elementwise_kernel<CUDAFunctor_add<float>, std::array<char*, 3ul>, 4, TrivialOff…` | copies, casts, layout | 0.88 | 1.7 |
| `pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cu…` | attention (flash / SDPA) | 0.87 | 1.7 |
| **head80** | ours: qk-norm, RoPE (ck_glue) | 0.86 | 1.7 |
| **swiglu_bf16** | ours: SwiGLU silu·mul (ck_glue) | 0.80 | 1.5 |
| `cutlass_tensorop_bf16_s16816dgrad_optimized_bf16_128x128_32x3_nhwc_unity_stride_align8` | convolution | 0.77 | 1.5 |
| **head64** | ours: qk-norm, RoPE (ck_glue) | 0.61 | 1.2 |

</details>

## Our kernels vs the ops they replace

Each distinct call of the first window stock fits, replayed on identical inputs.

| op | stock ms | ours ms | speed-up | stock SOL % | ours SOL % | stock error vs f64 (mean / max) | ours error vs f64 | precision |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|
| swiglu dw (8, 72, 120, 512) bfloat16 | 4.71 | 1.35 | 3.48× | 4 | 13 | 7.37e-06 / 3.36e-03 | 1.66e-06 / 9.37e-04 | ✓ |
| qkv dw (1, 8, 18, 30, 4608) bfloat16 | 1.60 | 0.51 | 3.13× | 6 | 19 | 1.03e-03 / 2.63e-02 | 7.61e-04 / 1.56e-02 | ✓ |
| swiglu dw (8, 18, 30, 1536) bfloat16 | 0.86 | 0.28 | 3.10× | 4 | 12 | 1.02e-04 / 2.65e-02 | 2.70e-05 / 7.38e-03 | ✓ |
| qkv dw (1, 8, 72, 120, 1536) bfloat16 | 7.08 | 2.24 | 3.16× | 7 | 23 | 9.83e-04 / 2.49e-02 | 7.29e-04 / 1.56e-02 | ✓ |
| swiglu dw (8, 72, 120, 512) float32 | 5.49 | 1.42 | 3.87× | 6 | 24 | 1.17e-05 / 5.35e-03 | 4.64e-10 / 2.51e-07 | ✓ |
| qkv dw (1, 8, 80, 128, 1536) bfloat16 | 8.42 | 2.54 | 3.31× | 7 | 24 | 8.93e-04 / 3.01e-02 | 6.61e-04 / 1.56e-02 | ✓ |
| attn regions (40, 8, 2048, 64) | 4.04 | 3.60 | 1.12× | 44 | 49 | 6.63e-04 / 4.50e-03 | 6.63e-04 / 4.46e-03 | ✓ |
| swiglu dw (1, 72, 120, 1280) float32 | 1.61 | 0.44 | 3.65× | 7 | 24 | 2.61e-05 / 3.62e-02 | 9.28e-10 / 2.01e-06 | ✓ |
| qkv dw (1, 80, 128, 3840) bfloat16 | 2.64 | 0.81 | 3.26× | 7 | 24 | 8.14e-04 / 3.31e-02 | 5.62e-04 / 1.54e-02 | ✓ |
| attn regions (40, 16, 256, 80) | 1.32 | 0.18 | 7.45× | 10 | 72 | 5.90e-04 / 4.37e-03 | 5.90e-04 / 4.39e-03 | ≈ |
| qkv dw (1, 72, 120, 3840) bfloat16 | 2.22 | 0.72 | 3.08× | 7 | 22 | 7.95e-04 / 2.07e-02 | 5.29e-04 / 1.55e-02 | ✓ |

## Precision detail

| stage | config | ms / call | ms / frame | speed-up | mean err | p99.9 err | err ÷ stock (mean / p99.9) | frames better | precision |
|---|---|--:|--:|--:|--:|--:|--:|:-:|:-:|
| stage 1 2160p | stock | 1956.17 | 244.52 | 1.00× | — | — | — / — |  |  |
| stage 2 2160p | stock | OOM | OOM | — | — | — | — / — |  |  |
| end-to-end 2160p | stock | OOM | OOM | — | — | — | — / — |  |  |
| stage 1 2160p | oracle | — | — | — | — | — | — / — |  |  |
| stage 2 2160p | oracle | — | — | — | — | — | — / — |  |  |
| end-to-end 2160p | oracle | — | — | — | — | — | — / — |  |  |
| stage 1 2160p | ours | 1377.35 | 172.17 | 1.42× | — | — | — / — |  | — |
| stage 2 2160p | ours | OOM | OOM | — | — | — | — / — |  | — |
| end-to-end 2160p | ours | OOM | OOM | — | — | — | — / — |  | — |
| stage 1 1080p | stock | 443.99 | 55.50 | 1.00× | 3.25e-03 | 9.74e-02 | 1.000 / 1.000 | 0/32 |  |
| stage 2 1080p | stock | 113.05 | 113.05 | 1.00× | 8.16e-04 | 1.57e-02 | 1.000 / 1.000 | 0/32 |  |
| end-to-end 1080p | stock | 1453.12 | 181.64 | 1.00× | 9.75e-04 | 9.99e-02 | 1.000 / 1.000 | 0/32 |  |
| stage 1 1080p | oracle | — | — | — | 1.62e-03 | 3.13e-02 | 0.525 / 0.548 | 32/32 |  |
| stage 2 1080p | oracle | — | — | — | 7.53e-04 | 1.73e-02 | 0.898 / 0.940 | 32/32 |  |
| end-to-end 1080p | oracle | — | — | — | 1.06e-03 | 1.80e-01 | 0.967 / 1.124 | 24/32 |  |
| stage 1 1080p | ours | 288.14 | 36.02 | 1.54× | 3.02e-03 | 8.74e-02 | 0.940 / 0.926 | 32/32 | ✓ |
| stage 2 1080p | ours | 75.46 | 75.46 | 1.50× | 8.10e-04 | 1.48e-02 | 0.990 / 0.989 | 32/32 | ✓ |
| end-to-end 1080p | ours | 990.21 | 123.78 | 1.47× | 1.00e-03 | 1.13e-01 | 1.008 / 1.026 | 24/32 | ≈ |
| stage 1 720p | stock | 174.30 | 21.79 | 1.00× | 2.72e-03 | 4.65e-02 | 1.000 / 1.000 | 0/32 |  |
| stage 2 720p | stock | 46.83 | 46.83 | 1.00× | 7.40e-04 | 1.24e-02 | 1.000 / 1.000 | 0/32 |  |
| end-to-end 720p | stock | 603.49 | 75.44 | 1.00× | 2.87e-03 | 6.37e-01 | 1.000 / 1.000 | 0/32 |  |
| stage 1 720p | oracle | — | — | — | 1.73e-03 | 5.98e-02 | 0.582 / 0.861 | 32/32 |  |
| stage 2 720p | oracle | — | — | — | 6.60e-04 | 9.56e-03 | 0.880 / 0.883 | 32/32 |  |
| end-to-end 720p | oracle | — | — | — | 3.26e-03 | 6.37e-01 | 1.005 / 1.048 | 23/32 |  |
| stage 1 720p | ours | 115.33 | 14.42 | 1.51× | 2.58e-03 | 4.58e-02 | 0.946 / 0.873 | 32/32 | ✓ |
| stage 2 720p | ours | 30.58 | 30.58 | 1.53× | 7.24e-04 | 1.03e-02 | 0.982 / 0.891 | 31/32 | ✓ |
| end-to-end 720p | ours | 415.96 | 51.99 | 1.45× | 2.86e-03 | 6.35e-01 | 0.994 / 0.995 | 27/32 | ✓ |

## What ours is

- `ck_attn`: flash attention, bf16, head sizes 64/80/96, with Swin's shifted-window regions as one launch (key tiles no query can see are skipped)
- `ck_dw`: the 7×7 depthwise SwiGLU / QKV mixers, NHWC, padding read in place, f32 arithmetic, residual fused
- `ck_glue`: RMSNorm → bf16 with Swin's pad and roll folded in; roll back, crop and residual add as one pass; SwiGLU as one gate|up GEMM and a fused silu·mul; qk-norm and RoPE as one pass each

## Method

- **GPU:** NVIDIA GeForce RTX 5080 (sm_120), torch 2.9.1+cu128. Measured peaks: 123 TFLOPS 16-bit tensor, 41 TFLOPS f32, 818 GB/s DRAM.
- **Kernels:** `ck_attn` sm_120, `ck_dw` sm_120, `ck_glue` sm_120. `ck_attn` runs bf16 `m16n8k16` MMA with `cp.async` double-buffered K/V tiles; unmasked attention stays on PyTorch's flash kernel, which it does not beat.
- **Data:** 4 clips × 8 consecutive frames per resolution from public green-screen footage (`just prep`), through the runtime's own preprocessing, VFMs (C-RADIO, MoGe, RVM), stage 1 over the window and stage 2 per frame, without a hint.
- **Configs:** `stock` is upstream (bf16 autocast). `ours` patches the kernels in at load (`corridorkey.ptx.enable`). `oracle` is stock with our depthwise and MLP slots in strict f32: what exact ops there would score.
- **Time:** CUDA events on each resolution's first clip, stock and ours alternating call by call on shared inputs so clock and thermal drift fall on both alike; median per config. Per frame = per call ÷ frames in the call (stage 1 and end to end run the whole window); fps = 1000 ÷ ms per frame. The oracle is a precision bound only, not timed.
- **Memory:** per call, the peak torch allocates above what is already live (after a warm-up call); peak = that plus the 4.6 GiB of weights (model and VFMs, f32).
- **Kernel split:** CUPTI device time per kernel (`torch.profiler`) over one call each of the VFMs, stage 1 and stage 2, per frame; idle = wall − kernel time.
- **SOL %:** FLOPs counted by `FlopCounterMode` at the measured tensor peak (model rows), or each op's own FLOP / byte bound (`bench/sol.py`), ÷ measured time.
- **Precision:** stage rows take the fp32 reference's inputs, so judge that stage alone; end to end chains everything. Errors are per frame (mean, 99.9th percentile), compared with stock by geometric mean ratio. End-to-end error is dominated by bf16 VFM noise that stage 2 can amplify, so exact ops can score worse than stock there, hence the oracle bound. Components are scored against float64 on identical inputs. ✓ no worse than stock (0.1 % tie); ≈ worse, but no worse than exact ops in our slots (the oracle), within 2 % (components) or not significantly (two standard errors over clips); ✗ a loss; — not judged.
