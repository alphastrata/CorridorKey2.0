# CorridorKey 2.0 on NVIDIA TITAN RTX: stock vs ours

Time per frame end to end (VFMs, stage 1, stage 2) at every resolution; kernels `ck_attn` sm_75, `ck_dw` sm_75, `ck_glue` sm_75.

## Time per frame

| resolution | stock ms / frame | ours ms / frame | speed-up | stock fps | ours fps | stock peak GiB | ours peak GiB | memory saved | precision |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| 2160p (3840×2176) | OOM | 3763.4 | stock OOM | OOM | 0.27 | OOM | 10.7 | — | — |
| 1080p (1920×1152) | OOM | 908.3 | stock OOM | OOM | 1.10 | OOM | 6.4 | — | — |
| 720p (1280×768) | 536.0 | 389.2 | 1.38× | 1.87 | 2.57 | 13.1 | 5.3 | 60% | ≈ |

## Per part

| resolution | part | stock ms / frame | ours ms / frame | speed-up | saved ms / frame | stock SOL % | ours SOL % | stock activations GiB | ours activations GiB | error ÷ stock (mean / p99.9) | precision |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| 2160p | VFMs | 588.8 | 587.4 | 1.00× | 1.4 | 5 | 5 | 5.2 | 5.2 | — / — | not ours |
| 2160p | stage 1 | OOM | 914.7 | — | — | — | — | OOM | 5.6 | — / — | — |
| 2160p | stage 2 | OOM | 2194.2 | — | — | — | — | OOM | 1.7 | — / — | — |
| 2160p | end-to-end | OOM | 3763.4 | — | — | — | — | OOM | 6.1 | — / — | — |
| 1080p | VFMs | 108.6 | 108.7 | 1.00× | -0.1 | 6 | 6 | 0.5 | 0.5 | — / — | not ours |
| 1080p | stage 1 | OOM | 224.6 | — | — | — | — | OOM | 1.6 | — / — | — |
| 1080p | stage 2 | 886.5 | 566.1 | 1.57× | 320.4 | 5 | 8 | 10.4 | 0.5 | — / — | — |
| 1080p | end-to-end | OOM | 908.3 | — | — | — | — | OOM | 1.8 | — / — | — |
| 720p | VFMs | 48.1 | 48.2 | 1.00× | -0.0 | 6 | 6 | 0.0 | 0.2 | — / — | not ours |
| 720p | stage 1 | 165.7 | 97.2 | 1.70× | 68.5 | 5 | 8 | 8.4 | 0.6 | 0.903 / 0.830 | ✓ |
| 720p | stage 2 | 322.2 | 241.0 | 1.34× | 81.2 | 6 | 8 | 2.2 | 0.2 | 0.987 / 0.941 | ✓ |
| 720p | end-to-end | 536.0 | 389.2 | 1.38× | 146.8 | 5 | 7 | 8.5 | 0.7 | 1.026 / 1.020 | ≈ |

## Where a frame's time goes

GPU time by kernel category, stock and ours; **bold** rows are our kernels.

### 2160p

| kernels | stock ms / frame | stock % | ours ms / frame | ours % | saved ms / frame |
|---|--:|--:|--:|--:|--:|
| GEMM | 0.00 | — | 2661.19 | 70.7 | -2661.19 |
| **ours: attention (ck_attn)** | 0.00 | — | 398.94 | 10.6 | -398.94 |
| **ours: qk-norm, RoPE (ck_glue)** | 0.00 | — | 139.60 | 3.7 | -139.60 |
| copies, casts, layout | 0.00 | — | 132.22 | 3.5 | -132.22 |
| **ours: depthwise SwiGLU / QKV (ck_dw)** | 0.00 | — | 127.78 | 3.4 | -127.78 |
| elementwise | 0.00 | — | 112.52 | 3.0 | -112.52 |
| other | 0.00 | — | 50.07 | 1.3 | -50.07 |
| **ours: norm, roll, residual (ck_glue)** | 0.00 | — | 26.10 | 0.7 | -26.10 |
| attention (flash / SDPA) | 0.00 | — | 18.76 | 0.5 | -18.76 |
| **ours: SwiGLU silu·mul (ck_glue)** | 0.00 | — | 16.87 | 0.4 | -16.87 |
| norms | 0.00 | — | 6.23 | 0.2 | -6.23 |
| resampling | 0.00 | — | 1.56 | 0.0 | -1.56 |
| convolution | 0.00 | — | 0.02 | 0.0 | -0.02 |
| GPU idle (launches, host, sync) | — | — | 71.58 | 1.9 | — |
| **total (wall)** | — | 100 | 3763.45 | 100 | — |

| kernel (ours, end to end) | category | ms / frame | % of frame |
|---|---|--:|--:|
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, true, false, 6, 4, 6, 3, 4>` | GEMM | 2340.70 | 62.2 |
| **attn80** | ours: attention (ck_attn) | 227.67 | 6.0 |
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, false, false, 6, 3, 5, 3, 3>` | GEMM | 159.40 | 4.2 |
| `volta_sgemm_64x64_nn` | GEMM | 114.78 | 3.0 |
| **attn64** | ours: attention (ck_attn) | 100.46 | 2.7 |
| **head80** | ours: qk-norm, RoPE (ck_glue) | 92.64 | 2.5 |
| **mix_f32** | ours: depthwise SwiGLU / QKV (ck_dw) | 69.99 | 1.9 |
| `elementwise_kernel<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorItera…` | copies, casts, layout | 55.40 | 1.5 |
| **dw_bf16** | ours: depthwise SwiGLU / QKV (ck_dw) | 43.08 | 1.1 |
| `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lam…` | copies, casts, layout | 42.32 | 1.1 |
| `elementwise_kernel<128, 2, gpu_kernel_impl_nocast<where_kernel_impl(at::TensorIterator&):…` | elementwise | 42.09 | 1.1 |
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, false, true, 6, 4, 6, 3, 4>` | GEMM | 41.72 | 1.1 |
| **attn96** | ours: attention (ck_attn) | 39.29 | 1.0 |
| **head64** | ours: qk-norm, RoPE (ck_glue) | 38.07 | 1.0 |
| `cunn_SoftMaxForwardReg<float, float, float, SoftMaxForwardEpilogue, long, 3>` | other | 37.61 | 1.0 |

<details><summary>1080p</summary>

### 1080p

| kernels | stock ms / frame | stock % | ours ms / frame | ours % | saved ms / frame |
|---|--:|--:|--:|--:|--:|
| GEMM | 0.00 | — | 692.50 | 76.2 | -692.50 |
| **ours: qk-norm, RoPE (ck_glue)** | 0.00 | — | 37.41 | 4.1 | -37.41 |
| **ours: depthwise SwiGLU / QKV (ck_dw)** | 0.00 | — | 36.44 | 4.0 | -36.44 |
| **ours: attention (ck_attn)** | 0.00 | — | 36.38 | 4.0 | -36.38 |
| copies, casts, layout | 0.00 | — | 32.25 | 3.6 | -32.25 |
| elementwise | 0.00 | — | 16.70 | 1.8 | -16.70 |
| **ours: norm, roll, residual (ck_glue)** | 0.00 | — | 6.80 | 0.7 | -6.80 |
| **ours: SwiGLU silu·mul (ck_glue)** | 0.00 | — | 4.46 | 0.5 | -4.46 |
| attention (flash / SDPA) | 0.00 | — | 3.16 | 0.3 | -3.16 |
| other | 0.00 | — | 2.26 | 0.2 | -2.26 |
| norms | 0.00 | — | 1.68 | 0.2 | -1.68 |
| resampling | 0.00 | — | 0.93 | 0.1 | -0.93 |
| GPU idle (launches, host, sync) | — | — | 37.36 | 4.1 | — |
| **total (wall)** | — | 100 | 908.31 | 100 | — |

| kernel (ours, end to end) | category | ms / frame | % of frame |
|---|---|--:|--:|
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, true, false, 6, 4, 6, 3, 4>` | GEMM | 635.17 | 69.9 |
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, false, false, 6, 3, 5, 3, 3>` | GEMM | 37.38 | 4.1 |
| **head80** | ours: qk-norm, RoPE (ck_glue) | 24.62 | 2.7 |
| **mix_f32** | ours: depthwise SwiGLU / QKV (ck_dw) | 19.89 | 2.2 |
| **attn80** | ours: attention (ck_attn) | 17.02 | 1.9 |
| `elementwise_kernel<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorItera…` | copies, casts, layout | 15.81 | 1.7 |
| **dw_bf16** | ours: depthwise SwiGLU / QKV (ck_dw) | 12.45 | 1.4 |
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, false, true, 6, 4, 6, 3, 4>` | GEMM | 10.80 | 1.2 |
| **head64** | ours: qk-norm, RoPE (ck_glue) | 10.70 | 1.2 |
| `volta_sgemm_64x64_nn` | GEMM | 8.47 | 0.9 |
| `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lam…` | copies, casts, layout | 8.11 | 0.9 |
| **attn64** | ours: attention (ck_attn) | 6.97 | 0.8 |
| **attn64_masked** | ours: attention (ck_attn) | 6.85 | 0.8 |
| **swiglu_bf16** | ours: SwiGLU silu·mul (ck_glue) | 4.46 | 0.5 |
| `unrolled_elementwise_kernel<CUDAFunctor_add<float>, std::array<char*, 3ul>, 4, TrivialOff…` | copies, casts, layout | 4.40 | 0.5 |

</details>

<details><summary>720p</summary>

### 720p

| kernels | stock ms / frame | stock % | ours ms / frame | ours % | saved ms / frame |
|---|--:|--:|--:|--:|--:|
| GEMM | 356.09 | 66.4 | 297.93 | 76.5 | +58.16 |
| copies, casts, layout | 54.71 | 10.2 | 15.12 | 3.9 | +39.59 |
| elementwise | 51.71 | 9.6 | 6.37 | 1.6 | +45.34 |
| depthwise conv | 21.90 | 4.1 | 0.00 | 0.0 | +21.90 |
| **ours: depthwise SwiGLU / QKV (ck_dw)** | 0.00 | 0.0 | 15.30 | 3.9 | -15.30 |
| **ours: qk-norm, RoPE (ck_glue)** | 0.00 | 0.0 | 12.93 | 3.3 | -12.93 |
| other | 11.60 | 2.2 | 1.13 | 0.3 | +10.47 |
| attention (flash / SDPA) | 9.18 | 1.7 | 0.62 | 0.2 | +8.56 |
| **ours: attention (ck_attn)** | 0.00 | 0.0 | 9.11 | 2.3 | -9.11 |
| norms | 8.48 | 1.6 | 0.79 | 0.2 | +7.69 |
| **ours: norm, roll, residual (ck_glue)** | 0.00 | 0.0 | 3.03 | 0.8 | -3.03 |
| **ours: SwiGLU silu·mul (ck_glue)** | 0.00 | 0.0 | 1.98 | 0.5 | -1.98 |
| resampling | 0.46 | 0.1 | 0.45 | 0.1 | +0.01 |
| GPU idle (launches, host, sync) | 21.91 | 4.1 | 24.46 | 6.3 | -2.55 |
| **total (wall)** | 536.05 | 100 | 389.23 | 100 | +146.82 |

| kernel (ours, end to end) | category | ms / frame | % of frame |
|---|---|--:|--:|
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, true, false, 6, 4, 6, 3, 4>` | GEMM | 271.21 | 69.7 |
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, false, false, 6, 3, 5, 3, 3>` | GEMM | 19.77 | 5.1 |
| **head80** | ours: qk-norm, RoPE (ck_glue) | 8.55 | 2.2 |
| **mix_f32** | ours: depthwise SwiGLU / QKV (ck_dw) | 8.44 | 2.2 |
| `elementwise_kernel<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorItera…` | copies, casts, layout | 5.91 | 1.5 |
| `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lam…` | copies, casts, layout | 5.25 | 1.3 |
| `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, false, true, 6, 4, 6, 3, 4>` | GEMM | 4.92 | 1.3 |
| **dw_bf16** | ours: depthwise SwiGLU / QKV (ck_dw) | 4.80 | 1.2 |
| **head64** | ours: qk-norm, RoPE (ck_glue) | 3.94 | 1.0 |
| **attn80** | ours: attention (ck_attn) | 3.72 | 1.0 |
| **attn64** | ours: attention (ck_attn) | 2.12 | 0.5 |
| `unrolled_elementwise_kernel<CUDAFunctor_add<float>, std::array<char*, 3ul>, 4, TrivialOff…` | copies, casts, layout | 2.01 | 0.5 |
| **swiglu_bf16** | ours: SwiGLU silu·mul (ck_glue) | 1.98 | 0.5 |
| **norm_bf16** | ours: norm, roll, residual (ck_glue) | 1.93 | 0.5 |
| **attn64_masked** | ours: attention (ck_attn) | 1.85 | 0.5 |

</details>

## Our kernels vs the ops they replace

Each distinct call of the first window stock fits, replayed on identical inputs.

| op | stock ms | ours ms | speed-up | stock SOL % | ours SOL % | stock error vs f64 (mean / max) | ours error vs f64 | precision |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|
| swiglu dw (8, 48, 80, 512) bfloat16 | 4.12 | 1.88 | 2.19× | 5 | 11 | 6.89e-06 / 2.97e-03 | 1.61e-06 / 4.88e-04 | ✓ |
| qkv dw (1, 8, 12, 20, 4608) bfloat16 | 1.28 | 0.90 | 1.43× | 5 | 7 | 1.03e-03 / 2.58e-02 | 7.57e-04 / 1.56e-02 | ✓ |
| swiglu dw (8, 12, 20, 1536) bfloat16 | 0.85 | 0.60 | 1.43× | 5 | 7 | 1.00e-04 / 2.22e-02 | 2.67e-05 / 3.90e-03 | ✓ |
| qkv dw (1, 8, 48, 80, 1536) bfloat16 | 6.16 | 2.80 | 2.20× | 6 | 12 | 9.86e-04 / 2.51e-02 | 7.31e-04 / 1.56e-02 | ✓ |
| swiglu dw (8, 48, 80, 512) float32 | 4.18 | 1.86 | 2.24× | 5 | 12 | 1.18e-05 / 4.32e-03 | 4.65e-10 / 1.98e-07 | ✓ |
| attn regions (15, 8, 2048, 64) | 25.01 | 2.39 | 10.48× | 3 | 30 | 5.85e-04 / 7.80e-03 | 5.85e-04 / 7.80e-03 | ✓ |
| swiglu dw (1, 48, 80, 1280) float32 | 1.50 | 0.67 | 2.22× | 5 | 11 | 2.58e-05 / 4.74e-02 | 9.26e-10 / 1.53e-06 | ✓ |
| qkv dw (1, 48, 80, 3840) bfloat16 | 2.03 | 0.92 | 2.21× | 5 | 12 | 9.52e-04 / 3.24e-02 | 6.55e-04 / 1.55e-02 | ✓ |
| attn regions (15, 16, 256, 80) | 3.21 | 0.29 | 11.08× | 2 | 25 | 5.40e-04 / 6.23e-03 | 5.41e-04 / 6.23e-03 | ✓ |

## Precision detail

| stage | config | ms / call | ms / frame | speed-up | mean err | p99.9 err | err ÷ stock (mean / p99.9) | frames better | precision |
|---|---|--:|--:|--:|--:|--:|--:|:-:|:-:|
| stage 1 2160p | stock | OOM | OOM | — | — | — | — / — |  |  |
| stage 2 2160p | stock | OOM | OOM | — | — | — | — / — |  |  |
| end-to-end 2160p | stock | OOM | OOM | — | — | — | — / — |  |  |
| stage 1 2160p | oracle | — | — | — | — | — | — / — |  |  |
| stage 2 2160p | oracle | — | — | — | — | — | — / — |  |  |
| end-to-end 2160p | oracle | — | — | — | — | — | — / — |  |  |
| stage 1 2160p | ours | 7317.75 | 914.72 | — | 1.73e-03 | 1.18e-02 | — / — |  | — |
| stage 2 2160p | ours | 2194.16 | 2194.16 | — | 4.73e-04 | 6.17e-03 | — / — |  | — |
| end-to-end 2160p | ours | 30107.57 | 3763.45 | — | 4.83e-04 | 8.12e-03 | — / — |  | — |
| stage 1 1080p | stock | OOM | OOM | — | — | — | — / — |  |  |
| stage 2 1080p | stock | 886.53 | 886.53 | 1.00× | — | — | — / — |  |  |
| end-to-end 1080p | stock | OOM | OOM | — | — | — | — / — |  |  |
| stage 1 1080p | oracle | — | — | — | — | — | — / — |  |  |
| stage 2 1080p | oracle | — | — | — | — | — | — / — |  |  |
| end-to-end 1080p | oracle | — | — | — | — | — | — / — |  |  |
| stage 1 1080p | ours | 1797.00 | 224.62 | — | 3.12e-03 | 9.74e-02 | — / — |  | — |
| stage 2 1080p | ours | 566.12 | 566.12 | 1.57× | 9.15e-04 | 1.52e-02 | — / — |  | — |
| end-to-end 1080p | ours | 7266.49 | 908.31 | — | 1.20e-03 | 1.66e-01 | — / — |  | — |
| stage 1 720p | stock | 1325.50 | 165.69 | 1.00× | 2.87e-03 | 5.28e-02 | 1.000 / 1.000 | 0/32 |  |
| stage 2 720p | stock | 322.21 | 322.21 | 1.00× | 8.06e-04 | 9.89e-03 | 1.000 / 1.000 | 0/32 |  |
| end-to-end 720p | stock | 4288.38 | 536.05 | 1.00× | 1.60e-03 | 1.62e-01 | 1.000 / 1.000 | 0/32 |  |
| stage 1 720p | oracle | — | — | — | 1.65e-03 | 4.54e-02 | 0.546 / 0.728 | 32/32 |  |
| stage 2 720p | oracle | — | — | — | 7.32e-04 | 9.07e-03 | 0.895 / 0.944 | 32/32 |  |
| end-to-end 720p | oracle | — | — | — | 2.01e-03 | 2.13e-01 | 1.015 / 1.076 | 24/32 |  |
| stage 1 720p | ours | 777.57 | 97.20 | 1.70× | 2.53e-03 | 4.10e-02 | 0.903 / 0.830 | 32/32 | ✓ |
| stage 2 720p | ours | 241.01 | 241.01 | 1.34× | 7.95e-04 | 9.30e-03 | 0.987 / 0.941 | 28/32 | ✓ |
| end-to-end 720p | ours | 3113.83 | 389.23 | 1.38× | 1.73e-03 | 1.53e-01 | 1.026 / 1.020 | 20/32 | ≈ |

## What ours is

- `ck_attn`: flash attention, bf16, head sizes 64/80/96, with Swin's shifted-window regions as one launch (key tiles no query can see are skipped)
- `ck_dw`: the 7×7 depthwise SwiGLU / QKV mixers, NHWC, padding read in place, f32 arithmetic, residual fused
- `ck_glue`: RMSNorm → bf16 with Swin's pad and roll folded in; roll back, crop and residual add as one pass; SwiGLU as one gate|up GEMM and a fused silu·mul; qk-norm and RoPE as one pass each

## Method

- **GPU:** NVIDIA TITAN RTX (sm_75), torch 2.9.1+cu128. Measured peaks: 95 TFLOPS 16-bit tensor, 15 TFLOPS f32, 551 GB/s DRAM.
- **Kernels:** `ck_attn` sm_75, `ck_dw` sm_75, `ck_glue` sm_75. Turing has no bf16 MMA: `ck_attn` widens bf16 to f16 as it stages K/V (synchronously, one stage) and runs two `m16n8k8`; all attention goes through it, as PyTorch has no bf16 flash or memory-efficient SDPA before sm_80.
- **Data:** 4 clips × 8 consecutive frames per resolution from public green-screen footage (`just prep`), through the runtime's own preprocessing, VFMs (C-RADIO, MoGe, RVM), stage 1 over the window and stage 2 per frame, without a hint.
- **Configs:** `stock` is upstream (bf16 autocast). `ours` patches the kernels in at load (`corridorkey.ptx.enable`). `oracle` is stock with our depthwise and MLP slots in strict f32: what exact ops there would score.
- **Time:** CUDA events on each resolution's first clip, stock and ours alternating call by call on shared inputs so clock and thermal drift fall on both alike; median per config. Per frame = per call ÷ frames in the call (stage 1 and end to end run the whole window); fps = 1000 ÷ ms per frame. The oracle is a precision bound only, not timed.
- **Memory:** per call, the peak torch allocates above what is already live (after a warm-up call); peak = that plus the 4.6 GiB of weights (model and VFMs, f32).
- **Kernel split:** CUPTI device time per kernel (`torch.profiler`) over one call each of the VFMs, stage 1 and stage 2, per frame; idle = wall − kernel time.
- **SOL %:** FLOPs counted by `FlopCounterMode` at the measured tensor peak (model rows), or each op's own FLOP / byte bound (`bench/sol.py`), ÷ measured time.
- **Precision:** stage rows take the fp32 reference's inputs, so judge that stage alone; end to end chains everything. Errors are per frame (mean, 99.9th percentile), compared with stock by geometric mean ratio. End-to-end error is dominated by bf16 VFM noise that stage 2 can amplify, so exact ops can score worse than stock there, hence the oracle bound. Components are scored against float64 on identical inputs. ✓ no worse than stock (0.1 % tie); ≈ worse, but no worse than exact ops in our slots (the oracle), within 2 % (components) or not significantly (two standard errors over clips); ✗ a loss; — not judged.
