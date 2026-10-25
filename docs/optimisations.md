# Optimisations

Paths are from the repository root. "Python" means `infer.py` with `corridorkey/ptx`; "ck" means the Rust launcher.

## Attention

| Thing | Kernel | Why | ELI5 | Source |
|---|---|---|---|---|
| Masked attention as one flash-style launch | `ck_attn` | Stock hands SDPA a dense Swin region mask, which rules out its flash kernel and materialises the mask. Unmasked attention stays on PyTorch's flash kernel in Python. | Work out who may look at whom on the fly instead of writing a giant seating chart first. | `kernels/ck_attn/src/mma.rs`, `corridorkey/ptx/attn.py` |
| Skip key tiles no query can see | `ck_attn` | Fully masked tiles cost nothing. | Don't read pages nobody is allowed to read. | `kernels/ck_attn/src/lib.rs` (`span`) |
| Sort each window's tokens by region | `ck_ops` `split` / `merge` | Grouping regions makes whole tiles fully masked, so the skip above fires far more often. | Seat each party at its own tables so whole tables can be ignored. | `corridorkey/ptx/grid.py` (`_sorted`), `launcher/crates/ck/src/attn.rs` |
| Heads gathered with qk-norm and RoPE on the way in, gated back on the way out | `ck_ops` `split` / `merge` | Replaces the window reshape/copy/permute chain around every attention call. | Sort the post while carrying it, not in a separate trip. | `kernels/ck_ops/src/heads.rs`, `corridorkey/ptx/grid.py` |
| qk-norm and RoPE as one pass each | `ck_glue` `head64/80/96` | Several elementwise passes over the heads become one. | One lap of the track instead of four. | `kernels/ck_glue/src/head.rs`, `corridorkey/ptx/glue.py` |
| Rows per warp tuned per GPU generation | `ck_attn` | Occupancy differs by architecture. | Different team sizes for different kitchens. | `kernels/ck_attn/src/tune/` |

## Blocks

| Thing | Kernel | Why | ELI5 | Source |
|---|---|---|---|---|
| Depthwise SwiGLU / QKV mixers read NHWC in place, padding and residual fused | `ck_dw` `mix_*`, `dw_bf16`, `rms_*` | Stock permutes to NCHW, pads, convolves and adds: several full-tensor copies. | Paint on the canvas instead of copying it four times first. | `kernels/ck_dw/src/lib.rs`, `corridorkey/ptx/dw.py` |
| 7×7 tap loads hoisted, output columns unrolled | `ck_dw` | Neighbouring outputs share inputs; load once, reuse. | Read each pixel once, not seven times. | `kernels/ck_dw/src/lib.rs` |
| Norm → Swin pad → roll as one bf16 write; roll back → crop → residual add as one pass | `ck_glue` `norm_bf16`, `unshift_add` | Swin's shift is otherwise a chain of copies. | Shuffle and add in one trip. | `kernels/ck_glue/src/grid.rs`, `corridorkey/ptx/block.py` |
| AdaLN modulation and gated residual, vectorised | `ck_glue` `norm_mod_bf16`, `add_gate` | Same: fused elementwise passes. | — | `kernels/ck_glue/src/grid.rs` |
| SwiGLU as one gate\|up GEMM and a fused `silu · mul` | `ck_glue` `swiglu_bf16` | Two GEMMs and three elementwise passes become one GEMM and one pass. | Make both halves of the sandwich at once. | `kernels/ck_glue/src/lib.rs`, `corridorkey/ptx/glue.py` |
| Thin-input (patchify) convolutions as im2col and one GEMM | — | cuDNN is slow with 3–4 input channels. | Unroll the tiny filter into one big multiply. | `corridorkey/ptx/half.py` |
| Turing: Linears as f16 GEMMs into f32; dense convolutions in f16 | — | sm_75 has no bf16 tensor cores, so bf16 falls back to slow paths. | Use the units the card actually has. | `corridorkey/ptx/half.py` |

## GEMM (ck)

| Thing | Kernel | Why | ELI5 | Source |
|---|---|---|---|---|
| Own GEMM; cuBLASLt removed | `ck_gemm` | Tiles chosen for this model's shapes; no library heuristics or workspace. | A tool cut for one job. | `kernels/ck_gemm/src/tensor.rs` |
| Tiles: WIDE 256×128, SQUARE 128×128, THIN 128×16, SMALL 64×64 | `ck_gemm` | Matches tile to matrix shape so few threads idle. | Pick the right-size box. | `kernels/ck_gemm/src/tensor.rs`, `launcher/crates/ck-gpu/src/gemm.rs` (`tile`) |
| Epilogue staged through shared memory | `ck_gemm` | 16-byte coalesced stores. | Pack the van fully before driving. | `kernels/ck_gemm/src/tensor.rs` |
| Deterministic split-K for GEMMs under half a wave, or full-wave with K ≥ 8192 | `ck_gemm` | Too few output tiles to fill the GPU; split the long sum instead. | More hands on one long sum. | `launcher/crates/ck-gpu/src/gemm.rs` (`split`) |
| 1–1.75-wave GEMMs: first wave whole, split-K the tail | `ck_gemm` | A part-full last wave leaves SMs idle. | Don't send a bus out half empty. | `launcher/crates/ck-gpu/src/gemm.rs` |
| Bias rides the GEMM as a ones column (hi + lo halves) | `ck_gemm` | Saves a bias pass; hi/lo split keeps f32 accuracy in bf16. | Slip the extra item into the same delivery. | `launcher/crates/ck/src/weights.rs` (`tail`) |
| CUDA-core path for f32 inputs or K not a multiple of 8 | `ck_gemm` `any` | Tensor cores need aligned K. | Fallback for odd sizes. | `kernels/ck_gemm/src/lib.rs` |
| f16 tensor cores on Turing | `ck_gemm`, `ck_attn` | No bf16 tensor cores before sm_80. | — | `launcher/crates/ck-gpu/src/lib.rs` (`half`) |
| Pascal: SIMT GEMM and attention, shuffle-emulated mma/ldmatrix | `ck_gemm`, `ck_attn`, `ck_arch` | No tensor cores on sm_61. | Do it by hand when there's no machine. | `kernels/ck_gemm/src/simt.rs`, `kernels/ck_attn/src/simt.rs`, `kernels/ck_arch/src/sm61.rs` |
| im2col writes eight columns per thread, NHWC ordered (ky, kx, c) | `ck_ops` `im2col` | Coalesced reads and writes. | Carry eight at a time. | `kernels/ck_ops/src/conv.rs` |

## Pipeline (ck)

| Thing | Kernel | Why | ELI5 | Source |
|---|---|---|---|---|
| Whole model in Rust, including C-RADIO, MoGe and RVM | `ck_ops` `vit.rs` | No interpreter, no framework dispatch, own memory plan. | Skip the middleman. | `launcher/crates/ck/src/` (`vfm.rs`, `stage1.rs`, `stage2.rs`) |
| Frame prep, resize, render and colour on the GPU | `ck_ops` `frame.rs`, `resize.rs`, `render.rs` | CPU colour maths and resizing were a bottleneck. | Keep the work where the data is. | `kernels/ck_ops/src/` |
| Window sized from free GPU memory | — | Largest window that fits means fewer, fuller passes. | Take the biggest pan that fits the oven. | `launcher/crates/ck/src/vram.rs` |
| Exact chunking of block transients; stage 1 frees inputs once spent | — | Lower peak memory allows bigger windows. | Wash up as you go. | `launcher/crates/ck/src/block.rs`, `launcher/crates/ck/src/stage1.rs` |
| Memory pool keeps freed memory; trim and retry once on OOM | — | Device allocation is slow. | Reuse the boxes. | `launcher/crates/ck-gpu/src/tensor.rs` |
| Host copies through double-buffered pinned staging | — | Overlaps copy and compute. | Load one tray while the other's in the oven. | `launcher/crates/ck-gpu/src/staging.rs` |
| Weights convert to half types in one parallel pass from the mapping | — | Faster start-up. | — | `launcher/crates/ck/src/weights.rs` |
| VFMs read straight from the publishers' files | — | No 2 GiB export; npz and pickle tensors read in place. | — | `launcher/crates/ck/src/sources.rs` |
| Multi-GPU: fastest GPU leads, others take stage-2 jobs | — | Uses every card without the slow one holding up the fast one. | The quickest cook runs the kitchen; others plate up. | `launcher/crates/ck/src/farm.rs` |
| A lead's own stage-2 jobs keep the baton on its GPU; outputs render before download | — | Avoids device↔host round trips. | Don't post a letter to yourself. | `launcher/crates/ck/src/farm.rs` |
| `--codec hevc_nvenc` | — | x264 at 4K is CPU-bound. | Let the card's video chip encode. | `launcher/crates/ck/src/write.rs` |
| `--threads` per encoder | — | ffmpeg's default threading can take 13 GB of RAM per 4K encoder. | — | `launcher/crates/ck/src/write.rs` |

## Runtime (Python)

| Thing | Kernel | Why | ELI5 | Source |
|---|---|---|---|---|
| Weights stored in bf16 | — | Halves weight memory and removes per-call casts under autocast. | — | `corridorkey/runtime.py` (`bf16_storage`) |
| Skip torch's random initialisers while building | — | Every weight comes from the checkpoint anyway. | Don't paint a wall you're about to wallpaper. | `corridorkey/runtime.py` (`_no_random_init`) |
| Colour transforms on the GPU, on a side stream | — | numpy colour maths was CPU-bound. | — | `corridorkey/color_utils.py` |
| Outputs quantised to uint16 on the GPU | — | Smaller device→host copies. | Shrink it before shipping. | `corridorkey/runtime.py` (`_to_uint16`) |
