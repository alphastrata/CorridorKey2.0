# CorridorKey 2.0 — `ck` pipeline

Rust launcher `launcher/crates/ck/src/*`, kernels from `corridorkey/ptx/bin/sm_{75,120}/*.ptx`, GEMMs on `ck_gemm`
(`launcher/crates/ck-gpu/src/gemm.rs`: THIN n≤16, SQUARE on sm_120, WIDE/SQUARE on sm_75, `any` for f32 or K%8≠0). The Python path (`infer.py` → `corridorkey/runtime.py`) runs the same model; its
equivalents are noted in `[py: …]`. Shapes are for 1080p: input padded to 1152×2048 (quantum 64), `/16` = 72×128,
`/64` = 18×32. `T` = window length (auto, e.g. 80), `S` = stride (T/2).

```
 ck <clip…> -o out                                                        main.rs  (argh flags → Config)
 │
 ▼
 farm::run ─────────── one thread per GPU (devices from --device / all) ───────────────── farm.rs
 │   Shared: clip queue · stage-2 job queue (≤ QUEUE_BYTES) · abort flag
 │
 ├─► worker(GPU k)   Gpu::open (PTX modules)             Models::load (weights.rs: bf16 / f16 on sm_75)
 │    │
 │    └─ loop ──────────────────────────────────────────────────────────────────────────────────┐
 │        ├─ queued stage-2 job?  → stage2(job)            (any GPU helps the lead)              │
 │        ├─ next clip?           → lead(clip)             (this GPU becomes the clip's lead)    │
 │        └─ nothing left         → exit                                                          │
 │   ◄──────────────────────────────────────────────────────────────────────────────────────────┘
 │
 ▼
 lead(clip)                                                                                farm.rs
 │   Source::open (ffprobe)                                                                io.rs
 │   Geometry::scaled (pad to quantum, --model-scale Lanczos)                              frame.rs
 │   vram::window → T, S from free memory                                                  vram.rs
 │   spawn sink thread: reorders Done frames → Writer per output (ffmpeg pipe / EXR)      write.rs
 │
 ▼
 Clip::run — for each window [t0, t0+T) stepping S ─────────────────────────────────────── clip.rs
 │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
 │  │ read(t1)       ffmpeg decode thread → raw frames (rgb24 / rgb48le / gbrpf32le)   io.rs     │
 │  │                optional hint video / first-frame hint (grey or alpha)                     │
 │  │                                                                                            │
 │  │ features(t1)   only new frames, batches of --vfm-batch          clip.rs → vfm.rs, vit.rs  │
 │  │   prep (Display encoding, f32)                                   frame.rs · ck_ops         │
 │  │   ├─ RVM   resize /4 → patch16 → 12 ViT blocks → 4 cross blocks                           │
 │  │   │        + GRU over frames (state carried window to window)     gru_kv / gru_mix         │
 │  │   ├─ C-RADIO resize /4 (area) → patch16 → 27 ViT blocks → standardise                      │
 │  │   └─ MoGe  resize 7/32 → patch14 → 12 ViT blocks (+ taps)  → standardise                   │
 │  │   each ViT block: LN → QKV GEMM → ck_attn (attnD) → out GEMM (+bias tail) → LN → MLP       │
 │  │   → per-frame features at /64              [py: foundations.py, rvm.py]                    │
 │  │                                                                                            │
 │  │ x = prep(raw[t0:t1]) → [T, 4, H, W] f32  (RGB Model encoding + hint channel)  frame.rs     │
 │  │ quality = hints(): hint video · first-frame hint · carry from previous window   clip.rs    │
 │  │                                                                                            │
 │  │ ═══ Stage 1  (temporal, whole window at once) ══════════════════════════ stage1.rs ═══     │
 │  │   stem      conv k32 s16 (im2col 8-wide + GEMM)       → patch  [T,72,128,512]              │
 │  │   enc_16 ×2 Full spatial attention per frame, AdaLN on hint quality (norm_mod/add_gate)   │
 │  │   down      conv k8 s4 on enc (NHWC im2col) ┐                                              │
 │  │   VFM feats → NHWC concat ──────────────────┴→ fuse (SwiGLU MLP) → f64 [T,18,32,1536]      │
 │  │   body_64 ×5   Full TEMPORAL attention over T·18·32 tokens   ← the cross-frame context     │
 │  │   mid_up    proj + 3×3 smooth + shuffle ×4 → mid [T,72,128,512] (+ scaled enc skip)        │
 │  │   mid_16 ×4    Swin 16×16 windows across all T frames (shift phases 0/8)                   │
 │  │   mid_down  conv k8 s4 → += f64                                                            │
 │  │   body_64 ×5   Full temporal attention                                                     │
 │  │   up        proj + smooth + shuffle → dec (+ scaled mid skip)                              │
 │  │   dec_16 ×6    Swin / CSwin (rows|cols) temporal; late patch skip before last 2            │
 │  │   → baton [T,72,128,512] f32 ;  aux = deconv k16 s8 (GEMM + col2im) → [T,4,H/2,W/2]        │
 │  │                                                                    [py: stage1.py]         │
 │  │                                                                                            │
 │  │ blend      acc[f] += taper(j) · baton[j];  weight[f] += taper(j)   (axpy, f32)  clip.rs    │
 │  │ carry      aux alpha → resize to H×W → next window's hint channel (unless --no-carry)      │
 │  │                                                                                            │
 │  │ flush(<next t0)  frames no later window touches → Job{baton/weight, raw}                  │
 │  │   submit: another GPU idle and queue has room → download baton, enqueue                    │
 │  │           else run stage 2 here (draining any queued jobs first)        farm.rs            │
 │  └────────────────────────────────────────────────────────────────────────────────────────────┘
 │
 ▼
 stage2(job)  — one frame at a time, on whichever GPU took it                       farm.rs → stage2.rs
 │   throttle: frame < written + AHEAD (bounds memory ahead of the writer)
 │   rgb = prep(raw) full res f32
 │   x[…,0:512]   = place(baton · 1/weight)                         (stage-1 context, /16)
 │   x[…,512:1280]= patch1 GEMM on im2col(rgb, k32 s16)             (fresh full-res pixels)
 │   pre ×8   Full / Swin / CSwin spatial blocks, d=1280, 16 heads
 │   gate     x += SiLU(gate(x)) · patch2(rgb cols)   on the RGB channels   (add_gated)
 │   last ×2  Swin (shift 8,0 / 0,8)
 │   depatch  GEMM → col2im (k32 s16, f32) → logits [1,4,H,W]           [py: stage2.py]
 │   render   alpha · fg · cutout · checker (despill, brightness/contrast, linear)  frame.rs · ck_ops
 │            or render_exr (half / float)
 │   download → Done{frame, images} → sink
 │
 ▼
 sink (per clip)  reorder by frame index → Writer::write → ffmpeg encode (x264 / NVENC …) or EXR   write.rs
                  `written` counter releases the stage-2 throttle
```

## Inside every transformer block — `block.rs`, `attn.rs`

```
 x [b,t,h,w,c] f32 (residual stream)
 │
 ├─ norm1  RMSNorm → bf16, Swin roll/pad folded in            ck_glue norm_bf16 / norm_mod (AdaLN)
 ├─ QKV    GEMM (ck_gemm, bias epilogue) → 7×7 depthwise on QKV slice      ck_dw dw_qkv
 ├─ gate   GEMM → per-head sigmoid gate (f32)
 ├─ attend per head group:                                                  attn.rs
 │    split    qkv → q,k,v per window, qk-norm + RoPE, region-sorted order  ck_ops split
 │    attn     flash attention bf16, D 64/80/96, masked windows skip tiles   ck_attn attnD[_masked]
 │    merge    gate · heads → [.., c + 8] with [1,1,0…] tail                ck_ops merge
 ├─ proj   GEMM; bias rides the ones-column tail; += x  (or unshift_add)   ck_glue unshift_add
 ├─ mix    7×7 depthwise local mix, residual fused                         ck_dw / ck_glue mix_add
 └─ MLP    norm2 → gate|up GEMM → SiLU·mul (+tail) → down GEMM (+bias tail) += x   ck_glue swiglu
```

## Loops at a glance

| Loop | Where | Over |
|---|---|---|
| GPU workers | `farm::run` | devices, one thread each |
| clip queue | `worker` | input clips; first free GPU leads |
| windows | `Clip::run` | `[t0, t0+T)` every `S` frames |
| VFM batches | `Clip::features` | new frames, `--vfm-batch` at a time |
| RVM GRU | `Rvm::forward` | frames in order, state carried across windows |
| encoder chunks | `Stage1::encode` | frames, sized to `ENC_BYTES` |
| blocks | `Stage1::forward`, `Stage2::forward` | 2 + 5 + 4 + 5 + 6 stage-1, 8 + 2 stage-2 |
| head chunks | `attn::attend` | heads, sized to `HEAD_BYTES` |
| stage-2 jobs | `submit` / `worker` | frames, any GPU |
| writer | `sink` | frames in order, one ffmpeg per output |

## Precision

- Decode: native bit depth (8-bit ×257, 16-bit, float); frames enter both stages as full-resolution f32.
- Stage 1 attends at /16 and /64; stage 2 sees full-resolution pixels per frame.
- GEMMs and attention bf16 (f16 on sm_75) with f32 accumulation; residual stream, window blend, logits f32.
