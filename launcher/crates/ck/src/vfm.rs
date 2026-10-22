//! The frozen foundation models (`weights/vfms.safetensors`, from `launcher/parity/export.py`): C-RADIOv4-SO400M at
//! `H/4`, MoGe-2's `DINOv2` ViT-B/14 at `7H/32` (layers 5 and 11, projected and summed), and RVM-S at `H/4`, recurrent
//! over frames. Each yields standardised NCHW features on the `/64` grid.

use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::Result;

use crate::ops::{Conv, Into, conv, to_half, to_nhwc};
use crate::vit::{Attn, Block, Mlp, Norm, Resize, embed, gru_kv, gru_mix, resize, rvm_attn, rvm_mlp, standardise};
use crate::weights::{Loader, Weights};

/// `[h · w, c]` rows → `[1, c, h, w]`.
fn planes(gpu: &Gpu, v: &[f32], (h, w, c): (usize, usize, usize)) -> Result<Tensor> {
    let t: Vec<f32> = (0..v.len()).map(|i| v[i % (h * w) * c + i / (h * w)]).collect();
    gpu.upload_f32(&t, &[1, c, h, w], Dtype::F32)
}

/// Prefix tokens and the patch grid's positions, resampled from `[1, c, h, w]` to `grid`.
struct Tokens {
    prefix: Tensor,
    pos: Tensor,
    mode: Resize,
}

impl Tokens {
    fn table(
        &self,
        gpu: &Gpu,
        (gh, gw): (usize, usize),
        scale: (f32, f32),
        unit: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let [p, c] = self.prefix.shape[..] else {
            eyre::bail!("prefix wants [p, c]")
        };
        let grid = resize(gpu, &self.pos, (gh, gw), scale, self.mode, unit)?;
        let table = gpu.zeros(&[p + gh * gw, c], Dtype::F32)?;
        to_nhwc(gpu, &grid, &table.at(p * c, &[gh * gw, c])?, (c, 0))?;
        Ok(table)
    }
}

/// Patch embedding of NCHW `x` by `w [c, 3 · k · k]`, then prefix and positions.
fn tokenise(
    gpu: &Gpu,
    x: &Tensor,
    (w, b): (&Tensor, &Tensor),
    k: usize,
    table: &Tensor,
    prefix: &Tensor,
) -> Result<Tensor> {
    let [n, _, h, wd] = x.shape[..] else {
        eyre::bail!("tokenise wants 4-d")
    };
    let c = w.shape[0];
    let patches = gpu.empty(&[n * (h / k) * (wd / k), c], Dtype::F32)?;
    let geo = Conv { k, s: k, p: 0 };
    conv(
        gpu,
        x,
        false,
        geo,
        (w, b),
        Into {
            out: &patches,
            ld: c,
            off: 0,
            accumulate: false,
        },
    )?;
    embed(gpu, &patches, prefix, table, n)
}

struct Stats {
    mean: Tensor,
    std: Tensor,
}

impl Stats {
    fn load(ld: &Loader, name: &str) -> Result<Self> {
        Ok(Self {
            mean: ld.f32(&format!("stats.{name}_mean"))?,
            std: ld.f32(&format!("stats.{name}_std"))?,
        })
    }
}

struct Radio {
    cond: Stats,
    embedder: Tensor,
    bias: Tensor,
    tokens: Tokens,
    blocks: Vec<Block>,
    stats: Stats,
}

impl Radio {
    fn load(ld: &Loader) -> Result<Self> {
        let pg = "cradio.model.patch_generator";
        let (_, pos) = ld.get(&format!("{pg}.pos_embed"))?;
        let prefix = ld.f32(&format!("{pg}.cls_token.token"))?;
        let c = prefix.shape[1];
        Ok(Self {
            cond: Stats {
                mean: ld.f32("cradio.input_conditioner.norm_mean")?,
                std: ld.f32("cradio.input_conditioner.norm_std")?,
            },
            embedder: ld.mat(&format!("{pg}.embedder.weight"))?,
            bias: ld.gpu.zeros(&[c], Dtype::F32)?,
            tokens: Tokens {
                prefix,
                pos: planes(ld.gpu, &pos, (128, 128, c))?,
                mode: Resize::Bilinear,
            },
            blocks: (0..27)
                .map(|i| Block::timm(ld, &format!("cradio.model.blocks.{i}"), 16, 1e-6))
                .collect::<Result<_>>()?,
            stats: Stats::load(ld, "cradio")?,
        })
    }

    fn forward(&self, gpu: &Gpu, display: &Tensor, unit: (&Tensor, &Tensor), ones: &Tensor) -> Result<Tensor> {
        let [t, _, h, w] = display.shape[..] else {
            eyre::bail!("display wants 4-d")
        };
        let x = resize(
            gpu,
            display,
            (h / 4, w / 4),
            (4.0, 4.0),
            Resize::Area,
            (&self.cond.mean, &self.cond.std),
        )?;
        let (gh, gw) = (h / 64, w / 64);
        let s = 128.0 / gh.max(gw) as f32;
        let table = self.tokens.table(gpu, (gh, gw), (s, s), unit)?;
        let x = tokenise(gpu, &x, (&self.embedder, &self.bias), 16, &table, &self.tokens.prefix)?;
        let n = x.shape[1];
        self.blocks.iter().try_for_each(|b| b.forward(gpu, &x, (t, n), ones))?;
        let skip = self.tokens.prefix.shape[0];
        standardise(gpu, &x, (&self.stats.mean, &self.stats.std), (t, skip, gh, gw), false)
    }
}

struct Moge {
    image: Stats,
    patch: (Tensor, Tensor),
    tokens: Tokens,
    blocks: Vec<Block>,
    norm: Norm,
    proj: [crate::vit::Linear; 2],
    stats: Stats,
}

impl Moge {
    fn load(ld: &Loader) -> Result<Self> {
        let bb = "moge.backbone";
        let (_, pos) = ld.get(&format!("{bb}.pos_embed"))?;
        let (_, cls) = ld.get(&format!("{bb}.cls_token"))?;
        let c = cls.len();
        let prefix: Vec<f32> = cls.iter().zip(&pos).map(|(a, b)| a + b).collect();
        let proj = |i: usize| crate::vit::Linear::load(ld, &[&format!("moge.output_projections.{i}")], None);
        Ok(Self {
            image: Stats {
                mean: ld.f32("moge.image_mean")?,
                std: ld.f32("moge.image_std")?,
            },
            patch: (
                ld.mat(&format!("{bb}.patch_embed.proj.weight"))?,
                ld.f32(&format!("{bb}.patch_embed.proj.bias"))?,
            ),
            tokens: Tokens {
                prefix: ld.gpu.upload_f32(&prefix, &[1, c], Dtype::F32)?,
                pos: planes(ld.gpu, &pos[c..], (37, 37, c))?,
                mode: Resize::Bicubic,
            },
            blocks: (0..12)
                .map(|i| Block::timm(ld, &format!("{bb}.blocks.{i}"), 12, 1e-6))
                .collect::<Result<_>>()?,
            norm: Norm::load(ld, &format!("{bb}.norm"), 1e-6)?,
            proj: [proj(0)?, proj(1)?],
            stats: Stats::load(ld, "moge")?,
        })
    }

    fn forward(&self, gpu: &Gpu, display: &Tensor, unit: (&Tensor, &Tensor), ones: &Tensor) -> Result<Tensor> {
        let [t, _, h, w] = display.shape[..] else {
            eyre::bail!("display wants 4-d")
        };
        let (hm, wm) = (h * 7 / 32, w * 7 / 32);
        let x = resize(
            gpu,
            display,
            (hm, wm),
            (0.0, 0.0),
            Resize::Area,
            (&self.image.mean, &self.image.std),
        )?;
        let (gh, gw) = (hm / 14, wm / 14);
        let step = |g: usize| (1.0 / ((g as f64 + 0.1) / 37.0)) as f32;
        let table = self.tokens.table(gpu, (gh, gw), (step(gh), step(gw)), unit)?;
        let x = tokenise(gpu, &x, (&self.patch.0, &self.patch.1), 14, &table, &self.tokens.prefix)?;
        let n = x.shape[1];
        let out = gpu.empty(&[t * n, self.norm.w.numel()], Dtype::F32)?;
        self.blocks.iter().enumerate().try_for_each(|(i, b)| {
            b.forward(gpu, &x, (t, n), ones)?;
            match i {
                5 | 11 => self.proj[i / 6].run(gpu, &self.norm.apply(gpu, &x, Dtype::Bf16)?, &out, i == 11),
                _ => Ok(()),
            }
        })?;
        standardise(gpu, &out, (&self.stats.mean, &self.stats.std), (t, 1, gh, gw), false)
    }
}

struct Cross {
    ca: Attn,
    mlp: Mlp,
    sa: Attn,
}

struct Rvm {
    patch: (Tensor, Tensor),
    tokens: Tokens,
    blocks: Vec<Block>,
    norm: Norm,
    input: Tensor,
    state: Tensor,
    state_norm: Tensor,
    cross: Vec<Cross>,
    out_norm: Norm,
    stats: Stats,
}

impl Rvm {
    fn load(ld: &Loader) -> Result<Self> {
        let (_, pos) = ld.get("rvm.base_posenc")?;
        let prefix = ld.f32("rvm.cls_token")?;
        let c = prefix.shape[1];
        let enc = |i: usize| -> Result<Block> {
            let p = format!("rvm.encoder_layers.{i}");
            Ok(Block {
                attn: rvm_attn(ld, &format!("{p}.attention_norm"), &format!("{p}.attention"), 6)?,
                mlp: rvm_mlp(ld, &format!("{p}.mlp_norm"), &format!("{p}.mlp"))?,
            })
        };
        let core = "rvm.rnn_core";
        let cross = |i: usize| -> Result<Cross> {
            let p = format!("{core}.transformer.xa_blocks.{i}");
            Ok(Cross {
                ca: rvm_attn(ld, &format!("{p}.ca_attention_norm"), &format!("{p}.ca_attention"), 8)?,
                mlp: rvm_mlp(ld, &format!("{p}.mlp_norm"), &format!("{p}.mlp"))?,
                sa: rvm_attn(ld, &format!("{p}.attention_norm"), &format!("{p}.attention"), 8)?,
            })
        };
        let gate = |a: &str, b: &str| ld.rows(&[&format!("{core}.{a}.weight"), &format!("{core}.{b}.weight")]);
        Ok(Self {
            patch: (ld.mat("rvm.patch_embed.weight")?, ld.f32("rvm.patch_embed.bias")?),
            tokens: Tokens {
                prefix,
                pos: planes(ld.gpu, &pos, (16, 16, c))?,
                mode: Resize::Bicubic,
            },
            blocks: (0..12).map(enc).collect::<Result<_>>()?,
            norm: Norm::load(ld, "rvm.encoder_final_norm", 1e-5)?,
            input: gate("input_update", "input_reset")?,
            state: gate("state_update", "state_reset")?,
            state_norm: ld.f32(&format!("{core}.state_layer_norm.weight"))?,
            cross: (0..4).map(cross).collect::<Result<_>>()?,
            out_norm: Norm::load(ld, &format!("{core}.transformer.output_norm"), 1e-5)?,
            stats: Stats::load(ld, "rvm")?,
        })
    }

    /// One recurrent step: `prev` state `[n, c]` and this frame's encoding → the new state in `out`.
    fn step(
        &self,
        gpu: &Gpu,
        (enc, half): (&Tensor, &Tensor),
        prev: &Tensor,
        out: &Tensor,
        ones: &Tensor,
    ) -> Result<()> {
        let [n, c] = enc.shape[..] else {
            eyre::bail!("step wants [n, c]")
        };
        let gates = gpu.empty(&[n, 2 * c], Dtype::F32)?;
        gpu.linear(half, &self.input, None, &gates, false)?;
        gpu.linear(&to_half(gpu, prev)?, &self.state, None, &gates, true)?;
        let kv = gru_kv(gpu, &gates, prev, &self.state_norm)?;
        let x = gpu.copy(enc)?;
        self.cross.iter().try_for_each(|b| {
            b.ca.forward(gpu, &x, (1, n), Some(&kv), ones)?;
            b.mlp.forward(gpu, &x)?;
            b.sa.forward(gpu, &x, (1, n), None, ones)
        })?;
        gru_mix(gpu, &gates, &self.out_norm.apply(gpu, &x, Dtype::F32)?, prev, out)
    }

    fn forward(
        &self,
        gpu: &Gpu,
        display: &Tensor,
        state: Option<&Tensor>,
        unit: (&Tensor, &Tensor),
        ones: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let [t, _, h, w] = display.shape[..] else {
            eyre::bail!("display wants 4-d")
        };
        let x = resize(gpu, display, (h / 4, w / 4), (4.0, 4.0), Resize::Bilinear, unit)?;
        let (gh, gw) = (h / 64, w / 64);
        let table = self
            .tokens
            .table(gpu, (gh, gw), (16.0 / gh as f32, 16.0 / gw as f32), unit)?;
        let x = tokenise(gpu, &x, (&self.patch.0, &self.patch.1), 16, &table, &self.tokens.prefix)?;
        let (n, c) = (x.shape[1], x.shape[2]);
        self.blocks.iter().try_for_each(|b| b.forward(gpu, &x, (t, n), ones))?;
        let enc = self.norm.apply(gpu, &x, Dtype::F32)?;
        let half = to_half(gpu, &enc)?;
        let feats = gpu.empty(&[t * n, c], Dtype::F32)?;
        let first = state.cloned().map_or_else(|| gpu.zeros(&[n, c], Dtype::F32), Ok)?;
        let last = (0..t).try_fold(first, |prev, i| {
            let out = feats.rows(i * n, n)?;
            self.step(gpu, (&enc.rows(i * n, n)?, &half.rows(i * n, n)?), &prev, &out, ones)?;
            Ok::<_, eyre::Report>(out)
        })?;
        let rvm = standardise(gpu, &feats, (&self.stats.mean, &self.stats.std), (t, 1, gh, gw), true)?;
        Ok((rvm, last))
    }
}

/// Per-frame features `[t, c, H/64, W/64]` and RVM's state after the last frame.
pub struct Features {
    pub cradio: Tensor,
    pub moge: Tensor,
    pub rvm: Tensor,
    pub state: Tensor,
}

pub struct Vfms {
    radio: Radio,
    moge: Moge,
    rvm: Rvm,
    ones: Tensor,
    zero: Tensor,
}

impl Vfms {
    /// # Errors
    /// When a weight is missing or an upload fails.
    pub fn load(gpu: &Gpu, weights: &Weights) -> Result<Self> {
        let ld = Loader {
            gpu,
            weights,
            prefix: "",
            half: Dtype::Bf16,
        };
        Ok(Self {
            radio: Radio::load(&ld)?,
            moge: Moge::load(&ld)?,
            rvm: Rvm::load(&ld)?,
            ones: gpu.upload_f32(&[1.0; 3], &[3], Dtype::F32)?,
            zero: gpu.zeros(&[1], Dtype::F32)?,
        })
    }

    /// Each model's transformer blocks, and the attention's unit scales.
    #[must_use]
    pub fn blocks(&self) -> ([(&'static str, &[Block]); 3], &Tensor) {
        let blocks = [
            ("cradio", &self.radio.blocks[..]),
            ("moge", &self.moge.blocks[..]),
            ("rvm", &self.rvm.blocks[..]),
        ];
        (blocks, &self.ones)
    }

    /// Features of display frames `[t, 3, H, W]` in `[0, 1]` (`H`, `W` multiples of 64), RVM continuing from `state`.
    ///
    /// # Errors
    /// On a launch or GEMM failure.
    pub fn forward(&self, gpu: &Gpu, display: &Tensor, state: Option<&Tensor>) -> Result<Features> {
        let one = self.ones.rows(0, 1)?;
        let unit = (&self.zero, &one);
        let (rvm, state) = self.rvm.forward(gpu, display, state, unit, &self.ones)?;
        Ok(Features {
            cradio: self.radio.forward(gpu, display, unit, &self.ones)?,
            moge: self.moge.forward(gpu, display, unit, &self.ones)?,
            rvm,
            state,
        })
    }
}
