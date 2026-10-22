//! The pre-norm transformer blocks of both stages on f32 NHWC `x [b, t, h, w, c]`: attention (full, Swin or `CSwin`,
//! spatial or spatio-temporal), optional AdaLN-Zero, local depthwise mix and `SwiGLU`.

use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::Result;

use crate::attn::{Cache, Group, Layout, Qk, attend};
use crate::ops::{self, Act, Gate, Mix, Shift};
use crate::weights::Loader;

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Full,
    Swin { window: usize, shift: (usize, usize) },
    CSwin,
}

/// `per_frame`: attention within each frame with 2-D `RoPE` (stage 1's encoder); `temporal`: windows span all `t`
/// with 3-D `RoPE`; neither: plain 2-D (stage 2).
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub kind: Kind,
    pub temporal: bool,
    pub heads: usize,
}

struct AdaLn {
    in_w: Tensor,
    in_b: Tensor,
    w: Tensor,
    b: Tensor,
}

pub struct Block {
    spec: Spec,
    norm1: Tensor,
    qkv_w: Tensor,
    qkv_b: Tensor,
    dw: Option<(Tensor, Tensor)>,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_w: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    mix: Option<Mix>,
    norm2: Tensor,
    gu_w: Tensor,
    gu_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
    adaln: Option<AdaLn>,
}

/// `AdaLN`-Zero rows `[b · t, 6c]` from per-frame `cond [b · t]`.
fn modulation(gpu: &Gpu, a: &AdaLn, cond: &Tensor) -> Result<Tensor> {
    let frames = cond.numel();
    let hidden = gpu.empty(&[frames, a.in_w.shape[0]], Dtype::F32)?;
    gpu.linear(&cond.reshape(&[frames, 1])?, &a.in_w, Some(&a.in_b), &hidden, false)?;
    let hidden = ops::act(gpu, &hidden, Act::Silu, Dtype::F32)?;
    let params = gpu.empty(&[frames, a.w.shape[0]], Dtype::F32)?;
    gpu.linear(&hidden, &a.w, Some(&a.b), &params, false)?;
    Ok(params)
}

impl Block {
    /// # Errors
    /// When a weight is missing.
    pub fn load(ld: &Loader, spec: Spec) -> Result<Self> {
        let mix = ld
            .has("local_mix.gamma")
            .then(|| -> Result<Mix> {
                Ok(Mix {
                    norm: ld.f32("local_mix.norm.weight")?,
                    taps: ld.taps(&["local_mix.value.weight", "local_mix.gate.weight"])?,
                    bias: ld.cat(&["local_mix.value.bias", "local_mix.gate.bias"])?,
                    gamma: ld.f32("local_mix.gamma")?,
                })
            })
            .transpose()?;
        let dw = ld
            .has("attn.qkv_dw.dw.weight")
            .then(|| -> Result<_> { Ok((ld.taps(&["attn.qkv_dw.dw.weight"])?, ld.f32("attn.qkv_dw.dw.bias")?)) })
            .transpose()?;
        let adaln = ld
            .has("adaln.proj.weight")
            .then(|| -> Result<AdaLn> {
                Ok(AdaLn {
                    in_w: ld.f32("adaln.in_proj.weight")?,
                    in_b: ld.f32("adaln.in_proj.bias")?,
                    w: ld.f32("adaln.proj.weight")?,
                    b: ld.f32("adaln.proj.bias")?,
                })
            })
            .transpose()?;
        Ok(Self {
            spec,
            norm1: ld.f32("norm1.weight")?,
            qkv_w: ld.mat("attn.qkv.qkv.weight")?,
            qkv_b: ld.f32("attn.qkv.qkv.bias")?,
            dw,
            q_norm: ld.f32("attn.q_norm.weight")?,
            k_norm: ld.f32("attn.k_norm.weight")?,
            gate_w: ld.mat("attn.attn_gate.proj.weight")?,
            proj_w: ld.mat("attn.proj.weight")?,
            proj_b: ld.f32("attn.proj.bias")?,
            mix,
            norm2: ld.f32("norm2.weight")?,
            gu_w: ld.rows(&["mlp.w_gate.weight", "mlp.w_up.weight"])?,
            gu_b: ld.cat(&["mlp.w_gate.bias", "mlp.w_up.bias"])?,
            down_w: ld.mat("mlp.w_down.weight")?,
            down_b: ld.f32("mlp.w_down.bias")?,
            adaln,
        })
    }

    fn groups(&self, b: usize, (t, h, w): (usize, usize, usize)) -> (Vec<Group>, Shift, (usize, usize)) {
        let heads = self.spec.heads;
        let base = Layout {
            b,
            t,
            gh: h,
            gw: w,
            ph: h,
            pw: w,
            wh: h,
            ww: w,
        };
        let one = |layout: Layout, rope, shift| Group {
            head0: 0,
            heads,
            layout,
            rope,
            spatial: !self.spec.temporal,
            shift,
            real: (h, w),
        };
        match self.spec.kind {
            Kind::Full if self.spec.temporal => (vec![one(base, (t, h, w), (0, 0))], Shift::none(h, w), (h, w)),
            Kind::Full => {
                let layout = Layout { b: b * t, t: 1, ..base };
                (vec![one(layout, (1, h, w), (0, 0))], Shift::none(h, w), (h, w))
            }
            Kind::Swin { window, shift } => {
                let (hp, wp) = (h.div_ceil(window) * window, w.div_ceil(window) * window);
                let s = Shift {
                    hp,
                    wp,
                    sh: shift.0,
                    sw: shift.1,
                };
                let shifted = shift != (0, 0);
                let (gh, gw) = if shifted { (hp, wp) } else { (h, w) };
                let layout = Layout {
                    gh,
                    gw,
                    ph: hp,
                    pw: wp,
                    wh: window,
                    ww: window,
                    ..base
                };
                let s = if shifted { s } else { Shift::none(h, w) };
                (vec![one(layout, (t, window, window), shift)], s, (gh, gw))
            }
            Kind::CSwin => {
                let half = heads / 2;
                let rows = Group {
                    heads: half,
                    layout: Layout { ww: 1, ..base },
                    rope: (t, h, 1),
                    ..one(base, (t, h, w), (0, 0))
                };
                let cols = Group {
                    head0: half,
                    heads: half,
                    layout: Layout { wh: 1, ..base },
                    rope: (t, 1, w),
                    ..rows
                };
                (vec![rows, cols], Shift::none(h, w), (h, w))
            }
        }
    }

    /// `x` f32 `[b, t, h, w, c]` through the block (its buffer is updated in place); `cond [b · t]` for `AdaLN` blocks.
    ///
    /// # Errors
    /// On a launch or GEMM failure.
    pub fn forward(&self, gpu: &Gpu, cache: &Cache, x: &Tensor, cond: Option<&Tensor>) -> Result<Tensor> {
        let [b, t, h, w, c] = x.shape[..] else {
            eyre::bail!("block wants [b, t, h, w, c], got {:?}", x.shape)
        };
        let heads = self.spec.heads;
        let head = c / heads;
        let frames = x.reshape(&[b * t, h, w, c])?;
        let (groups, shift, (gh, gw)) = self.groups(b, (t, h, w));
        let shifted = shift.moved(h, w);
        let params = match (&self.adaln, cond) {
            (Some(a), Some(cond)) => Some(modulation(gpu, a, cond)?),
            (Some(a), None) => Some(modulation(gpu, a, &gpu.zeros(&[b * t], Dtype::F32)?)?),
            _ => None,
        };
        let per = h * w;
        let xn = match &params {
            Some(p) => ops::norm_mod(gpu, &frames, &self.norm1, p, per, (0, c))?,
            None => ops::norm(gpu, &frames, &self.norm1, shift)?,
        };
        let qkv = gpu.empty(&[b * t, gh, gw, 3 * c], Dtype::Bf16)?;
        gpu.linear(&xn, &self.qkv_w, Some(&self.qkv_b), &qkv, false)?;
        let qkv = match &self.dw {
            Some((taps, bias)) => ops::dw_qkv(gpu, &qkv, taps, bias)?,
            None => qkv,
        };
        let gate = gpu.empty(&[b * t * gh * gw, heads], Dtype::F32)?;
        gpu.linear(&xn, &self.gate_w, None, &gate, false)?;
        let ao = gpu.empty(&[b * t, gh, gw, c], Dtype::Bf16)?;
        let qk = Qk {
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            heads,
            head,
        };
        groups
            .into_iter()
            .try_for_each(|g| attend(gpu, cache, &qkv, &gate, &ao, &qk, g))?;

        match &params {
            Some(p) => {
                let pa = gpu.empty(&[b * t * h * w, c], Dtype::F32)?;
                gpu.linear(&ao, &self.proj_w, Some(&self.proj_b), &pa, false)?;
                ops::add_gated(
                    gpu,
                    &frames,
                    &pa,
                    Gate::Frame {
                        gate: &p.at(2 * c, &[p.numel() - 2 * c])?,
                        per,
                        ld: 6 * c,
                    },
                    (c, 0),
                )?;
            }
            None if shifted => {
                let pa = gpu.empty(&[b * t, gh, gw, c], Dtype::Bf16)?;
                gpu.linear(&ao, &self.proj_w, Some(&self.proj_b), &pa, false)?;
                ops::unshift_add(gpu, &frames, &pa, shift)?;
            }
            None => gpu.linear(&ao, &self.proj_w, Some(&self.proj_b), &frames, true)?,
        }

        let frames = match &self.mix {
            Some(m) => ops::mix_add(gpu, &frames, m)?,
            None => frames,
        };
        let xn = match &params {
            Some(p) => ops::norm_mod(gpu, &frames, &self.norm2, p, per, (3 * c, 4 * c))?,
            None => ops::norm(gpu, &frames, &self.norm2, Shift::none(h, w))?,
        };
        let gu = gpu.empty(&[b * t * h * w, self.gu_w.shape[0]], Dtype::Bf16)?;
        gpu.linear(&xn, &self.gu_w, Some(&self.gu_b), &gu, false)?;
        let mid = ops::swiglu(gpu, &gu)?;
        match &params {
            Some(p) => {
                let m = gpu.empty(&[b * t * h * w, c], Dtype::F32)?;
                gpu.linear(&mid, &self.down_w, Some(&self.down_b), &m, false)?;
                ops::add_gated(
                    gpu,
                    &frames,
                    &m,
                    Gate::Frame {
                        gate: &p.at(5 * c, &[p.numel() - 5 * c])?,
                        per,
                        ld: 6 * c,
                    },
                    (c, 0),
                )?;
            }
            None => gpu.linear(&mid, &self.down_w, Some(&self.down_b), &frames, true)?,
        }
        frames.reshape(&[b, t, h, w, c])
    }
}

/// Blocks `{prefix}.{i}.` of `kinds`.
///
/// # Errors
/// When a weight is missing.
pub fn load_blocks(ld: &Loader, prefix: &str, kinds: &[Kind], temporal: bool, heads: usize) -> Result<Vec<Block>> {
    kinds
        .iter()
        .enumerate()
        .map(|(i, &kind)| {
            let at = format!("{}{prefix}.{i}.", ld.prefix);
            Block::load(&Loader { prefix: &at, ..*ld }, Spec { kind, temporal, heads })
        })
        .collect()
}
