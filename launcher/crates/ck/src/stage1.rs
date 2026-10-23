use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::Result;

use crate::attn::Cache;
use crate::block::{Block, Kind, load_blocks};
use crate::ops::{self, Conv, Into, Shift};
use crate::weights::Loader;

const STEM: Conv = Conv { k: 32, s: 16, p: 8 };
const DOWN: Conv = Conv { k: 8, s: 4, p: 2 };
const SMOOTH: Conv = Conv { k: 3, s: 1, p: 1 };
const AUX: Conv = Conv { k: 16, s: 8, p: 4 };

const ENC_BYTES: usize = 1 << 30;
const AUX_BYTES: usize = 512 << 20;

type Pair = (Tensor, Tensor);

struct Up {
    proj: Pair,
    smooth: Pair,
}

impl Up {
    fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        let [n, h, w, c] = x.shape[..] else {
            eyre::bail!("up wants [n, h, w, c]")
        };
        let p = gpu.empty(&[n, h, w, self.proj.0.shape[0]], Dtype::F32)?;
        gpu.linear(
            &ops::to_half(gpu, &x.reshape(&[n * h * w, c])?)?,
            &self.proj.0,
            Some(&self.proj.1),
            &p,
            false,
        )?;
        let y = ops::shuffle(gpu, &p, 4)?;
        let into = Into {
            out: &y,
            ld: y.shape[3],
            off: 0,
            accumulate: true,
        };
        ops::conv(gpu, &y, true, SMOOTH, (&self.smooth.0, &self.smooth.1), into)?;
        Ok(y)
    }
}

struct Fuse {
    short: Pair,
    norm: Tensor,
    gu: Pair,
    down: Tensor,
}

impl Fuse {
    fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        let c = *x.shape.last().unwrap_or(&0);
        let rows = x.numel() / c;
        let out = gpu.empty(&[rows, self.short.0.shape[0]], Dtype::F32)?;
        gpu.linear(&ops::to_half(gpu, x)?, &self.short.0, Some(&self.short.1), &out, false)?;
        let flat = x.reshape(&[rows, 1, 1, c])?;
        let h = ops::norm(gpu, &flat, &self.norm, Shift::none(1, 1))?;
        let gu = gpu.empty(&[rows, self.gu.0.shape[0]], Dtype::Bf16)?;
        gpu.linear(&h, &self.gu.0, Some(&self.gu.1), &gu, false)?;
        gpu.linear(&ops::swiglu(gpu, &gu)?, &self.down, None, &out, true)?;
        Ok(out)
    }
}

pub struct Inputs<'a> {
    pub b: usize,
    pub t: usize,
    pub x: Tensor,
    pub quality: &'a Tensor,
    pub features: [&'a Tensor; 3],
}

pub struct Outputs {
    pub baton: Tensor,
    pub aux: Tensor,
}

pub struct Stage1 {
    stem: Pair,
    enc: Vec<Block>,
    down: Pair,
    fuse: Fuse,
    body: Vec<Block>,
    mid_up: Up,
    mid_scale: Tensor,
    mid: Vec<Block>,
    mid_down: Pair,
    up: Up,
    skip_scale: Tensor,
    late_scale: Tensor,
    dec: Vec<Block>,
    aux: Pair,
}

impl Stage1 {
    pub fn load(ld: &Loader) -> Result<Self> {
        let pair = |n: &str| -> Result<Pair> { Ok((ld.mat(&format!("{n}.weight"))?, ld.f32(&format!("{n}.bias"))?)) };
        let nhwc = |n: &str| -> Result<Pair> { Ok((ld.nhwc(&format!("{n}.weight"))?, ld.f32(&format!("{n}.bias"))?)) };
        let count = |p: &str| (0..64).take_while(|i| ld.has(&format!("{p}.{i}.norm1.weight"))).count();
        let (w, hw) = (16, 8);
        let swin = |shift| Kind::Swin { window: w, shift };
        let phases = [swin((0, 0)), swin((hw, hw)), swin((hw, 0)), swin((0, hw))];
        let mid: Vec<Kind> = (0..count("mid_16")).map(|i| phases[i % 4]).collect();
        let dec = if count("dec_16") == 8 {
            vec![
                phases[0],
                phases[1],
                Kind::CSwin,
                phases[2],
                phases[3],
                Kind::CSwin,
                phases[2],
                phases[3],
            ]
        } else {
            vec![phases[0], phases[1], Kind::CSwin, Kind::CSwin, phases[2], phases[3]]
        };
        let up = |n: &str| -> Result<Up> {
            Ok(Up {
                proj: pair(&format!("{n}.proj"))?,
                smooth: nhwc(&format!("{n}.smooth"))?,
            })
        };
        Ok(Self {
            stem: pair("stem.conv")?,
            enc: load_blocks(ld, "enc_16", &vec![Kind::Full; count("enc_16")], false, 8)?,
            down: nhwc("down.conv")?,
            fuse: Fuse {
                short: pair("fuse.linear_shortcut")?,
                norm: ld.f32("fuse.norm.weight")?,
                gu: (
                    ld.rows(&["fuse.w_gate.weight", "fuse.w_up.weight"])?,
                    ld.cat(&["fuse.w_gate.bias", "fuse.w_up.bias"])?,
                ),
                down: ld.biased("fuse.w_down.weight", "fuse.w_down.bias")?,
            },
            body: load_blocks(ld, "body_64", &vec![Kind::Full; count("body_64")], true, 16)?,
            mid_up: up("mid_up")?,
            mid_scale: ld.f32("mid_skip_scale")?,
            mid: load_blocks(ld, "mid_16", &mid, true, 8)?,
            mid_down: nhwc("mid_down.conv")?,
            up: up("up")?,
            skip_scale: ld.f32("uskip_scale")?,
            late_scale: ld.f32("patch_late_skip_scale")?,
            dec: load_blocks(ld, "dec_16", &dec, true, 8)?,
            aux: (
                ld.deconv("aux_depatchify.deconv.weight")?,
                ld.f32("aux_depatchify.deconv.bias")?,
            ),
        })
    }

    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        self.enc.iter().chain(&self.body).chain(&self.mid).chain(&self.dec)
    }

    fn encode(&self, gpu: &Gpu, cache: &Cache, x: Tensor, quality: &Tensor) -> Result<(Tensor, Tensor)> {
        let [frames, _, hh, ww] = x.shape[..] else {
            eyre::bail!("x wants [b · t, 4, H, W]")
        };
        let (h16, w16) = (hh / 16, ww / 16);
        let d16 = self.stem.0.shape[0];
        let patch = gpu.empty(&[frames, h16, w16, d16], Dtype::F32)?;
        ops::conv(
            gpu,
            &x,
            false,
            STEM,
            (&self.stem.0, &self.stem.1),
            Into {
                out: &patch,
                ld: d16,
                off: 0,
                accumulate: false,
            },
        )?;
        drop(x);
        ops::probe(gpu, "stem");
        let enc = gpu.copy(&patch)?;
        let step = (ENC_BYTES / (h16 * w16 * d16 * 18)).clamp(1, frames);
        (0..frames).step_by(step).try_for_each(|f| {
            let n = step.min(frames - f);
            let x = enc.rows(f, n)?.reshape(&[1, n, h16, w16, d16])?;
            let q = quality.rows(f, n)?;
            self.enc
                .iter()
                .try_fold(x, |x, b| b.forward(gpu, cache, &x, Some(&q)))
                .map(drop)
        })?;
        Ok((patch, enc))
    }

    fn aux(&self, gpu: &Gpu, baton: &Tensor, (h16, w16): (usize, usize)) -> Result<Tensor> {
        let d16 = *baton.shape.last().unwrap_or(&0);
        let frames = baton.numel() / (h16 * w16 * d16);
        let (h2, w2) = AUX.inverse(h16, w16);
        let aux = gpu.empty(&[frames, 4, h2, w2], Dtype::F32)?;
        let taps = self.aux.0.shape[0];
        let rows = baton.reshape(&[frames, h16 * w16, d16])?;
        let step = (AUX_BYTES / (h16 * w16 * taps * 4)).clamp(1, frames);
        (0..frames).step_by(step).try_for_each(|f| {
            let n = step.min(frames - f);
            let cols = gpu.empty(&[n * h16 * w16, taps], Dtype::F32)?;
            gpu.linear(&ops::to_half(gpu, &rows.rows(f, n)?)?, &self.aux.0, None, &cols, false)?;
            ops::col2im_into(gpu, &cols, &self.aux.1, AUX, &aux.rows(f, n)?)
        })?;
        Ok(aux)
    }

    pub fn forward(&self, gpu: &Gpu, cache: &Cache, i: Inputs) -> Result<Outputs> {
        let Inputs {
            b,
            t,
            x,
            quality,
            features,
        } = i;
        let [frames, _, hh, ww] = x.shape[..] else {
            eyre::bail!("x wants [b · t, 4, H, W]")
        };
        let (h16, w16, h64, w64) = (hh / 16, ww / 16, hh / 64, ww / 64);
        let d16 = self.stem.0.shape[0];
        let d64 = self.fuse.short.0.shape[0];
        let grid16 = [b, t, h16, w16, d16];
        let (patch, enc) = self.encode(gpu, cache, x, quality)?;
        let (patch, enc) = (patch.into_shape(&grid16)?, enc.into_shape(&grid16)?);

        ops::probe(gpu, "enc");
        let vfm: usize = features.iter().map(|f| f.shape[1]).sum();
        let combined = gpu.empty(&[frames, h64, w64, vfm + d16], Dtype::F32)?;
        features.iter().try_fold(0, |off, f| {
            ops::to_nhwc(gpu, f, &combined, (vfm + d16, off)).map(|()| off + f.shape[1])
        })?;
        let enc_frames = enc.reshape(&[frames, h16, w16, d16])?;
        ops::conv(
            gpu,
            &enc_frames,
            true,
            DOWN,
            (&self.down.0, &self.down.1),
            Into {
                out: &combined,
                ld: vfm + d16,
                off: vfm,
                accumulate: false,
            },
        )?;
        let f64 = self.fuse.forward(gpu, &combined)?.into_shape(&[b, t, h64, w64, d64])?;
        drop(combined);

        ops::probe(gpu, "fuse");
        let split = self.body.len() / 2;
        let f64 = self.body[..split]
            .iter()
            .try_fold(f64, |x, b| b.forward(gpu, cache, &x, None))?;
        ops::probe(gpu, "body");
        let mid = self.mid_up.forward(gpu, &f64.reshape(&[frames, h64, w64, d64])?)?;
        ops::add_scaled(gpu, &mid, &enc, &self.mid_scale)?;
        drop((enc, enc_frames));
        let mid = self
            .mid
            .iter()
            .try_fold(mid.into_shape(&grid16)?, |x, b| b.forward(gpu, cache, &x, None))?;
        ops::probe(gpu, "mid");
        let mid_frames = mid.reshape(&[frames, h16, w16, d16])?;
        ops::conv(
            gpu,
            &mid_frames,
            true,
            DOWN,
            (&self.mid_down.0, &self.mid_down.1),
            Into {
                out: &f64,
                ld: d64,
                off: 0,
                accumulate: true,
            },
        )?;
        let f64 = self.body[split..]
            .iter()
            .try_fold(f64, |x, b| b.forward(gpu, cache, &x, None))?;

        ops::probe(gpu, "body2");
        let dec = self.up.forward(gpu, &f64.into_shape(&[frames, h64, w64, d64])?)?;
        ops::add_scaled(gpu, &dec, &mid, &self.skip_scale)?;
        drop((mid, mid_frames));
        let late = self.dec.len().saturating_sub(2);
        let baton = self
            .dec
            .iter()
            .enumerate()
            .try_fold(dec.into_shape(&grid16)?, |x, (k, b)| {
                if k == late {
                    ops::add_scaled(gpu, &x, &patch, &self.late_scale)?;
                }
                b.forward(gpu, cache, &x, None)
            })?;
        drop(patch);

        ops::probe(gpu, "dec");
        let aux = self.aux(gpu, &baton, (h16, w16))?;
        ops::probe(gpu, "aux");
        Ok(Outputs { baton, aux })
    }
}
