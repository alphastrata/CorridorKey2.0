use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::Result;

use crate::vit::{Resize, resize};

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub hn: usize,
    pub wn: usize,
    pub h0: usize,
    pub w0: usize,
    pub h: usize,
    pub w: usize,
}

impl Geometry {
    #[must_use]
    pub fn padded(h0: usize, w0: usize, quantum: usize) -> Self {
        Self::scaled(h0, w0, 1.0, quantum)
    }

    #[must_use]
    pub fn scaled(hn: usize, wn: usize, scale: f64, quantum: usize) -> Self {
        let by = |n: usize| {
            let scaled = n as f64 * scale;
            scaled.round_ties_even().max(0.0) as usize
        }
        .max(1);
        let up = |n: usize| n.div_ceil(quantum).max(1) * quantum;
        let (h0, w0) = (by(hn), by(wn));
        Self {
            hn,
            wn,
            h0,
            w0,
            h: up(h0),
            w: up(w0),
        }
    }

    #[must_use]
    pub fn native(self) -> bool {
        (self.hn, self.wn) == (self.h0, self.w0)
    }

    #[must_use]
    pub fn output(self) -> Self {
        Self::padded(self.hn, self.wn, 1)
    }

    fn args(self) -> [Arg; 6] {
        [
            self.h0,
            self.w0,
            self.h,
            self.w,
            (self.h - self.h0) / 2,
            (self.w - self.w0) / 2,
        ]
        .map(Arg::from)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Adjust {
    pub gain: f32,
    pub contrast: f32,
    pub pivot: f32,
}

impl Adjust {
    fn args(self) -> [Arg; 4] {
        let on = (self.gain - 1.0).abs() > 1e-6 || (self.contrast - 1.0).abs() > 1e-6;
        [
            usize::from(on).into(),
            Arg::F(self.gain),
            Arg::F(self.contrast),
            Arg::F(self.pivot),
        ]
    }
}

#[derive(Clone, Copy)]
pub enum Encoding {
    Display = 0,
    Model = 1,
}

fn launch(gpu: &Gpu, name: &str, n: usize, args: &[Arg]) -> Result<()> {
    gpu.launch("ck_ops", name, (n.div_ceil(256) as u32, 1, 1), 256, 0, args)
}

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub adjust: Adjust,
    pub linear: bool,
}

fn planar(gpu: &Gpu, raw: &Tensor, (h, w): (usize, usize), c: usize) -> Result<Tensor> {
    let n = raw.numel() / (h * w * c);
    let out = gpu.empty(&[n, c, h, w], Dtype::F32)?;
    let args = [[raw, &out].map(Arg::from).as_slice(), &[n, h, w, c].map(Arg::from)].concat();
    launch(gpu, "planar", out.numel(), &args)?;
    Ok(out)
}

pub fn lanczos(gpu: &Gpu, x: &Tensor, (oh, ow): (usize, usize)) -> Result<Tensor> {
    let (h, w) = (x.shape[2], x.shape[3]);
    let unit = (gpu.zeros(&[1], Dtype::F32)?, gpu.upload_f32(&[1.0], &[1], Dtype::F32)?);
    let (hf, ohf) = (f64::from(u32::try_from(h).unwrap_or(u32::MAX)), f64::from(u32::try_from(oh).unwrap_or(u32::MAX)));
    let (wf, owf) = (f64::from(u32::try_from(w).unwrap_or(u32::MAX)), f64::from(u32::try_from(ow).unwrap_or(u32::MAX)));
    let step = ((hf / ohf) as f32, (wf / owf) as f32);
    resize(gpu, x, (oh, ow), step, Resize::Lanczos, (&unit.0, &unit.1))
}

/// The native frame as `[n, 3, hn, wn]` planes (G, B, R) and those planes at the model's scale.
pub fn planes(gpu: &Gpu, raw: &Tensor, g: Geometry) -> Result<(Tensor, Tensor)> {
    let n = raw.numel() / (g.hn * g.wn * 3);
    let native = if raw.dtype == Dtype::F32 {
        raw.reshape(&[n, 3, g.hn, g.wn])?
    } else {
        planar(gpu, raw, (g.hn, g.wn), 3)?
    };
    let scaled = lanczos(gpu, &native, (g.h0, g.w0))?;
    Ok((native, scaled))
}

pub fn prep(gpu: &Gpu, raw: &Tensor, out: &Tensor, g: Geometry, ld: usize, enc: Encoding, input: Input) -> Result<()> {
    let float = raw.dtype == Dtype::F32;
    if !g.native() {
        let input = Input {
            linear: float && input.linear,
            ..input
        };
        return prep_planes(gpu, &planes(gpu, raw, g)?.1, out, g, ld, enc, input);
    }
    if float {
        return prep_planes(gpu, raw, out, g, ld, enc, input);
    }
    let n = raw.numel() / (g.h0 * g.w0 * 3);
    let args = [
        [raw.into(), out.into(), n.into()].as_slice(),
        &g.args(),
        &[ld.into(), Arg::U(enc as u32)],
        &input.adjust.args(),
    ]
    .concat();
    launch(gpu, "prep", n * 3 * g.h * g.w, &args)
}

pub fn prep_planes(
    gpu: &Gpu,
    raw: &Tensor,
    out: &Tensor,
    g: Geometry,
    ld: usize,
    enc: Encoding,
    input: Input,
) -> Result<()> {
    let n = raw.numel() / (g.h0 * g.w0 * 3);
    let args = [
        [raw.into(), out.into(), n.into()].as_slice(),
        &g.args(),
        &[ld.into(), Arg::U(enc as u32), usize::from(input.linear).into()],
        &input.adjust.args(),
    ]
    .concat();
    launch(gpu, "prep_f32", n * 3 * g.h * g.w, &args)
}

pub fn hint(gpu: &Gpu, raw: &Tensor, out: &Tensor, g: Geometry, ld: usize) -> Result<()> {
    let mut grey = planar(gpu, raw, (g.hn, g.wn), 1)?;
    if !g.native() {
        grey = lanczos(gpu, &grey, (g.h0, g.w0))?;
    }
    let n = grey.numel() / (g.h0 * g.w0);
    let args = [
        [(&grey).into(), out.into(), n.into()].as_slice(),
        &g.args(),
        &[ld.into()],
    ]
    .concat();
    launch(gpu, "hint", n * g.h * g.w, &args)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Output {
    Alpha = 0,
    Fg = 1,
    Cutout = 2,
    Checker = 3,
}

impl Output {
    #[must_use]
    pub fn channels(self) -> usize {
        if self == Self::Cutout { 4 } else { 3 }
    }

    #[must_use]
    pub fn planes(self) -> usize {
        if self == Self::Alpha { 1 } else { self.channels() }
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        ["alpha", "fg", "cutout", "checker"][self as usize]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub despill: f32,
    pub linear_cutout: bool,
    pub adjust: Adjust,
}

pub fn native(gpu: &Gpu, logits: &Tensor, g: Geometry) -> Result<(Tensor, Geometry)> {
    if g.native() {
        return Ok((logits.clone(), g));
    }
    let cropped = gpu.empty(&[1, 4, g.h0, g.w0], Dtype::F32)?;
    let dims = [4, g.h, g.w, g.h0, g.w0, (g.h - g.h0) / 2, (g.w - g.w0) / 2];
    let args = [[logits, &cropped].map(Arg::from).as_slice(), &dims.map(Arg::from)].concat();
    launch(gpu, "crop", cropped.numel(), &args)?;
    Ok((lanczos(gpu, &cropped, (g.hn, g.wn))?, g.output()))
}

pub fn render(gpu: &Gpu, logits: &Tensor, g: Geometry, out: Output, look: Look) -> Result<Tensor> {
    let (logits, g) = native(gpu, logits, g)?;
    let img = gpu.empty(&[g.h0, g.w0, out.channels()], Dtype::U16)?;
    let args = [
        [(&logits).into(), (&img).into()].as_slice(),
        &g.args(),
        &[
            Arg::U(out as u32),
            Arg::F(look.despill),
            usize::from(look.linear_cutout).into(),
        ],
        &look.adjust.args(),
    ]
    .concat();
    launch(gpu, "render", g.h0 * g.w0, &args)?;
    Ok(img)
}

pub fn render_exr(gpu: &Gpu, logits: &Tensor, g: Geometry, out: Output, look: Look, half: bool) -> Result<Tensor> {
    let (logits, g) = native(gpu, logits, g)?;
    let img = gpu.empty(&[out.planes(), g.h0, g.w0], Dtype::F32)?;
    let args = [
        [(&logits).into(), (&img).into()].as_slice(),
        &g.args(),
        &[Arg::U(out as u32), Arg::F(look.despill), usize::from(half).into()],
        &look.adjust.args(),
    ]
    .concat();
    launch(gpu, "render_exr", g.h0 * g.w0, &args)?;
    Ok(img)
}
