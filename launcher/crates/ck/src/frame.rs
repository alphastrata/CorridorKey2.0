//! Frames between u16 pixels and the model: padding, colour encoding, rendering (`ck_ops` `prep`, `hint`, `render`).

use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::Result;

/// A native `h0 × w0` frame centred in the model's `h × w`, reflect-padded.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub h0: usize,
    pub w0: usize,
    pub h: usize,
    pub w: usize,
}

impl Geometry {
    /// Native size padded up to multiples of `quantum`.
    #[must_use]
    pub fn padded(h0: usize, w0: usize, quantum: usize) -> Self {
        let up = |n: usize| n.div_ceil(quantum).max(1) * quantum;
        Self {
            h0,
            w0,
            h: up(h0),
            w: up(w0),
        }
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

/// Scene-linear exposure, `gain · (pivot + contrast · (v − pivot))`.
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

/// u16 RGB `raw [n, h0, w0, 3]` → f32 planes `[n, 3, h, w]` at `ld` floats per frame from `out`'s start.
///
/// # Errors
/// On a launch failure.
pub fn prep(gpu: &Gpu, raw: &Tensor, out: &Tensor, g: Geometry, ld: usize, enc: Encoding, adj: Adjust) -> Result<()> {
    let n = raw.numel() / (g.h0 * g.w0 * 3);
    let args = [
        [raw.into(), out.into(), n.into()].as_slice(),
        &g.args(),
        &[ld.into(), Arg::U(enc as u32)],
        &adj.args(),
    ]
    .concat();
    launch(gpu, "prep", n * 3 * g.h * g.w, &args)
}

/// u16 grey `raw [n, h0, w0]` → f32 `[n, h, w]` at `ld` floats per frame, clipped to `[0, 1]`.
///
/// # Errors
/// On a launch failure.
pub fn hint(gpu: &Gpu, raw: &Tensor, out: &Tensor, g: Geometry, ld: usize) -> Result<()> {
    let n = raw.numel() / (g.h0 * g.w0);
    let args = [[raw.into(), out.into(), n.into()].as_slice(), &g.args(), &[ld.into()]].concat();
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
}

/// How [`render`] finishes the foreground.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub despill: f32,
    pub linear_cutout: bool,
    pub adjust: Adjust,
}

/// Logits `[4, h, w]` cropped to native as u16 BGR(A) `[h0, w0, 3 | 4]`.
///
/// # Errors
/// On a launch failure.
pub fn render(gpu: &Gpu, logits: &Tensor, g: Geometry, out: Output, look: Look) -> Result<Tensor> {
    let img = gpu.empty(&[g.h0, g.w0, out.channels()], Dtype::U16)?;
    let args = [
        [logits.into(), (&img).into()].as_slice(),
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
