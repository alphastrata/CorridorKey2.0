//! Frames in. `prep`: u16 RGB `[n, h0, w0, 3]` reflect-padded about `(top, left)` into f32 planes `[n, 3, h, w]`
//! (`ld` floats per frame): `mode` 0 display sRGB (or the tone-limited adjusted linear), 1 the model's asinh-encoded
//! linear. `prep_f32`: the same from f32 G, B, R planes `[n, 3, h0, w0]`, scene-linear if `linear`, else display
//! clipped to `[0, 1]`. `hint`: u16 grey the same way, clipped to `[0, 1]`. `axpy`: `x += a · y`. `place`:
//! `out[r, off + j] = x[r, j] · s`, `ld`-wide rows.

use crate::gid;
use ptx::kernel;
use ptx::prelude::*;

pub(crate) const KNEE: f32 = 0.008;
pub(crate) const DENOM: f32 = 8.294_05;

#[derive(Clone, Copy)]
pub(crate) struct Adjust {
    pub on: bool,
    pub gain: f32,
    pub contrast: f32,
    pub pivot: f32,
}

impl Adjust {
    fn apply(self, v: f32) -> f32 {
        if self.on {
            self.gain * (self.pivot + self.contrast * (v - self.pivot))
        } else {
            v
        }
    }

    pub fn invert(self, v: f32) -> f32 {
        if self.on {
            (v / self.gain - self.pivot) / self.contrast + self.pivot
        } else {
            v
        }
    }
}

fn pow(x: f32, p: f32) -> f32 {
    (x.ln() * p).exp()
}

pub(crate) fn to_linear(x: f32) -> f32 {
    let a = x.abs();
    let v = if a <= 0.040_45 {
        a / 12.92
    } else {
        pow((a + 0.055) / 1.055, 2.4)
    };
    v.copysign(x)
}

pub(crate) fn to_srgb(x: f32) -> f32 {
    let a = x.abs();
    let v = if a <= 0.003_130_8 {
        a * 12.92
    } else {
        1.055 * pow(a, 1.0 / 2.4) - 0.055
    };
    v.copysign(x)
}

fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let i = if i < 0 { -i } else { i };
    (if i >= n { 2 * (n - 1) - i } else { i }) as usize
}

/// Frame, channel, padded and source coordinates of output `i`, if in range.
fn texel(i: usize, dims: [usize; 7], c: usize) -> Option<[usize; 5]> {
    let [frames, h0, w0, rows, cols, top, left] = dims;
    if i >= frames * c * rows * cols {
        return None;
    }
    let (col, row) = (i % cols, i / cols % rows);
    let (ch, f) = (i / (cols * rows) % c, i / (c * cols * rows));
    let sy = reflect(row as isize - top as isize, h0);
    let sx = reflect(col as isize - left as isize, w0);
    Some([f, ch, row * cols + col, sy, sx])
}

fn encode(v: f32, linear: bool, mode: u32, adj: Adjust) -> f32 {
    let lin = adj.apply(if linear { v } else { to_linear(v) });
    match mode {
        0 if !linear && !adj.on => v,
        0 => to_srgb(lin.max(0.0)).clamp(0.0, 1.0),
        _ => {
            let z = lin / KNEE;
            (z.abs() + (z * z + 1.0).sqrt()).ln().copysign(z) / DENOM
        }
    }
}

#[kernel]
pub fn prep(
    raw: &[u16],
    out: &mut [f32],
    n: u32,
    h0: u32,
    w0: u32,
    h: u32,
    w: u32,
    top: u32,
    left: u32,
    ld: u32,
    mode: u32,
    adjust: u32,
    gain: f32,
    contrast: f32,
    pivot: f32,
) {
    let dims = [n, h0, w0, h, w, top, left].map(|v| v as usize);
    let Some([f, ch, yx, sy, sx]) = texel(gid(), dims, 3) else {
        return;
    };
    let (h0, w0, plane) = (dims[1], dims[2], dims[3] * dims[4]);
    let v = f32::from(raw[((f * h0 + sy) * w0 + sx) * 3 + ch]) / 65535.0;
    let adj = Adjust {
        on: adjust != 0,
        gain,
        contrast,
        pivot,
    };
    out.set(f * ld as usize + ch * plane + yx, encode(v, false, mode, adj));
}

#[kernel]
pub fn prep_f32(
    raw: &[f32],
    out: &mut [f32],
    n: u32,
    h0: u32,
    w0: u32,
    h: u32,
    w: u32,
    top: u32,
    left: u32,
    ld: u32,
    mode: u32,
    linear: u32,
    adjust: u32,
    gain: f32,
    contrast: f32,
    pivot: f32,
) {
    let dims = [n, h0, w0, h, w, top, left].map(|v| v as usize);
    let Some([f, ch, yx, sy, sx]) = texel(gid(), dims, 3) else {
        return;
    };
    let (h0, w0, plane) = (dims[1], dims[2], dims[3] * dims[4]);
    let v = raw[((f * 3 + (ch + 2) % 3) * h0 + sy) * w0 + sx];
    let linear = linear != 0;
    let adj = Adjust {
        on: adjust != 0,
        gain,
        contrast,
        pivot,
    };
    let v = if linear { v } else { v.clamp(0.0, 1.0) };
    out.set(f * ld as usize + ch * plane + yx, encode(v, linear, mode, adj));
}

#[kernel]
pub fn hint(
    raw: &[u16],
    out: &mut [f32],
    n: u32,
    h0: u32,
    w0: u32,
    h: u32,
    w: u32,
    top: u32,
    left: u32,
    ld: u32,
) {
    let dims = [n, h0, w0, h, w, top, left].map(|v| v as usize);
    let Some([f, _, yx, sy, sx]) = texel(gid(), dims, 1) else {
        return;
    };
    let v = f32::from(raw[(f * dims[1] + sy) * dims[2] + sx]) / 65535.0;
    out.set(f * ld as usize + yx, v.clamp(0.0, 1.0));
}

#[kernel]
pub fn axpy(x: &mut [f32], y: &[f32], a: f32, n: u32) {
    let i = gid();
    if i < n as usize {
        let v = a.fma_rn(y[i], x[i]);
        x.set(i, v);
    }
}

#[kernel]
pub fn place(x: &[f32], out: &mut [f32], rows: u32, c: u32, ld: u32, off: u32, s: f32) {
    let i = gid();
    let [rows, c, ld, off] = [rows, c, ld, off].map(|v| v as usize);
    if i < rows * c {
        out.set(i / c * ld + off + i % c, x[i] * s);
    }
}
