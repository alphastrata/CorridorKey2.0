//! Frames in and out. `prep`: u16 RGB `[n, h0, w0, 3]` reflect-padded about `(top, left)` into f32 planes
//! `[n, 3, h, w]` (`ld` floats per frame): `mode` 0 display sRGB (or the tone-limited adjusted linear), 1 the model's
//! asinh-encoded linear. `hint`: u16 grey the same way, clipped to `[0, 1]`. `axpy`: `x += a · y`. `place`:
//! `out[r, off + j] = x[r, j] · s`, `ld`-wide rows. `render`: logits `[4, h, w]` cropped to `h0 × w0` as u16 BGR(A):
//! `kind` 0 alpha, 1 foreground, 2 cutout, 3 over a checker.

use crate::gid;
use ptx::kernel;
use ptx::prelude::*;

const KNEE: f32 = 0.008;
const DENOM: f32 = 8.294_05;

#[derive(Clone, Copy)]
struct Adjust {
    on: bool,
    gain: f32,
    contrast: f32,
    pivot: f32,
}

impl Adjust {
    fn apply(self, v: f32) -> f32 {
        if self.on {
            self.gain * (self.pivot + self.contrast * (v - self.pivot))
        } else {
            v
        }
    }

    fn invert(self, v: f32) -> f32 {
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

fn to_linear(x: f32) -> f32 {
    let a = x.abs();
    let v = if a <= 0.040_45 {
        a / 12.92
    } else {
        pow((a + 0.055) / 1.055, 2.4)
    };
    v.copysign(x)
}

fn to_srgb(x: f32) -> f32 {
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
    let i = gid();
    let [n, h0, w0, h, w, top, left, ld] = [n, h0, w0, h, w, top, left, ld].map(|v| v as usize);
    if i >= n * 3 * h * w {
        return;
    }
    let (x, y, ch, f) = (i % w, i / w % h, i / (w * h) % 3, i / (3 * w * h));
    let (sy, sx) = (
        reflect(y as isize - top as isize, h0),
        reflect(x as isize - left as isize, w0),
    );
    let v = f32::from(raw[((f * h0 + sy) * w0 + sx) * 3 + ch]) / 65535.0;
    let adj = Adjust {
        on: adjust != 0,
        gain,
        contrast,
        pivot,
    };
    let lin = adj.apply(to_linear(v));
    let val = match (mode, adj.on) {
        (0, false) => v,
        (0, true) => to_srgb(lin.max(0.0)).clamp(0.0, 1.0),
        _ => {
            let z = lin / KNEE;
            (z.abs() + (z * z + 1.0).sqrt()).ln().copysign(z) / DENOM
        }
    };
    out.set(f * ld + (ch * h + y) * w + x, val);
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
    let i = gid();
    let [n, h0, w0, h, w, top, left, ld] = [n, h0, w0, h, w, top, left, ld].map(|v| v as usize);
    if i >= n * h * w {
        return;
    }
    let (x, y, f) = (i % w, i / w % h, i / (w * h));
    let (sy, sx) = (
        reflect(y as isize - top as isize, h0),
        reflect(x as isize - left as isize, w0),
    );
    out.set(
        f * ld + y * w + x,
        (f32::from(raw[(f * h0 + sy) * w0 + sx]) / 65535.0).clamp(0.0, 1.0),
    );
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

fn despill([r, g, b]: [f32; 3], s: f32) -> [f32; 3] {
    let spill = (g - f32::midpoint(r, b)).max(0.0);
    let full = [r + spill * 0.5, g - spill, b + spill * 0.5];
    if s <= 0.0 {
        [r, g, b]
    } else if s >= 1.0 {
        full
    } else {
        [
            r * (1.0 - s) + full[0] * s,
            g * (1.0 - s) + full[1] * s,
            b * (1.0 - s) + full[2] * s,
        ]
    }
}

fn unorm(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0) as u16
}

#[kernel]
pub fn render(
    logits: &[f32],
    out: &mut [u16],
    h0: u32,
    w0: u32,
    h: u32,
    w: u32,
    top: u32,
    left: u32,
    kind: u32,
    strength: f32,
    linear_cutout: u32,
    adjust: u32,
    gain: f32,
    contrast: f32,
    pivot: f32,
) {
    let i = gid();
    let [h0, w0, h, w, top, left] = [h0, w0, h, w, top, left].map(|v| v as usize);
    if i >= h0 * w0 {
        return;
    }
    let (x, y) = (i % w0, i / w0);
    let at = |c: usize| logits[(c * h + y + top) * w + x + left];
    let a = at(0).clamp(0.0, 1.0);
    let adj = Adjust {
        on: adjust != 0,
        gain,
        contrast,
        pivot,
    };
    let lin = [1, 2, 3].map(|c| {
        let e = (at(c) * DENOM).exp();
        adj.invert((e - e.rcp()) * 0.5 * KNEE).max(0.0)
    });
    let srgb = lin.map(|v| to_srgb(v).clamp(0.0, 1.0));
    let (rgb, alpha) = match kind {
        0 => ([a; 3], None),
        1 => (srgb, None),
        2 => (
            despill(if linear_cutout == 0 { srgb } else { lin }, strength),
            Some(a),
        ),
        _ => {
            let bg = to_linear(if (y / 64 + x / 64) % 2 == 0 {
                0.75
            } else {
                0.5
            });
            (
                despill(srgb, strength)
                    .map(|v| to_srgb((to_linear(v) * a + bg * (1.0 - a)).max(0.0))),
                None,
            )
        }
    };
    let ch = 3 + usize::from(alpha.is_some());
    [rgb[2], rgb[1], rgb[0]]
        .into_iter()
        .chain(alpha)
        .enumerate()
        .for_each(|(c, v)| out.set(i * ch + c, unorm(v)));
}
