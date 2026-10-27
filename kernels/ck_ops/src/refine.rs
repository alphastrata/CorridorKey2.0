use crate::frame::{Adjust, DENOM, KNEE, to_linear, to_srgb};
use crate::{gelu, gid};
use ptx::kernel;
use ptx::mem::ld_global_nc_v4_f32;
use ptx::prelude::*;
use ptx::slice::GpuSlice;

const WIDTH: usize = 24;
const LOW: usize = 7;

fn ld(s: GpuSlice<f32>, i: usize) -> f32 {
    unsafe { *s.get_unchecked(i) }
}

fn display(v: f32, linear: bool) -> f32 {
    if linear {
        to_srgb(v.max(0.0)).clamp(0.0, 1.0)
    } else {
        v.clamp(0.0, 1.0)
    }
}

fn rgb(planes: GpuSlice<f32>, plane: usize, at: usize, linear: bool) -> [f32; 3] {
    [2, 0, 1].map(|p| display(planes[p * plane + at], linear))
}

fn window(
    (y, x): (usize, usize),
    (h, w): (usize, usize),
    r: usize,
    mut f: impl FnMut(usize),
) -> f32 {
    let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
    let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
    (y0..y1).for_each(|yy| (x0..x1).for_each(|xx| f(yy * w + xx)));
    ((y1 - y0) * (x1 - x0)) as f32
}

#[derive(Clone, Copy)]
struct Taps {
    at: [usize; 4],
    ty: f32,
    tx: f32,
    plane: usize,
}

impl Taps {
    fn new((h, w): (usize, usize), (oy, ox): (usize, usize), (oh, ow): (usize, usize)) -> Self {
        let axis = |o: usize, n: usize, on: usize| {
            let f = ((o as f32 + 0.5) * (n as f32 / on as f32) - 0.5).max(0.0);
            let i = (f as usize).min(n - 1);
            (i, (i + 1).min(n - 1), f - i as f32)
        };
        let (y0, y1, ty) = axis(oy, h, oh);
        let (x0, x1, tx) = axis(ox, w, ow);
        Self {
            at: [y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1],
            ty,
            tx,
            plane: h * w,
        }
    }

    fn get(self, x: GpuSlice<f32>, p: usize) -> f32 {
        let b = p * self.plane;
        let row = |a: usize| (1.0 - self.tx) * ld(x, b + self.at[a]) + self.tx * ld(x, b + self.at[a + 1]);
        (1.0 - self.ty) * row(0) + self.ty * row(2)
    }
}

#[kernel]
pub fn refine_low(
    lo: &[f32],
    logits: &[f32],
    out: &mut [f32],
    h: u32,
    w: u32,
    linear: u32,
    adjust: u32,
    gain: f32,
    contrast: f32,
    pivot: f32,
) {
    let i = gid();
    let n = (h * w) as usize;
    if i >= n {
        return;
    }
    let adj = Adjust {
        on: adjust != 0,
        gain,
        contrast,
        pivot,
    };
    let rgb = rgb(lo, n, i, linear != 0);
    let fg = [1, 2, 3].map(|c| {
        let e = (logits[c * n + i] * DENOM).exp();
        to_srgb(adj.invert((e - e.rcp()) * 0.5 * KNEE).max(0.0)).clamp(0.0, 1.0)
    });
    let vals = [
        rgb[0],
        rgb[1],
        rgb[2],
        logits[i].clamp(0.0, 1.0),
        fg[0],
        fg[1],
        fg[2],
    ];
    vals.into_iter()
        .enumerate()
        .for_each(|(c, v)| out.set(c * n + i, v));
}

#[kernel]
pub fn guided(low: &[f32], ab: &mut [f32], h: u32, w: u32, eps: f32) {
    let i = gid();
    let (h, w) = (h as usize, w as usize);
    let n = h * w;
    if i >= n {
        return;
    }
    let mut m = [0.0f32; 13];
    let count = window((i / w, i % w), (h, w), 2, |j| {
        let c = [low[j], low[n + j], low[2 * n + j]];
        let p = low[3 * n + j];
        let v = [
            c[0],
            c[1],
            c[2],
            p,
            c[0] * p,
            c[1] * p,
            c[2] * p,
            c[0] * c[0],
            c[0] * c[1],
            c[0] * c[2],
            c[1] * c[1],
            c[1] * c[2],
            c[2] * c[2],
        ];
        (0..13).for_each(|k| m[k] += v[k]);
    });
    let m = m.map(|v| v / count);
    let cov = [0, 1, 2].map(|k| f64::from(m[4 + k] - m[k] * m[3]));
    let s = |a: usize, b: usize, at: usize| f64::from(m[at] - m[a] * m[b]);
    let e = f64::from(eps);
    let (s00, s01, s02) = (s(0, 0, 7) + e, s(0, 1, 8), s(0, 2, 9));
    let (s11, s12, s22) = (s(1, 1, 10) + e, s(1, 2, 11), s(2, 2, 12) + e);
    let adj = [
        [s11 * s22 - s12 * s12, s02 * s12 - s01 * s22, s01 * s12 - s02 * s11],
        [s02 * s12 - s01 * s22, s00 * s22 - s02 * s02, s01 * s02 - s00 * s12],
        [s01 * s12 - s02 * s11, s01 * s02 - s00 * s12, s00 * s11 - s01 * s01],
    ];
    let det = s00 * adj[0][0] + s01 * adj[0][1] + s02 * adj[0][2];
    let a = adj.map(|r| ((r[0] * cov[0] + r[1] * cov[1] + r[2] * cov[2]) / det) as f32);
    let b = m[3] - (a[0] * m[0] + a[1] * m[1] + a[2] * m[2]);
    [a[0], a[1], a[2], b]
        .into_iter()
        .enumerate()
        .for_each(|(c, v)| ab.set(c * n + i, v));
}

#[kernel]
pub fn box_mean(x: &[f32], out: &mut [f32], planes: u32, h: u32, w: u32, r: u32) {
    let i = gid();
    let (h, w) = (h as usize, w as usize);
    let n = h * w;
    if i >= planes as usize * n {
        return;
    }
    let (p, at) = (i / n, i % n);
    let mut sum = 0.0f32;
    let count = window((at / w, at % w), (h, w), r as usize, |j| {
        sum += x[p * n + j];
    });
    out.set(i, sum / count);
}

#[kernel]
pub fn band(low: &[f32], out: &mut [f32], h: u32, w: u32, gain: f32, lo: f32) {
    let i = gid();
    let (h, w) = (h as usize, w as usize);
    let n = h * w;
    if i >= n {
        return;
    }
    let (mut unc, mut near) = (0.0f32, 0.0f32);
    window((i / w, i % w), (h, w), 2, |j| {
        let a = low[3 * n + j];
        if a > lo && a < 0.996 {
            unc = 1.0;
        }
        near = near.max(a);
    });
    out.set(i, unc * (near * gain).clamp(0.0, 1.0));
}

#[kernel]
pub fn bg_seed(low: &[f32], out: &mut [f32], n: u32) {
    let i = gid();
    let n = n as usize;
    if i >= n {
        return;
    }
    let wt = (1.0 - low[3 * n + i]).clamp(0.0, 1.0);
    let wt = (wt * wt) * (wt * wt);
    (0..3).for_each(|c| out.set(c * n + i, to_linear(low[c * n + i].clamp(0.0, 1.0)) * wt));
    out.set(3 * n + i, wt);
}

#[kernel]
pub fn pool2(x: &[f32], out: &mut [f32], h: u32, w: u32) {
    let i = gid();
    let (h, w) = (h as usize, w as usize);
    let (oh, ow) = (h.div_ceil(2), w.div_ceil(2));
    if i >= 4 * oh * ow {
        return;
    }
    let (p, oy, ox) = (i / (oh * ow), i / ow % oh, i % ow);
    let (y1, x1) = ((2 * oy + 2).min(h), (2 * ox + 2).min(w));
    let sum = (2 * oy..y1).fold(0.0f32, |s, y| {
        (2 * ox..x1).fold(s, |s, c| s + x[(p * h + y) * w + c])
    });
    out.set(i, sum / ((y1 - 2 * oy) * (x1 - 2 * ox)) as f32);
}

#[kernel]
pub fn pull(est: &[f32], lvl: &[f32], out: &mut [f32], hc: u32, wc: u32, h: u32, w: u32, last: u32) {
    let i = gid();
    let [hc, wc, h, w] = [hc, wc, h, w].map(|v| v as usize);
    let n = h * w;
    if i >= 3 * n {
        return;
    }
    let (c, at) = (i / n, i % n);
    let e = Taps::new((hc, wc), (at / w, at % w), (h, w)).get(est, c);
    let d = lvl[3 * n + at].clamp(0.0, 1.0);
    let v = (lvl[c * n + at] + (1.0 - d) * e) / (d + (1.0 - d)).max(1e-6);
    out.set(i, if last == 0 { v } else { to_srgb(v.clamp(0.0, 1.0)) });
}

#[kernel]
pub fn refine_in(
    native: &[f32],
    low: &[f32],
    coef: &[f32],
    bg: &[f32],
    x: &mut [f32],
    y0: u32,
    rows: u32,
    hn: u32,
    wn: u32,
    h: u32,
    w: u32,
    linear: u32,
) {
    let i = gid();
    let [y0, rows, hn, wn, h, w] = [y0, rows, hn, wn, h, w].map(|v| v as usize);
    let plane = rows * wn;
    if i >= plane {
        return;
    }
    let (yy, xx) = (y0 + i / wn, i % wn);
    let img = rgb(native, hn * wn, yy * wn + xx, linear != 0);
    let t = Taps::new((h, w), (yy, xx), (hn, wn));
    let mut put = |c: usize, v: f32| x.set(c * plane + i, v);
    (0..3).for_each(|c| {
        put(c, img[c]);
        put(3 + c, t.get(bg, c));
        put(6 + c, (img[c] + t.get(low, 4 + c) - t.get(low, c)).clamp(0.0, 1.0));
    });
    put(9, t.get(low, 3));
    let gf = (0..3).fold(t.get(coef, 3), |s, c| t.get(coef, c).fma_rn(img[c], s));
    put(10, gf.clamp(0.0, 1.0));
}

#[kernel]
pub fn refine_conv(
    x: &[f32],
    out: &mut [f32],
    wt: &[f32],
    bias: &[f32],
    cin: u32,
    width: u32,
    rows: u32,
    cols: u32,
    dil: u32,
    residual: u32,
) {
    let i = gid();
    let [cin, width, rows, cols, dil] = [cin, width, rows, cols, dil].map(|v| v as usize);
    let plane = rows * cols;
    if i >= plane {
        return;
    }
    let (y, xx) = (i / cols, i % cols);
    let mut acc: [f32; WIDTH] = core::array::from_fn(|o| ld(bias, o));
    (0..cin * 9).for_each(|k| {
        let (c, t) = (k / 9, k % 9);
        let sy = (y + (t / 3) * dil).wrapping_sub(dil);
        let sx = (xx + (t % 3) * dil).wrapping_sub(dil);
        if sy < rows && sx < cols {
            let v = ld(x, c * plane + sy * cols + sx);
            ptx::unroll!(6, |q| {
                let w4 = unsafe { ld_global_nc_v4_f32(wt.addr_at(k * WIDTH + 4 * q)) };
                acc[4 * q] = w4.0.fma_rn(v, acc[4 * q]);
                acc[4 * q + 1] = w4.1.fma_rn(v, acc[4 * q + 1]);
                acc[4 * q + 2] = w4.2.fma_rn(v, acc[4 * q + 2]);
                acc[4 * q + 3] = w4.3.fma_rn(v, acc[4 * q + 3]);
            });
        }
    });
    ptx::unroll!(24, |o| if o < width {
        let g = gelu(acc[o]);
        out.set(o * plane + i, if residual == 0 { g } else { ld(x, o * plane + i) + g });
    });
}

#[kernel]
pub fn refine_out(
    x: &[f32],
    hid: &[f32],
    wt: &[f32],
    bias: &[f32],
    band: &[f32],
    logits: &mut [f32],
    rows: u32,
    core: u32,
    take: u32,
    y0: u32,
    hn: u32,
    wn: u32,
    h: u32,
    w: u32,
    width: u32,
    net: u32,
    adjust: u32,
    gain: f32,
    contrast: f32,
    pivot: f32,
) {
    let i = gid();
    let [rows, core, take, y0, hn, wn, h, w, width] =
        [rows, core, take, y0, hn, wn, h, w, width].map(|v| v as usize);
    if i >= take * wn {
        return;
    }
    let plane = rows * wn;
    let p = core * wn + i;
    let (yy, xx) = (y0 + core + i / wn, i % wn);
    let gf = x[10 * plane + p];
    let fg_in = [6, 7, 8].map(|c| x[c * plane + p]);
    let (alpha, fg) = if net == 0 {
        (gf, fg_in)
    } else {
        let d: [f32; 4] = core::array::from_fn(|k| {
            (0..width).fold(ld(bias, k), |s, c| ld(wt, k * WIDTH + c).fma_rn(ld(hid, c * plane + p), s))
        });
        let b = Taps::new((h, w), (yy, xx), (hn, wn)).get(band, 0);
        let mix = |r: f32, base: f32| b * r.clamp(0.0, 1.0) + (1.0 - b) * base;
        (
            mix(gf + d[0], gf),
            [0, 1, 2].map(|c| mix(fg_in[c] + d[1 + c], fg_in[c])),
        )
    };
    let adj = Adjust {
        on: adjust != 0,
        gain,
        contrast,
        pivot,
    };
    let at = yy * wn + xx;
    let n = hn * wn;
    logits.set(at, alpha);
    fg.into_iter().enumerate().for_each(|(c, v)| {
        let z = adj.apply(to_linear(v)) / KNEE;
        logits.set((1 + c) * n + at, (z.abs() + (z * z + 1.0).sqrt()).ln().copysign(z) / DENOM);
    });
}
