//! Attention heads ↔ token grids. The grid is `[b, t, gh, gw]` tokens; windows of `wh × ww` (all `t`) are numbered
//! `(b, wy, wx)` and their tokens `(t, iy, ix)`: Swin windows, `CSwin` strips (`gh × 1`, `1 × gw`) or the whole grid.
//! `split`: bf16 `qkv [tokens, 3, heads, D]` → bf16 `q`, `k`, `v [windows, hc, N, D]` for heads `head0..head0 + hc`,
//! `mode` bit 0 RMS-norming q and k by `qw`/`kw [D]`, bit 1 rotating them by f32 `cos`/`sin [N, D / 2]`; one thread
//! per (window, head, token). `merge`: `out[token, head · D + d] = o · sigmoid(gate[token, head])`, bf16.

use crate::{bf, gid, rn, sigmoid};
use ptx::kernel;
use ptx::prelude::*;
use ptx::slice::{GpuSlice, GpuSliceMut};

#[derive(Clone, Copy)]
struct Windows {
    t: usize,
    gh: usize,
    gw: usize,
    wh: usize,
    ww: usize,
}

impl Windows {
    fn tokens(self) -> usize {
        self.t * self.wh * self.ww
    }

    /// Grid token of window `win`'s token `tok`.
    fn grid(self, win: usize, tok: usize) -> usize {
        let (nh, nw) = (self.gh / self.wh, self.gw / self.ww);
        let (b, wy, wx) = (win / (nh * nw), win / nw % nh, win % nw);
        let (tt, iy, ix) = (tok / (self.wh * self.ww), tok / self.ww % self.wh, tok % self.ww);
        ((b * self.t + tt) * self.gh + wy * self.wh + iy) * self.gw + wx * self.ww + ix
    }

    /// Window and token of grid token `g`.
    fn local(self, g: usize) -> (usize, usize) {
        let (nh, nw) = (self.gh / self.wh, self.gw / self.ww);
        let (x, y, tt, b) = (g % self.gw, g / self.gw % self.gh, g / (self.gw * self.gh) % self.t, g / (self.gw * self.gh * self.t));
        let win = (b * nh + y / self.wh) * nw + x / self.ww;
        (win, (tt * self.wh + y % self.wh) * self.ww + x % self.ww)
    }
}

fn head(
    src: GpuSlice<u16>,
    from: usize,
    norm: Option<GpuSlice<f32>>,
    rope: Option<(GpuSlice<f32>, GpuSlice<f32>, usize)>,
    eps: f32,
    dst: &mut GpuSliceMut<u16>,
    to: usize,
    d: usize,
) {
    let inv = norm.map_or(1.0, |_| {
        let sq = (0..d).fold(0.0f32, |acc, j| {
            let v = bf(src[from + j]);
            v.fma_rn(v, acc)
        });
        (sq / d as f32 + eps).rsqrt()
    });
    let val = |j: usize| bf(src[from + j]) * inv * norm.map_or(1.0, |w| w[j]);
    (0..d / 2).for_each(|j| {
        let (even, odd) = (val(2 * j), val(2 * j + 1));
        let (a, b) = rope.map_or((even, odd), |(cos, sin, at)| {
            let (c, s) = (cos[at + j], sin[at + j]);
            (even * c - odd * s, even * s + odd * c)
        });
        dst.set(to + 2 * j, rn(a));
        dst.set(to + 2 * j + 1, rn(b));
    });
}

#[kernel]
pub fn split(
    qkv: &[u16],
    qw: &[f32],
    kw: &[f32],
    cos: &[f32],
    sin: &[f32],
    q: &mut [u16],
    k: &mut [u16],
    v: &mut [u16],
    windows: u32,
    geo: &[u32],
    eps: f32,
    mode: u32,
) {
    let [t, gh, gw, wh, ww, heads, d, head0, hc] = core::array::from_fn(|i| geo[i] as usize);
    let win = Windows { t, gh, gw, wh, ww };
    let n = win.tokens();
    let i = gid();
    if i >= windows as usize * hc * n {
        return;
    }
    let (tok, hh, w) = (i % n, i / n % hc, i / (n * hc));
    let row = win.grid(w, tok) * 3 * heads * d + (head0 + hh) * d;
    let norm = mode & 1 != 0;
    let rope = (mode & 2 != 0).then_some((cos, sin, tok * d / 2));
    head(qkv, row, norm.then_some(qw), rope, eps, &mut q, i * d, d);
    head(qkv, row + heads * d, norm.then_some(kw), rope, eps, &mut k, i * d, d);
    head(qkv, row + 2 * heads * d, None, None, eps, &mut v, i * d, d);
}

#[kernel]
pub fn merge(o: &[u16], gate: &[f32], out: &mut [u16], tokens: u32, geo: &[u32]) {
    let [t, gh, gw, wh, ww, heads, d, head0, hc] = core::array::from_fn(|i| geo[i] as usize);
    let win = Windows { t, gh, gw, wh, ww };
    let i = gid();
    if i >= tokens as usize * hc * d {
        return;
    }
    let (j, hh, g) = (i % d, i / d % hc, i / (d * hc));
    let (w, tok) = win.local(g);
    let val = bf(o[((w * hc + hh) * win.tokens() + tok) * d + j]) * sigmoid(gate[g * heads + head0 + hh]);
    out.set(g * heads * d + (head0 + hh) * d + j, rn(val));
}
