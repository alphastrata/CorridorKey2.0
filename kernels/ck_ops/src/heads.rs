//! Attention heads ↔ token grids. The grid is `[b, t, gh, gw]` tokens, windowed as if zero-padded to `ph × pw`;
//! windows of `wh × ww` (all `t`) are numbered `(b, wy, wx)` and their tokens `(t, iy, ix)`: Swin windows, `CSwin`
//! strips (`gh × 1`, `1 × gw`) or the whole grid. Padding tokens split to zeros and merge nowhere.
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
    ph: usize,
    pw: usize,
    wh: usize,
    ww: usize,
}

impl Windows {
    fn tokens(self) -> usize {
        self.t * self.wh * self.ww
    }

    /// Grid token of window `win`'s token `tok`, unless padding.
    fn grid(self, win: usize, tok: usize) -> Option<usize> {
        let (nh, nw) = (self.ph / self.wh, self.pw / self.ww);
        let (b, wy, wx) = (win / (nh * nw), win / nw % nh, win % nw);
        let (tt, iy, ix) = (
            tok / (self.wh * self.ww),
            tok / self.ww % self.wh,
            tok % self.ww,
        );
        let (y, x) = (wy * self.wh + iy, wx * self.ww + ix);
        (y < self.gh && x < self.gw).then_some(((b * self.t + tt) * self.gh + y) * self.gw + x)
    }

    /// Window and token of grid token `g`.
    fn local(self, g: usize) -> (usize, usize) {
        let (nh, nw) = (self.ph / self.wh, self.pw / self.ww);
        let (x, y, tt, b) = (
            g % self.gw,
            g / self.gw % self.gh,
            g / (self.gw * self.gh) % self.t,
            g / (self.gw * self.gh * self.t),
        );
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
    t: u32,
    gh: u32,
    gw: u32,
    ph: u32,
    pw: u32,
    wh: u32,
    ww: u32,
    heads: u32,
    d: u32,
    head0: u32,
    hc: u32,
    eps: f32,
    mode: u32,
) {
    let [t, gh, gw, ph, pw, wh, ww] = [t, gh, gw, ph, pw, wh, ww].map(|v| v as usize);
    let [heads, d, head0, hc] = [heads, d, head0, hc].map(|v| v as usize);
    let win = Windows {
        t,
        gh,
        gw,
        ph,
        pw,
        wh,
        ww,
    };
    let n = win.tokens();
    let i = gid();
    if i >= windows as usize * hc * n {
        return;
    }
    let (tok, hh, w) = (i % n, i / n % hc, i / (n * hc));
    let Some(g) = win.grid(w, tok) else {
        (0..d).for_each(|j| {
            q.set(i * d + j, 0);
            k.set(i * d + j, 0);
            v.set(i * d + j, 0);
        });
        return;
    };
    let row = g * 3 * heads * d + (head0 + hh) * d;
    let norm = mode & 1 != 0;
    let rope = (mode & 2 != 0).then_some((cos, sin, tok * d / 2));
    head(qkv, row, norm.then_some(qw), rope, eps, &mut q, i * d, d);
    head(
        qkv,
        row + heads * d,
        norm.then_some(kw),
        rope,
        eps,
        &mut k,
        i * d,
        d,
    );
    head(qkv, row + 2 * heads * d, None, None, eps, &mut v, i * d, d);
}

#[kernel]
pub fn merge(
    o: &[u16],
    gate: &[f32],
    out: &mut [u16],
    tokens: u32,
    t: u32,
    gh: u32,
    gw: u32,
    ph: u32,
    pw: u32,
    wh: u32,
    ww: u32,
    heads: u32,
    d: u32,
    head0: u32,
    hc: u32,
) {
    let [t, gh, gw, ph, pw, wh, ww] = [t, gh, gw, ph, pw, wh, ww].map(|v| v as usize);
    let [heads, d, head0, hc] = [heads, d, head0, hc].map(|v| v as usize);
    let win = Windows {
        t,
        gh,
        gw,
        ph,
        pw,
        wh,
        ww,
    };
    let i = gid();
    if i >= tokens as usize * hc * d {
        return;
    }
    let (j, hh, g) = (i % d, i / d % hc, i / (d * hc));
    let (w, tok) = win.local(g);
    let val =
        bf(o[((w * hc + hh) * win.tokens() + tok) * d + j]) * sigmoid(gate[g * heads + head0 + hh]);
    out.set(g * heads * d + (head0 + hh) * d + j, rn(val));
}
