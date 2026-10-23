
use crate::{bf, gid, rn, sigmoid};
use ptx::kernel;
use ptx::prelude::*;
use ptx::slice::{GpuSlice, GpuSliceMut};
use ptx::warp_ops::shfl_bfly_sync_b32;

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

fn pairs(lane: usize, d: usize) -> impl Iterator<Item = usize> {
    [lane, lane + 32].into_iter().filter(move |&j| j < d / 2)
}

fn warp_sum(v: f32) -> f32 {
    [16, 8, 4, 2, 1].into_iter().fold(v, |acc, m| {
        acc + f32::from_bits(shfl_bfly_sync_b32(acc.to_bits(), m, 0x1f))
    })
}

fn head(
    src: GpuSlice<u16>,
    from: usize,
    norm: Option<GpuSlice<f32>>,
    rope: Option<(GpuSlice<f32>, GpuSlice<f32>, usize)>,
    (eps, d, lane): (f32, usize, usize),
    dst: &mut GpuSliceMut<u16>,
    to: usize,
) {
    let pair = |j: usize| (bf(src[from + 2 * j]), bf(src[from + 2 * j + 1]));
    let inv = norm.map_or(1.0, |_| {
        let sq = pairs(lane, d).fold(0.0f32, |acc, j| {
            let (x0, x1) = pair(j);
            x1.fma_rn(x1, x0.fma_rn(x0, acc))
        });
        (warp_sum(sq) / d as f32 + eps).rsqrt()
    });
    let scale = |i: usize| inv * norm.map_or(1.0, |w| w[i]);
    pairs(lane, d).for_each(|j| {
        let (x0, x1) = pair(j);
        let (even, odd) = (x0 * scale(2 * j), x1 * scale(2 * j + 1));
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
    let (i, lane) = (gid() / 32, thread_idx_x() as usize % 32);
    if i >= windows as usize * hc * n {
        return;
    }
    let (tok, hh, w) = (i % n, i / n % hc, i / (n * hc));
    let Some(g) = win.grid(w, tok) else {
        pairs(lane, d).flat_map(|j| [2 * j, 2 * j + 1]).for_each(|e| {
            q.set(i * d + e, 0);
            k.set(i * d + e, 0);
            v.set(i * d + e, 0);
        });
        return;
    };
    let row = g * 3 * heads * d + (head0 + hh) * d;
    let norm = mode & 1 != 0;
    let rope = (mode & 2 != 0).then_some((cos, sin, tok * d / 2));
    let geo = (eps, d, lane);
    head(qkv, row, norm.then_some(qw), rope, geo, &mut q, i * d);
    head(qkv, row + heads * d, norm.then_some(kw), rope, geo, &mut k, i * d);
    head(qkv, row + 2 * heads * d, None, None, geo, &mut v, i * d);
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
