use crate::{gid, rn, sigmoid};
use ptx::kernel;
use ptx::mem::{ld_global_nc_v4_f32, ld_global_nc_v4_u32, st_global_v4_u32};
use ptx::prelude::*;
use ptx::slice::{GpuSlice, GpuSliceMut};
use ptx::warp_ops::shfl_idx;

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

    fn pattern(self) -> usize {
        (self.ph / self.wh) * (self.pw / self.ww)
    }

    fn reorder(self, order: GpuSlice<u32>, inverse: bool, w: usize, tok: usize) -> usize {
        let n = self.tokens();
        let table = self.pattern() * n;
        if order.len() < 2 * table {
            return tok;
        }
        order[usize::from(inverse) * table + w % self.pattern() * n + tok] as usize
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

fn chunk(src: GpuSlice<u16>, at: usize) -> [f32; 8] {
    let w = unsafe { ld_global_nc_v4_u32(src.addr_at(at)) };
    let w = [w.0, w.1, w.2, w.3];
    core::array::from_fn(|j| {
        f32::from_bits(if j % 2 == 0 {
            w[j / 2] << 16
        } else {
            w[j / 2] & 0xffff_0000
        })
    })
}

fn quad(x: GpuSlice<f32>, at: usize) -> [f32; 4] {
    let q = unsafe { ld_global_nc_v4_f32(x.addr_at(at)) };
    [q.0, q.1, q.2, q.3]
}

pub(crate) fn store(dst: &mut GpuSliceMut<u16>, at: usize, v: [f32; 8]) {
    let w: [u32; 4] =
        core::array::from_fn(|j| u32::from(rn(v[2 * j])) | u32::from(rn(v[2 * j + 1])) << 16);
    unsafe { st_global_v4_u32(dst.addr_at(at), w[0], w[1], w[2], w[3]) };
}

struct Lanes {
    part: usize,
    first: u32,
    per: u32,
}

fn head(
    vals: [f32; 8],
    norm: Option<GpuSlice<f32>>,
    rope: Option<(GpuSlice<f32>, GpuSlice<f32>, usize)>,
    eps: f32,
    l: &Lanes,
) -> [f32; 8] {
    let at = 8 * l.part;
    let mut v = vals;
    if let Some(w) = norm {
        let sq = v.iter().fold(0.0f32, |acc, &e| e.fma_rn(e, acc));
        let sum = (0..l.per).fold(0.0, |acc, k| acc + shfl_idx(sq, (l.first + k) & 31));
        let inv = (sum / (8 * l.per) as f32 + eps).rsqrt();
        let (lo, hi) = (quad(w, at), quad(w, at + 4));
        ptx::unroll!(8, |j| v[j] *= inv * if j < 4 { lo[j] } else { hi[j - 4] });
    }
    if let Some((cos, sin, base)) = rope {
        let (c, s) = (quad(cos, base + at / 2), quad(sin, base + at / 2));
        ptx::unroll!(4, |j| {
            let (even, odd) = (v[2 * j], v[2 * j + 1]);
            v[2 * j] = even * c[j] - odd * s[j];
            v[2 * j + 1] = even * s[j] + odd * c[j];
        });
    }
    v
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
    order: &[u32],
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
    let per = d / 8;
    let lane = thread_idx_x() as usize % 32;
    let slot = lane / per;
    let i = (block_idx_x() as usize * 8 + thread_idx_x() as usize / 32) * (32 / per) + slot;
    let l = Lanes {
        part: lane % per,
        first: (slot * per) as u32,
        per: per as u32,
    };
    let live = slot < 32 / per && i < windows as usize * hc * n;
    let (hh, w) = (i / n % hc, i / (n * hc));
    let tok = if live {
        win.reorder(order, false, w, i % n)
    } else {
        0
    };
    let g = if live { win.grid(w, tok) } else { None };
    let row = g.map_or(0, |g| g * 3 * heads * d + (head0 + hh) * d) + 8 * l.part;
    let load = |at: usize| g.map_or([0.0; 8], |_| chunk(qkv, at));
    let norm = mode & 1 != 0;
    let rope = (mode & 2 != 0).then_some((cos, sin, tok * d / 2));
    let qv = head(load(row), norm.then_some(qw), rope, eps, &l);
    let kv = head(load(row + heads * d), norm.then_some(kw), rope, eps, &l);
    if !live {
        return;
    }
    let to = i * d + 8 * l.part;
    let zero = g.is_none();
    store(&mut q, to, if zero { [0.0; 8] } else { qv });
    store(&mut k, to, if zero { [0.0; 8] } else { kv });
    store(&mut v, to, load(row + 2 * heads * d));
}

#[kernel]
pub fn merge(
    o: &[u16],
    gate: &[f32],
    out: &mut [u16],
    order: &[u32],
    tokens: u32,
    ld: u32,
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
    let per = d / 8;
    if i >= tokens as usize * hc * per {
        return;
    }
    let (j, hh, g) = (8 * (i % per), i / per % hc, i / (per * hc));
    let (w, tok) = win.local(g);
    let tok = win.reorder(order, true, w, tok);
    let s = sigmoid(gate[g * heads + head0 + hh]);
    let vals = chunk(o, ((w * hc + hh) * win.tokens() + tok) * d + j).map(|x| x * s);
    let row = g * ld as usize;
    store(&mut out, row + (head0 + hh) * d + j, vals);
    if head0 + hh + j == 0 && ld as usize > heads * d {
        store(
            &mut out,
            row + heads * d,
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
    }
}
