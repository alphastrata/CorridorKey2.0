use ck_arch::bf16x2;
use ptx::mem::ld_global_nc_v4_u32;
use ptx::num::f32::ptx_exp2_approx;
use ptx::prelude::*;
use ptx::slice::{GpuSlice, GpuSliceMut};
use ptx::sync::shared::{shared_window, shld_v4_f32, shst_v4_f32};
use ptx::warp_ops::shfl_bfly_sync_b32;

use crate::{BC, BR, Shape, span};

const HALF: usize = 48;
const KEYS: usize = 32;

fn widen(w: u32) -> (f32, f32) {
    (f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000))
}

#[inline(always)]
fn load<const D: usize>((keys, values): (u64, u64), n: usize, kb: usize, smem: u32) {
    let tid = thread_idx_x() as usize;
    ptx::unroll!(6, |it| if it < 2 * BC * D / 8 / 256 {
        let i = tid + it * 256;
        let (which, rem) = (i / (BC * D / 8), i % (BC * D / 8));
        let (key, ch) = (rem / (D / 8), rem % (D / 8));
        let token = (kb * BC + key).min(n - 1);
        let w = unsafe { ld_global_nc_v4_u32(if which == 0 { keys } else { values } + 2 * (token * D + ch * 8) as u64) };
        let f = [w.0, w.1, w.2, w.3].map(widen);
        let at = smem + ((which * BC + key) * D + ch * 8) as u32 * 4;
        unsafe {
            shst_v4_f32(at, (f[0].0, f[0].1, f[1].0, f[1].1));
            shst_v4_f32(at + 16, (f[2].0, f[2].1, f[3].0, f[3].1));
        }
    });
}

#[inline(always)]
fn dot<const D: usize>(q: &[f32; HALF], at: u32) -> f32 {
    let mut s = [0.0f32; 4];
    ptx::unroll!(12, |c| if c < D / 8 {
        let k = unsafe { shld_v4_f32(at + 16 * c as u32) };
        s = [q[4 * c].fma_rn(k.0, s[0]), q[4 * c + 1].fma_rn(k.1, s[1]), q[4 * c + 2].fma_rn(k.2, s[2]), q[4 * c + 3].fma_rn(k.3, s[3])];
    });
    let s = (s[0] + s[1]) + (s[2] + s[3]);
    s + f32::from_bits(shfl_bfly_sync_b32(s.to_bits(), 1, 0x1f))
}

#[inline(always)]
pub fn attn<const D: usize, const MASK: bool>(
    (query, key, value): (GpuSlice<u32>, GpuSlice<u32>, GpuSlice<u32>),
    (rid, spans, mul): (GpuSlice<u32>, GpuSlice<u32>, GpuSlice<f32>),
    mut out: GpuSliceMut<u32>,
    shape: Shape,
    scale: f32,
) {
    let scale = scale / (mul[0] * mul[1]);
    let smem = shared_window();
    let tid = thread_idx_x() as usize;
    let (qx, bh) = (block_idx_x() as usize, block_idx_y() as usize);
    let (row, half) = (qx * BR + tid / 2, tid % 2);
    let window = if MASK { bh / shape.heads % shape.windows } else { 0 };
    let base = bh * shape.n * D;
    let kv = (key.addr_at(base / 2), value.addr_at(base / 2));
    let tiles = shape.n.div_ceil(BC);
    let own = if MASK {
        let first = span(spans, tiles, window, 2 * qx);
        let second = span(spans, tiles, window, (2 * qx + 1).min(tiles - 1));
        (first.0.min(second.0), first.1.max(second.1))
    } else {
        (0, 0)
    };
    let seen = |tile: usize| {
        let (lo, hi) = span(spans, tiles, window, tile);
        own.1 >= 0 && hi >= own.0 && lo <= own.1
    };
    let qrid = if MASK { rid[window * shape.n + row.min(shape.n - 1)] } else { 0 };
    let mut qv = [0.0f32; HALF];
    ptx::unroll!(24, |p| if p < D / 4 {
        let (lo, hi) = widen(query[(base + row.min(shape.n - 1) * D + half * D / 2) / 2 + p]);
        (qv[2 * p], qv[2 * p + 1]) = (lo, hi);
    });
    let dims = (half * D / 2) as u32 * 4;
    let init: ([f32; HALF], f32, f32) = ([0.0; HALF], f32::NEG_INFINITY, 0.0);
    let (ov, _, norm) = (0..tiles).fold(init, #[inline(always)] |acc, tile| {
        if MASK && !seen(tile) {
            return acc;
        }
        load::<D>(kv, shape.n, tile, smem);
        sync_threads();
        let valid = (shape.n - tile * BC).min(BC);
        let masked = MASK && {
            let (lo, hi) = span(spans, tiles, window, tile);
            !(own.0 == own.1 && lo == hi && lo == own.0)
        };
        let acc = (0..BC / KEYS).fold(acc, #[inline(always)] |(mut ov, peak, norm), part| {
            let mut probs = [0.0f32; KEYS];
            ptx::unroll!(32, |j| {
                let key = part * KEYS + j;
                let hidden = key >= valid || (MASK && masked && rid[window * shape.n + (tile * BC + key).min(shape.n - 1)] != qrid);
                let v = dot::<D>(&qv, smem + (key * D) as u32 * 4 + dims);
                probs[j] = if hidden { f32::NEG_INFINITY } else { v };
            });
            let mut top = peak;
            ptx::unroll!(32, |j| top = top.max(probs[j]));
            let shift = if top == f32::NEG_INFINITY { 0.0 } else { -top * scale };
            let corr = ptx_exp2_approx(peak.fma_rn(scale, shift));
            ptx::unroll!(32, |j| probs[j] = ptx_exp2_approx(probs[j].fma_rn(scale, shift)));
            let mut sum = 0.0f32;
            ptx::unroll!(32, |j| sum += probs[j]);
            let norm = norm.fma_rn(corr, sum);
            ptx::unroll!(12, |e| if e < D / 8 { ptx::unroll!(4, |i| ov[4 * e + i] *= corr) });
            ptx::unroll!(32, |j| {
                let at = smem + ((BC + part * KEYS + j) * D) as u32 * 4 + dims;
                ptx::unroll!(12, |e| if e < D / 8 {
                    let v = unsafe { shld_v4_f32(at + 16 * e as u32) };
                    (ov[4 * e], ov[4 * e + 1], ov[4 * e + 2], ov[4 * e + 3]) =
                        (probs[j].fma_rn(v.0, ov[4 * e]), probs[j].fma_rn(v.1, ov[4 * e + 1]), probs[j].fma_rn(v.2, ov[4 * e + 2]), probs[j].fma_rn(v.3, ov[4 * e + 3]));
                });
            });
            (ov, top, norm)
        });
        sync_threads();
        acc
    });
    let outside = MASK && (qrid as i32) < 0;
    let vmul: f32 = mul[2];
    let inv = if outside { 0.0 } else { 1.0 / (norm * vmul) };
    if row < shape.n {
        let at = (base + row * D + half * D / 2) / 2;
        ptx::unroll!(24, |p| if p < D / 4 {
            out.set(at + p, bf16x2(ov[2 * p] * inv, ov[2 * p + 1] * inv));
        });
    }
}
