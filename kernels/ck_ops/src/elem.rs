//! Pointwise maps on rows of `c` channels. `act_*`: `act(x)` (0 `SiLU`, 1 exact GELU, 2 tanh GELU) to f32 or bf16.
//! `to_bf16`: cast. `to_nhwc`: f32 NCHW `[n, c, hw]` into channels `off..off + c` of an `ld`-wide NHWC f32 grid.
//! `add_scaled`: `x += scale[c] · y`. `add_gated`: `x[r, off + j] += y[r, j] · g` with `g = silu(gate[r, j])`
//! (`per = 0`) or `gate[r / per, j]` (one gate row, `gld` apart, per frame of `per` pixels). `norm_mod`: AdaLN-modulated
//! RMS norm, `rms(x) · w · (1 + scale) + shift` with `scale`, `shift` the frame's rows, one warp per pixel.

use crate::{bf, gelu, gelu_tanh, gid, rn, silu};
use ptx::kernel;
use ptx::prelude::*;
use ptx::warp_ops::shfl_bfly_sync_b32;

fn act(x: f32, kind: u32) -> f32 {
    match kind {
        0 => silu(x),
        1 => gelu(x),
        _ => gelu_tanh(x),
    }
}

#[kernel]
pub fn act_f32(x: &[f32], out: &mut [f32], n: u32, kind: u32) {
    let i = gid();
    if i < n as usize {
        out.set(i, act(x[i], kind));
    }
}

#[kernel]
pub fn act_bf16(x: &[f32], out: &mut [u16], n: u32, kind: u32) {
    let i = gid();
    if i < n as usize {
        out.set(i, rn(act(x[i], kind)));
    }
}

#[kernel]
pub fn act_bf16_bf16(x: &[u16], out: &mut [u16], n: u32, kind: u32) {
    let i = gid();
    if i < n as usize {
        out.set(i, rn(act(bf(x[i]), kind)));
    }
}

#[kernel]
pub fn to_bf16(x: &[f32], out: &mut [u16], n: u32) {
    let i = gid();
    if i < n as usize {
        out.set(i, rn(x[i]));
    }
}

#[kernel]
pub fn to_nhwc(x: &[f32], out: &mut [f32], n: u32, c: u32, hw: u32, ld: u32, off: u32) {
    let i = gid();
    let [n, c, hw, ld, off] = [n, c, hw, ld, off].map(|v| v as usize);
    if i >= n * c * hw {
        return;
    }
    let (ch, p, b) = (i % c, i / c % hw, i / (c * hw));
    out.set((b * hw + p) * ld + off + ch, x[(b * c + ch) * hw + p]);
}

#[kernel]
pub fn add_scaled(x: &mut [f32], y: &[f32], scale: &[f32], n: u32, c: u32) {
    let i = gid();
    if i < n as usize {
        let v = x[i] + scale[i % c as usize] * y[i];
        x.set(i, v);
    }
}

#[kernel]
pub fn add_gated(
    x: &mut [f32],
    y: &[f32],
    gate: &[f32],
    rows: u32,
    c: u32,
    ld: u32,
    off: u32,
    per: u32,
    gld: u32,
) {
    let i = gid();
    let [rows, c, ld, off, per, gld] = [rows, c, ld, off, per, gld].map(|v| v as usize);
    if i >= rows * c {
        return;
    }
    let (r, j) = (i / c, i % c);
    let g = r
        .checked_div(per)
        .map_or_else(|| silu(gate[i]), |f| gate[f * gld + j]);
    let at = r * ld + off + j;
    let v = x[at] + y[i] * g;
    x.set(at, v);
}

#[kernel]
pub fn norm_mod(
    x: &[f32],
    weight: &[f32],
    params: &[f32],
    out: &mut [u16],
    pixels: u32,
    c: u32,
    eps: f32,
    per: u32,
    ld: u32,
    shift: u32,
    scale: u32,
) {
    let p = (block_idx_x() * 8 + thread_idx_x() / 32) as usize;
    if p >= pixels as usize {
        return;
    }
    let (c, lane) = (c as usize, thread_idx_x() as usize % 32);
    let sq = (lane..c).step_by(32).fold(0.0f32, |acc, j| {
        let v = x[p * c + j];
        v.fma_rn(v, acc)
    });
    let sq = [16, 8, 4, 2, 1].into_iter().fold(sq, |acc, m| {
        acc + f32::from_bits(shfl_bfly_sync_b32(acc.to_bits(), m, 0x1f))
    });
    let inv = (sq / c as f32 + eps).rsqrt();
    let row = p / per as usize * ld as usize;
    (lane..c).step_by(32).for_each(|j| {
        let v = x[p * c + j] * inv * weight[j];
        let v = v * (1.0 + params[row + scale as usize + j]) + params[row + shift as usize + j];
        out.set(p * c + j, rn(v));
    });
}
