
use crate::{gid, rn, sigmoid};
use ptx::kernel;
use ptx::num::f16::{cvt_f32_f16, cvt_rn_f16_f32};
use ptx::prelude::*;
use ptx::slice::GpuSlice;
use ptx::warp_ops::shfl_bfly_sync_b32;

fn warp_sum(v: f32) -> f32 {
    [16, 8, 4, 2, 1].into_iter().fold(v, |acc, m| {
        acc + f32::from_bits(shfl_bfly_sync_b32(acc.to_bits(), m, 0x1f))
    })
}

fn row(rows: u32) -> Option<(usize, usize)> {
    let r = (block_idx_x() * 8 + thread_idx_x() / 32) as usize;
    (r < rows as usize).then_some((r, thread_idx_x() as usize % 32))
}

fn moments(x: GpuSlice<f32>, base: usize, c: usize, lane: usize, eps: f32) -> (f32, f32) {
    let mean = warp_sum((lane..c).step_by(32).fold(0.0, |a, j| a + x[base + j])) / c as f32;
    let var = warp_sum((lane..c).step_by(32).fold(0.0f32, |a, j| {
        let d = x[base + j] - mean;
        d.fma_rn(d, a)
    })) / c as f32;
    (mean, (var + eps).rsqrt())
}

#[kernel]
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [u16], rows: u32, c: u32, eps: f32) {
    let Some((r, lane)) = row(rows) else { return };
    let (c, base) = (c as usize, r * c as usize);
    let (mean, inv) = moments(x, base, c, lane, eps);
    (lane..c)
        .step_by(32)
        .for_each(|j| out.set(base + j, rn((x[base + j] - mean) * inv * w[j] + b[j])));
}

#[kernel]
pub fn layer_norm_f32(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    out: &mut [f32],
    rows: u32,
    c: u32,
    eps: f32,
) {
    let Some((r, lane)) = row(rows) else { return };
    let (c, base) = (c as usize, r * c as usize);
    let (mean, inv) = moments(x, base, c, lane, eps);
    (lane..c)
        .step_by(32)
        .for_each(|j| out.set(base + j, (x[base + j] - mean) * inv * w[j] + b[j]));
}

#[kernel]
pub fn heads(
    qkv: &[u16],
    q: &mut [u16],
    k: &mut [u16],
    v: &mut [u16],
    b: u32,
    n: u32,
    heads: u32,
    d: u32,
    dp: u32,
) {
    let i = gid();
    let [b, n, heads, d, dp] = [b, n, heads, d, dp].map(|v| v as usize);
    let per = b * heads * n * dp;
    if i >= 3 * per {
        return;
    }
    let (which, o) = (i / per, i % per);
    let (j, tok, h, bb) = (
        o % dp,
        o / dp % n,
        o / (dp * n) % heads,
        o / (dp * n * heads),
    );
    let val = if j < d {
        qkv[(((bb * n + tok) * 3 + which) * heads + h) * d + j]
    } else {
        0
    };
    match which {
        0 => q.set(o, val),
        1 => k.set(o, val),
        _ => v.set(o, val),
    }
}

#[kernel]
pub fn unheads(o: &[u16], out: &mut [u16], b: u32, n: u32, heads: u32, d: u32, dp: u32) {
    let i = gid();
    let [b, n, heads, d, dp] = [b, n, heads, d, dp].map(|v| v as usize);
    if i >= b * n * heads * d {
        return;
    }
    let (j, h, tok, bb) = (
        i % d,
        i / d % heads,
        i / (d * heads) % n,
        i / (d * heads * n),
    );
    out.set(i, o[((bb * heads + h) * n + tok) * dp + j]);
}

#[kernel]
pub fn embed(
    patches: &[f32],
    prefix: &[f32],
    pos: &[f32],
    out: &mut [f32],
    b: u32,
    p: u32,
    hw: u32,
    c: u32,
) {
    let i = gid();
    let [b, p, hw, c] = [b, p, hw, c].map(|v| v as usize);
    let n = p + hw;
    if i >= b * n * c {
        return;
    }
    let (j, tok, bb) = (i % c, i / c % n, i / (c * n));
    let base = if tok < p {
        prefix[tok * c + j]
    } else {
        patches[(bb * hw + tok - p) * c + j]
    };
    out.set(i, base + pos[tok * c + j]);
}

#[kernel]
pub fn standardise(
    x: &[f32],
    mean: &[f32],
    std: &[f32],
    out: &mut [f32],
    b: u32,
    skip: u32,
    hw: u32,
    c: u32,
    half: u32,
) {
    let i = gid();
    let [b, skip, hw, c] = [b, skip, hw, c].map(|v| v as usize);
    if i >= b * c * hw {
        return;
    }
    let (px, ch, bb) = (i % hw, i / hw % c, i / (hw * c));
    let v = (x[(bb * (skip + hw) + skip + px) * c + ch] - mean[ch]) / std[ch].max(1e-6);
    out.set(
        i,
        if half == 0 {
            v
        } else {
            cvt_f32_f16(cvt_rn_f16_f32(v))
        },
    );
}

#[kernel]
pub fn gru_kv(gates: &[f32], s: &[f32], w: &[f32], out: &mut [u16], rows: u32, c: u32, eps: f32) {
    let Some((r, lane)) = row(rows) else { return };
    let (c, base) = (c as usize, r * c as usize);
    let (mean, inv) = moments(s, base, c, lane, eps);
    (lane..c).step_by(32).for_each(|j| {
        let reset = sigmoid(gates[2 * base + c + j]);
        out.set(base + j, rn(reset * (s[base + j] - mean) * inv * w[j]));
    });
}

#[kernel]
pub fn gru_mix(gates: &[f32], h: &[f32], s: &[f32], out: &mut [f32], rows: u32, c: u32) {
    let i = gid();
    let c = c as usize;
    if i >= rows as usize * c {
        return;
    }
    let u = sigmoid(gates[i / c * 2 * c + i % c]);
    out.set(i, (1.0 - u) * s[i] + u * h[i]);
}
