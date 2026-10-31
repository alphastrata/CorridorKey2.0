use crate::{bf, gelu, gelu_tanh, gid, rn, silu};
use ck_arch::convert::{Conv, F16Bits, Host};
use ptx::kernel;
use ptx::mem::{ld_global_nc_v4_f32, st_global_v4_f32};
use ptx::num::f16::cvt_rn_f16_f32;
use ptx::prelude::*;
use ptx::slice::GpuSliceMut;
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
pub fn act_bf16(x: &[f32], out: &mut [u16], n: u32, kind: u32, c: u32, ld: u32) {
    let i = gid();
    if i < n.usize() {
        let (r, j, c) = (i / c.usize(), i % c.usize(), c.usize());
        out.set(r * ld.usize() + j, rn(act(x[i], kind)));
        if j == 0 && ld as usize > c {
            tail(&mut out, r * ld as usize + c);
        }
    }
}

pub(crate) fn tail(out: &mut GpuSliceMut<u16>, at: usize) {
    (0..8).for_each(|k| out.set(at + k, if k < 2 { 0x3f80 } else { 0 }));
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
#[allow(clippy::cast_possible_truncation)]
pub fn bf16_f16(x: &[u16], out: &mut [u16], n: u32) {
    let i = gid();
    if i < n as usize {
        out.set(i, cvt_rn_f16_f32(bf(x[i]).clamp(-65504.0, 65504.0)).u16());
    }
}

#[kernel]
pub fn f32_bf16(x: &[f32], out: &mut [u16], rows: u32, n: u32, ld: u32, off: u32) {
    let i = gid();
    let [rows, n, ld, off] = [rows, n, ld, off].map(|v| v as usize);
    if i < rows * n {
        out.set(i / n * ld + off + i % n, rn(x[i]));
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
pub fn add_bias(y: &mut [f32], b: &[f32], rows: u32, n: u32, ld: u32, off: u32) {
    let (i, quads) = (gid(), n as usize / 4);
    if i >= rows as usize * quads {
        return;
    }
    let (r, j) = (i / quads, 4 * (i % quads));
    let at = r * ld as usize + off as usize + j;

    let (v, w) = unsafe {
        (
            ld_global_nc_v4_f32(y.addr_at(at)),
            ld_global_nc_v4_f32(b.addr_at(j)),
        )
    };
    unsafe { st_global_v4_f32(y.addr_at(at), v.0 + w.0, v.1 + w.1, v.2 + w.2, v.3 + w.3) };
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
#[allow(clippy::cast_precision_loss)]
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

    let inv = (sq / c.f32() + eps).rsqrt();
    let row = p / per as usize * ld as usize;
    (lane..c).step_by(32).for_each(|j| {
        let v = x[p * c + j] * inv * weight[j];
        let v = v * (1.0 + params[row + scale as usize + j]) + params[row + shift as usize + j];
        out.set(p * c + j, rn(v));
    });
}
