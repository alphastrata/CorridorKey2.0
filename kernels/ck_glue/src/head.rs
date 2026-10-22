//! Per-head ops on bf16 rows `[rows, D]` (input rows in blocks of `inner`, `stride` pairs apart; output
//! contiguous), one thread per row, f32 arithmetic: `MODE` bit 0 RMS-norms by
//! f32 `weight [D]`; bit 1 applies interleaved `RoPE` with f32 `cos`/`sin` `[n, D / 2]` of token `row % n`.

use crate::{high, low, pack};
use ptx::kernel;
use ptx::mem::{ld_global_nc_v4_u32, st_global_v4_u32};
use ptx::prelude::*;
use ptx::slice::{GpuSlice, GpuSliceMut};

fn quad(x: GpuSlice<f32>, i: usize) -> [f32; 4] {
    let q = unsafe { ld_global_nc_v4_u32(x.addr_window::<16>(i)) };
    [q.0, q.1, q.2, q.3].map(f32::from_bits)
}

fn chunk(src: GpuSlice<u32>, at: usize) -> [f32; 8] {
    let word = unsafe { ld_global_nc_v4_u32(src.addr_window::<16>(at)) };
    let words = [word.0, word.1, word.2, word.3];
    core::array::from_fn(|j| {
        if j % 2 == 0 {
            low(words[j / 2])
        } else {
            high(words[j / 2])
        }
    })
}

/// Streams the row in chunks of 8 (a sum-of-squares pass, then normalise, rotate and store), so no
/// `D`-wide array is held.
#[inline(always)]
fn head<const D: usize>(
    src: GpuSlice<u32>,
    weight: GpuSlice<f32>,
    (cos, sin): (GpuSlice<f32>, GpuSlice<f32>),
    out: &mut GpuSliceMut<u32>,
    (rows, tokens, inner, stride): (u32, u32, u32, u32),
    eps: f32,
    mode: u32,
) {
    let row = (block_idx_x() * 256 + thread_idx_x()) as usize;
    if row >= rows as usize {
        return;
    }
    let (inner, base) = (inner as usize, row * D / 2);
    let from = row / inner * stride as usize + row % inner * D / 2;
    let inv = if mode & 1 == 0 {
        1.0
    } else {
        let sq = (0..D / 8)
            .flat_map(|i| chunk(src, from + 4 * i))
            .fold(0.0f32, |acc, e| e.fma_rn(e, acc));
        (sq / D as f32 + eps).rsqrt()
    };
    let at = row % tokens as usize * D / 2;
    (0..D / 8).for_each(|i| {
        let mut vals = chunk(src, from + 4 * i);
        if mode & 1 != 0 {
            let (lo, hi) = (quad(weight, 8 * i), quad(weight, 8 * i + 4));
            ptx::unroll!(8, |j| vals[j] *=
                inv * if j < 4 { lo[j] } else { hi[j - 4] });
        }
        if mode & 2 != 0 {
            let (cs, sn) = (quad(cos, at + 4 * i), quad(sin, at + 4 * i));
            ptx::unroll!(4, |j| {
                let (even, odd) = (vals[2 * j], vals[2 * j + 1]);
                vals[2 * j] = even * cs[j] - odd * sn[j];
                vals[2 * j + 1] = even * sn[j] + odd * cs[j];
            });
        }
        let w: [u32; 4] = core::array::from_fn(|j| pack(vals[2 * j], vals[2 * j + 1]));
        unsafe { st_global_v4_u32(out.addr_at(base + 4 * i), w[0], w[1], w[2], w[3]) };
    });
}

macro_rules! entry {
    ($name:ident, $d:literal) => {
        /// `x`, `out` bf16 pairs `[rows, D / 2]`; grid `⌈rows / 256⌉`, block 256.
        #[kernel]
        pub fn $name(
            x: &[u32],
            w: &[f32],
            cos: &[f32],
            sin: &[f32],
            out: &mut [u32],
            rows: u32,
            n: u32,
            inner: u32,
            stride: u32,
            eps: f32,
            mode: u32,
        ) {
            head::<$d>(
                x,
                w,
                (cos, sin),
                &mut out,
                (rows, n, inner, stride),
                eps,
                mode,
            );
        }
    };
}

entry!(head64, 64);
entry!(head80, 80);
entry!(head96, 96);
