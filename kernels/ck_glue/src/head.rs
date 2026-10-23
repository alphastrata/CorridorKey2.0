use crate::{high, low, pack};
use ptx::kernel;
use ptx::mem::{ld_global_nc_v4_u32, st_global_v4_u32};
use ptx::prelude::*;
use ptx::slice::{GpuSlice, GpuSliceMut};
use ptx::warp_ops::shfl_idx;

fn quad(x: GpuSlice<f32>, i: usize) -> [f32; 4] {
    let q = unsafe { ld_global_nc_v4_u32(x.addr_window::<16>(i)) };
    [q.0, q.1, q.2, q.3].map(f32::from_bits)
}

pub(crate) fn chunk(src: GpuSlice<u32>, at: usize) -> [f32; 8] {
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

pub(crate) struct Head<'a> {
    pub weight: GpuSlice<'a, f32>,
    pub cos: GpuSlice<'a, f32>,
    pub sin: GpuSlice<'a, f32>,
    pub eps: f32,
    pub mode: u32,
}

pub(crate) struct Lane {
    pub row: usize,
    pub part: usize,
    pub first: u32,
    pub live: bool,
}

pub(crate) fn lane<const D: usize>() -> Lane {
    let per = D / 8;
    let lanes = thread_idx_x() as usize % 32;
    let slot = lanes / per;
    let row = (block_idx_x() as usize * 8 + thread_idx_x() as usize / 32) * (32 / per) + slot;
    Lane {
        row,
        part: lanes % per,
        first: (slot * per) as u32,
        live: slot < 32 / per,
    }
}

pub(crate) fn group_sum<const D: usize>(v: f32, first: u32) -> f32 {
    (0..D as u32 / 8).fold(0.0, |acc, k| acc + shfl_idx(v, (first + k) & 31))
}

#[inline(always)]
pub(crate) fn emit<const D: usize>(
    src: GpuSlice<u32>,
    from: usize,
    p: &Head,
    token: usize,
    l: &Lane,
    out: &mut GpuSliceMut<u32>,
    base: usize,
    live: bool,
) {
    let mut vals = if live {
        chunk(src, from + 4 * l.part)
    } else {
        [0.0; 8]
    };
    if p.mode & 1 != 0 {
        let sq = vals.iter().fold(0.0f32, |acc, &e| e.fma_rn(e, acc));
        let inv = (group_sum::<D>(sq, l.first) / D as f32 + p.eps).rsqrt();
        if !live {
            return;
        }
        let i = 8 * l.part;
        let (lo, hi) = (quad(p.weight, i), quad(p.weight, i + 4));
        ptx::unroll!(8, |j| vals[j] *= inv * if j < 4 { lo[j] } else { hi[j - 4] });
    }
    if !live {
        return;
    }
    if p.mode & 2 != 0 {
        let at = token * D / 2 + 4 * l.part;
        let (cs, sn) = (quad(p.cos, at), quad(p.sin, at));
        ptx::unroll!(4, |j| {
            let (even, odd) = (vals[2 * j], vals[2 * j + 1]);
            vals[2 * j] = even * cs[j] - odd * sn[j];
            vals[2 * j + 1] = even * sn[j] + odd * cs[j];
        });
    }
    let w: [u32; 4] = core::array::from_fn(|j| pack(vals[2 * j], vals[2 * j + 1]));
    unsafe { st_global_v4_u32(out.addr_at(base + 4 * l.part), w[0], w[1], w[2], w[3]) };
}

macro_rules! entry {
    ($name:ident, $d:literal) => {
        #[kernel]
        pub fn $name(
            x: &[u32],
            weight: &[f32],
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
            let l = lane::<$d>();
            let live = l.live && l.row < rows as usize;
            let row = if live { l.row } else { 0 };
            let inner = inner as usize;
            let from = row / inner * stride as usize + row % inner * $d / 2;
            let p = Head {
                weight,
                cos,
                sin,
                eps,
                mode,
            };
            emit::<$d>(x, from, &p, row % n as usize, &l, &mut out, row * $d / 2, live);
        }
    };
}

entry!(head64, 64);
entry!(head80, 80);
entry!(head96, 96);
