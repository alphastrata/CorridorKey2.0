use ck_arch::{bf16x2, commit, fetch16, mma16, put16, wait};
use ptx::mem::{ld_global_cs_v4_u32, ld_global_nc_v2_u32, st_global_v4_u32};
use ptx::kernel;
use ptx::mma::ldmatrix_x4_b16_u32;
use ptx::prelude::*;
use ptx::sync::shared::{shared_window, shld_v4_f32, shst_v2_f32};
use ptx::warp_ops::warp_sync;
use tune::{SMALL, SQUARE, THIN, WIDE};

use crate::{ACCUMULATE, BIAS, Shape, THREADS, Tile, place};

#[cfg_attr(feature = "sm_86", path = "tune/sm86.rs")]
#[cfg_attr(feature = "sm_89", path = "tune/sm89.rs")]
#[cfg_attr(all(feature = "sm_80", not(any(feature = "sm_86", feature = "sm_89"))), path = "tune/sm120.rs")]
#[cfg_attr(not(feature = "sm_80"), path = "tune/sm75.rs")]
mod tune;

const BK: usize = 32;
const MTS: usize = 4;
const NTS: usize = 8;
const ROW: u32 = (BK * 2) as u32;

const fn chunk(row: usize, c: usize) -> u32 {
    ((c ^ ((row >> 1) & 3)) * 16) as u32
}

type Acc = [[[f32; 4]; NTS]; MTS];

impl Tile {
    const fn wm(self) -> usize {
        THREADS / 32 / self.wn
    }

    const fn mt(self) -> usize {
        self.bm / self.wm() / 16
    }

    const fn nt(self) -> usize {
        self.bn / self.wn / 8
    }

    const fn slot(self) -> u32 {
        (self.bm + self.bn) as u32 * ROW
    }

    const fn ragged(self) -> bool {
        (self.bm + self.bn) % 64 != 0
    }

    const fn feeds(self) -> usize {
        (self.bm + self.bn).div_ceil(64)
    }

    const fn stride(self) -> u32 {
        (self.bn / self.wn * 4 + 16) as u32
    }
}
#[derive(Clone, Copy)]
struct Feed {
    src: [usize; 8],
    dst: [u32; 8],
    kc: usize,
}

impl Feed {
    #[inline(always)]
    fn new<const T: Tile>((input, weight): (usize, usize), shape: Shape, (r0, c0): (usize, usize)) -> Self {
        let tid = thread_idx_x() as usize;
        let kc = tid % 4 * 8;
        let mut f = Self {
            src: [0; 8],
            dst: [0; 8],
            kc,
        };
        ptx::unroll!(8, |it| if it < T.feeds() {
            let r = if T.ragged() { (tid / 4 + it * 64).min(T.bm + T.bn - 1) } else { tid / 4 + it * 64 };
            f.dst[it] = if !T.ragged() || tid / 4 + it * 64 < T.bm + T.bn { r as u32 * ROW + chunk(r, kc / 8) } else { u32::MAX };
            f.src[it] = if r < T.bm {
                input + 2 * ((r0 + r).min(shape.rows - 1) * shape.k + kc)
            } else {
                weight + 2 * ((c0 + r - T.bm).min(shape.n - 1) * shape.k + kc)
            };
        });
        f
    }

    #[inline(always)]
    fn used<const T: Tile>(&self, it: usize) -> bool {
        !T.ragged() || self.dst[it] != u32::MAX
    }

    #[inline(always)]
    fn fetch<const T: Tile>(&self, k: usize, kb: usize, slot: u32) -> [[u32; 4]; 8] {
        let edge = kb + BK > k;
        let mut w = [[0u32; 4]; 8];
        ptx::unroll!(8, |it| if it < T.feeds() && self.used::<T>(it) {
            w[it] = fetch16(slot + self.dst[it], self.src[it] + 2 * kb, !edge || kb + self.kc < k);
        });
        w
    }

    #[inline(always)]
    fn put<const T: Tile, const CONVERT: bool>(&self, w: &[[u32; 4]; 8], slot: u32) {
        ptx::unroll!(8, |it| if it < T.feeds() && self.used::<T>(it) {
            put16(slot + self.dst[it], w[it], CONVERT && it < T.bm / 64);
        });
    }
}

#[inline(always)]
fn pending<const T: Tile>() {
    match T.stages {
        4 => wait::<2>(),
        3 => wait::<1>(),
        _ => wait::<0>(),
    }
}

type Frag = ([[u32; 4]; MTS], [[u32; 4]; NTS / 2]);

#[inline(always)]
fn frag<const T: Tile>(slot: u32, ks: usize, (wm, wn): (usize, usize)) -> Frag {
    let lane = thread_idx_x() as usize % 32;
    let mut f = ([[0u32; 4]; MTS], [[0u32; 4]; NTS / 2]);
    ptx::unroll!(4, |mt| if mt < T.mt() {
        let r = wm * (T.bm / T.wm()) + mt * 16 + lane % 16;
        f.0[mt] = unsafe { ldmatrix_x4_b16_u32(slot + r as u32 * ROW + chunk(r, 2 * ks + lane / 16)).0 };
    });
    ptx::unroll!(4, |p| if p < T.nt() / 2 {
        let r = T.bm + wn * (T.bn / T.wn) + p * 16 + (lane / 16) * 8 + lane % 8;
        f.1[p] = unsafe { ldmatrix_x4_b16_u32(slot + r as u32 * ROW + chunk(r, 2 * ks + (lane / 8) % 2)).0 };
    });
    f
}

#[inline(always)]
fn mma<const T: Tile>(mut acc: Acc, (a, b): &Frag) -> Acc {
    ptx::unroll!(4, |mt| if mt < T.mt() {
        ptx::unroll!(8, |nt| if nt < T.nt() {
            let h = 2 * (nt % 2);
            acc[mt][nt] = mma16(a[mt], [b[nt / 2][h], b[nt / 2][h + 1]], acc[mt][nt]);
        });
    });
    acc
}

#[inline(always)]
pub fn gemm<const T: Tile, const F32: bool, const FLAGS: u32, const CONVERT: bool>((input, weight, bias, out): (u64, u64, u64, u64), s: Shape) {
    let smem = shared_window();
    let tid = thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / T.wn, warp % T.wn);
    let (r0, c0) = place::<T>(s);
    let (split, k0) = (block_idx_y() as usize, block_idx_y() as usize * s.span);
    let extent = s.span.min(s.k - k0);
    let feed = Feed::new::<T>((input as usize + 2 * k0, weight as usize + 2 * k0), s, (r0, c0));
    let out = out + (split * s.rows * s.ld * if F32 { 4 } else { 2 }) as u64;
    let tiles = extent.div_ceil(BK);
    let last = smem + (T.stages - 1) as u32 * T.slot();
    let next = |slot: u32| if slot == last { smem } else { slot + T.slot() };
    let fetch = |t: usize, slot: u32| {
        let w = feed.fetch::<T>(if t < tiles { extent } else { 0 }, t * BK, slot);
        commit();
        w
    };
    let fill = (0..T.stages - 1).fold(smem, |slot, t| {
        feed.put::<T, CONVERT>(&fetch(t, slot), slot);
        next(slot)
    });
    let warp = (wm, wn);
    pending::<T>();
    sync_threads();
    let first = frag::<T>(smem, 0, warp);
    let (acc, ..) = (0..tiles).fold(([[[0.0f32; 4]; NTS]; MTS], first, smem, fill), |(acc, f0, read, write), t| {
        let staged = fetch(t + T.stages - 1, write);
        let f1 = frag::<T>(read, 1, warp);
        let acc = mma::<T>(acc, &f0);
        feed.put::<T, CONVERT>(&staged, write);
        pending::<T>();
        sync_threads();
        let after = next(read);
        let f0 = frag::<T>(after, 0, warp);
        (mma::<T>(acc, &f1), f0, after, next(write))
    });
    wait::<0>();
    sync_threads();
    let stage = smem + (tid / 32) as u32 * 16 * T.stride();
    let (group, quad) = (lane / 4, lane % 4);
    let mut add = [[0.0f32; 2]; NTS];
    ptx::unroll!(8, |nt| if FLAGS & BIAS != 0 && nt < T.nt() {
        let col = (c0 + wn * (T.bn / T.wn) + nt * 8 + quad * 2).min(s.n - 2);
        let (lo, hi) = unsafe { ld_global_nc_v2_u32(bias + 4 * col as u64) };
        add[nt] = [f32::from_bits(lo), f32::from_bits(hi)];
    });
    ptx::unroll!(4, |mt| if mt < T.mt() {
        ptx::unroll!(8, |nt| if nt < T.nt() {
            ptx::unroll!(2, |h| {
                let at = stage + (group + 8 * h) as u32 * T.stride() + ((nt * 8 + quad * 2) * 4) as u32;
                let v = (acc[mt][nt][2 * h] + add[nt][0], acc[mt][nt][2 * h + 1] + add[nt][1]);
                unsafe { shst_v2_f32(at, v) };
            });
        });
        warp_sync();
        let top = r0 + wm * (T.bm / T.wm()) + mt * 16;
        let left = c0 + wn * (T.bn / T.wn);
        drain::<T, F32, FLAGS>(stage, out, s, (top, left));
        warp_sync();
    });
}

#[inline(always)]
fn drain<const T: Tile, const F32: bool, const FLAGS: u32>(stage: u32, out: u64, shape: Shape, (top, left): (usize, usize)) {
    let lane = thread_idx_x() as usize % 32;
    let width = T.bn / T.wn;
    let lanes = width / if F32 { 4 } else { 8 };
    ptx::unroll!(16, |i| if i < 16 * lanes / 32 {
        let (sr, sc) = ((i * 32 + lane) / lanes, (i * 32 + lane) % lanes);
        let (row, col) = (top + sr, left + sc * if F32 { 4 } else { 8 });
        if row < shape.rows && col < shape.n {
            let src = stage + sr as u32 * T.stride() + (sc * if F32 { 16 } else { 32 }) as u32;
            let at = out + ((row * shape.ld + shape.off + col) * if F32 { 4 } else { 2 }) as u64;
            let old = if FLAGS & ACCUMULATE != 0 { unsafe { ld_global_cs_v4_u32(at) } } else { (0, 0, 0, 0) };
            unsafe {
                if F32 {
                    let v = shld_v4_f32(src);
                    let prior = if FLAGS & ACCUMULATE != 0 {
                        [f32::from_bits(old.0), f32::from_bits(old.1), f32::from_bits(old.2), f32::from_bits(old.3)]
                    } else {
                        [0.0; 4]
                    };
                    st_global_v4_u32(at, (v.0 + prior[0]).to_bits(), (v.1 + prior[1]).to_bits(), (v.2 + prior[2]).to_bits(), (v.3 + prior[3]).to_bits());
                } else {
                    let (lo4, hi4) = (shld_v4_f32(src), shld_v4_f32(src + 16));
                    let v = [lo4.0, lo4.1, lo4.2, lo4.3, hi4.0, hi4.1, hi4.2, hi4.3];
                    let words = [old.0, old.1, old.2, old.3];
                    let packed: [u32; 4] = core::array::from_fn(|j| {
                        let (lo, hi) = if FLAGS & ACCUMULATE != 0 {
                            (f32::from_bits(words[j] << 16), f32::from_bits(words[j] & 0xffff_0000))
                        } else {
                            (0.0, 0.0)
                        };
                        bf16x2(v[2 * j] + lo, v[2 * j + 1] + hi)
                    });
                    st_global_v4_u32(at, packed[0], packed[1], packed[2], packed[3]);
                }
            }
        }
    });
}

entry!(gemm_wide_f32, WIDE, true, 0, true);
entry!(gemm_wide_f32_acc, WIDE, true, ACCUMULATE, true);
entry!(gemm_wide_f32_bias, WIDE, true, BIAS, true);
entry!(gemm_wide_f32_acc_bias, WIDE, true, ACCUMULATE | BIAS, true);
entry!(gemm_wide_bf16, WIDE, false, 0, true);
entry!(gemm_wide_bf16_acc, WIDE, false, ACCUMULATE, true);
entry!(gemm_wide_bf16_bias, WIDE, false, BIAS, true);
entry!(gemm_wide_bf16_acc_bias, WIDE, false, ACCUMULATE | BIAS, true);
entry!(gemm_square_f32, SQUARE, true, 0, true);
entry!(gemm_square_f32_acc, SQUARE, true, ACCUMULATE, true);
entry!(gemm_square_f32_bias, SQUARE, true, BIAS, true);
entry!(gemm_square_f32_acc_bias, SQUARE, true, ACCUMULATE | BIAS, true);
entry!(gemm_square_bf16, SQUARE, false, 0, true);
entry!(gemm_square_bf16_acc, SQUARE, false, ACCUMULATE, true);
entry!(gemm_square_bf16_bias, SQUARE, false, BIAS, true);
entry!(gemm_square_bf16_acc_bias, SQUARE, false, ACCUMULATE | BIAS, true);
entry!(gemm_small_f32, SMALL, true, 0, true);
entry!(gemm_small_f32_acc, SMALL, true, ACCUMULATE, true);
entry!(gemm_small_f32_bias, SMALL, true, BIAS, true);
entry!(gemm_small_f32_acc_bias, SMALL, true, ACCUMULATE | BIAS, true);
entry!(gemm_small_bf16, SMALL, false, 0, true);
entry!(gemm_small_bf16_acc, SMALL, false, ACCUMULATE, true);
entry!(gemm_small_bf16_bias, SMALL, false, BIAS, true);
entry!(gemm_small_bf16_acc_bias, SMALL, false, ACCUMULATE | BIAS, true);
entry!(gemm_thin_f32, THIN, true, 0, true);
entry!(gemm_thin_f32_acc, THIN, true, ACCUMULATE, true);
entry!(gemm_thin_f32_bias, THIN, true, BIAS, true);
entry!(gemm_thin_f32_acc_bias, THIN, true, ACCUMULATE | BIAS, true);
entry!(gemm_thin_bf16, THIN, false, 0, true);
entry!(gemm_thin_bf16_acc, THIN, false, ACCUMULATE, true);
entry!(gemm_thin_bf16_bias, THIN, false, BIAS, true);
entry!(gemm_thin_bf16_acc_bias, THIN, false, ACCUMULATE | BIAS, true);

stages!(STAGES_WIDE = WIDE, STAGES_SQUARE = SQUARE, STAGES_THIN = THIN, STAGES_SMALL = SMALL);
