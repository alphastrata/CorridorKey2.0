#![no_std]
#![feature(abi_ptx, adt_const_params)]

use ck_arch::{bf16x2, commit, fetch16, ldsm4, mma16, put16, wait, widen};
use ptx::kernel;
use ptx::prelude::*;
use ptx::mem::{
    ld_global_cs_u32, ld_global_cs_u64, ld_global_cs_v4_u32, ld_global_nc_u16, ld_global_nc_u32, ld_global_nc_v2_u32, ld_global_nc_v4_f32,
    st_global_u32, st_global_v2_u32, st_global_v4_u32,
};
use ptx::warp_ops::warp_sync;
use ptx::sync::shared::{shared_window, shld_v4_f32, shst_f32, shst_v2_f32};
use tune::{SMALL, SQUARE, THIN, WIDE};

#[cfg_attr(feature = "sm_120", path = "tune/sm120.rs")]
#[cfg_attr(not(feature = "sm_120"), path = "tune/sm75.rs")]
mod tune;

const BK: usize = 32;
const THREADS: usize = 256;
const MTS: usize = 4;
const NTS: usize = 8;
const ROW: u32 = (BK * 2) as u32;

const fn chunk(row: usize, c: usize) -> u32 {
    ((c ^ ((row >> 1) & 3)) * 16) as u32
}

type Acc = [[[f32; 4]; NTS]; MTS];

#[derive(Clone, Copy, PartialEq, Eq, core::marker::ConstParamTy)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub wn: usize,
    pub stages: usize,
}

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
const ACCUMULATE: u32 = 1;
const BIAS: u32 = 2;

#[derive(Clone, Copy)]
struct Shape {
    rows: usize,
    n: usize,
    k: usize,
    ld: usize,
    off: usize,
    span: usize,
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

const GROUP: usize = 8;

#[inline(always)]
fn pending<const T: Tile>() {
    match T.stages {
        4 => wait::<2>(),
        3 => wait::<1>(),
        _ => wait::<0>(),
    }
}

#[inline(always)]
fn place<const T: Tile>(s: Shape) -> (usize, usize) {
    let (rows, cols) = (s.rows.div_ceil(T.bm), s.n.div_ceil(T.bn));
    let id = block_idx_x() as usize;
    let (group, at) = (id / (GROUP * cols), id % (GROUP * cols));
    let height = (rows - group * GROUP).min(GROUP);
    ((group * GROUP + at % height) * T.bm, at / height * T.bn)
}

type Frag = ([[u32; 4]; MTS], [[u32; 4]; NTS / 2]);

#[inline(always)]
fn frag<const T: Tile>(slot: u32, ks: usize, (wm, wn): (usize, usize)) -> Frag {
    let lane = thread_idx_x() as usize % 32;
    let mut f = ([[0u32; 4]; MTS], [[0u32; 4]; NTS / 2]);
    ptx::unroll!(4, |mt| if mt < T.mt() {
        let r = wm * (T.bm / T.wm()) + mt * 16 + lane % 16;
        f.0[mt] = ldsm4(slot + r as u32 * ROW + chunk(r, 2 * ks + lane / 16));
    });
    ptx::unroll!(4, |p| if p < T.nt() / 2 {
        let r = T.bm + wn * (T.bn / T.wn) + p * 16 + (lane / 16) * 8 + lane % 8;
        f.1[p] = ldsm4(slot + r as u32 * ROW + chunk(r, 2 * ks + (lane / 8) % 2));
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
fn gemm<const T: Tile, const F32: bool, const FLAGS: u32, const CONVERT: bool>((input, weight, bias, out): (u64, u64, u64, u64), s: Shape) {
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

const AT: usize = 64;
const AK: usize = 16;
const AROW: u32 = ((AT + 4) * 4) as u32;

#[inline(always)]
fn element<const F32: bool, const HALF_IS_BF16: bool>(base: u64, at: usize) -> f32 {
    unsafe {
        if F32 {
            f32::from_bits(ld_global_nc_u32(base + 4 * at as u64))
        } else if HALF_IS_BF16 {
            f32::from_bits(u32::from(ld_global_nc_u16(base + 2 * at as u64)) << 16)
        } else {
            widen(ld_global_nc_u16(base + 2 * at as u64))
        }
    }
}

#[inline(always)]
fn any<const IN32: bool, const FLAGS: u32>((input, weight, bias, out): (u64, u64, u64, u64), shape: Shape) {
    let smem = shared_window();
    let tid = thread_idx_x() as usize;
    let (r0, c0) = (block_idx_y() as usize * AT, block_idx_x() as usize * AT);
    let (tr, tc) = (tid / 16 * 4, tid % 16 * 4);
    let (xs, ws) = (smem, smem + AK as u32 * AROW);
    let acc = (0..shape.k.div_ceil(AK)).fold([[0.0f32; 4]; 4], |mut acc, slab| {
        ptx::unroll!(4, |e| {
            let i = tid + e * THREADS;
            let (r, kk) = (i / AK, slab * AK + i % AK);
            let xv = if r0 + r < shape.rows && kk < shape.k { element::<IN32, true>(input, (r0 + r) * shape.k + kk) } else { 0.0 };
            let wv = if c0 + r < shape.n && kk < shape.k { element::<IN32, false>(weight, (c0 + r) * shape.k + kk) } else { 0.0 };
            unsafe {
                shst_f32(xs + (i % AK) as u32 * AROW + 4 * r as u32, xv);
                shst_f32(ws + (i % AK) as u32 * AROW + 4 * r as u32, wv);
            }
        });
        sync_threads();
        ptx::unroll!(16, |kk| {
            let (av, bv) = unsafe {
                (shld_v4_f32(xs + kk as u32 * AROW + 4 * tr as u32), shld_v4_f32(ws + kk as u32 * AROW + 4 * tc as u32))
            };
            let (av, bv) = ([av.0, av.1, av.2, av.3], [bv.0, bv.1, bv.2, bv.3]);
            ptx::unroll!(4, |i| ptx::unroll!(4, |j| acc[i][j] = av[i].fma_rn(bv[j], acc[i][j])));
        });
        sync_threads();
        acc
    });
    ptx::unroll!(4, |i| ptx::unroll!(4, |j| {
        let (row, col) = (r0 + tr + i, c0 + tc + j);
        if row < shape.rows && col < shape.n {
            let mut val = acc[i][j];
            if FLAGS & BIAS != 0 {
                val += unsafe { f32::from_bits(ld_global_nc_u32(bias + 4 * col as u64)) };
            }
            let at = out + 4 * (row * shape.ld + shape.off + col) as u64;
            unsafe {
                if FLAGS & ACCUMULATE != 0 {
                    val += f32::from_bits(ld_global_cs_u32(at));
                }
                st_global_u32(at, val.to_bits());
            }
        }
    }));
}

#[inline(always)]
fn reduce<const F32: bool, const FLAGS: u32>((parts, bias, out): (u64, u64, u64), s: Shape, splits: usize) {
    let at = (block_idx_x() as usize * THREADS + thread_idx_x() as usize) * 4;
    if at >= s.rows * s.n {
        return;
    }
    let mut v = (0..splits).fold([0.0f32; 4], |v, p| {
        let w = unsafe { ld_global_cs_v4_u32(parts + 4 * (p * s.rows * s.n + at) as u64) };
        [v[0] + f32::from_bits(w.0), v[1] + f32::from_bits(w.1), v[2] + f32::from_bits(w.2), v[3] + f32::from_bits(w.3)]
    });
    let (row, col) = (at / s.n, at % s.n);
    if FLAGS & BIAS != 0 {
        let b = unsafe { ld_global_nc_v4_f32(bias + 4 * col as u64) };
        v = [v[0] + b.0, v[1] + b.1, v[2] + b.2, v[3] + b.3];
    }
    let dst = row * s.ld + s.off + col;
    unsafe {
        if F32 {
            let dst = out + 4 * dst as u64;
            if FLAGS & ACCUMULATE != 0 {
                let o = ld_global_cs_v4_u32(dst);
                v = [v[0] + f32::from_bits(o.0), v[1] + f32::from_bits(o.1), v[2] + f32::from_bits(o.2), v[3] + f32::from_bits(o.3)];
            }
            st_global_v4_u32(dst, v[0].to_bits(), v[1].to_bits(), v[2].to_bits(), v[3].to_bits());
        } else {
            let dst = out + 2 * dst as u64;
            if FLAGS & ACCUMULATE != 0 {
                let o = ld_global_cs_u64(dst);
                let half = |w: u64| f32::from_bits((w as u32) << 16);
                v = [v[0] + half(o), v[1] + half(o >> 16), v[2] + half(o >> 32), v[3] + half(o >> 48)];
            }
            st_global_v2_u32(dst, bf16x2(v[0], v[1]), bf16x2(v[2], v[3]));
        }
    }
}

macro_rules! entry {
    ($name:ident, any, $in32:literal, $flags:expr) => {
        entry!(@ $name, |at, s| any::<$in32, { $flags }>(at, s), $flags);
    };
    ($name:ident, reduce, $f32:literal, $flags:expr) => {
        entry!(@ $name, |(_, w, b, y), s: Shape| reduce::<$f32, { $flags }>((w, b, y), s, s.span), $flags);
    };
    ($name:ident, $tile:expr, $f32:literal, $flags:expr, $convert:literal) => {
        entry!(@ $name, |at, s| gemm::<{ $tile }, $f32, { $flags }, $convert>(at, s), $flags);
    };
    (@ $name:ident, $body:expr, $flags:expr) => {
        #[kernel]
        pub fn $name(
            x: &[u32],
            w: &[u32],
            bias: &[f32],
            y: &mut [u32],
            rows: u32,
            n: u32,
            k: u32,
            ld: u32,
            off: u32,
            span: u32,
        ) {
            let s = Shape {
                rows: rows as usize,
                n: n as usize,
                k: k as usize,
                ld: ld as usize,
                off: off as usize,
                span: span as usize,
            };
            let b = if $flags & BIAS == 0 { 0 } else { bias.addr_at(0) };
            let at = (x.addr_at(0), w.addr_at(0), b, y.addr_at(0));
            ($body)(at, s);
        }
    };
}

entry!(gemm_wide_f32, WIDE, true, 0, true);
entry!(gemm_wide_f32_acc, WIDE, true, ACCUMULATE, true);
entry!(gemm_wide_f32_bias, WIDE, true, BIAS, true);
entry!(gemm_wide_f32_acc_bias, WIDE, true, ACCUMULATE | BIAS, true);
entry!(gemm_wide_bf16, WIDE, false, 0, true);
entry!(gemm_wide_bf16_acc, WIDE, false, ACCUMULATE, true);
entry!(gemm_wide_bf16_bias, WIDE, false, BIAS, true);
entry!(gemm_wide_bf16_acc_bias, WIDE, false, ACCUMULATE | BIAS, true);
entry!(reduce_f32, reduce, true, 0);
entry!(reduce_f32_acc, reduce, true, ACCUMULATE);
entry!(reduce_f32_bias, reduce, true, BIAS);
entry!(reduce_f32_acc_bias, reduce, true, ACCUMULATE | BIAS);
entry!(reduce_bf16, reduce, false, 0);
entry!(reduce_bf16_acc, reduce, false, ACCUMULATE);
entry!(reduce_bf16_bias, reduce, false, BIAS);
entry!(reduce_bf16_acc_bias, reduce, false, ACCUMULATE | BIAS);
entry!(any_f32in_f32, any, true, 0);
entry!(any_f32in_f32_acc, any, true, ACCUMULATE);
entry!(any_f32in_f32_bias, any, true, BIAS);
entry!(any_f32in_f32_acc_bias, any, true, ACCUMULATE | BIAS);
entry!(any_f32, any, false, 0);
entry!(any_f32_acc, any, false, ACCUMULATE);
entry!(any_f32_bias, any, false, BIAS);
entry!(any_f32_acc_bias, any, false, ACCUMULATE | BIAS);
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

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}
