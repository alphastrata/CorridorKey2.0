//! Device kernels: the release build sets `#![no_std]`; `--no-default-features` yields a host `rlib` for tests.
#![cfg_attr(target_arch = "nvptx64", no_std)]
#![feature(abi_ptx, adt_const_params)]

#[cfg(feature = "device")]
use ck_arch::{bf16x2, widen};

#[cfg(feature = "device")]
use ck_arch::convert::{Host, Idx};
#[cfg(feature = "device")]
use ptx::kernel;
#[cfg(feature = "device")]
use ptx::mem::{
    ld_global_cs_u32, ld_global_cs_u64, ld_global_cs_v4_u32, ld_global_nc_u16, ld_global_nc_u32,
    ld_global_nc_v4_f32, st_global_u32, st_global_v2_u32, st_global_v4_u32,
};
#[cfg(feature = "device")]
use ptx::prelude::*;
#[cfg(feature = "device")]
use ptx::sync::shared::{shared_window, shld_v4_f32, shst_f32};

const THREADS: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, core::marker::ConstParamTy, core::fmt::Debug)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub wn: usize,
    pub stages: usize,
}

const ACCUMULATE: u32 = 1;
const BIAS: u32 = 2;

#[derive(Clone, Copy)]
pub struct Shape {
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    pub ld: usize,
    pub off: usize,
    pub span: usize,
}

pub const GROUP: usize = 8;

#[cfg(feature = "device")]
#[inline(always)]
#[allow(clippy::inline_always)]
fn place<const T: Tile>(s: Shape) -> (usize, usize) {
    let (rows, cols) = (s.rows.div_ceil(T.bm), s.n.div_ceil(T.bn));
    coords::<T>(rows, cols, block_idx_x() as usize)
}

#[must_use]
pub fn coords<const T: Tile>(rows: usize, cols: usize, id: usize) -> (usize, usize) {
    let (group, at) = (id / (GROUP * cols), id % (GROUP * cols));
    let height = (rows - group * GROUP).min(GROUP);
    ((group * GROUP + at % height) * T.bm, at / height * T.bn)
}

const AT: usize = 64;
const AK: usize = 16;
#[allow(clippy::cast_possible_truncation)]
const AROW: u32 = ((AT + 4) * 4) as u32;

#[cfg(feature = "device")]
#[inline(always)]
#[allow(clippy::inline_always)]
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

#[cfg(feature = "device")]
#[inline(always)]
#[allow(clippy::cast_possible_truncation, clippy::inline_always)]
fn any<const IN32: bool, const FLAGS: u32>(
    (input, weight, bias, out): (u64, u64, u64, u64),
    shape: Shape,
) {
    let smem = shared_window();

    let tid = thread_idx_x().usize();
    let (r0, c0) = (block_idx_y().usize() * AT, block_idx_x().usize() * AT);
    let (tr, tc) = (tid / 16 * 4, tid % 16 * 4);
    let (xs, ws) = (smem, smem + AK.u32() * AROW);
    let acc = (0..shape.k.div_ceil(AK)).fold([[0.0f32; 4]; 4], |mut acc, slab| {
        ptx::unroll!(4, |e| {
            let i = tid + e * THREADS;
            let (r, kk) = (i / AK, slab * AK + i % AK);

            let xv = if r0 + r < shape.rows && kk < shape.k {
                element::<IN32, true>(input, (r0 + r) * shape.k + kk)
            } else {
                0.0
            };
            let wv = if c0 + r < shape.n && kk < shape.k {
                element::<IN32, false>(weight, (c0 + r) * shape.k + kk)
            } else {
                0.0
            };
            unsafe {
                shst_f32(xs + (i % AK).u32() * AROW + 4 * r.u32(), xv);

                shst_f32(ws + (i % AK).u32() * AROW + 4 * r.u32(), wv);
            }
        });
        sync_threads();
        ptx::unroll!(16, |kk| {
            let (av, bv) = unsafe {
                (
                    shld_v4_f32(xs + kk.u32() * AROW + 4 * tr.u32()),
                    shld_v4_f32(ws + kk.u32() * AROW + 4 * tc.u32()),
                )
            };

            let (av, bv) = ([av.0, av.1, av.2, av.3], [bv.0, bv.1, bv.2, bv.3]);
            ptx::unroll!(4, |i| ptx::unroll!(4, |j| acc[i][j] =
                av[i].fma_rn(bv[j], acc[i][j])));
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

#[cfg(feature = "device")]
#[inline(always)]
#[allow(clippy::cast_possible_truncation, clippy::inline_always)]
fn reduce<const F32: bool, const FLAGS: u32>(
    (parts, bias, out): (u64, u64, u64),
    s: Shape,
    splits: usize,
) {
    let at = (block_idx_x() as usize * THREADS + thread_idx_x() as usize) * 4;
    if at >= s.rows * s.n {
        return;
    }
    let mut v = (0..splits).fold([0.0f32; 4], |v, p| {
        let w = unsafe { ld_global_cs_v4_u32(parts + 4 * (p * s.rows * s.n + at) as u64) };

        [
            v[0] + f32::from_bits(w.0),
            v[1] + f32::from_bits(w.1),
            v[2] + f32::from_bits(w.2),
            v[3] + f32::from_bits(w.3),
        ]
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
                v = [
                    v[0] + f32::from_bits(o.0),
                    v[1] + f32::from_bits(o.1),
                    v[2] + f32::from_bits(o.2),
                    v[3] + f32::from_bits(o.3),
                ];

            }
            st_global_v4_u32(
                dst,
                v[0].to_bits(),
                v[1].to_bits(),
                v[2].to_bits(),
                v[3].to_bits(),
            );
        } else {
            let dst = out + 2 * dst as u64;

            if FLAGS & ACCUMULATE != 0 {
                let o = ld_global_cs_u64(dst);
                let half = |w: u64| f32::from_bits((w as u32) << 16);
                v = [
                    v[0] + half(o),
                    v[1] + half(o >> 16),
                    v[2] + half(o >> 32),
                    v[3] + half(o >> 48),
                ];
            }
            st_global_v2_u32(dst, bf16x2(v[0], v[1]), bf16x2(v[2], v[3]));
        }
    }
}

macro_rules! entry {
    ($name:ident, any, $in32:literal, $flags:expr) => {
        entry!(@ $name, |at, s| any::<$in32, { $flags }>(at, s), $flags);
    };
    ($name:ident, simt, $f32:literal, $flags:expr) => {
        entry!(@ $name, |at, s| simt::simt::<$f32, { $flags }>(at, s), $flags);
    };
    ($name:ident, reduce, $f32:literal, $flags:expr) => {
        entry!(@ $name, |(_, w, b, y), s: Shape| reduce::<$f32, { $flags }>((w, b, y), s, s.span), $flags);
    };
    ($name:ident, $tile:expr, $f32:literal, $flags:expr, $convert:literal) => {
        entry!(@ $name, |at, s| crate::tensor::gemm::<{ $tile }, $f32, { $flags }, $convert>(at, s), $flags);
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

macro_rules! stages {
    ($($name:ident = $tile:expr),*) => {
        $(
            #[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation)]
#[cfg(feature = "device")]
            pub static $name: u32 = $tile.stages as u32;
        )*
    };
}

#[cfg(feature = "sm_100a")]
mod blackwell;
#[cfg(feature = "sm_90a")]
mod hopper;
#[cfg(all(feature = "device", feature = "sm_61"))]
mod simt;
#[cfg(any(feature = "sm_90a", feature = "sm_100a"))]
mod sw128;
#[cfg(all(feature = "device", not(feature = "sm_61")))]
mod tensor;

#[cfg(feature = "device")]
entry!(reduce_f32, reduce, true, 0);
#[cfg(feature = "device")]
entry!(reduce_f32_acc, reduce, true, ACCUMULATE);
#[cfg(feature = "device")]
entry!(reduce_f32_bias, reduce, true, BIAS);
#[cfg(feature = "device")]
entry!(reduce_f32_acc_bias, reduce, true, ACCUMULATE | BIAS);
#[cfg(feature = "device")]
entry!(reduce_bf16, reduce, false, 0);
#[cfg(feature = "device")]
entry!(reduce_bf16_acc, reduce, false, ACCUMULATE);
#[cfg(feature = "device")]
entry!(reduce_bf16_bias, reduce, false, BIAS);

#[cfg(feature = "device")]
entry!(reduce_bf16_acc_bias, reduce, false, ACCUMULATE | BIAS);
#[cfg(feature = "device")]
entry!(any_f32in_f32, any, true, 0);
#[cfg(feature = "device")]
entry!(any_f32in_f32_acc, any, true, ACCUMULATE);
#[cfg(feature = "device")]
entry!(any_f32in_f32_bias, any, true, BIAS);
#[cfg(feature = "device")]
entry!(any_f32in_f32_acc_bias, any, true, ACCUMULATE | BIAS);
#[cfg(feature = "device")]
entry!(any_f32, any, false, 0);

#[cfg(feature = "device")]
entry!(any_f32_acc, any, false, ACCUMULATE);
#[cfg(feature = "device")]
entry!(any_f32_bias, any, false, BIAS);
#[cfg(feature = "device")]
entry!(any_f32_acc_bias, any, false, ACCUMULATE | BIAS);
#[cfg(feature = "sm_61")]
#[cfg(feature = "device")]
entry!(gemm_simt_f32, simt, true, 0);
#[cfg(feature = "sm_61")]
#[cfg(feature = "device")]
entry!(gemm_simt_f32_acc, simt, true, ACCUMULATE);
#[cfg(feature = "sm_61")]
#[cfg(feature = "device")]
entry!(gemm_simt_f32_bias, simt, true, BIAS);

#[cfg(feature = "sm_61")]
#[cfg(feature = "device")]
entry!(gemm_simt_f32_acc_bias, simt, true, ACCUMULATE | BIAS);
#[cfg(feature = "sm_61")]
#[cfg(feature = "device")]
entry!(gemm_simt_bf16, simt, false, 0);
#[cfg(feature = "sm_61")]
#[cfg(feature = "device")]
entry!(gemm_simt_bf16_acc, simt, false, ACCUMULATE);
#[cfg(feature = "sm_61")]
#[cfg(feature = "device")]
entry!(gemm_simt_bf16_bias, simt, false, BIAS);
#[cfg(feature = "sm_61")]
#[cfg(feature = "device")]
entry!(gemm_simt_bf16_acc_bias, simt, false, ACCUMULATE | BIAS);

#[cfg(all(target_arch = "nvptx64", feature = "device", not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}

#[cfg(test)]
mod tests {
    #![cfg(not(feature = "device"))]

    use super::{GROUP, Shape, Tile, coords};

    const T: Tile = Tile {
        bm: 128,
        bn: 64,
        wn: 32,
        stages: 2,
    };

    #[test]
    fn coords_snake_order_within_groups() {
        let (rows, cols) = (1000usize.div_ceil(T.bm), 200usize.div_ceil(T.bn));
        assert_eq!(coords::<T>(rows, cols, 0), (0, 0));
        assert_eq!(coords::<T>(rows, cols, 1), (T.bm, 0));
        assert_eq!(coords::<T>(rows, cols, 8), (0, T.bn));
    }

    #[test]
    fn coords_clamps_final_group_height() {
        let rows = 9; // one partial bm-row group
        assert_eq!(coords::<T>(rows, 1, 4).0, 4 * T.bm);
        assert!(T.bm * GROUP > rows);
        assert_eq!(GROUP, 8);
    }

    #[test]
    fn shape_holds_gemm_geometry() {
        let s = Shape {
            rows: 7,
            n: 5,
            k: 3,
            ld: 7,
            off: 0,
            span: 1,
        };
        assert_eq!(s.rows * s.n, 35);

        assert_eq!(
            T,
            Tile {
                bm: 128,
                bn: 64,
                wn: 32,
                stages: 2
            }
        );
    }
}
