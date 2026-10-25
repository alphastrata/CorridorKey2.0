use ck_arch::{bf16x2, wait};
use ptx::cluster_mem::fence_proxy_async_shared_cta;
use ptx::kernel;
use ptx::mem::{ld_global_cs_v4_u32, ld_global_nc_v2_u32, st_global_v4_u32};
use ptx::mma::{wgmma_commit_group, wgmma_fence, wgmma_m64n128k16_f32_bf16_bf16, wgmma_smem_desc, wgmma_wait_group};
use ptx::prelude::*;
use ptx::sync::shared::{shared_window, shld_v4_f32, shst_v2_f32};

use crate::sw128::{BK, Feed, PANE, ROW, SLOT};
use crate::{ACCUMULATE, BIAS, Shape, Tile, place};

const HOPPER: Tile = Tile { bm: 128, bn: 128, wn: 1, stages: 3 };
const STRIDE: u32 = (128 + 4) * 4;

#[inline(always)]
fn desc(at: u32) -> u64 {
    wgmma_smem_desc(at, 16, 8 * ROW, 1)
}

#[inline(always)]
pub fn gemm<const F32: bool, const FLAGS: u32>((input, weight, bias, out): (u64, u64, u64, u64), s: Shape) {
    let smem = (shared_window() + 1023) & !1023;
    let tid = thread_idx_x() as usize;
    let (group, warp, lane) = (tid / 128, tid % 128 / 32, tid % 32);
    let (r0, c0) = place::<HOPPER>(s);
    let (split, k0) = (block_idx_y() as usize, block_idx_y() as usize * s.span);
    let extent = s.span.min(s.k - k0);
    let out = out + (split * s.rows * s.ld * if F32 { 4 } else { 2 }) as u64;
    let feed = Feed::<8>::new((input, weight), s, (r0, c0), (k0, extent));
    let tiles = extent.div_ceil(BK);
    let slot = |t: usize| smem + (t % HOPPER.stages) as u32 * SLOT;
    (0..HOPPER.stages - 1).for_each(|t| feed.load(t, slot(t)));
    let acc = (0..tiles).fold([0.0f32; 64], #[inline(always)] |mut acc, t| {
        wait::<{ HOPPER.stages - 2 }>();
        unsafe { fence_proxy_async_shared_cta() };
        sync_threads();
        let (a, b) = (slot(t) + group as u32 * 64 * ROW, slot(t) + PANE);
        unsafe {
            wgmma_fence();
            let steps = feed.steps(t);
            ptx::unroll!(4, |kk| if kk < steps {
                wgmma_m64n128k16_f32_bf16_bf16(&mut acc, desc(a + 32 * kk as u32), desc(b + 32 * kk as u32), true);
            });
            wgmma_commit_group();
            wgmma_wait_group::<1>();
        }
        sync_threads();
        feed.load(t + HOPPER.stages - 1, slot(t + HOPPER.stages - 1));
        acc
    });
    unsafe { wgmma_wait_group::<0>() };
    wait::<0>();
    sync_threads();
    let stage = smem + group as u32 * 64 * STRIDE;
    let r = warp * 16 + lane / 4;
    ptx::unroll!(16, |j| {
        let col = 8 * j + 2 * (lane % 4);
        let add = if FLAGS & BIAS != 0 {
            let (lo, hi) = unsafe { ld_global_nc_v2_u32(bias + 4 * (c0 + col).min(s.n - 2) as u64) };
            (f32::from_bits(lo), f32::from_bits(hi))
        } else {
            (0.0, 0.0)
        };
        ptx::unroll!(2, |h| unsafe {
            shst_v2_f32(stage + (r + 8 * h) as u32 * STRIDE + 4 * col as u32, (acc[4 * j + 2 * h] + add.0, acc[4 * j + 2 * h + 1] + add.1));
        });
    });
    sync_threads();
    drain::<F32, FLAGS>(stage, out, s, (r0 + group * 64, c0));
}

#[inline(always)]
fn drain<const F32: bool, const FLAGS: u32>(stage: u32, out: u64, s: Shape, (top, left): (usize, usize)) {
    let t = thread_idx_x() as usize % 128;
    let width = if F32 { 4 } else { 8 };
    let per = 128 / width;
    ptx::unroll!(16, |p| if p < 64 * per / 128 {
        let (sr, sc) = ((p * 128 + t) / per, (p * 128 + t) % per);
        let (row, col) = (top + sr, left + sc * width);
        if row < s.rows && col < s.n {
            let src = stage + sr as u32 * STRIDE + (sc * width * 4) as u32;
            let at = out + ((row * s.ld + s.off + col) * if F32 { 4 } else { 2 }) as u64;
            let old = if FLAGS & ACCUMULATE != 0 { unsafe { ld_global_cs_v4_u32(at) } } else { (0, 0, 0, 0) };
            let old = [old.0, old.1, old.2, old.3];
            unsafe {
                if F32 {
                    let v = shld_v4_f32(src);
                    let v = [v.0, v.1, v.2, v.3];
                    let w: [u32; 4] = core::array::from_fn(|e| (v[e] + if FLAGS & ACCUMULATE != 0 { f32::from_bits(old[e]) } else { 0.0 }).to_bits());
                    st_global_v4_u32(at, w[0], w[1], w[2], w[3]);
                } else {
                    let (lo4, hi4) = (shld_v4_f32(src), shld_v4_f32(src + 16));
                    let v = [lo4.0, lo4.1, lo4.2, lo4.3, hi4.0, hi4.1, hi4.2, hi4.3];
                    let w: [u32; 4] = core::array::from_fn(|e| {
                        let (lo, hi) = if FLAGS & ACCUMULATE != 0 { (f32::from_bits(old[e] << 16), f32::from_bits(old[e] & 0xffff_0000)) } else { (0.0, 0.0) };
                        bf16x2(v[2 * e] + lo, v[2 * e + 1] + hi)
                    });
                    st_global_v4_u32(at, w[0], w[1], w[2], w[3]);
                }
            }
        }
    });
}

macro_rules! hopper {
    ($($name:ident, $f32:literal, $flags:expr;)*) => {
        $(entry!(@ $name, |at, s| gemm::<$f32, { $flags }>(at, s), $flags);)*
    };
}

hopper! {
    gemm_hopper_f32, true, 0;
    gemm_hopper_f32_acc, true, ACCUMULATE;
    gemm_hopper_f32_bias, true, BIAS;
    gemm_hopper_f32_acc_bias, true, ACCUMULATE | BIAS;
    gemm_hopper_bf16, false, 0;
    gemm_hopper_bf16_acc, false, ACCUMULATE;
    gemm_hopper_bf16_bias, false, BIAS;
    gemm_hopper_bf16_acc_bias, false, ACCUMULATE | BIAS;
}
