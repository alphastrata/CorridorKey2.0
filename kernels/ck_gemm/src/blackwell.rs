use ck_arch::{bf16x2, wait};
use ptx::cluster_mem::fence_proxy_async_shared_cta;
use ptx::kernel;
use ptx::mem::{ld_global_cs_v4_u32, ld_global_nc_u32, st_global_v4_u32};
use ptx::pipe::{mbarrier_init, mbarrier_wait_parity};
use ptx::prelude::*;
use ptx::sync::shared::{shared_window, shld_u32};
use ptx::tcgen05::{
    alloc, commit, dealloc, fence_after_thread_sync, fence_before_thread_sync, instr_desc_f16, ld_32x32b_x8, mma_f16,
    relinquish_alloc_permit, smem_desc, wait_ld,
};

use crate::sw128::{BK, Feed, PANE, SLOT};
use crate::{ACCUMULATE, BIAS, Shape, Tile, place};

const BLACKWELL: Tile = Tile { bm: 128, bn: 128, wn: 1, stages: 3 };
const BF16: u32 = (1 << 7) | (1 << 10);

#[inline(always)]
pub fn gemm<const F32: bool, const FLAGS: u32>((input, weight, bias, out): (u64, u64, u64, u64), s: Shape) {
    let smem = (shared_window() + 1023) & !1023;
    let (bars, held) = (smem + BLACKWELL.stages as u32 * SLOT, smem + BLACKWELL.stages as u32 * SLOT + 16);
    let tid = thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (r0, c0) = place::<BLACKWELL>(s);
    let (split, k0) = (block_idx_y() as usize, block_idx_y() as usize * s.span);
    let extent = s.span.min(s.k - k0);
    let out = out + (split * s.rows * s.ld * if F32 { 4 } else { 2 }) as u64;
    let feed = Feed::<8>::new((input, weight), s, (r0, c0), (k0, extent));
    let slot = |t: usize| smem + (t % BLACKWELL.stages) as u32 * SLOT;
    let bar = |t: usize| (bars + 8 * (t % 2) as u32) as usize;
    if warp == 0 {
        unsafe {
            alloc(held, 128);
            relinquish_alloc_permit();
        }
    }
    if tid == 0 {
        unsafe {
            mbarrier_init(bar(0), 1);
            mbarrier_init(bar(1), 1);
        }
    }
    unsafe { fence_before_thread_sync() };
    sync_threads();
    unsafe { fence_after_thread_sync() };
    let tmem = unsafe { shld_u32(held) };
    let idesc = instr_desc_f16(128, 128) | BF16;
    let tiles = extent.div_ceil(BK);
    let done = |t: usize| unsafe { mbarrier_wait_parity(bar(t), (t / 2 % 2) as u32) };
    (0..BLACKWELL.stages - 1).for_each(|t| feed.load(t, slot(t)));
    (0..tiles).for_each(#[inline(always)] |t| {
        wait::<{ BLACKWELL.stages - 2 }>();
        unsafe { fence_proxy_async_shared_cta() };
        sync_threads();
        if tid == 0 {
            unsafe {
                fence_after_thread_sync();
                let steps = feed.steps(t);
                ptx::unroll!(4, |kk| if kk < steps {
                    let at = 32 * kk as u32;
                    mma_f16(tmem, smem_desc(slot(t) + at), smem_desc(slot(t) + PANE + at), idesc, t > 0 || kk > 0);
                });
                commit(bar(t) as u64);
            }
        }
        if t > 0 {
            done(t - 1);
        }
        feed.load(t + BLACKWELL.stages - 1, slot(t + BLACKWELL.stages - 1));
    });
    done(tiles - 1);
    wait::<0>();
    unsafe { fence_after_thread_sync() };
    let (quarter, half) = (warp % 4, warp / 4);
    let row = r0 + 32 * quarter + lane;
    ptx::unroll!(8, |j| {
        let j = 8 * half + j;
        let mut w = [0u32; 8];
        unsafe {
            ld_32x32b_x8(&mut w, tmem + (((32 * quarter) as u32) << 16) + 8 * j as u32);
            wait_ld();
        }
        let col = c0 + 8 * j;
        if row < s.rows && col < s.n {
            let mut v: [f32; 8] = core::array::from_fn(|e| f32::from_bits(w[e]));
            if FLAGS & BIAS != 0 {
                v = core::array::from_fn(|e| if col + e < s.n { v[e] + f32::from_bits(unsafe { ld_global_nc_u32(bias + 4 * (col + e) as u64) }) } else { v[e] });
            }
            store::<F32, FLAGS>(out, (row * s.ld + s.off + col) as u64, v);
        }
    });
    unsafe { fence_before_thread_sync() };
    sync_threads();
    if warp == 0 {
        unsafe { dealloc(tmem, 128) };
    }
}

#[inline(always)]
fn store<const F32: bool, const FLAGS: u32>(out: u64, at: u64, v: [f32; 8]) {
    unsafe {
        if F32 {
            ptx::unroll!(2, |h| {
                let at = out + 4 * at + 16 * h as u64;
                let old = if FLAGS & ACCUMULATE != 0 { ld_global_cs_v4_u32(at) } else { (0, 0, 0, 0) };
                let old = [old.0, old.1, old.2, old.3].map(f32::from_bits);
                let w: [u32; 4] = core::array::from_fn(|e| (v[4 * h + e] + old[e]).to_bits());
                st_global_v4_u32(at, w[0], w[1], w[2], w[3]);
            });
        } else {
            let at = out + 2 * at;
            let old = if FLAGS & ACCUMULATE != 0 { ld_global_cs_v4_u32(at) } else { (0, 0, 0, 0) };
            let old = [old.0, old.1, old.2, old.3];
            let w: [u32; 4] = core::array::from_fn(|e| bf16x2(v[2 * e] + f32::from_bits(old[e] << 16), v[2 * e + 1] + f32::from_bits(old[e] & 0xffff_0000)));
            st_global_v4_u32(at, w[0], w[1], w[2], w[3]);
        }
    }
}

macro_rules! blackwell {
    ($($name:ident, $f32:literal, $flags:expr;)*) => {
        $(entry!(@ $name, |at, s| gemm::<$f32, { $flags }>(at, s), $flags);)*
    };
}

blackwell! {
    gemm_blackwell_f32, true, 0;
    gemm_blackwell_f32_acc, true, ACCUMULATE;
    gemm_blackwell_f32_bias, true, BIAS;
    gemm_blackwell_f32_acc_bias, true, ACCUMULATE | BIAS;
    gemm_blackwell_bf16, false, 0;
    gemm_blackwell_bf16_acc, false, ACCUMULATE;
    gemm_blackwell_bf16_bias, false, BIAS;
    gemm_blackwell_bf16_acc_bias, false, ACCUMULATE | BIAS;
}

stages!(STAGES_BLACKWELL = BLACKWELL);
