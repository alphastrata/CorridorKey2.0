use crate::gelu;
use ptx::kernel;
use ptx::mem::{ld_global_nc_v4_f32, ld_global_nc_v4_u32};
use ptx::mma::{FragC16x8F32, ldmatrix_x2_b16_u32, ldmatrix_x4_b16_u32};
use ptx::num::f16::cvt_rn_f16x2_f32;
use ptx::prelude::*;
use ptx::slice::GpuSlice;
use ptx::sync::shared::{shared_window, shst_v2_u32, shst_v4_u32};

const C: usize = 24;
const PX: u32 = 2 * C as u32;
const TH: usize = 8;
const TW: usize = 32;
const HALO: usize = 4;
const TILE: u32 = ((TH + 2 * HALO) * (TW + 2 * HALO)) as u32 * PX;
const CHUNKS: usize = 29;
const ROW: u32 = 16 * CHUNKS as u32;
pub const SHARED: u32 = TILE + 16 + C as u32 * ROW;

#[unsafe(no_mangle)]
pub static REFINE_SHARED: u32 = SHARED;

#[cfg(feature = "sm_80")]
fn mma(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    use ptx::mma::{FragA16x16F16, FragB16x8F16, mma_m16n8k16_f16f16f32};
    mma_m16n8k16_f16f16f32(FragA16x16F16(a), FragB16x8F16(b), FragC16x8F32(c)).0
}

#[cfg(not(feature = "sm_80"))]
fn mma(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    use ptx::mma::{FragA16x8F16, FragB8x8F16, mma_m16n8k8_f16f16f32};
    let c = mma_m16n8k8_f16f16f32(FragA16x8F16([a[0], a[1]]), FragB8x8F16([b[0]]), FragC16x8F32(c));
    mma_m16n8k8_f16f16f32(FragA16x8F16([a[2], a[3]]), FragB8x8F16([b[1]]), c).0
}

fn ld(s: GpuSlice<f32>, i: usize) -> f32 {
    unsafe { *s.get_unchecked(i) }
}

#[kernel]
pub fn refine_mma(
    x: &[f32],
    out: &mut [f32],
    wt: &[u16],
    bias: &[f32],
    cin8: u32,
    rows: u32,
    cols: u32,
    dil: u32,
    residual: u32,
) {
    let [cin8, rows, cols, d] = [cin8, rows, cols, dil].map(|v| v as usize);
    let smem = shared_window();
    let (zero, wbase) = (smem + TILE, smem + TILE + 16);
    let tid = thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (y0, x0) = (block_idx_y() as usize * TH, block_idx_x() as usize * TW);
    let (th, tw) = (TH + 2 * d, TW + 2 * d);

    (0..(C * CHUNKS).div_ceil(256)).for_each(|k| {
        let i = tid + 256 * k;
        if i < C * CHUNKS {
            unsafe { shst_v4_u32(wbase + 16 * i as u32, ld_global_nc_v4_u32(wt.addr_at(8 * i))) };
        }
    });
    if tid == 0 {
        unsafe { shst_v4_u32(zero, (0, 0, 0, 0)) };
    }
    let quads = th * tw * (C / 4);
    (0..quads.div_ceil(256)).for_each(|k| {
        let i = tid + 256 * k;
        if i >= quads {
            return;
        }
        let (p, q) = (i / (C / 4), i % (C / 4));
        let (sy, sx) = ((y0 + p / tw).wrapping_sub(d), (x0 + p % tw).wrapping_sub(d));
        let v = if sy < rows && sx < cols {
            unsafe { ld_global_nc_v4_f32(x.addr_at((sy * cols + sx) * C + 4 * q)) }
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        unsafe { shst_v2_u32(smem + p as u32 * PX + 8 * q as u32, (cvt_rn_f16x2_f32(v.0, v.1), cvt_rn_f16x2_f32(v.2, v.3))) };
    });
    sync_threads();

    let chunks = 9 * cin8;
    let a_addr = |j: usize, px: usize| -> u32 {
        if j >= chunks {
            return zero;
        }
        let (t, cc) = (j / cin8, j % cin8);
        let p = (warp + (t / 3) * d) * tw + px + (t % 3) * d;
        smem + p as u32 * PX + 16 * cc as u32
    };
    let b_addr = |n: usize, j: usize| wbase + n as u32 * ROW + 16 * j as u32;
    let mut acc = [[[0.0f32; 4]; 3]; 2];
    (0..chunks.div_ceil(2)).for_each(|s| {
        let a: [[u32; 4]; 2] = core::array::from_fn(|mt| unsafe {
            ldmatrix_x4_b16_u32(a_addr(2 * s + lane / 16, mt * 16 + lane % 16)).0
        });
        let j = 2 * s + (lane / 8) % 2;
        let b01 = unsafe { ldmatrix_x4_b16_u32(b_addr((lane / 16) * 8 + lane % 8, j)).0 };
        let b2 = unsafe { ldmatrix_x2_b16_u32(b_addr(16 + lane % 8, j)) };
        let b = [[b01[0], b01[1]], [b01[2], b01[3]], b2];
        ptx::unroll!(2, |mt| ptx::unroll!(3, |nt| acc[mt][nt] = mma(a[mt], b[nt], acc[mt][nt])));
    });

    let y = y0 + warp;
    if y >= rows {
        return;
    }
    let (g, tig) = (lane / 4, lane % 4);
    ptx::unroll!(2, |mt| ptx::unroll!(2, |half| {
        let xx = x0 + mt * 16 + g + 8 * half;
        if xx < cols {
            let at = (y * cols + xx) * C;
            ptx::unroll!(3, |nt| ptx::unroll!(2, |e| {
                let o = nt * 8 + 2 * tig + e;
                let v = gelu(acc[mt][nt][2 * half + e] + ld(bias, o));
                out.set(at + o, if residual == 0 { v } else { ld(x, at + o) + v });
            }));
        }
    }));
}
