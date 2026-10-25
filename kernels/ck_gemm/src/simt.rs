use ptx::mem::{ld_global_cs_u32, ld_global_nc_u32, ld_global_nc_v4_u32, st_global_u32};
use ptx::prelude::*;
use ptx::sync::shared::{shared_window, shld_v4_f32, shst_f32};

use crate::{ACCUMULATE, BIAS, Shape, Tile, bf16x2, place};

const SIMT: Tile = Tile { bm: 128, bn: 128, wn: 4, stages: 2 };
const SK: usize = 16;
const SROW: u32 = 128 * 4;
const PANE: u32 = SK as u32 * SROW;

fn quad(at: u32) -> [f32; 4] {
    let v = unsafe { shld_v4_f32(at) };
    [v.0, v.1, v.2, v.3]
}

#[inline(always)]
fn fetch(src: [u64; 2], t: usize, live: bool) -> [[u32; 4]; 2] {
    let one = |at: u64| if live { let w = unsafe { ld_global_nc_v4_u32(at + 2 * (t * SK) as u64) }; [w.0, w.1, w.2, w.3] } else { [0; 4] };
    [one(src[0]), one(src[1])]
}

#[inline(always)]
fn put(w: [[u32; 4]; 2], buf: u32, (lr, kc): (usize, usize)) {
    ptx::unroll!(2, |m| ptx::unroll!(4, |j| unsafe {
        let at = buf + m as u32 * PANE + (kc + 2 * j) as u32 * SROW + 4 * lr as u32;
        shst_f32(at, f32::from_bits(w[m][j] << 16));
        shst_f32(at + SROW, f32::from_bits(w[m][j] & 0xffff_0000));
    }));
}

#[inline(always)]
fn step(mut acc: [[f32; 8]; 8], buf: u32, (ay, bx): (u32, u32)) -> [[f32; 8]; 8] {
    ptx::unroll!(16, |kk| {
        let row = buf + kk as u32 * SROW;
        let (a0, a1) = (quad(row + ay), quad(row + 256 + ay));
        let (b0, b1) = (quad(row + PANE + bx), quad(row + PANE + 256 + bx));
        let (a, b) = ([a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]], [b0[0], b0[1], b0[2], b0[3], b1[0], b1[1], b1[2], b1[3]]);
        ptx::unroll!(8, |i| ptx::unroll!(8, |j| acc[i][j] = a[i].fma_rn(b[j], acc[i][j])));
    });
    acc
}

#[inline(always)]
pub fn simt<const F32: bool, const FLAGS: u32>((input, weight, bias, out): (u64, u64, u64, u64), s: Shape) {
    let smem = shared_window();
    let tid = thread_idx_x() as usize;
    let (r0, c0) = place::<SIMT>(s);
    let (split, k0) = (block_idx_y() as usize, block_idx_y() as usize * s.span);
    let extent = s.span.min(s.k - k0);
    let out = out + (split * s.rows * s.ld * if F32 { 4 } else { 2 }) as u64;
    let (lr, kc) = (tid / 2, tid % 2 * 8);
    let src = [
        input + 2 * ((r0 + lr).min(s.rows - 1) * s.k + k0 + kc) as u64,
        weight + 2 * ((c0 + lr).min(s.n - 1) * s.k + k0 + kc) as u64,
    ];
    let live = |t: usize| t * SK + kc < extent;
    let (ty, tx) = (tid / 16, tid % 16);
    let tiles = extent.div_ceil(SK);
    put(fetch(src, 0, live(0)), smem, (lr, kc));
    sync_threads();
    let acc = (0..tiles).fold([[0.0f32; 8]; 8], #[inline(always)] |acc, t| {
        let buf = smem + (t % 2) as u32 * 2 * PANE;
        let next = fetch(src, t + 1, live(t + 1));
        let acc = step(acc, buf, (16 * ty as u32, 16 * tx as u32));
        put(next, smem + ((t + 1) % 2) as u32 * 2 * PANE, (lr, kc));
        sync_threads();
        acc
    });
    ptx::unroll!(8, |i| {
        let row = r0 + i / 4 * 64 + ty * 4 + i % 4;
        ptx::unroll!(4, |p| {
            let col = c0 + p / 2 * 64 + tx * 4 + p % 2 * 2;
            if row < s.rows && col < s.n {
                let whole = col + 1 < s.n;
                let at = row * s.ld + s.off + col;
                let mut v = [acc[i][2 * p], acc[i][2 * p + 1]];
                if FLAGS & BIAS != 0 {
                    v[0] += f32::from_bits(unsafe { ld_global_nc_u32(bias + 4 * col as u64) });
                    v[1] += f32::from_bits(unsafe { ld_global_nc_u32(bias + 4 * (col + usize::from(whole)) as u64) });
                }
                unsafe {
                    if F32 {
                        let at = out + 4 * at as u64;
                        ptx::unroll!(2, |e| if e == 0 || whole {
                            let prior = if FLAGS & ACCUMULATE != 0 { f32::from_bits(ld_global_cs_u32(at + 4 * e as u64)) } else { 0.0 };
                            st_global_u32(at + 4 * e as u64, (v[e] + prior).to_bits());
                        });
                    } else {
                        let at = out + 2 * at as u64;
                        let old = if FLAGS & ACCUMULATE != 0 || !whole { ld_global_cs_u32(at) } else { 0 };
                        if FLAGS & ACCUMULATE != 0 {
                            v = [v[0] + f32::from_bits(old << 16), v[1] + f32::from_bits(old & 0xffff_0000)];
                        }
                        let packed = bf16x2(v[0], v[1]);
                        st_global_u32(at, if whole { packed } else { packed & 0xffff | old & 0xffff_0000 });
                    }
                }
            }
        });
    });
}
