//! Swin's shifted windows around attention, on NHWC grids `[lead, h, w, c]`. `norm_bf16`: `rms_norm(x) · w`
//! of f32 pixels into the bf16 grid `[lead, hp, wp, c]` rolled by `(-sh, -sw)` and zero-padded past `h`/`w`,
//! one warp per output pixel. `unshift_add`: `x + roll(a, (sh, sw))[.., :h, :w]` with bf16 `a` on the padded
//! grid, eight channels per thread. Without a shift (`hp = h`, `sh = 0`, ...), plain norm and add.

use crate::{f32x8, high, low, pack};
use ptx::kernel;
use ptx::mem::{ld_global_nc_v4_u32, st_global_v4_f32, st_global_v4_u32};
use ptx::prelude::*;
use ptx::warp_ops::shfl_bfly_sync_b32;

#[derive(Clone, Copy)]
struct Grid {
    h: usize,
    w: usize,
    hp: usize,
    wp: usize,
    sh: usize,
    sw: usize,
}

impl Grid {
    fn new(h: u32, w: u32, hp: u32, wp: u32, sh: u32, sw: u32) -> Self {
        let [h, w, hp, wp, sh, sw] = [h, w, hp, wp, sh, sw].map(|v| v as usize);
        Self {
            h,
            w,
            hp,
            wp,
            sh,
            sw,
        }
    }

    /// Input pixel behind padded pixel `p`, if not padding.
    fn source(self, p: usize) -> Option<usize> {
        let (lead, y, x) = (p / (self.hp * self.wp), p / self.wp % self.hp, p % self.wp);
        let (sy, sx) = ((y + self.sh) % self.hp, (x + self.sw) % self.wp);
        (sy < self.h && sx < self.w).then_some((lead * self.h + sy) * self.w + sx)
    }

    /// Padded pixel that lands on input pixel `p` once rolled back.
    fn back(self, p: usize) -> usize {
        let (lead, y, x) = (p / (self.h * self.w), p / self.w % self.h, p % self.w);
        let (sy, sx) = (
            (y + self.hp - self.sh) % self.hp,
            (x + self.wp - self.sw) % self.wp,
        );
        (lead * self.hp + sy) * self.wp + sx
    }
}

#[kernel]
pub fn norm_bf16(
    x: &[f32],
    weight: &[f32],
    out: &mut [u32],
    pixels: u32,
    c: u32,
    eps: f32,
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    sh: u32,
    sw: u32,
) {
    let p = (block_idx_x() * 8 + thread_idx_x() / 32) as usize;
    if p >= pixels as usize {
        return;
    }
    let (c, lane) = (c as usize, thread_idx_x() as usize % 32);
    let octets = || (lane..c / 8).step_by(32).map(|o| 8 * o);
    let Some(src) = Grid::new(h, w, hp, wp, sh, sw).source(p) else {
        octets()
            .for_each(|i| unsafe { st_global_v4_u32(out.addr_at(p * c / 2 + i / 2), 0, 0, 0, 0) });
        return;
    };
    let sq = octets()
        .flat_map(|i| f32x8(x, src * c + i))
        .fold(0.0f32, |acc, v| v.fma_rn(v, acc));
    let sq = [16, 8, 4, 2, 1].into_iter().fold(sq, |acc, m| {
        acc + f32::from_bits(shfl_bfly_sync_b32(acc.to_bits(), m, 0x1f))
    });
    let inv = (sq / c as f32 + eps).rsqrt();
    octets().for_each(|i| {
        let (v, g) = (f32x8(x, src * c + i), f32x8(weight, i));
        let q: [u32; 4] = core::array::from_fn(|k| {
            pack(v[2 * k] * inv * g[2 * k], v[2 * k + 1] * inv * g[2 * k + 1])
        });
        unsafe { st_global_v4_u32(out.addr_at(p * c / 2 + i / 2), q[0], q[1], q[2], q[3]) };
    });
}

#[kernel]
pub fn unshift_add(
    x: &[f32],
    a: &[u32],
    out: &mut [f32],
    pixels: u32,
    c: u32,
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    sh: u32,
    sw: u32,
) {
    let i = (block_idx_x() * 256 + thread_idx_x()) as usize;
    let per = c as usize / 8;
    if i >= pixels as usize * per {
        return;
    }
    let (p, o) = (i / per, 8 * (i % per));
    let c = c as usize;
    let src = Grid::new(h, w, hp, wp, sh, sw).back(p);
    let v = f32x8(x, p * c + o);
    let q = unsafe { ld_global_nc_v4_u32(a.addr_window::<16>(src * c / 2 + o / 2)) };
    let s: [f32; 8] = core::array::from_fn(|k| {
        let pair = [q.0, q.1, q.2, q.3][k / 2];
        v[k] + if k % 2 == 0 { low(pair) } else { high(pair) }
    });
    unsafe {
        st_global_v4_f32(out.addr_at(p * c + o), s[0], s[1], s[2], s[3]);
        st_global_v4_f32(out.addr_at(p * c + o + 4), s[4], s[5], s[6], s[7]);
    }
}
