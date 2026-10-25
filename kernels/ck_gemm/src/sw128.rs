use ck_arch::{commit, fetch16};
use ptx::prelude::*;

use crate::Shape;

pub const BK: usize = 64;
pub const ROW: u32 = (BK * 2) as u32;
pub const PANE: u32 = 128 * ROW;
pub const SLOT: u32 = 2 * PANE;

pub struct Feed<const N: usize> {
    src: [usize; N],
    extent: usize,
}

impl<const N: usize> Feed<N> {
    #[inline(always)]
    pub fn new((input, weight): (u64, u64), s: Shape, (r0, c0): (usize, usize), (k0, extent): (usize, usize)) -> Self {
        let tid = thread_idx_x() as usize;
        let c = tid % 8;
        let src = core::array::from_fn(|i| {
            let r = tid / 8 + 256 / N * i;
            if r < 128 {
                input as usize + 2 * ((r0 + r).min(s.rows - 1) * s.k + k0 + c * 8)
            } else {
                weight as usize + 2 * ((c0 + r - 128).min(s.n - 1) * s.k + k0 + c * 8)
            }
        });
        Self { src, extent }
    }

    #[inline(always)]
    pub fn load(&self, t: usize, slot: u32) {
        let tid = thread_idx_x() as usize;
        let c = tid % 8;
        let live = t * BK + c * 8 < self.extent;
        ptx::unroll!(16, |i| if i < N {
            let r = tid / 8 + 256 / N * i;
            let _ = fetch16(slot + r as u32 * ROW + ((c ^ (r & 7)) * 16) as u32, self.src[i] + 2 * t * BK, live);
        });
        commit();
    }

    #[inline(always)]
    pub fn steps(&self, t: usize) -> usize {
        (self.extent - t * BK).min(BK).div_ceil(16)
    }
}
