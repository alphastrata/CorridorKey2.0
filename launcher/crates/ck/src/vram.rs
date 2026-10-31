use crate::convert::Idx;
use crate::frame::Geometry;

#[must_use]
pub fn estimate(g: Geometry, frames: usize, sm: u32) -> f64 {
    ck_plan::estimate((g.h.u32(), g.w.u32()), frames.u32(), sm)
}

#[must_use]
pub fn window(g: Geometry, total: usize, free: f64, sm: u32) -> (usize, usize, f64) {
    let (chunk, stride, est) = ck_plan::window((g.h.u32(), g.w.u32()), total.u32(), free, sm);
    let n = |v: u32| usize::try_from(v).unwrap_or(1);
    (n(chunk), n(stride), est)
}
