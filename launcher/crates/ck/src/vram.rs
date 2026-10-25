use crate::frame::Geometry;

const A: f64 = 2.5;
const B: f64 = 0.17;
const MARGIN: f64 = 1.15;
const OVERHEAD: f64 = 0.9;
const CHUNKS: [usize; 8] = [80, 48, 24, 16, 8, 4, 2, 1];

#[must_use]
pub fn estimate(g: Geometry, frames: usize, sm: u32) -> f64 {
    let mp = (g.h * g.w) as f64 / 1e6;
    let c = if sm >= 80 { 0.041 } else { 0.07 };
    (A + (B * mp).max(c * mp * frames as f64)) * MARGIN + OVERHEAD
}

#[must_use]
pub fn window(g: Geometry, total: usize, free: f64, sm: u32) -> (usize, usize, f64) {
    let chunk = CHUNKS
        .into_iter()
        .find(|&c| estimate(g, c.min(total), sm) <= free)
        .unwrap_or(1);
    (chunk, (chunk / 2).max(1), estimate(g, chunk.min(total), sm))
}
