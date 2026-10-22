//! Window size from free GPU memory. Tensor peaks (`CK_MEM`, 720p to 2160p, 1 to 80 frames) fit
//! `A + max(B · MP, C · MP · frames)` GiB at `MP` model megapixels to within 0.15 GiB: weights, then the larger of the
//! per-frame models (foundations, stage 2) and the window; the process needs a margin for the pool and cuBLAS.

use crate::frame::Geometry;

const A: f64 = 2.5;
const B: f64 = 0.17;
const C: f64 = 0.041;
const MARGIN: f64 = 1.15;
const OVERHEAD: f64 = 0.9;
const CHUNKS: [usize; 8] = [80, 48, 24, 16, 8, 4, 2, 1];

/// Estimated process peak in GiB for a window of `frames` frames.
#[must_use]
pub fn estimate(g: Geometry, frames: usize) -> f64 {
    let mp = (g.h * g.w) as f64 / 1e6;
    (A + (B * mp).max(C * mp * frames as f64)) * MARGIN + OVERHEAD
}

/// The largest window (and half of it as the stride) estimated to fit `free` GiB, with its estimate; one frame if
/// none does.
#[must_use]
pub fn window(g: Geometry, total: usize, free: f64) -> (usize, usize, f64) {
    let chunk = CHUNKS
        .into_iter()
        .find(|&c| estimate(g, c.min(total)) <= free)
        .unwrap_or(1);
    (chunk, (chunk / 2).max(1), estimate(g, chunk.min(total)))
}
