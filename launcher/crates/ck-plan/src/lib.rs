//! GPU memory planning shared by `ck`, which sizes its windows, and `ck-gui`, which refuses jobs that cannot fit.
#![no_std]

const A: f64 = 2.5;
const B: f64 = 0.17;
const MARGIN: f64 = 1.15;
const OVERHEAD: f64 = 0.9;
/// Window sizes the automatic planner tries, largest first.
pub const CHUNKS: [u32; 8] = [80, 48, 24, 16, 8, 4, 2, 1];
pub const QUANTUM: u32 = 128;

/// `n / 2^halvings`, rounded half to even, at least 1.
fn halve(n: u32, halvings: u32) -> u32 {
    let (q, r, half) = (n >> halvings, n & ((1 << halvings) - 1), (1 << halvings) >> 1);
    let up = halvings > 0 && (r > half || r == half && q & 1 == 1);
    (q + u32::from(up)).max(1)
}

/// The padded size the model runs at for a `(h, w)` source scaled by `2^-halvings`.
#[must_use]
pub fn model_size((h, w): (u32, u32), halvings: u32, quantum: u32) -> (u32, u32) {
    let up = |n: u32| n.div_ceil(quantum).max(1) * quantum;
    (up(halve(h, halvings)), up(halve(w, halvings)))
}

/// Estimated peak GiB for keying `frames` at once at the padded model size.
#[must_use]
pub fn estimate((h, w): (u32, u32), frames: u32, sm: u32) -> f64 {
    let mp = f64::from(h) * f64::from(w) / 1e6;
    let c = if sm >= 80 { 0.041 } else { 0.07 };
    (A + (B * mp).max(c * mp * f64::from(frames))) * MARGIN + OVERHEAD
}

/// The largest window, its stride and estimated peak, that fits `free` GiB for a clip of `total` frames.
#[must_use]
pub fn window(size: (u32, u32), total: u32, free: f64, sm: u32) -> (u32, u32, f64) {
    let chunk = CHUNKS.into_iter().find(|&c| estimate(size, c.min(total), sm) <= free).unwrap_or(1);
    (chunk, (chunk / 2).max(1), estimate(size, chunk.min(total), sm))
}

/// GiB the NVENC sessions take for `streams` video outputs of a `(w, h)` source.
#[must_use]
pub fn encoder_gib(streams: u32, (w, h): (u32, u32)) -> f64 {
    f64::from(streams) * 1.7 * f64::from(w) * f64::from(h) / (4096.0 * 2160.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_rounds_half_to_even() {
        assert_eq!([halve(2160, 0), halve(2160, 1), halve(5, 1), halve(7, 1), halve(1, 3)], [2160, 1080, 2, 4, 1]);
        assert_eq!(model_size((2160, 4096), 0, QUANTUM), (2176, 4096));
    }

    #[test]
    fn window_shrinks_to_fit() {
        let size = model_size((2160, 4096), 0, QUANTUM);
        assert_eq!(window(size, 604, 10.5, 120).0, 16);
        assert!(estimate(size, 48, 120) > 20.0);
    }
}
