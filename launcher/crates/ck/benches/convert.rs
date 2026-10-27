use std::hint::black_box;
use std::time::Instant;

use ck::convert::{Float, Idx, Metric, Narrow, Quant};

fn main() {
    const N: u32 = 1 << 20;
    let mut sink = 0u64;
    let start = Instant::now();
    for i in 0..N {
        let idx = black_box(i as usize) % 4096;
        let scaled = black_box(idx.f32());
        sink += black_box(u64::from(scaled.u32()) + u64::from(idx.u32()) + idx.f64().usize() as u64);
    }
    let elapsed = start.elapsed();
    println!("{N} conversions in {elapsed:?} (sink {sink})");
}
