use std::hint::black_box;
use std::time::Instant;

use ck_gui::convert::{Float, Metric, Narrow, Quant};

fn main() {
    const N: u32 = 1 << 20;
    let mut sink = 0u32;
    let start = Instant::now();
    for i in 0..N {
        let idx = black_box(i as usize) % 4096;
        let v = black_box(idx.f32() / 3600.0);
        sink += black_box(u32::from(v.u8()) + idx.f64().u32() % 4096);
    }
    let elapsed = start.elapsed();
    println!("{N} pixel conversions in {elapsed:?} (sink {sink})");
}
