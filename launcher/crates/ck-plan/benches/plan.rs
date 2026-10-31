use std::hint::black_box;
use std::time::Instant;

use ck_plan::{QUANTUM, estimate, model_size};

fn main() {
    const N: u32 = 1 << 20;
    let mut sink = 0u64;
    let start = Instant::now();
    for i in 0..N {
        let halvings = black_box(i % 8);
        let (h, w) = model_size(black_box((2160, 4096)), halvings, QUANTUM);
        sink += black_box(u64::from(h) + u64::from(w) + u64::from(i));
        black_box(estimate((h, w), black_box(i % 64), 120));
    }
    let elapsed = start.elapsed();
    println!("{N} (model_size, estimate) passes in {elapsed:?} (sink {sink})");
}