use std::hint::black_box;
use std::time::Instant;

use ck_attn::{BC, BR, Shape};

fn main() {
    const N: u32 = 1 << 20;
    let mut sink = 0u32;
    let start = Instant::now();
    for i in 0..N {
        let s = Shape {
            n: black_box(i as usize % 1024) + 1,
            heads: 8,
            windows: 4,
        };
        sink += black_box(u32::try_from(s.n * BR * BC % 4096).unwrap_or(0));
    }
    let elapsed = start.elapsed();
    println!("{N} tile shapes in {elapsed:?} (sink {sink})");
}
