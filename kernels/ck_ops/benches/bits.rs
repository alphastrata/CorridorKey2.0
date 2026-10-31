use std::hint::black_box;
use std::time::Instant;

use ck_ops::{bf, rn};

fn main() {
    const N: u32 = 1 << 20;
    let mut sink = 0u32;
    let start = Instant::now();
    for i in 0..N {
        let v = bf(((i & 0x7fff) + 0x3800) as u16).max(1.0);
        sink += black_box(u32::from(rn(black_box(v))));
    }
    let elapsed = start.elapsed();
    println!("{N} bf16 bits in {elapsed:?} (sink {sink})");
}
