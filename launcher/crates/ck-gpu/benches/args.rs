use std::hint::black_box;
use std::time::Instant;

use ck_gpu::Arg;

fn main() {
    const N: u32 = 1 << 20;
    let mut sink = 0u32;
    let start = Instant::now();
    for i in 0..N {
        let arg = Arg::from(black_box(i as usize) % 4096);
        sink += black_box(match arg {
            Arg::U(v) => v,
            _ => i,
        });
    }
    let elapsed = start.elapsed();
    println!("{N} kernel args in {elapsed:?} (sink {sink})");
}
