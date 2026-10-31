use std::hint::black_box;
use std::time::Instant;

use ck_glue::pack;

fn main() {
    const N: u32 = 1 << 20;
    let mut sink = 0.0f32;
    let start = Instant::now();
    for i in 0..N {
        let v = (i & 0xffff) as f32 / 128.0 - 256.0;
        sink += black_box(pack(black_box(v), black_box(-v)));
    }
    let elapsed = start.elapsed();
    println!("{N} bf16 packs in {elapsed:?} (sink {sink})");
}
