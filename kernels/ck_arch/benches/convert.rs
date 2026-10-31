use std::hint::black_box;
use std::time::Instant;

fn main() {
    const N: u32 = 1 << 20;
    let mut sink = 0.0f32;
    let start = Instant::now();
    for i in 0..N {
        sink += black_box(ck_arch::convert::Conv::f32(black_box(
            ck_arch::convert::Idx::u32(black_box(i as usize)),
        )));
    }
    let elapsed = start.elapsed();
    println!("{N} index conversions in {elapsed:?} (sink {sink})");
}
