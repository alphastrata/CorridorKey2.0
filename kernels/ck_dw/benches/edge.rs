use std::hint::black_box;
use std::time::Instant;

use ck_dw::Dims;

fn main() {
    const N: u32 = 1 << 20;
    let (w, n) = (32, 32u32);
    let d = Dims {
        h: 0,
        w,
        c: 1,
        reflect: true,
    };

    let mut sink = 0usize;
    let start = Instant::now();
    for i in 0..N {
        sink += black_box(d.edge(black_box((i % n) as usize), black_box(n as usize)));
    }
    let elapsed = start.elapsed();
    println!("{N} reflected edges in {elapsed:?} (sink {sink})");
}
