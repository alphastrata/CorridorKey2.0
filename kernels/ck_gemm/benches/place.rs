use std::hint::black_box;
use std::time::Instant;

use ck_gemm::{GROUP, Shape, Tile, coords};

const T: Tile = Tile {
    bm: 128,
    bn: 64,
    wn: 32,
    stages: 4,
};

fn main() {
    const N: u32 = 1 << 20;
    let shape = Shape {
        rows: 4096,
        n: 4096,
        k: 512,
        ld: 4096,
        off: 0,
        span: 1,
    };
    let (rows, cols) = (shape.rows.div_ceil(T.bm), shape.n.div_ceil(T.bn));

    let mut sink = 0usize;
    let start = Instant::now();
    for i in 0..N {
        let (r, c) = black_box(coords::<T>(
            rows,
            cols,
            black_box((i as usize) % (GROUP * cols)),
        ));
        sink += r * cols + c;
    }
    let elapsed = start.elapsed();
    println!("{N} tile placements in {elapsed:?} (sink {sink})");
}
