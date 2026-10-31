use std::hint::black_box;
use std::time::Instant;

use ck_ipc::proto::Frame;
use ck_ipc::{decode, encode};

fn main() {
    const N: u32 = 1 << 20;
    let msg = Frame { output: "out".into(), index: 7, w: 2, h: 1, rgb: vec![1, 2, 3, 4, 5, 6] };
    let golden = encode(&msg).unwrap();
    let mut sink = 0u64;
    let start = Instant::now();
    for i in 0..N {
        let round: Frame = decode(black_box(golden.as_slice())).unwrap();
        let packed = encode(black_box(&round)).unwrap();
        sink += black_box(u64::from(round.index + i) + u64::try_from(packed.len()).unwrap_or(u64::MAX));
    }
    let elapsed = start.elapsed();
    println!("{N} round-trips of {} B in {elapsed:?} (sink {sink})", golden.len());
}