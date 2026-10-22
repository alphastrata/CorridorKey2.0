use std::path::Path;

use ck_gpu::{Dtype, Gpu};
use half::bf16;

fn bin() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../corridorkey/ptx/bin"))
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    (0..n as u32)
        .map(|i| {
            let h = (i ^ seed)
                .wrapping_mul(0x9E37_79B9)
                .rotate_left(13)
                .wrapping_mul(0x85EB_CA6B);
            (h >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f32::from(bf16::from_f32(x))).collect()
}

#[test]
fn linear_bias_accumulate() -> eyre::Result<()> {
    let gpu = Gpu::open(0, bin())?;
    let (rows, out, inner) = (37, 24, 40);
    let x = noise(rows * inner, 1);
    let w = noise(out * inner, 2);
    let bias = noise(out, 3);
    let start = noise(rows * out, 4);
    let xt = gpu.upload_f32(&x, &[rows, inner], Dtype::Bf16)?;
    let wt = gpu.upload_f32(&w, &[out, inner], Dtype::Bf16)?;
    let bt = gpu.upload_f32(&bias, &[out], Dtype::F32)?;
    let yt = gpu.upload_f32(&start, &[rows, out], Dtype::F32)?;
    gpu.linear(&xt, &wt, Some(&bt), &yt, true)?;
    let got = gpu.download_f32(&yt)?;
    let (x, w) = (round(&x), round(&w));
    let worst = (0..rows * out)
        .map(|i| {
            let (row, col) = (i / out, i % out);
            let dot = (0..inner).map(|j| x[row * inner + j] * w[col * inner + j]).sum::<f32>();
            (got[i] - start[i] - bias[col] - dot).abs()
        })
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "worst {worst}");
    Ok(())
}
