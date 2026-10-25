use std::path::{Path, PathBuf};

use ck::weights::Weights;
use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::Result;

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

pub fn device() -> usize {
    std::env::var("CK_DEVICE")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(0)
}

pub fn upload(gpu: &Gpu, w: &Weights, name: &str) -> Result<Tensor> {
    let (shape, v) = w.get(name)?;
    gpu.upload_f32(&v, &shape, Dtype::F32)
}

pub fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let d = a.iter().zip(b).map(|(x, y)| f64::from((x - y).abs()));
    let (sum, max) = d.fold((0.0, 0.0f64), |(s, m), v| (s + v, m.max(v)));
    (sum / a.len() as f64, max)
}
