//! Stage 2 against Python's stock bf16 and fp32 reference on real footage (`launcher/parity/dump.py`).

use std::path::{Path, PathBuf};

use ck::attn::Cache;
use ck::stage2::Stage2;
use ck::weights::{Loader, Weights};
use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::Result;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn device() -> usize {
    std::env::var("CK_DEVICE")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(0)
}

fn upload(gpu: &Gpu, w: &Weights, name: &str) -> Result<Tensor> {
    let (shape, v) = w.get(name)?;
    gpu.upload_f32(&v, &shape, Dtype::F32)
}

/// Mean and max absolute difference.
fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let d = a.iter().zip(b).map(|(x, y)| f64::from((x - y).abs()));
    let (sum, max) = d.fold((0.0, 0.0f64), |(s, m), v| (s + v, m.max(v)));
    (sum / a.len() as f64, max)
}

#[test]
fn stage2_720() -> Result<()> {
    stage2("720")
}

#[test]
fn stage2_1080() -> Result<()> {
    stage2("1080")
}

fn stage2(tier: &str) -> Result<()> {
    let root = repo();
    let fx = Weights::open(&root.join(format!("launcher/fixtures/stage2_{tier}.safetensors")))?;
    let gpu = Gpu::open(device(), &root.join("corridorkey/ptx/bin"))?;
    let weights = Weights::open(&root.join("weights/corridorkey_v2.safetensors"))?;
    let ld = Loader {
        gpu: &gpu,
        weights: &weights,
        prefix: "stage2.",
        half: Dtype::Bf16,
    };
    let s2 = Stage2::load(&ld)?;
    let cache = Cache::default();

    let names: Vec<String> = (0..8)
        .map(|i| format!("pre{i}"))
        .chain((0..2).map(|i| format!("final{i}")))
        .collect();
    let blocks: Vec<&ck::block::Block> = s2.blocks().collect();
    names.iter().zip(&blocks).try_for_each(|(n, b)| -> Result<()> {
        let x = upload(&gpu, &fx, &format!("stock.{n}.in"))?;
        let [_, h, w, c] = x.shape[..] else {
            eyre::bail!("block input")
        };
        let got = gpu.download_f32(&b.forward(&gpu, &cache, &x.reshape(&[1, 1, h, w, c])?, None)?)?;
        let (stock, reference) = (fx.get(&format!("stock.{n}.out"))?.1, fx.get(&format!("ref.{n}.out"))?.1);
        let (ours, theirs) = (diff(&got, &reference), diff(&stock, &reference));
        let vs = diff(&got, &stock);
        println!(
            "{n:7} ours-ref {:.3e}/{:.3e} stock-ref {:.3e}/{:.3e} ours-stock {:.3e}/{:.3e}",
            ours.0, ours.1, theirs.0, theirs.1, vs.0, vs.1
        );
        Ok(())
    })?;

    let (rgb, baton) = (upload(&gpu, &fx, "rgb")?, upload(&gpu, &fx, "baton")?);
    let got = gpu.download_f32(&s2.forward(&gpu, &cache, &rgb, &baton)?)?;
    let (stock, reference) = (fx.get("stock.logits")?.1, fx.get("ref.logits")?.1);
    let (ours, theirs) = (diff(&got, &reference), diff(&stock, &reference));
    println!(
        "logits  ours-ref {:.3e}/{:.3e} stock-ref {:.3e}/{:.3e}",
        ours.0, ours.1, theirs.0, theirs.1
    );
    assert!(ours.0 <= theirs.0 * 1.1, "stage 2 less precise than stock");
    Ok(())
}
