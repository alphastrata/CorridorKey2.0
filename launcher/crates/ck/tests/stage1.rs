//! Stage 1 against Python's stock bf16 and fp32 reference on real footage (`launcher/parity/dump.py`).

mod common;

use ck::attn::Cache;
use ck::stage1::{Inputs, Stage1};
use ck::weights::{Loader, Weights};
use ck_gpu::Gpu;
use common::{device, diff, repo, upload};
use eyre::Result;

fn report(name: &str, got: &[f32], fx: &Weights, key: &str) -> Result<((f64, f64), (f64, f64))> {
    let (stock, reference) = (fx.get(&format!("stock.{key}"))?.1, fx.get(&format!("ref.{key}"))?.1);
    let (ours, theirs) = (diff(got, &reference), diff(&stock, &reference));
    let vs = diff(got, &stock);
    println!(
        "{name:8} ours-ref {:.3e}/{:.3e} stock-ref {:.3e}/{:.3e} ours-stock {:.3e}/{:.3e}",
        ours.0, ours.1, theirs.0, theirs.1, vs.0, vs.1
    );
    Ok((ours, theirs))
}

/// NHWC `[.., h, w, c]` → NCHW `[.., c, h, w]` on the host.
fn nchw(v: &[f32], (h, w, c): (usize, usize, usize)) -> Vec<f32> {
    let per = h * w * c;
    (0..v.len())
        .map(|i| v[i / per * per + (i % (h * w)) * c + i % per / (h * w)])
        .collect()
}

#[test]
fn stage1_720() -> Result<()> {
    let root = repo();
    let fx = Weights::open(&root.join("launcher/fixtures/stage1_720.safetensors"))?;
    let gpu = Gpu::open(device(), &root.join("corridorkey/ptx/bin"))?;
    let weights = Weights::open(&root.join("weights/corridorkey_v2.safetensors"))?;
    let ld = Loader {
        gpu: &gpu,
        weights: &weights,
        prefix: "stage1.",
        half: gpu.half(),
    };
    let s1 = Stage1::load(&ld)?;
    let cache = Cache::default();
    let quality = upload(&gpu, &fx, "quality")?;

    let parts = [("enc_16", 2), ("body_64", 10), ("mid_16", 4), ("dec_16", 6)];
    let names: Vec<String> = parts
        .iter()
        .flat_map(|(p, n)| (0..*n).map(move |i| format!("{p}{i}")))
        .collect();
    names.iter().zip(s1.blocks()).try_for_each(|(n, b)| -> Result<()> {
        let x = upload(&gpu, &fx, &format!("stock.{n}.in"))?;
        let cond = n.starts_with("enc").then_some(&quality);
        let got = gpu.download_f32(&b.forward(&gpu, &cache, &x, cond)?)?;
        report(n, &got, &fx, &format!("{n}.out")).map(|_| ())
    })?;

    let [x, cradio, moge, rvm] = ["x", "cradio", "moge", "rvm"].map(|k| upload(&gpu, &fx, k));
    let (x, cradio, moge, rvm) = (x?, cradio?, moge?, rvm?);
    let inputs = Inputs {
        b: 1,
        t: x.shape[0],
        x,
        quality: &quality,
        features: [&cradio, &moge, &rvm],
    };
    let out = s1.forward(&gpu, &cache, inputs)?;
    let [_, _, h, w, c] = out.baton.shape[..] else {
        eyre::bail!("baton shape")
    };
    let baton = nchw(&gpu.download_f32(&out.baton)?, (h, w, c));
    let (ours, theirs) = report("baton", &baton, &fx, "out0")?;
    let (aux, aux_stock) = report("aux", &gpu.download_f32(&out.aux)?, &fx, "out1")?;
    assert!(
        ours.0 <= theirs.0 * 1.1 && aux.0 <= aux_stock.0 * 1.1,
        "stage 1 less precise than stock"
    );
    Ok(())
}
