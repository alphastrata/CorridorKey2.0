mod common;

use ck::attn::Cache;
use ck::stage2::{Baton, Stage2};
use ck::weights::{Loader, Weights};
use ck_gpu::Gpu;
use common::{device, diff, repo, upload};
use eyre::Result;

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
        half: gpu.half(),
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
    let got = gpu.download_f32(&s2.forward(&gpu, &cache, &rgb, Baton::Nchw(&baton))?)?;
    let (stock, reference) = (fx.get("stock.out0")?.1, fx.get("ref.out0")?.1);
    let (ours, theirs) = (diff(&got, &reference), diff(&stock, &reference));
    println!(
        "logits  ours-ref {:.3e}/{:.3e} stock-ref {:.3e}/{:.3e}",
        ours.0, ours.1, theirs.0, theirs.1
    );
    assert!(ours.0 <= theirs.0 * 1.1, "stage 2 less precise than stock");
    Ok(())
}
