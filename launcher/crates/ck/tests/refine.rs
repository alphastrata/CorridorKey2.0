mod common;

use ck::frame::{Adjust, Geometry, Input};
use ck::refine::Refiner;
use ck::weights::Weights;
use ck_gpu::Gpu;
use common::{device, diff, repo, upload};
use eyre::Result;

const KNEE: f32 = 0.008;
const DENOM: f32 = 8.294_05;

fn to_srgb(x: f32) -> f32 {
    if x <= 0.003_130_8 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

#[test]
fn refine_matches_reference() -> Result<()> {
    let root = repo();
    let fx = Weights::open(&root.join("launcher/fixtures/refine.safetensors"))?;
    let gpu = Gpu::open(device(), &root.join("corridorkey/ptx/bin"))?;
    let net = Refiner::load(&gpu, &root.join("weights/corridorkey_edge_refiner_v10w.safetensors"))?;
    let native = upload(&gpu, &fx, "native")?;
    let low = upload(&gpu, &fx, "low")?;
    let (hn, wn, h, w) = (native.shape[2], native.shape[3], low.shape[2], low.shape[3]);
    let g = Geometry { hn, wn, h0: h, w0: w, h, w };
    let input = Input { adjust: Adjust { gain: 1.0, contrast: 1.0, pivot: 0.18 }, linear: false };
    let out = net.native(&gpu, &upload(&gpu, &fx, "logits")?, (&native, &low), g, input)?;
    let out = gpu.download_f32(&out)?;
    let n = hn * wn;
    let fg: Vec<f32> = out[n..].iter().map(|&v| to_srgb((v * DENOM).sinh() * KNEE).clamp(0.0, 1.0)).collect();
    let (ma, xa) = diff(&out[..n], &fx.get("alpha")?.1);
    let (mf, xf) = diff(&fg, &fx.get("fg")?.1);
    let (mb, _) = diff(&fx.get("bilinear")?.1, &fx.get("alpha")?.1);
    println!("alpha mean {ma:.2e} max {xa:.2e} · fg mean {mf:.2e} max {xf:.2e} · bilinear vs reference {mb:.2e}");
    assert!(ma < 2e-4 && xa < 3e-3, "alpha");
    assert!(mf < 2e-4 && xf < 3e-3, "fg");
    Ok(())
}
