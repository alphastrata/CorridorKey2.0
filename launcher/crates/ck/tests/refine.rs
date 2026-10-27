mod common;

use ck::frame::{Adjust, Geometry, Input};
use ck::refine::Refiner;
use ck::weights::Weights;
use ck_gpu::{Dtype, Gpu};
use common::{device, diff, repo};
use eyre::Result;

const KNEE: f32 = 0.008;
const DENOM: f32 = 8.294_05;

fn to_linear(x: f32) -> f32 {
    if x <= 0.040_45 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}

fn to_srgb(x: f32) -> f32 {
    if x <= 0.003_130_8 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn gbr(v: &[f32]) -> Vec<f32> {
    let n = v.len() / 3;
    [1, 2, 0].iter().flat_map(|&c| v[c * n..(c + 1) * n].iter().copied()).collect()
}

#[test]
fn refine_matches_reference() -> Result<()> {
    let root = repo();
    let fx = Weights::open(&root.join("launcher/fixtures/refine.safetensors"))?;
    let gpu = Gpu::open(device(), &root.join("corridorkey/ptx/bin"))?;
    let net = Refiner::load(&gpu, &root.join("weights/corridorkey_edge_refiner_v10w.safetensors"))?;
    let (rs, rgb) = fx.get("rgb")?;
    let (ls, rgb_low) = fx.get("rgb_low")?;
    let (_, alpha_low) = fx.get("alpha_low")?;
    let (_, fg_low) = fx.get("fg_low")?;
    let (hn, wn, h, w) = (rs[2], rs[3], ls[2], ls[3]);
    let logits: Vec<f32> = alpha_low
        .iter()
        .copied()
        .chain(fg_low.iter().map(|&v| {
            let z = to_linear(v) / KNEE;
            z.asinh() / DENOM
        }))
        .collect();
    let g = Geometry { hn, wn, h0: h, w0: w, h, w };
    let adj = Adjust { gain: 1.0, contrast: 1.0, pivot: 0.18 };
    let out = net.native(
        &gpu,
        &gpu.upload_f32(&logits, &[1, 4, h, w], Dtype::F32)?,
        (&gpu.upload_f32(&gbr(&rgb), &[3, hn, wn], Dtype::F32)?, &gpu.upload_f32(&gbr(&rgb_low), &[3, h, w], Dtype::F32)?),
        g,
        Input { adjust: adj, linear: false },
    )?;
    let out = gpu.download_f32(&out)?;
    let n = hn * wn;
    let fg: Vec<f32> = out[n..].iter().map(|&v| to_srgb((v * DENOM).sinh() * KNEE).clamp(0.0, 1.0)).collect();
    let (ma, xa) = diff(&out[..n], &fx.get("alpha")?.1);
    let (mf, xf) = diff(&fg, &fx.get("fg")?.1);
    let (mb, _) = diff(&fx.get("bilinear")?.1, &fx.get("alpha")?.1);
    println!("alpha mean {ma:.2e} max {xa:.2e} · fg mean {mf:.2e} max {xf:.2e} · bilinear vs reference {mb:.2e}");
    assert!(ma < 1e-5 && xa < 2e-3, "alpha");
    assert!(mf < 1e-5 && xf < 2e-3, "fg");
    Ok(())
}
