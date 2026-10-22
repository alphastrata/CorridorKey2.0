//! The foundation models against Python's stock bf16 and fp32 reference on real footage (`launcher/parity/dump.py`).

mod common;

use ck::vfm::Vfms;
use ck::weights::Weights;
use ck_gpu::Gpu;
use common::{device, diff, repo, upload};
use eyre::Result;

fn report(name: &str, got: &[f32], fx: &Weights, key: &str) -> Result<((f64, f64), (f64, f64))> {
    let (stock, reference) = (fx.get(&format!("stock.{key}"))?.1, fx.get(&format!("ref.{key}"))?.1);
    let (ours, theirs) = (diff(got, &reference), diff(&stock, &reference));
    println!(
        "{name:9} ours-ref {:.3e}/{:.3e} stock-ref {:.3e}/{:.3e}",
        ours.0, ours.1, theirs.0, theirs.1
    );
    Ok((ours, theirs))
}

#[test]
fn vfm_720() -> Result<()> {
    let root = repo();
    let fx = Weights::open(&root.join("launcher/fixtures/vfm_720.safetensors"))?;
    let gpu = Gpu::open(device(), &root.join("corridorkey/ptx/bin"))?;
    let vfms = Vfms::load(&gpu, &Weights::open(&root.join("weights/vfms.safetensors"))?)?;

    let (blocks, ones) = vfms.blocks();
    blocks.iter().try_for_each(|(tag, list)| {
        (1..list.len()).try_for_each(|i| -> Result<()> {
            let x = upload(&gpu, &fx, &format!("stock.{tag}{}.out", i - 1))?;
            let (t, n) = (x.shape[0], x.shape[1]);
            list[i].forward(&gpu, &x, (t, n), ones)?;
            report(
                &format!("{tag}{i}"),
                &gpu.download_f32(&x)?,
                &fx,
                &format!("{tag}{i}.out"),
            )
            .map(|_| ())
        })
    })?;

    let display = upload(&gpu, &fx, "display")?;
    let f = vfms.forward(&gpu, &display, None)?;
    let worse = [(&f.cradio, "cradio"), (&f.moge, "moge"), (&f.rvm, "rvm")]
        .into_iter()
        .enumerate()
        .map(|(i, (t, name))| {
            let (ours, theirs) = report(name, &gpu.download_f32(t)?, &fx, &format!("out{i}"))?;
            Ok((ours.0 > theirs.0 * 1.1).then_some(name))
        })
        .collect::<Result<Vec<_>>>()?;
    assert!(worse.iter().all(Option::is_none), "less precise than stock: {worse:?}");
    Ok(())
}
