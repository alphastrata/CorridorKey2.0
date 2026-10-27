use std::path::Path;

use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::{Result, ensure};

use crate::frame::{Adjust, Geometry};
use crate::weights::Weights;

const WIDTH: usize = 24;
const DILATIONS: [usize; 4] = [1, 2, 4, 1];
const HALO: usize = 1 + DILATIONS[0] + DILATIONS[1] + DILATIONS[2] + DILATIONS[3];
const BUDGET: usize = 256 << 20;
const EPS: f32 = 5e-4;
const GAIN: f32 = 10.0;
const LO: f32 = 0.004;

struct Layer {
    w: Tensor,
    b: Tensor,
    cin: usize,
    dil: usize,
}

/// The real-time edge refiner (mantelmoritz/CorridorKey-Realtime-EdgeRefiner): restores native edge detail on a
/// matte keyed at reduced resolution, in place of plain upsampling.
pub struct Refiner {
    layers: Vec<Layer>,
    out: (Tensor, Tensor),
    width: usize,
}

fn launch(gpu: &Gpu, name: &str, n: usize, args: &[Arg]) -> Result<()> {
    gpu.launch("ck_ops", name, (n.div_ceil(256) as u32, 1, 1), 256, 0, args)
}

fn adjust(a: Adjust) -> [Arg; 4] {
    let on = (a.gain - 1.0).abs() > 1e-6 || (a.contrast - 1.0).abs() > 1e-6;
    [usize::from(on).into(), Arg::F(a.gain), Arg::F(a.contrast), Arg::F(a.pivot)]
}

impl Refiner {
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self> {
        let w = Weights::open(path)?;
        let layer = |name: &str, dil: usize| -> Result<Layer> {
            let (shape, v) = w.get(&format!("{name}.weight"))?;
            let [cout, cin, 3, 3] = shape[..] else { eyre::bail!("{name}: wants a 3x3 conv, got {shape:?}") };
            ensure!(cout <= WIDTH, "{name}: width {cout} > {WIDTH}");
            let taps: Vec<f32> = (0..cin * 9)
                .flat_map(|k| (0..WIDTH).map(move |o| (k, o)))
                .map(|(k, o)| if o < cout { v[(o * cin + k / 9) * 9 + k % 9] } else { 0.0 })
                .collect();
            let (_, b) = w.get(&format!("{name}.bias"))?;
            let b: Vec<f32> = (0..WIDTH).map(|o| b.get(o).copied().unwrap_or(0.0)).collect();
            Ok(Layer {
                w: gpu.upload_f32(&taps, &[cin * 9, WIDTH], Dtype::F32)?,
                b: gpu.upload_f32(&b, &[WIDTH], Dtype::F32)?,
                cin,
                dil,
            })
        };
        let layers = std::iter::once(layer("inp", 1))
            .chain(DILATIONS.iter().enumerate().map(|(i, &d)| layer(&format!("blocks.{i}"), d)))
            .collect::<Result<Vec<_>>>()?;
        let (shape, ow) = w.get("out.weight")?;
        let width = shape[1];
        let ow: Vec<f32> = (0..4 * WIDTH).map(|i| if i % WIDTH < width { ow[i / WIDTH * width + i % WIDTH] } else { 0.0 }).collect();
        let (_, ob) = w.get("out.bias")?;
        Ok(Self {
            out: (gpu.upload_f32(&ow, &[4, WIDTH], Dtype::F32)?, gpu.upload_f32(&ob, &[4], Dtype::F32)?),
            width,
            layers,
        })
    }

    /// Native `[1, 4, hn, wn]` logits from the model's `[1, 4, h, w]` (padded) logits, the native frame's
    /// `[3, hn, wn]` planes and the reduced `[3, h0, w0]` planes the model saw (both in G, B, R order).
    pub fn native(&self, gpu: &Gpu, logits: &Tensor, native: &Tensor, low: &Tensor, g: Geometry, linear: bool, adj: Adjust) -> Result<Tensor> {
        let (h, w) = (g.h0, g.w0);
        let n = h * w;
        let crop = gpu.empty(&[4, h, w], Dtype::F32)?;
        let dims = [4, g.h, g.w, h, w, (g.h - h) / 2, (g.w - w) / 2];
        launch(gpu, "crop", 4 * n, &[[logits, &crop].map(Arg::from).as_slice(), &dims.map(Arg::from)].concat())?;
        let lows = gpu.empty(&[7, h, w], Dtype::F32)?;
        let args = [[low, &crop, &lows].map(Arg::from).as_slice(), &[h, w, usize::from(linear)].map(Arg::from), &adjust(adj)].concat();
        launch(gpu, "refine_low", n, &args)?;

        let ab = gpu.empty(&[4, h, w], Dtype::F32)?;
        launch(gpu, "guided", n, &[(&lows).into(), (&ab).into(), h.into(), w.into(), Arg::F(EPS)])?;
        let coef = gpu.empty(&[4, h, w], Dtype::F32)?;
        launch(gpu, "box_mean", 4 * n, &[(&ab).into(), (&coef).into(), 4usize.into(), h.into(), w.into(), 2usize.into()])?;
        let pre = gpu.empty(&[1, h, w], Dtype::F32)?;
        launch(gpu, "band", n, &[(&lows).into(), (&pre).into(), h.into(), w.into(), Arg::F(GAIN), Arg::F(LO)])?;
        let band = gpu.empty(&[1, h, w], Dtype::F32)?;
        launch(gpu, "box_mean", n, &[(&pre).into(), (&band).into(), 1usize.into(), h.into(), w.into(), 1usize.into()])?;
        let bg = background(gpu, &lows, (h, w))?;

        let (hn, wn) = (g.hn, g.wn);
        let mask = gpu.download_f32(&band)?;
        let out = gpu.empty(&[1, 4, hn, wn], Dtype::F32)?;
        let core = (BUDGET / (4 * wn * (11 + 2 * WIDTH))).saturating_sub(2 * HALO).max(16).min(hn);
        let mut hid = (gpu.empty(&[WIDTH * (core + 2 * HALO) * wn], Dtype::F32)?, gpu.empty(&[WIDTH * (core + 2 * HALO) * wn], Dtype::F32)?);
        let x = gpu.empty(&[11 * (core + 2 * HALO) * wn], Dtype::F32)?;
        (0..hn).step_by(core).try_for_each(|c0| -> Result<()> {
            let take = core.min(hn - c0);
            let y0 = c0.saturating_sub(HALO);
            let rows = (c0 + take + HALO).min(hn) - y0;
            let (ly0, ly1) = (y0 * h / hn, ((y0 + rows) * h).div_ceil(hn).min(h));
            let net = mask[ly0.saturating_sub(1) * w..(ly1 + 1).min(h) * w].iter().any(|&b| b > 0.0);
            let plane = rows * wn;
            let args = [[native, &lows, &coef, &bg, &x].map(Arg::from).as_slice(), &[y0, rows, hn, wn, h, w, usize::from(linear)].map(Arg::from)].concat();
            launch(gpu, "refine_in", plane, &args)?;
            if net {
                self.layers.iter().enumerate().try_for_each(|(i, l)| {
                    let src = if i == 0 { &x } else { &hid.0 };
                    let args = [[src, &hid.1, &l.w, &l.b].map(Arg::from).as_slice(), &[l.cin, self.width, rows, wn, l.dil, usize::from(i > 0)].map(Arg::from)].concat();
                    launch(gpu, "refine_conv", plane, &args)?;
                    std::mem::swap(&mut hid.0, &mut hid.1);
                    Ok::<_, eyre::Report>(())
                })?;
            }
            let dims = [rows, c0 - y0, take, y0, hn, wn, h, w, self.width, usize::from(net)];
            let args = [[&x, &hid.0, &self.out.0, &self.out.1, &band, &out].map(Arg::from).as_slice(), &dims.map(Arg::from), &adjust(adj)].concat();
            launch(gpu, "refine_out", take * wn, &args)
        })?;
        Ok(out)
    }
}

fn background(gpu: &Gpu, lows: &Tensor, (h, w): (usize, usize)) -> Result<Tensor> {
    let seed = gpu.empty(&[4, h, w], Dtype::F32)?;
    launch(gpu, "bg_seed", h * w, &[lows.into(), (&seed).into(), (h * w).into()])?;
    let mut levels = vec![(seed, (h, w))];
    while levels.last().is_some_and(|(_, (lh, lw))| *lh.min(lw) > 4) {
        let (prev, (lh, lw)) = levels.last().ok_or_else(|| eyre::eyre!("no level"))?;
        let size = (lh.div_ceil(2), lw.div_ceil(2));
        let next = gpu.empty(&[4, size.0, size.1], Dtype::F32)?;
        launch(gpu, "pool2", 4 * size.0 * size.1, &[prev.into(), (&next).into(), (*lh).into(), (*lw).into()])?;
        levels.push((next, size));
    }
    let (top, _) = levels.last().ok_or_else(|| eyre::eyre!("no level"))?;
    let v = gpu.download_f32(top)?;
    let m = v.len() / 4;
    let den = v[3 * m..].iter().map(|&x| f64::from(x)).sum::<f64>().max(1e-6);
    let mean: Vec<f32> = (0..3).map(|c| (v[c * m..(c + 1) * m].iter().map(|&x| f64::from(x)).sum::<f64>() / den) as f32).collect();
    let seed = (gpu.upload_f32(&mean, &[3, 1, 1], Dtype::F32)?, (1, 1));
    levels.iter().enumerate().rev().try_fold(seed, |(est, (eh, ew)), (i, (lvl, (lh, lw)))| {
        let out = gpu.empty(&[3, *lh, *lw], Dtype::F32)?;
        let args = [&est, lvl, &out].map(Arg::from);
        let dims = [eh, ew, *lh, *lw, usize::from(i == 0)].map(Arg::from);
        launch(gpu, "pull", 3 * lh * lw, &[args.as_slice(), &dims].concat())?;
        Ok((out, (*lh, *lw)))
    }).map(|(t, _)| t)
}
