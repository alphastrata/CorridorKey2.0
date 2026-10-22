//! A clip end to end: Hann-windowed stage 1 (the aux alpha carried as the next window's hint), foundation features
//! once per frame (RVM recurrent over the clip), batons blended per frame, then stage 2 and the renders as each frame
//! completes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::{OptionExt, Result, ensure};

use crate::attn::Cache;
use crate::frame::{self, Encoding, Geometry, Input, Look, Output};
use crate::io::{Plane, Source};
use crate::ops::{axpy, mem};
use crate::stage1::{Inputs, Stage1};
use crate::stage2::{Baton, Stage2};
use crate::vfm::Vfms;
use crate::weights::{Loader, Weights};
use crate::write::{Encoder, Exr, Format, Writer};

pub struct Models {
    pub vfms: Vfms,
    pub s1: Stage1,
    pub s2: Stage2,
    pub cache: Cache,
}

impl Models {
    /// # Errors
    /// When a file or weight is missing.
    pub fn load(gpu: &Gpu, model: &Path, vfms: &Path) -> Result<Self> {
        let weights = Weights::open(model)?;
        let ld = |prefix| Loader {
            gpu,
            weights: &weights,
            prefix,
            half: Dtype::Bf16,
        };
        Ok(Self {
            vfms: Vfms::load(gpu, &Weights::open(vfms)?)?,
            s1: Stage1::load(&ld("stage1."))?,
            s2: Stage2::load(&ld("stage2."))?,
            cache: Cache::default(),
        })
    }
}

pub struct Config {
    pub chunk: usize,
    pub stride: usize,
    pub carry: bool,
    pub hint_quality: f32,
    pub carry_quality: f32,
    pub quantum: usize,
    /// Lanczos scale of the frames the model sees; outputs stay native.
    pub scale: f64,
    pub frames: Option<usize>,
    pub dir_fps: f64,
    pub outputs: Vec<Output>,
    pub look: Look,
    /// Float frames are scene-linear (else display-coded, clipped to `[0, 1]`).
    pub linear_input: bool,
    pub encoder: Encoder,
    /// EXR sequences in place of videos.
    pub exr: Option<Exr>,
    pub vfm_batch: usize,
}

/// Optional user hints: a first-frame matte, or a matte video for every frame.
#[derive(Default)]
pub struct Hints {
    pub first: Option<PathBuf>,
    pub video: Option<PathBuf>,
}

/// Window starts covering `total` frames, the last flush with the end.
#[must_use]
pub fn starts(total: usize, chunk: usize, stride: usize) -> Vec<usize> {
    if total <= chunk {
        return vec![0];
    }
    let mut s: Vec<usize> = (0..=total - chunk).step_by(stride).collect();
    if s.last() != Some(&(total - chunk)) {
        s.push(total - chunk);
    }
    s
}

/// Raised-cosine ramps over `overlap` frames at a window's shared ends.
#[must_use]
pub fn taper(t: usize, overlap: usize, first: bool, last: bool) -> Vec<f32> {
    let mut w = vec![1.0f32; t];
    if overlap == 0 || t <= 1 {
        return w;
    }
    let n = overlap.min(t);
    let ramp = |k: usize| (0.5 * (1.0 - (std::f64::consts::PI * (k + 1) as f64 / (overlap + 1) as f64).cos())) as f32;
    if !first {
        (0..n).for_each(|k| w[k] = ramp(k));
    }
    if !last {
        (0..n).for_each(|k| w[t - n + k] = ramp(n - 1 - k));
    }
    w
}

#[derive(Default)]
struct Frame {
    raw: Vec<u8>,
    hint: Option<Vec<u8>>,
    feats: Option<[Tensor; 3]>,
    carry: Option<Tensor>,
    acc: Option<Tensor>,
    weight: f32,
}

struct Clip<'a> {
    gpu: &'a Gpu,
    m: &'a Models,
    cfg: &'a Config,
    g: Geometry,
    frames: BTreeMap<usize, Frame>,
    decoded: usize,
    featured: usize,
    state: Option<Tensor>,
    pix: Dtype,
    rgb: flume::Receiver<Result<Vec<u8>>>,
    hints: Option<flume::Receiver<Result<Vec<u8>>>>,
    first: Option<Vec<u8>>,
    unit: (Tensor, Tensor),
    writers: Vec<(Output, Writer)>,
}

fn upload(gpu: &Gpu, parts: &[&[u8]], shape: &[usize], dtype: Dtype) -> Result<Tensor> {
    gpu.upload(&parts.concat(), shape, dtype)
}

impl Clip<'_> {
    fn read(&mut self, until: usize) -> Result<()> {
        while self.decoded < until {
            let raw = self.rgb.recv()??;
            let hint = self.hints.as_ref().map(|h| h.recv()?).transpose()?;
            self.frames.insert(
                self.decoded,
                Frame {
                    raw,
                    hint,
                    ..Frame::default()
                },
            );
            self.decoded += 1;
        }
        Ok(())
    }

    fn raw(&self, range: std::ops::Range<usize>) -> Result<Tensor> {
        let parts = range
            .clone()
            .map(|t| Ok(self.frames.get(&t).ok_or_eyre("frame gone")?.raw.as_slice()));
        let g = self.g;
        upload(
            self.gpu,
            &parts.collect::<Result<Vec<_>>>()?,
            &[range.len(), 3 * g.hn * g.wn],
            self.pix,
        )
    }

    fn features(&mut self, until: usize) -> Result<()> {
        let g = self.g;
        while self.featured < until {
            let n = (until - self.featured).min(self.cfg.vfm_batch);
            let raw = self.raw(self.featured..self.featured + n)?;
            let display = self.gpu.empty(&[n, 3, g.h, g.w], Dtype::F32)?;
            frame::prep(
                self.gpu,
                &raw,
                &display,
                g,
                3 * g.h * g.w,
                Encoding::Display,
                self.cfg.input(),
            )?;
            let f = self.m.vfms.forward(self.gpu, &display, self.state.as_ref())?;
            (0..n).try_for_each(|j| -> Result<()> {
                let feats = [f.cradio.rows(j, 1)?, f.moge.rows(j, 1)?, f.rvm.rows(j, 1)?];
                self.frames
                    .get_mut(&(self.featured + j))
                    .ok_or_eyre("frame gone")?
                    .feats = Some(feats);
                Ok(())
            })?;
            self.state = Some(f.state);
            self.featured += n;
        }
        Ok(())
    }

    /// Hints into channel 3 of `x`, and each frame's quality (−1 throughout when no frame has one).
    fn hints(&self, x: &Tensor, t0: usize, t: usize) -> Result<Tensor> {
        let (g, plane) = (self.g, self.g.h * self.g.w);
        let mut quality = vec![None; t];
        if self.hints.is_some() {
            let parts = (t0..t0 + t)
                .map(|f| self.frames[&f].hint.as_deref().ok_or_eyre("hint frame missing"))
                .collect::<Result<Vec<_>>>()?;
            let raw = upload(self.gpu, &parts, &[t, g.hn, g.wn], Dtype::U16)?;
            frame::hint(self.gpu, &raw, &x.at(3 * plane, &[(4 * t - 3) * plane])?, g, 4 * plane)?;
            quality.fill(Some(self.cfg.hint_quality));
        }
        (0..t).try_for_each(|j| -> Result<()> {
            if let Some(c) = &self.frames[&(t0 + j)].carry {
                self.gpu.copy_into(c, &x.at((4 * j + 3) * plane, &[plane])?)?;
                quality[j] = Some(self.cfg.carry_quality);
            }
            Ok(())
        })?;
        if let (Some(first), 0) = (&self.first, t0) {
            let raw = upload(self.gpu, &[first], &[1, g.hn, g.wn], Dtype::U16)?;
            frame::hint(self.gpu, &raw, &x.at(3 * plane, &[plane])?, g, 4 * plane)?;
            quality[0] = Some(self.cfg.hint_quality);
        }
        let none = if quality.iter().any(Option::is_some) { 0.0 } else { -1.0 };
        let q: Vec<f32> = quality.into_iter().map(|q| q.unwrap_or(none)).collect();
        self.gpu.upload_f32(&q, &[t], Dtype::F32)
    }

    /// The aux alpha of window frame `j` at full size, clamped to `[0, 1]`.
    fn carry(&self, aux: &Tensor, j: usize) -> Result<Tensor> {
        let (h, w) = (self.g.h, self.g.w);
        let (hh, hw) = (h / 2, w / 2);
        let out = self.gpu.empty(&[h, w], Dtype::F32)?;
        let src = aux.at(4 * j * hh * hw, &[hh * hw])?;
        let dims = [1, 1, hh, hw, h, w].map(Arg::from);
        let args = [
            [&src, &self.unit.0, &self.unit.1, &out].map(Arg::from).as_slice(),
            &dims,
            &[Arg::F(0.5), Arg::F(0.5), Arg::U(1), Arg::F(0.0), Arg::F(1.0)],
        ]
        .concat();
        self.gpu
            .launch("ck_ops", "resize", ((h * w).div_ceil(256) as u32, 1, 1), 256, 0, &args)?;
        Ok(out)
    }

    fn window(&mut self, (t0, t1): (usize, usize), next: usize, (first, last): (bool, bool)) -> Result<()> {
        self.read(t1)?;
        self.features(t1)?;
        mem(self.gpu, "features");
        let (g, t, plane) = (self.g, t1 - t0, self.g.h * self.g.w);
        let x = self.gpu.zeros(&[t, 4, g.h, g.w], Dtype::F32)?;
        frame::prep(
            self.gpu,
            &self.raw(t0..t1)?,
            &x,
            g,
            4 * plane,
            Encoding::Model,
            self.cfg.input(),
        )?;
        let quality = self.hints(&x, t0, t)?;
        let feats = (0..3)
            .map(|k| {
                let one = &self.frames[&t0].feats.as_ref().ok_or_eyre("features missing")?[k];
                let all = self
                    .gpu
                    .empty(&[[t].as_slice(), &one.shape[1..]].concat(), Dtype::F32)?;
                (0..t).try_for_each(|j| {
                    let f = &self.frames[&(t0 + j)].feats.as_ref().ok_or_eyre("features missing")?[k];
                    self.gpu.copy_into(f, &all.rows(j, 1)?)
                })?;
                Ok(all)
            })
            .collect::<Result<Vec<_>>>()?;
        let inputs = Inputs {
            b: 1,
            t,
            x: &x,
            quality: &quality,
            features: [&feats[0], &feats[1], &feats[2]],
        };
        mem(self.gpu, "inputs");
        let out = self.m.s1.forward(self.gpu, &self.m.cache, &inputs)?;
        let baton = out.baton.reshape(&[t, out.baton.numel() / t])?;
        let weights = taper(t, self.cfg.chunk.saturating_sub(self.cfg.stride), first, last);
        (0..t).try_for_each(|j| -> Result<()> {
            let carry = self.cfg.carry.then(|| self.carry(&out.aux, j)).transpose()?;
            let f = self.frames.get_mut(&(t0 + j)).ok_or_eyre("frame gone")?;
            if carry.is_some() {
                f.carry = carry;
            }
            let acc = match f.acc.take() {
                Some(a) => a,
                None => self.gpu.zeros(&[1, baton.shape[1]], Dtype::F32)?,
            };
            axpy(self.gpu, &acc, &baton.rows(j, 1)?, weights[j])?;
            f.acc = Some(acc);
            f.weight += weights[j];
            Ok(())
        })?;
        mem(self.gpu, "blend");
        self.flush(next, &x, t0)?;
        mem(self.gpu, "stage 2");
        Ok(())
    }

    fn flush(&mut self, limit: usize, x: &Tensor, t0: usize) -> Result<()> {
        let (g, ready): (Geometry, Vec<usize>) = (self.g, self.frames.range(..limit).map(|(k, _)| *k).collect());
        ready.into_iter().try_for_each(|k| {
            let f = self.frames.remove(&k).ok_or_eyre("frame gone")?;
            let acc = f
                .acc
                .ok_or_eyre("frame never blended")?
                .reshape(&[1, g.h / 16, g.w / 16, 512])?;
            let rgb = x.rows(k - t0, 1)?.at(0, &[1, 3, g.h, g.w])?;
            let baton = Baton::Nhwc(&acc, 1.0 / f.weight.max(1e-6));
            let logits = self.m.s2.forward(self.gpu, &self.m.cache, &rgb, baton)?;
            self.writers.iter().try_for_each(|&(o, ref w)| {
                let img = match self.cfg.exr {
                    Some(t) => frame::render_exr(self.gpu, &logits, g, o, self.cfg.look, t == Exr::Half)?,
                    None => frame::render(self.gpu, &logits, g, o, self.cfg.look)?,
                };
                w.write(self.gpu.download(&img)?)
            })
        })
    }
}

fn clip_name(input: &Path) -> String {
    let stem = input.file_stem().map_or_default(|s| s.to_string_lossy().into_owned());
    match stem.to_lowercase().as_str() {
        "input" | "frames" | "rgb" => input
            .parent()
            .and_then(Path::file_name)
            .map_or(stem, |p| p.to_string_lossy().into_owned()),
        _ => stem,
    }
}

fn stem(o: Output, despill: f32) -> String {
    match o {
        Output::Alpha => "alpha".into(),
        Output::Fg => "fg".into(),
        Output::Cutout => "cutout".into(),
        Output::Checker => format!("checker_comp_despill{:03}", (despill * 100.0) as u32),
    }
}

impl Config {
    fn input(&self) -> Input {
        Input {
            adjust: self.look.adjust,
            linear: self.linear_input,
        }
    }

    /// `{clip}_{stem}.mp4|mov`, or frames `{stem}_000000.exr` in `{clip}_{stem}_exr/`.
    fn writer(&self, out_dir: &Path, clip: &str, src: &Source, o: Output) -> Result<Writer> {
        let stem = stem(o, self.look.despill);
        let size = (src.w, src.h);
        if let Some(t) = self.exr {
            let dir = out_dir.join(format!("{clip}_{stem}_exr"));
            std::fs::create_dir_all(&dir)?;
            return Writer::spawn(
                &dir.join(format!("{stem}_%06d.exr")),
                src.fps,
                size,
                o.planes(),
                Format::Exr(t),
            );
        }
        let ext = if o == Output::Cutout { "mov" } else { "mp4" };
        let format = Format::Video {
            enc: &self.encoder,
            linear: o == Output::Cutout && self.look.linear_cutout,
        };
        Writer::spawn(
            &out_dir.join(format!("{clip}_{stem}.{ext}")),
            src.fps,
            size,
            o.channels(),
            format,
        )
    }
}

/// Keys `input` (a video or frame directory) into `out_dir`, one file per requested output.
///
/// # Errors
/// On decode, GPU or encode failure.
pub fn run(gpu: &Gpu, m: &Models, input: &Path, out_dir: &Path, cfg: &Config, hints: &Hints) -> Result<()> {
    let src = Source::open(input, cfg.dir_fps)?;
    let total = cfg.frames.map_or(src.frames, |n| n.min(src.frames));
    ensure!(total > 0, "{}: no frames", input.display());
    let g = Geometry::scaled(src.h, src.w, cfg.scale, cfg.quantum);
    if src.float() && !cfg.linear_input {
        println!("warning: float frames read as display-coded and clipped to [0, 1]; pass --linear for scene-linear");
    }
    std::fs::create_dir_all(out_dir)?;
    let name = clip_name(input);
    let writers = cfg
        .outputs
        .iter()
        .map(|&o| Ok((o, cfg.writer(out_dir, &name, &src, o)?)))
        .collect::<Result<Vec<_>>>()?;
    let side = |p: &Path, n: usize, first: bool| -> Result<flume::Receiver<Result<Vec<u8>>>> {
        let s = Source::open(p, cfg.dir_fps)?;
        ensure!(
            (s.w, s.h) == (src.w, src.h),
            "{}: hint size {}x{} != {}x{}",
            p.display(),
            s.w,
            s.h,
            src.w,
            src.h
        );
        Ok(s.decode(if first && s.alpha() { Plane::Alpha } else { Plane::Grey }, n))
    };
    let first = hints.first.as_deref().map(|p| side(p, 1, true)?.recv()?).transpose()?;
    let mut clip = Clip {
        gpu,
        m,
        cfg,
        g,
        frames: BTreeMap::new(),
        decoded: 0,
        featured: 0,
        state: None,
        pix: src.dtype(),
        rgb: src.decode(Plane::Rgb, total),
        hints: hints.video.as_deref().map(|p| side(p, total, false)).transpose()?,
        first,
        unit: (gpu.zeros(&[1], Dtype::F32)?, gpu.upload_f32(&[1.0], &[1], Dtype::F32)?),
        writers,
    };
    let s = starts(total, cfg.chunk, cfg.stride);
    println!(
        "{name}: {total} frames {}x{} → {}x{}, {} windows",
        src.w,
        src.h,
        g.w,
        g.h,
        s.len()
    );
    s.iter().enumerate().try_for_each(|(i, &t0)| {
        let next = s.get(i + 1).copied().unwrap_or(total);
        clip.window((t0, (t0 + cfg.chunk).min(total)), next, (i == 0, i + 1 == s.len()))?;
        println!("  window {}/{} [{t0}:{}]", i + 1, s.len(), (t0 + cfg.chunk).min(total));
        Ok::<_, eyre::Report>(())
    })?;
    gpu.sync()?;
    clip.writers.into_iter().try_for_each(|(_, w)| w.finish())
}
