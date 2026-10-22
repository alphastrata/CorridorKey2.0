//! A clip's lead: Hann-windowed stage 1 on one GPU (the aux alpha carried as the next window's hint), foundation
//! features once per frame (RVM recurrent over the clip), batons blended per frame and handed on as stage-2 jobs.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::{OptionExt, Result, ensure};

use crate::farm::{Job, Meta};
use crate::frame::{self, Encoding};
use crate::io::{Plane, Source};
use crate::ops::{axpy, probe};
use crate::run::{Config, Hints, Models, starts, taper};
use crate::stage1::Inputs;

#[derive(Default)]
struct Frame {
    raw: Vec<u8>,
    hint: Option<Vec<u8>>,
    feats: Option<[Tensor; 3]>,
    carry: Option<Tensor>,
    acc: Option<Tensor>,
    weight: f32,
}

pub struct Clip<'a> {
    gpu: &'a Gpu,
    m: &'a Models,
    cfg: &'a Config,
    meta: Arc<Meta>,
    frames: BTreeMap<usize, Frame>,
    decoded: usize,
    featured: usize,
    state: Option<Tensor>,
    rgb: flume::Receiver<Result<Vec<u8>>>,
    hints: Option<flume::Receiver<Result<Vec<u8>>>>,
    first: Option<Vec<u8>>,
    unit: (Tensor, Tensor),
}

pub(crate) fn upload(gpu: &Gpu, parts: &[&[u8]], shape: &[usize], dtype: Dtype) -> Result<Tensor> {
    gpu.upload(&parts.concat(), shape, dtype)
}

impl<'a> Clip<'a> {
    /// Opens `src` (and any hints) for keying on `gpu`; finished frames go to `meta`.
    ///
    /// # Errors
    /// When a hint cannot be opened or does not match the clip.
    pub fn open(
        (gpu, m, cfg): (&'a Gpu, &'a Models, &'a Config),
        src: &Source,
        total: usize,
        hints: &Hints,
        meta: Arc<Meta>,
    ) -> Result<Self> {
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
        Ok(Self {
            gpu,
            m,
            cfg,
            meta,
            frames: BTreeMap::new(),
            decoded: 0,
            featured: 0,
            state: None,
            rgb: src.decode(Plane::Rgb, total),
            hints: hints.video.as_deref().map(|p| side(p, total, false)).transpose()?,
            first: hints.first.as_deref().map(|p| side(p, 1, true)?.recv()?).transpose()?,
            unit: (gpu.zeros(&[1], Dtype::F32)?, gpu.upload_f32(&[1.0], &[1], Dtype::F32)?),
        })
    }

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
        let g = self.meta.g;
        upload(
            self.gpu,
            &parts.collect::<Result<Vec<_>>>()?,
            &[range.len(), 3 * g.hn * g.wn],
            self.meta.pix,
        )
    }

    fn features(&mut self, until: usize) -> Result<()> {
        let g = self.meta.g;
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
        let (g, plane) = (self.meta.g, self.meta.g.h * self.meta.g.w);
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
        let (h, w) = (self.meta.g.h, self.meta.g.w);
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

    fn window(&mut self, (t0, t1): (usize, usize), (first, last): (bool, bool)) -> Result<()> {
        self.read(t1)?;
        self.features(t1)?;
        probe(self.gpu, "features");
        let (g, t) = (self.meta.g, t1 - t0);
        let plane = g.h * g.w;
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
            x,
            quality: &quality,
            features: [&feats[0], &feats[1], &feats[2]],
        };
        probe(self.gpu, "inputs");
        let out = self.m.s1.forward(self.gpu, &self.m.cache, inputs)?;
        drop((feats, quality));
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
        probe(self.gpu, "blend");
        Ok(())
    }

    /// Frames before `limit`, fully blended, as stage-2 jobs.
    fn flush(&mut self, limit: usize, submit: &mut impl FnMut(Job) -> Result<()>) -> Result<()> {
        let ready: Vec<usize> = self.frames.range(..limit).map(|(k, _)| *k).collect();
        ready.into_iter().try_for_each(|k| {
            let f = self.frames.remove(&k).ok_or_eyre("frame gone")?;
            let acc = f.acc.ok_or_eyre("frame never blended")?;
            submit(Job {
                meta: self.meta.clone(),
                frame: k,
                baton: self.gpu.download(&acc)?,
                scale: 1.0 / f.weight.max(1e-6),
                raw: f.raw,
            })
        })
    }

    /// Every window in turn, each flushing what it completes through `submit`.
    ///
    /// # Errors
    /// On decode or GPU failure, or when `submit` fails.
    pub fn run(mut self, total: usize, mut submit: impl FnMut(Job) -> Result<()>) -> Result<()> {
        let (chunk, stride) = (self.cfg.chunk, self.cfg.stride);
        let s = starts(total, chunk, stride);
        s.iter().enumerate().try_for_each(|(i, &t0)| {
            let t1 = (t0 + chunk).min(total);
            self.window((t0, t1), (i == 0, i + 1 == s.len()))?;
            self.flush(s.get(i + 1).copied().unwrap_or(total), &mut submit)?;
            probe(self.gpu, "stage 2");
            println!("  {} window {}/{} [{t0}:{t1}]", self.meta.name, i + 1, s.len());
            Ok(())
        })
    }
}
