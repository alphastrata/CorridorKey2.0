use crate::convert::{Metric, Narrow, Quant};
use std::path::{Path, PathBuf};

use ck_gpu::Gpu;
use eyre::Result;

use crate::attn::Cache;
use crate::frame::{Input, Look, Output};
use crate::io::Source;
use crate::refine::Refiner;
use crate::stage1::Stage1;
use crate::stage2::Stage2;
use crate::vfm::Vfms;
use crate::weights::{Loader, Weights};
use crate::write::{Encoder, Exr, Format, Writer};

pub struct Front {
    pub vfms: Vfms,
    pub s1: Stage1,
}

pub struct Models {
    pub s2: Stage2,
    pub refiner: Option<Refiner>,
    pub cache: Cache,
    front: std::cell::OnceCell<Front>,
    paths: (PathBuf, PathBuf),
}

fn loader<'a>(gpu: &'a Gpu, weights: &'a Weights, prefix: &'a str) -> Loader<'a> {
    Loader {
        gpu,
        weights,
        prefix,
        half: gpu.half(),
    }
}

impl Models {
    pub fn load(gpu: &Gpu, model: &Path, vfms: &Path, refiner: Option<&Path>) -> Result<Self> {
        Ok(Self {
            s2: Stage2::load(&loader(gpu, &Weights::open(model)?, "stage2."))?,
            refiner: refiner.map(|p| Refiner::load(gpu, p)).transpose()?,
            cache: Cache::default(),
            front: std::cell::OnceCell::new(),
            paths: (model.to_path_buf(), vfms.to_path_buf()),
        })
    }

    pub fn front(&self, gpu: &Gpu) -> Result<&Front> {
        if let Some(f) = self.front.get() {
            return Ok(f);
        }
        let front = Front {
            vfms: Vfms::load(
                gpu,
                &Weights::vfms(
                    &self.paths.1,

                    &self.paths.0.with_file_name("foundation_stats.safetensors"),
                )?,
            )?,
            s1: Stage1::load(&loader(gpu, &Weights::open(&self.paths.0)?, "stage1."))?,
        };
        Ok(self.front.get_or_init(|| front))
    }
}

#[derive(Clone)]
pub struct Config {
    pub chunk: usize,
    pub stride: usize,
    pub auto: bool,
    pub carry: bool,
    pub hint_quality: f32,
    pub carry_quality: f32,
    pub quantum: usize,
    pub scale: f64,
    pub refine: Option<PathBuf>,
    pub frames: Option<usize>,

    pub dir_fps: f64,
    pub outputs: Vec<Output>,
    pub look: Look,
    pub linear_input: bool,
    pub encoder: Encoder,
    pub exr: Option<Exr>,
    pub vfm_batch: usize,
    pub preview: Option<PreviewTo>,
}

/// Where live-view snapshots go: `<output>.ppm` files, or frames on a socket `ckd` handed down.
#[derive(Clone, Debug)]
pub enum PreviewTo {
    Dir(PathBuf),
    Fd(i32),
}

#[derive(Default)]
pub struct Hints {
    pub first: Option<PathBuf>,
    pub video: Option<PathBuf>,
}

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

#[must_use]
pub fn taper(t: usize, overlap: usize, first: bool, last: bool) -> Vec<f32> {
    let mut w = vec![1.0f32; t];
    if overlap == 0 || t <= 1 {
        return w;
    }
    let n = overlap.min(t);
    let overlap_f = (overlap + 1).f64();
    let ramp = |k: usize| {
        let t = std::f64::consts::PI * f64::from(u32::try_from(k + 1).unwrap_or(u32::MAX)) / overlap_f;
        (0.5 * (1.0 - t.cos())).f32()
    };

    if !first {
        (0..n).for_each(|k| w[k] = ramp(k));
    }
    if !last {
        (0..n).for_each(|k| w[t - n + k] = ramp(n - 1 - k));
    }
    w
}

pub(crate) fn clip_name(input: &Path) -> String {
    let stem = input.file_stem().map_or_default(|s| s.to_string_lossy().into_owned());
    match stem.to_lowercase().as_str() {
        "input" | "frames" | "rgb" => input
            .parent()
            .and_then(Path::file_name)
            .map_or(stem, |p| p.to_string_lossy().into_owned()),
        _ => stem,
    }
}

pub(crate) fn clip_names(inputs: &[PathBuf]) -> Vec<String> {
    let dupes = |names: &[String], n: &String| names.iter().filter(|m| *m == n).count() > 1;
    let base: Vec<String> = inputs.iter().map(|p| clip_name(p.as_path())).collect();
    let named: Vec<String> = inputs
        .iter()
        .zip(&base)
        .map(|(p, n)| match p.parent().and_then(Path::file_name) {
            Some(parent) if dupes(&base, n) => format!("{}_{n}", parent.to_string_lossy()),
            _ => n.clone(),
        })
        .collect();

    named
        .iter()
        .enumerate()
        .map(|(i, n)| {
            if dupes(&named, n) {
                format!("{n}_{i}")
            } else {
                n.clone()
            }
        })
        .collect()
}

fn stem(o: Output, despill: f32) -> String {
    match o {
        Output::Alpha => "alpha".into(),
        Output::Fg => "fg".into(),
        Output::Cutout => "cutout".into(),
        Output::Checker => format!("checker_comp_despill{:03}", (despill * 100.0).clamp(0.0, 100.0).u32()),
    }
}

impl Config {
    pub(crate) fn input(&self) -> Input {
        Input {
            adjust: self.look.adjust,
            linear: self.linear_input,
        }
    }

    /// GiB of GPU memory the NVENC encoders of a `w`×`h` clip hold (about 1.7 per 4K stream); none for others.
    #[must_use]
    pub fn encoder_gib(&self, (w, h): (usize, usize)) -> f64 {
        if self.exr.is_some() || !self.encoder.codec.ends_with("_nvenc") {
            return 0.0;
        }
        let streams = self.outputs.iter().filter(|o| o.channels() == 3).count();
        streams.f64() * 1.7 * (w * h).f64() / (4096.0 * 2160.0)
    }

    /// `{clip}_{stem}.mp4|mov`, or frames `{stem}_000000.exr` in `{clip}_{stem}_exr/`.
    pub(crate) fn writer(&self, out_dir: &Path, clip: &str, src: &Source, o: Output) -> Result<Writer> {
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
