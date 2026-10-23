//! The models, run settings, and the window schedule shared by [`crate::clip`] and [`crate::farm`].

use std::path::{Path, PathBuf};

use ck_gpu::Gpu;
use eyre::Result;

use crate::attn::Cache;
use crate::frame::{Input, Look, Output};
use crate::io::Source;
use crate::stage1::Stage1;
use crate::stage2::Stage2;
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
            half: gpu.half(),
        };
        Ok(Self {
            vfms: Vfms::load(gpu, &Weights::open(vfms)?)?,
            s1: Stage1::load(&ld("stage1."))?,
            s2: Stage2::load(&ld("stage2."))?,
            cache: Cache::default(),
        })
    }
}

#[derive(Clone)]
pub struct Config {
    pub chunk: usize,
    pub stride: usize,
    /// Pick `chunk` and `stride` per clip from the leading GPU's free memory.
    pub auto: bool,
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

/// Output names for `inputs`, unique: a repeated name takes its parent directory's, then its position.
pub(crate) fn clip_names(inputs: &[PathBuf]) -> Vec<String> {
    let dupes = |names: &[String], n: &String| names.iter().filter(|m| *m == n).count() > 1;
    let base: Vec<String> = inputs.iter().map(|p| clip_name(p)).collect();
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
        Output::Checker => format!("checker_comp_despill{:03}", (despill * 100.0) as u32),
    }
}

impl Config {
    pub(crate) fn input(&self) -> Input {
        Input {
            adjust: self.look.adjust,
            linear: self.linear_input,
        }
    }

    /// `{clip}_{stem}.mp4|mov`, or frames `{stem}_000000.exr` in `{clip}_{stem}_exr/`.
    /// GiB of GPU memory the NVENC encoders of a `w`×`h` clip hold (about 1.7 per 4K stream); none for others.
    #[must_use]
    pub fn encoder_gib(&self, (w, h): (usize, usize)) -> f64 {
        if self.exr.is_some() || !self.encoder.codec.ends_with("_nvenc") {
            return 0.0;
        }
        let streams = self.outputs.iter().filter(|o| o.channels() == 3).count();
        streams as f64 * 1.7 * (w * h) as f64 / (4096.0 * 2160.0)
    }

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
