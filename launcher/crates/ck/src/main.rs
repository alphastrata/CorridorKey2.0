//! `ck`: key a clip on the GPU.

use std::path::PathBuf;

use argh::FromArgs;
use ck::frame::{Adjust, Look, Output};
use ck::io::Encoder;
use ck::run::{Config, Hints, Models, run};
use ck_gpu::Gpu;
use eyre::{Result, bail};

/// Key a video or frame directory with `CorridorKey` v2.
#[derive(FromArgs)]
struct Args {
    /// input video or frame directory
    #[argh(positional)]
    input: PathBuf,
    /// output directory
    #[argh(option, short = 'o', default = "PathBuf::from(\"output\")")]
    output: PathBuf,
    /// comma-separated outputs: alpha, fg, cutout, checker
    #[argh(option, default = "String::from(\"alpha,fg\")")]
    outputs: String,
    /// CUDA device index
    #[argh(option, default = "0")]
    device: usize,
    /// model weights
    #[argh(option, default = "PathBuf::from(\"weights/corridorkey_v2.safetensors\")")]
    weights: PathBuf,
    /// foundation-model weights (`launcher/parity/export.py`)
    #[argh(option, default = "PathBuf::from(\"weights/vfms.safetensors\")")]
    vfms: PathBuf,
    /// PTX directory
    #[argh(option, default = "PathBuf::from(\"corridorkey/ptx/bin\")")]
    ptx: PathBuf,
    /// frames per stage-1 window
    #[argh(option, default = "80")]
    chunk: usize,
    /// frames between window starts
    #[argh(option, default = "40")]
    stride: usize,
    /// do not carry each window's alpha into the next as a hint
    #[argh(switch)]
    no_carry: bool,
    /// first-frame alpha hint (image)
    #[argh(option)]
    hint: Option<PathBuf>,
    /// alpha hint video or frame directory for every frame
    #[argh(option)]
    hint_video: Option<PathBuf>,
    /// trust in user hints
    #[argh(option, default = "0.95")]
    hint_quality: f32,
    /// trust in carried hints
    #[argh(option, default = "0.95")]
    carry_quality: f32,
    /// key only the first N frames
    #[argh(option)]
    frames: Option<usize>,
    /// frame rate of a frame directory
    #[argh(option, default = "24.0")]
    fps: f64,
    /// despill strength in [0, 1]
    #[argh(option, default = "0.5")]
    despill: f32,
    /// write the cutout in linear light
    #[argh(switch)]
    cutout_linear: bool,
    /// scene-linear gain
    #[argh(option, default = "1.0")]
    brightness: f32,
    /// scene-linear contrast
    #[argh(option, default = "1.0")]
    contrast: f32,
    /// contrast pivot
    #[argh(option, default = "0.18")]
    pivot: f32,
    /// encoder for 3-channel outputs
    #[argh(option, default = "String::from(\"libx264\")")]
    codec: String,
    /// encoder quality
    #[argh(option, default = "12")]
    crf: u32,
    /// encoder preset
    #[argh(option, default = "String::from(\"medium\")")]
    preset: String,
    /// encoder pixel format
    #[argh(option, default = "String::from(\"yuv444p10le\")")]
    pix_fmt: String,
}

fn outputs(list: &str) -> Result<Vec<Output>> {
    list.split(',')
        .map(|s| match s.trim() {
            "alpha" => Ok(Output::Alpha),
            "fg" => Ok(Output::Fg),
            "cutout" => Ok(Output::Cutout),
            "checker" => Ok(Output::Checker),
            other => bail!("unknown output {other:?}"),
        })
        .collect()
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let a: Args = argh::from_env();
    let gpu = Gpu::open(a.device, &a.ptx)?;
    let models = Models::load(&gpu, &a.weights, &a.vfms)?;
    let cfg = Config {
        chunk: a.chunk,
        stride: a.stride.max(1),
        carry: !a.no_carry,
        hint_quality: a.hint_quality,
        carry_quality: a.carry_quality,
        quantum: 128,
        frames: a.frames,
        dir_fps: a.fps,
        outputs: outputs(&a.outputs)?,
        look: Look {
            despill: a.despill,
            linear_cutout: a.cutout_linear,
            adjust: Adjust {
                gain: a.brightness,
                contrast: a.contrast,
                pivot: a.pivot,
            },
        },
        encoder: Encoder {
            codec: a.codec,
            crf: a.crf,
            preset: a.preset,
            pix_fmt: a.pix_fmt,
            threads: 2,
        },
        vfm_batch: 8,
    };
    let hints = Hints {
        first: a.hint,
        video: a.hint_video,
    };
    let start = std::time::Instant::now();
    run(&gpu, &models, &a.input, &a.output, &cfg, &hints)?;
    println!("done in {:.1} s", start.elapsed().as_secs_f64());
    Ok(())
}
