//! `ck`: key a clip on the GPU.

use std::path::PathBuf;

use argh::FromArgs;
use ck::farm::{Assets, run};
use ck::frame::{Adjust, Look, Output};
use ck::run::{Config, Hints};
use ck::write::{Encoder, Exr};
use ck_gpu::Gpu;
use eyre::{Result, bail, ensure};

/// Key a video or frame directory with `CorridorKey` v2.
#[derive(FromArgs)]
struct Args {
    /// input videos or frame directories
    #[argh(positional)]
    inputs: Vec<PathBuf>,
    /// output directory
    #[argh(option, short = 'o', default = "PathBuf::from(\"output\")")]
    output: PathBuf,
    /// comma-separated outputs: alpha, fg, cutout, checker
    #[argh(option, default = "String::from(\"alpha,fg\")")]
    outputs: String,
    /// CUDA devices to use, e.g. `0` or `0,1` (default: every device)
    #[argh(option, from_str_fn(devices))]
    device: Vec<Vec<usize>>,
    /// model weights
    #[argh(option, default = "PathBuf::from(\"weights/corridorkey_v2.safetensors\")")]
    weights: PathBuf,
    /// foundation-model weights (`launcher/parity/export.py`)
    #[argh(option, default = "PathBuf::from(\"weights/vfms.safetensors\")")]
    vfms: PathBuf,
    /// PTX directory
    #[argh(option, default = "PathBuf::from(\"corridorkey/ptx/bin\")")]
    ptx: PathBuf,
    /// scale of the frames the model sees, by Lanczos-4 (outputs stay native)
    #[argh(option, default = "1.0")]
    model_scale: f64,
    /// frames per stage-1 window, or `auto` to pick it (and the stride) per clip from free GPU memory
    #[argh(option, default = "Some(80)", from_str_fn(window))]
    chunk: Option<usize>,
    /// frames between window starts
    #[argh(option, default = "40")]
    stride: usize,
    /// model frames are padded to multiples of this
    #[argh(option, default = "128")]
    quantum: usize,
    /// frames per foundation-model batch
    #[argh(option, default = "8")]
    vfm_batch: usize,
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
    /// float (EXR) input frames are scene-linear
    #[argh(switch)]
    linear: bool,
    /// write EXR sequences: half or float
    #[argh(option)]
    exr: Option<String>,
    /// scene-linear gain
    #[argh(option, default = "1.0")]
    brightness: f32,
    /// scene-linear contrast
    #[argh(option, default = "1.0")]
    contrast: f32,
    /// contrast pivot
    #[argh(option, default = "0.18")]
    pivot: f32,
    /// encoder for 3-channel outputs (`hevc_nvenc` encodes on the GPU, far faster at 4K)
    #[argh(option, default = "String::from(\"libx264\")")]
    codec: String,
    /// encoder quality (`-crf`, or `-cq` for NVENC)
    #[argh(option, default = "12")]
    crf: u32,
    /// encoder preset
    #[argh(option, default = "String::from(\"medium\")")]
    preset: String,
    /// encoder threads per output (0: ffmpeg decides, which at 4K can take over 13 GB of RAM per encoder)
    #[argh(option, default = "8")]
    threads: usize,
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

fn devices(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|d| d.trim().parse().map_err(|_| format!("--device wants indices like 0 or 0,1, got {s:?}")))
        .collect()
}

fn window(s: &str) -> Result<Option<usize>, String> {
    match s {
        "auto" => Ok(None),
        n => match n.parse() {
            Ok(0) | Err(_) => Err(format!("--chunk wants a positive frame count or `auto`, got {n:?}")),
            Ok(n) => Ok(Some(n)),
        },
    }
}

fn exr(s: &str) -> Result<Exr> {
    match s {
        "half" => Ok(Exr::Half),
        "float" => Ok(Exr::Float),
        other => bail!("unknown EXR type {other:?}"),
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let a: Args = argh::from_env();
    ensure!(!a.inputs.is_empty(), "no inputs");
    if let Some(p) = a.inputs.iter().find(|p| !p.exists()) {
        bail!("{}: no such file or directory", p.display());
    }
    ensure!(
        a.inputs.len() == 1 || (a.hint.is_none() && a.hint_video.is_none()),
        "hints need a single input"
    );
    let devices = if a.device.is_empty() {
        (0..Gpu::count()?).collect()
    } else {
        a.device.concat()
    };
    let chunk = a.chunk.unwrap_or(80);
    ensure!((1..=chunk).contains(&a.stride), "want 0 < stride <= chunk");
    ensure!(a.quantum > 0 && a.quantum % 64 == 0, "--quantum must be a multiple of 64");
    ensure!(a.vfm_batch > 0, "--vfm-batch must be positive");
    let cfg = Config {
        chunk,
        stride: a.stride,
        auto: a.chunk.is_none(),
        carry: !a.no_carry,
        hint_quality: a.hint_quality,
        carry_quality: a.carry_quality,
        quantum: a.quantum,
        scale: a.model_scale,
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
        linear_input: a.linear,
        exr: a.exr.as_deref().map(exr).transpose()?,
        encoder: Encoder {
            codec: a.codec,
            crf: a.crf,
            preset: a.preset,
            pix_fmt: a.pix_fmt,
            threads: a.threads,
        },
        vfm_batch: a.vfm_batch,
    };
    let hints = Hints {
        first: a.hint,
        video: a.hint_video,
    };
    let start = std::time::Instant::now();
    let assets = Assets {
        weights: a.weights,
        vfms: a.vfms,
        ptx: a.ptx,
    };
    run(&assets, &devices, &a.inputs, &a.output, &cfg, &hints)?;
    println!("done in {:.1} s", start.elapsed().as_secs_f64());
    Ok(())
}
