use std::io::IsTerminal;
use std::sync::atomic::AtomicBool;
use std::path::{Path, PathBuf};
use std::time::Instant;

use argh::FromArgs;

#[cfg(unix)]
mod serve;
use ck::farm::{Assets, Session, run_in};
use ck::frame::{Adjust, Look, Output};
use ck::run::{Config, Hints, PreviewTo};
use ck::write::{Encoder, Exr};
use ck_gpu::Gpu;
use color_eyre::config::{HookBuilder, Theme};
use eyre::{Result, bail, ensure, eyre};
use tracing::info;
use tracing_error::ErrorLayer;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

#[derive(FromArgs)]
#[argh(description = "Key a video or frame directory with `CorridorKey` v2.")]
struct Args {
    #[argh(description = "input videos or frame directories")]
    #[argh(positional)]
    inputs: Vec<PathBuf>,
    #[argh(description = "stay resident: take jobs from ckd on fd 4, keeping models loaded between them")]
    #[argh(switch)]
    serve: bool,
    #[argh(description = "output directory")]
    #[argh(option, short = 'o', default = "PathBuf::from(\"output\")")]
    output: PathBuf,

    #[argh(description = "comma-separated outputs: alpha, fg, cutout, checker")]
    #[argh(option, default = "String::from(\"alpha,fg\")")]
    outputs: String,
    #[argh(description = "log filter, e.g. `debug` or `info,ck::time=debug` (default: $CK_LOG, else `info`)")]
    #[argh(option)]
    log: Option<String>,
    #[argh(description = "CUDA devices to use, e.g. `0` or `0,1` (default: every device)")]
    #[argh(option, from_str_fn(devices))]
    device: Vec<Vec<usize>>,

    #[argh(description = "model weights")]
    #[argh(option, default = "PathBuf::from(\"weights/corridorkey_v2.safetensors\")")]
    weights: PathBuf,
    #[argh(description = "foundation models, as `scripts/download_weights.py` fetches them")]
    #[argh(option, default = "PathBuf::from(\"vfms\")")]
    vfms: PathBuf,
    #[argh(description = "PTX directory")]
    #[argh(option, default = "PathBuf::from(\"corridorkey/ptx/bin\")")]
    ptx: PathBuf,

    #[argh(description = "scale of the frames the model sees, by Lanczos-4 (outputs stay native)")]
    #[argh(option, default = "1.0")]
    model_scale: f64,
    #[argh(
        description = "how outputs return to native size below model scale 1: `lanczos` or `refine` (the real-time edge refiner)"
    )]
    #[argh(option, default = "String::from(\"lanczos\")")]
    upscale: String,

    #[argh(description = "edge refiner weights")]
    #[argh(
        option,
        default = "PathBuf::from(\"weights/corridorkey_edge_refiner_v10w.safetensors\")"
    )]
    refiner: PathBuf,
    #[argh(
        description = "frames per stage-1 window; by default (`auto`) picked with the stride per clip from free GPU memory"
    )]
    #[argh(option, default = "None", from_str_fn(window))]
    chunk: Option<usize>,

    #[argh(description = "frames between window starts (default: half the window)")]
    #[argh(option)]
    stride: Option<usize>,
    #[argh(description = "model frames are padded to multiples of this")]
    #[argh(option, default = "128")]
    quantum: usize,

    #[argh(description = "frames per foundation-model batch")]
    #[argh(option, default = "8")]
    vfm_batch: usize,
    #[argh(description = "write downscaled snapshots of the latest frame here (`<output>.ppm`), for a live view")]
    #[argh(option)]
    preview: Option<PathBuf>,
    #[argh(description = "send live-view snapshots over this inherited socket instead (set by ckd)")]
    #[argh(option)]
    preview_fd: Option<i32>,
    #[argh(description = "do not carry each window's alpha into the next as a hint")]
    #[argh(switch)]
    no_carry: bool,
    #[argh(description = "first-frame alpha hint (image)")]
    #[argh(option)]
    hint: Option<PathBuf>,

    #[argh(description = "alpha hint video or frame directory for every frame")]
    #[argh(option)]
    hint_video: Option<PathBuf>,
    #[argh(description = "trust in user hints")]
    #[argh(option, default = "0.95")]
    hint_quality: f32,
    #[argh(description = "trust in carried hints")]
    #[argh(option, default = "0.95")]
    carry_quality: f32,
    #[argh(description = "key only the first N frames")]
    #[argh(option)]
    frames: Option<usize>,
    #[argh(description = "frame rate of a frame directory")]
    #[argh(option, default = "24.0")]
    fps: f64,

    #[argh(description = "despill strength in [0, 1]")]
    #[argh(option, default = "0.5")]
    despill: f32,
    #[argh(description = "write the cutout in linear light")]
    #[argh(switch)]
    cutout_linear: bool,
    #[argh(description = "float (EXR) input frames are scene-linear")]
    #[argh(switch)]
    linear: bool,
    #[argh(description = "write EXR sequences: half or float")]
    #[argh(option)]
    exr: Option<String>,

    #[argh(description = "scene-linear gain")]
    #[argh(option, default = "1.0")]
    brightness: f32,
    #[argh(description = "scene-linear contrast")]
    #[argh(option, default = "1.0")]
    contrast: f32,
    #[argh(description = "contrast pivot")]
    #[argh(option, default = "0.18")]
    pivot: f32,
    #[argh(description = "encoder for 3-channel outputs (`hevc_nvenc` encodes on the GPU, far faster at 4K)")]
    #[argh(option, default = "String::from(\"libx264\")")]
    codec: String,
    #[argh(description = "encoder quality (`-crf`, or `-cq` for NVENC)")]
    #[argh(option, default = "12")]
    crf: u32,

    #[argh(description = "encoder preset")]
    #[argh(option, default = "String::from(\"medium\")")]
    preset: String,
    #[argh(
        description = "encoder threads per output (0: ffmpeg decides, which at 4K can take over 13 GB of RAM per encoder)"
    )]
    #[argh(option, default = "8")]
    threads: usize,
    #[argh(description = "encoder pixel format")]
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
        .map(|d| {
            d.trim()
                .parse()
                .map_err(|_| format!("--device wants indices like 0 or 0,1, got {s:?}"))
        })
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

fn asset(path: &Path, fix: &str) -> Result<PathBuf> {
    let found = std::iter::once(PathBuf::new())
        .chain(std::env::current_exe()?.ancestors().map(PathBuf::from))
        .chain(Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().map(PathBuf::from))
        .map(|dir| dir.join(path))
        .find(|p| p.exists());
    found.ok_or_else(|| {
        eyre!(
            "{}: not found here or beside ck; create it with `{fix}`",
            path.display()
        )
    })
}

struct Since(Instant);

impl FormatTime for Since {
    fn format_time(&self, w: &mut Writer<'_>) -> std::fmt::Result {
        write!(w, "{:8.2}s", self.0.elapsed().as_secs_f64())
    }
}

fn logging(filter: Option<&str>) -> Result<()> {
    let filter = match filter {
        Some(f) => EnvFilter::try_new(f)?,
        None => EnvFilter::try_from_env("CK_LOG").or_else(|_| EnvFilter::try_new("info"))?,
    };
    tracing_subscriber::registry()
        .with(filter)
        .with(

            fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(std::io::stderr().is_terminal())
                .with_target(false)
                .with_timer(Since(Instant::now())),
        )
        .with(ErrorLayer::default())
        .try_init()?;

    Ok(())
}

fn main() -> Result<()> {
    let a: Args = argh::from_env();
    logging(a.log.as_deref())?;
    let theme = if std::io::stderr().is_terminal() {
        Theme::dark()
    } else {
        Theme::new()
    };
    HookBuilder::default().theme(theme).install()?;
    #[cfg(unix)]
    if a.serve {
        return serve::run();
    }
    let abort = AtomicBool::new(false);
    job(a, Session { resident: None, abort: &abort })
}

fn job(a: Args, session: Session<'_>) -> Result<()> {
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
    let stride = a.stride.unwrap_or((chunk / 2).max(1));
    ensure!((1..=chunk).contains(&stride), "want 0 < stride <= chunk");

    ensure!(
        a.quantum > 0 && a.quantum % 64 == 0,
        "--quantum must be a multiple of 64"
    );
    ensure!(a.vfm_batch > 0, "--vfm-batch must be positive");
    ensure!(
        ["lanczos", "refine"].contains(&a.upscale.as_str()),
        "--upscale wants lanczos or refine"
    );
    let cfg = Config {
        chunk,
        stride,
        auto: a.chunk.is_none(),

        carry: !a.no_carry,
        hint_quality: a.hint_quality,
        carry_quality: a.carry_quality,
        quantum: a.quantum,
        scale: a.model_scale,
        refine: (a.upscale == "refine" && a.model_scale < 1.0).then(|| a.refiner.clone()),
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
        preview: a.preview_fd.map(PreviewTo::Fd).or(a.preview.map(PreviewTo::Dir)),
    };
    let hints = Hints {
        first: a.hint,
        video: a.hint_video,
    };
    let start = Instant::now();
    let assets = Assets {

        weights: asset(&a.weights, "uv run python scripts/download_weights.py")?,
        vfms: asset(&a.vfms, "uv run python scripts/download_weights.py")?,
        ptx: asset(&a.ptx, "just ptx")?,
    };
    run_in(session, &assets, &devices, &a.inputs, &a.output, &cfg, &hints)?;
    info!("done in {:.1} s", start.elapsed().as_secs_f64());
    Ok(())
}
