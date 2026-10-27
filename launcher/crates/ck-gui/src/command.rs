use std::path::{Path, PathBuf};

use crate::clips::{Clip, Hint, OUTPUTS};
use crate::settings::Settings;

pub struct Command {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(&'static str, &'static str)>,
}

#[must_use]
pub fn repo() -> Option<PathBuf> {
    let from = |p: &Path| p.ancestors().find(|a| a.join("infer.py").is_file()).map(Path::to_owned);
    std::env::var_os("CK_REPO")
        .map(PathBuf::from)
        .or_else(|| std::env::current_exe().ok().and_then(|e| from(&e)))
        .or_else(|| std::env::current_dir().ok().and_then(|d| from(&d)))
        .or_else(|| from(Path::new(env!("CARGO_MANIFEST_DIR"))))
}

fn exe(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_owned() }
}

#[must_use]
pub fn ck_binary() -> Option<PathBuf> {
    let beside = std::env::current_exe().ok().map(|e| e.with_file_name(exe("ck")));
    let on_path = std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join(exe("ck"))).find(|c| c.is_file()));
    let built = repo().map(|r| r.join("launcher/target/release").join(exe("ck")));
    [std::env::var_os("CK_BIN").map(PathBuf::from), beside, on_path, built].into_iter().flatten().find(|p| p.is_file())
}

/// Weights found under `weights/` and the GPUs NVML reports, in PCI order (as ck sees them under
/// `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
#[derive(Default)]
pub struct Catalogue {
    pub models: Vec<PathBuf>,
    pub refiners: Vec<PathBuf>,
    pub gpus: Vec<String>,
}

fn file_name(p: &Path) -> String {
    p.file_name().unwrap_or_default().to_string_lossy().into_owned()
}

impl Catalogue {
    #[must_use]
    pub fn scan(gpus: Vec<String>) -> Self {
        let mut all: Vec<PathBuf> = repo()
            .and_then(|r| std::fs::read_dir(r.join("weights")).ok())
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| file_name(p).starts_with("corridorkey_") && p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        all.sort();
        let (refiners, models) = all.into_iter().partition(|p| file_name(p).contains("refiner"));
        Self { models, refiners, gpus }
    }

    #[must_use]
    pub fn label(p: &Path) -> String {
        let stem = file_name(p).trim_end_matches(".safetensors").trim_start_matches("corridorkey_").replace("edge_refiner_", "");
        let (name, dtype) = stem.split_once('.').unwrap_or((&stem, "fp32"));
        let size = p.metadata().map_or(0.0, |m| m.len() as f64 / f64::from(1u32 << 30));
        if size < 0.1 { name.to_owned() } else { format!("{name} · {dtype} · {size:.1} GB") }
    }

    #[must_use]
    pub fn index(list: &[PathBuf], name: &str, fallback: &str) -> usize {
        let find = |n: &str| list.iter().position(|p| file_name(p) == n);
        find(name).or_else(|| find(fallback)).unwrap_or_default()
    }

    #[must_use]
    pub fn name(list: &[PathBuf], i: i32) -> String {
        usize::try_from(i).ok().and_then(|i| list.get(i)).map_or_default(|p| file_name(p))
    }
}

const SCALES: [&str; 4] = ["1", "0.5", "0.25", "0.125"];

fn linear(clip: &Clip, s: &Settings) -> bool {
    match s.frame_colour {
        1 => true,
        2 => false,
        _ => crate::clips::frame_files(&clip.source).first().is_some_and(|f| f.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr"))),
    }
}

fn outputs(s: &Settings) -> Vec<&'static str> {
    OUTPUTS.iter().zip(s.outputs).filter(|&(_, on)| on).map(|((name, _), _)| *name).collect()
}

fn hint(clip: &Clip, flags: [&str; 2]) -> Vec<String> {
    match (clip.hint, &clip.hint_path) {
        (Hint::FirstFrame, Some(p)) => vec![flags[0].into(), p.display().to_string()],
        (Hint::Video, Some(p)) => vec![flags[1].into(), p.display().to_string()],
        _ => vec![],
    }
}

fn ck(clip: &Clip, s: &Settings, out: &Path) -> Vec<String> {
    let mut a = vec!["-o".into(), out.display().to_string(), "--outputs".into(), outputs(s).join(",")];
    a.extend(hint(clip, ["--hint", "--hint-video"]));
    a.extend(["--hint-quality".into(), s.hint_quality.to_string()]);
    match s.carry {
        Some(q) => a.extend(["--carry-quality".into(), q.to_string()]),
        None => a.push("--no-carry".into()),
    }
    if let Some(float) = s.exr { a.extend(["--exr".into(), if float { "float" } else { "half" }.into()]) } else {
        a.extend(["--codec".into(), if s.nvenc { "hevc_nvenc" } else { "libx264" }.into()]);
        if s.cutout_linear {
            a.push("--cutout-linear".into());
        }
    }
    a.extend(["--despill".into(), s.despill.to_string()]);
    if clip.frames {
        a.extend(["--fps".into(), s.fps.to_string()]);
        if linear(clip, s) {
            a.push("--linear".into());
        }
    }
    if let Some((chunk, stride)) = s.window {
        a.extend(["--chunk".into(), chunk.to_string(), "--stride".into(), stride.to_string()]);
    }
    if let Some(scale) = SCALES.get(usize::from(s.scale)).filter(|_| s.scale > 0) {
        a.extend(["--model-scale".into(), (*scale).into()]);
        if s.refine {
            a.extend(["--upscale".into(), "refine".into()]);
        }
    }
    if s.frames > 0 {
        a.extend(["--frames".into(), s.frames.to_string()]);
    }
    a.push(clip.source.display().to_string());
    a
}

fn placement(s: &Settings, gpus: &[usize], preview: &Path) -> Vec<String> {
    let weights = |name: &str| repo().map(|r| r.join("weights").join(name).display().to_string()).filter(|_| !name.is_empty());
    let mut a = vec!["--preview".into(), preview.display().to_string()];
    if !gpus.is_empty() {
        a.extend(["--device".into(), gpus.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")]);
    }
    a.extend(weights(&s.model).into_iter().flat_map(|w| ["--weights".into(), w]));
    a.extend(weights(&s.refiner).into_iter().flat_map(|w| ["--refiner".into(), w]));
    a
}

pub fn build(clip: &Clip, s: &Settings, out: &Path, (gpus, preview): (&[usize], &Path)) -> eyre::Result<Command> {
    let program = ck_binary().ok_or_else(|| eyre::eyre!("ck is not built: cargo build --release -p ck in launcher/"))?;
    let mut args = placement(s, gpus, preview);
    args.extend(ck(clip, s, out));
    Ok(Command { program, args, env: vec![("CUDA_DEVICE_ORDER", "PCI_BUS_ID")] })
}
