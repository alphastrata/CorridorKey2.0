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

pub fn build(clip: &Clip, s: &Settings, out: &Path) -> eyre::Result<Command> {
    let program = ck_binary().ok_or_else(|| eyre::eyre!("ck is not built: cargo build --release -p ck in launcher/"))?;
    Ok(Command { program, args: ck(clip, s, out), env: vec![] })
}
