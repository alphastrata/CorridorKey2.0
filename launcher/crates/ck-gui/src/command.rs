use std::path::{Path, PathBuf};

use crate::clips::{Clip, Hint, OUTPUTS};
use crate::settings::{Backend, Settings};

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

fn python() -> PathBuf {
    let venv = repo().map(|r| if cfg!(windows) { r.join(".venv/Scripts/python.exe") } else { r.join(".venv/bin/python") });
    venv.filter(|p| p.is_file()).unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "python" } else { "python3" }))
}

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
    a.extend(["--hint-quality".into(), format!("{}", s.hint_quality)]);
    match s.carry {
        Some(q) => a.extend(["--carry-quality".into(), format!("{q}")]),
        None => a.push("--no-carry".into()),
    }
    if let Some(float) = s.exr { a.extend(["--exr".into(), if float { "float" } else { "half" }.into()]) } else {
        a.extend(["--codec".into(), if s.nvenc { "hevc_nvenc" } else { "libx264" }.into()]);
        if s.cutout_linear {
            a.push("--cutout-linear".into());
        }
    }
    a.extend(["--despill".into(), format!("{}", s.despill)]);
    if clip.frames {
        a.extend(["--fps".into(), format!("{}", s.fps)]);
        if linear(clip, s) {
            a.push("--linear".into());
        }
    }
    if let Some((chunk, stride)) = s.window {
        a.extend(["--chunk".into(), chunk.to_string(), "--stride".into(), stride.to_string()]);
    }
    if s.frames > 0 {
        a.extend(["--frames".into(), s.frames.to_string()]);
    }
    a.push(clip.source.display().to_string());
    a
}

fn infer(clip: &Clip, s: &Settings, out: &Path, root: &Path) -> Vec<String> {
    let mut a = vec![
        root.join("infer.py").display().to_string(),
        "--input".into(),
        clip.source.display().to_string(),
        "--output_dir".into(),
        out.display().to_string(),
    ];
    a.extend(hint(clip, ["--hint_first_frame", "--hint_video"]));
    a.extend(["--hint_quality".into(), format!("{}", s.hint_quality)]);
    match s.carry {
        Some(q) => a.extend(["--carry_hint_quality".into(), format!("{q}")]),
        None => a.push("--no_carry_hint".into()),
    }
    a.push("--outputs".into());
    a.extend(outputs(s).into_iter().map(Into::into));
    if let Some(float) = s.exr { a.extend(["--output_format".into(), "exr".into(), "--exr_dtype".into(), if float { "float" } else { "half" }.into()]) } else {
        a.extend(["--output_format".into(), "video".into()]);
        if s.cutout_linear {
            a.push("--cutout_linear".into());
        }
    }
    a.extend(["--despill_strength".into(), format!("{}", s.despill)]);
    if clip.frames {
        a.extend(["--frame_dir_fps".into(), format!("{}", s.fps)]);
        if linear(clip, s) {
            a.push("--frame_dir_linear".into());
        }
    }
    if let Some((chunk, stride)) = s.window {
        if s.low_vram {
            a.push("--low_vram".into());
        }
        a.extend(["--hann_chunk".into(), chunk.to_string(), "--hann_stride".into(), stride.to_string()]);
    }
    a.extend(["--temp_dir".into(), s.temp_dir().display().to_string()]);
    if s.frames > 0 {
        let take = ["begin", "middle", "random_middle"][usize::from(s.start_mode.min(2))];
        a.extend(["--num_frames".into(), s.frames.to_string(), "--start_mode".into(), take.into()]);
    }
    a
}

pub fn build(clip: &Clip, s: &Settings, out: &Path) -> Result<Command, String> {
    match s.backend {
        Backend::Ck => {
            let program = ck_binary().ok_or("ck is not built: cargo build --release -p ck in launcher/")?;
            Ok(Command { program, args: ck(clip, s, out), env: vec![] })
        }
        Backend::PythonPtx | Backend::Stock => {
            let root = repo().ok_or("cannot find the repository (set CK_REPO)")?;
            let ptx = if s.backend == Backend::PythonPtx { "1" } else { "0" };
            Ok(Command {
                program: python(),
                args: infer(clip, s, out, &root),
                env: vec![("CORRIDORKEY_PTX", ptx), ("PYTHONUNBUFFERED", "1"), ("PYTHONIOENCODING", "utf-8"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")],
            })
        }
    }
}
