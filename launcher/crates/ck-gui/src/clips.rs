use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::settings::Settings;

pub const VIDEO: [&str; 5] = ["mp4", "mov", "avi", "mkv", "webm"];
pub const FRAMES: [&str; 8] = ["png", "jpg", "jpeg", "exr", "tif", "tiff", "webp", "dpx"];
pub const OUTPUTS: [(&str, &str); 4] = [
    ("alpha", "alpha"),
    ("fg", "fg"),
    ("checker", "checker_comp_despill"),
    ("cutout", "cutout"),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Hint {
    None,
    FirstFrame,
    Video,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Status {
    Ready,
    Queued,
    Running,
    Done,
    Failed,
    Stopped,
}

impl Status {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::Queued => "Queued",
            Self::Running => "Running",
            Self::Done => "Done",
            Self::Failed => "Failed",
            Self::Stopped => "Stopped",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Clip {
    pub path: PathBuf,
    pub source: PathBuf,
    pub frames: bool,
    pub hint: Hint,
    pub hint_path: Option<PathBuf>,
    pub status: Status,
    pub progress: f32,
    pub detail: String,
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub settings: Option<Settings>,
    /// The `ckd` job id while it runs, so a restarted GUI can reattach.
    #[serde(default)]
    pub job: Option<String>,
}

fn ext(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn stem(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_owned()
}

fn entries(dir: &Path) -> Vec<PathBuf> {
    let mut all: Vec<PathBuf> =
        std::fs::read_dir(dir).map_or_default(|d| d.filter_map(Result::ok).map(|e| e.path()).collect());
    all.sort();
    all
}

#[must_use]
pub fn frame_files(dir: &Path) -> Vec<PathBuf> {
    entries(dir)
        .into_iter()
        .filter(|p| p.is_file() && FRAMES.contains(&ext(p).as_str()))
        .collect()
}

fn is_video(p: &Path) -> bool {
    p.is_file() && VIDEO.contains(&ext(p).as_str()) && !p.to_string_lossy().to_lowercase().contains("_alphahint")
}

fn sequence(dir: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    let sub = |name: &str| {
        entries(dir)
            .into_iter()
            .find(|p| p.is_dir() && p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(name)))
    };
    let input = sub("input")?;
    (!frame_files(&input).is_empty()).then(|| (input, sub("alphahint")))
}

fn alpha_hint(video: &Path) -> Option<PathBuf> {
    let prefix = format!("{}_alphahint", stem(video).to_lowercase());
    let allowed = ["mp4", "mov", "avi", "mkv", "webm", "png", "jpg", "jpeg", "exr"];
    entries(video.parent()?).into_iter().find(|p| {
        p.is_file()
            && allowed.contains(&ext(p).as_str())
            && p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.to_lowercase().starts_with(&prefix))
    })
}

fn first_frame_hint(video: &Path) -> Option<PathBuf> {
    let base = stem(video).to_lowercase();
    let images = ["png", "jpg", "jpeg", "exr"];
    let tagged = |tag: &str| {
        entries(video.parent()?).into_iter().find(|p| {
            let s = stem(p).to_lowercase();
            p.is_file()
                && images.contains(&ext(p).as_str())
                && s.strip_prefix(&base)
                    .is_some_and(|r| r.trim_start_matches(['_', '-', ' ']).starts_with(tag))
        })
    };
    tagged("framehint").or_else(|| tagged("alphahint"))
}

fn video_clip(path: &Path) -> Clip {
    let (hint, hint_path) = match alpha_hint(path) {
        Some(h) if VIDEO.contains(&ext(&h).as_str()) => (Hint::Video, Some(h)),
        Some(h) => (Hint::FirstFrame, Some(h)),
        None => first_frame_hint(path).map_or((Hint::None, None), |h| (Hint::FirstFrame, Some(h))),
    };
    Clip::new(path, path, false, hint, hint_path)
}

impl Clip {
    fn new(path: &Path, source: &Path, frames: bool, hint: Hint, hint_path: Option<PathBuf>) -> Self {
        Self {
            path: path.to_owned(),
            source: source.to_owned(),
            frames,
            hint,
            hint_path,
            status: Status::Ready,
            progress: 0.0,
            detail: String::new(),
            output_dir: None,
            settings: None,
            job: None,
        }
    }

    #[must_use]
    pub fn name(&self) -> String {
        let s = stem(&self.source);
        if ["input", "frames", "rgb"].contains(&s.to_lowercase().as_str()) {
            return self.source.parent().map_or(s, |p| {
                p.file_name()
                    .map_or(String::new(), |n| n.to_string_lossy().into_owned())
            });
        }
        s
    }

    #[must_use]
    pub fn hint_label(&self) -> String {
        match (&self.hint, &self.hint_path) {
            (Hint::None, _) | (_, None) => "None".into(),
            (kind, Some(p)) => format!(
                "{} · {}",
                if *kind == Hint::FirstFrame { "Frame" } else { "Video" },
                p.file_name().unwrap_or_default().to_string_lossy()
            ),
        }
    }

    #[must_use]
    pub fn output_dir_for(&self, parent: &str) -> PathBuf {
        let base = if parent.is_empty() {
            self.path.parent().map_or_default(Path::to_owned)
        } else {
            PathBuf::from(parent)
        };
        base.join(format!("out_{}", stem(&self.path)))
    }

    #[must_use]
    pub fn outputs(&self, dir: &Path) -> [Option<PathBuf>; 4] {
        let name = self.name();
        let all = entries(dir);
        OUTPUTS.map(|(_, suffix)| {
            all.iter()
                .filter(|p| {
                    let n = p.file_name().unwrap_or_default().to_string_lossy();
                    let Some(rest) = n.strip_prefix(&format!("{name}_{suffix}")) else {
                        return false;
                    };

                    let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit());
                    (p.is_file() && (rest == ".mp4" || rest == ".mov")) || (p.is_dir() && rest == "_exr")
                })
                .max_by_key(|p| {
                    p.metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                })
                .cloned()
        })
    }
}

#[must_use]
pub fn discover(path: &Path) -> Vec<Clip> {
    if path.is_file() {
        return if is_video(path) { vec![video_clip(path)] } else { vec![] };
    }
    if !path.is_dir() {
        return vec![];
    }
    if let Some((input, hint)) = sequence(path) {
        let kind = if hint.is_some() { Hint::Video } else { Hint::None };

        return vec![Clip::new(path, &input, true, kind, hint)];
    }
    if !frame_files(path).is_empty() {
        return vec![Clip::new(path, path, true, Hint::None, None)];
    }
    entries(path)
        .into_iter()
        .filter(|p| is_video(p) || (p.is_dir() && sequence(p).is_some()))
        .flat_map(|p| discover(&p))
        .collect()
}
