use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::clips::Clip;
use crate::runner::Progress;

/// How queued clips share the enabled GPUs: one clip on each at once, or each clip across all of them.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    PerGpu,
    Together,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub nvenc: bool,
    pub outputs: [bool; 4],
    pub exr: Option<bool>,
    pub cutout_linear: bool,
    pub despill: f32,
    pub hint_quality: f32,
    pub carry: Option<f32>,
    pub window: Option<(u32, u32)>,
    pub frame_colour: u8,
    pub fps: f32,
    pub frames: u32,
    pub scale: u8,
    pub refine: bool,
    pub out_parent: Option<String>,
    pub model: String,
    pub refiner: String,
    /// Padding quantum and foundation batch; `None` follows the Preferences defaults.
    pub engine: Option<Engine>,
    /// ck log filter; empty = default `info`.
    pub log: String,
    /// Scene-linear gain, 1.0 = ungraded.
    pub brightness: f32,
    /// Scene-linear contrast, 1.0 = ungraded.
    pub contrast: f32,
    /// x264 `-crf`, or NVENC `-cq`: lower is better quality and bigger files.
    pub crf: u32,
    /// Index into `command::PRESETS`.
    pub preset: u8,
    /// Encoder threads per output; 0 = ffmpeg decides.
    pub threads: u32,
    /// This job's own resource shares; `None` follows the Preferences defaults.
    pub share: Option<Share>,
    /// GPUs this job runs on together; `None` follows the Preferences placement.
    pub gpus: Option<Vec<usize>>,
}

/// Percentages of the machine a job may take.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(default)]
pub struct Share {
    pub priority: u8,
    pub vram: u8,
    pub cpu: u8,
}

impl Default for Share {
    fn default() -> Self {
        Self { priority: 50, vram: 90, cpu: 75 }
    }
}

impl Share {
    pub fn nice(self) -> i32 {
        (19 * (100 - i32::from(self.priority.min(100))) + 50) / 100
    }

    /// Compute threads, or `None` for every core.
    pub fn threads(self) -> Option<usize> {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        (self.cpu < 100).then(|| (cores * usize::from(self.cpu)).div_ceil(100).max(1))
    }
}

/// Engine knobs a clip can share with the machine's defaults.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(default)]
pub struct Engine {
    pub quantum: u32,
    pub vfm_batch: u32,
}

impl Default for Engine {
    fn default() -> Self {
        Self { quantum: 128, vfm_batch: 8 }
    }
}

/// Machine-wide preferences, separate from any clip's settings.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Prefs {
    pub gpus_off: Vec<u32>,
    pub placement: Placement,
    pub share: Share,
    /// Minutes an idle resident worker keeps its models loaded.
    pub keep_alive: u32,
    /// Defaults every clip carries unless the Inspector's Engine override is on.
    pub engine: Engine,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            gpus_off: vec![],
            placement: Placement::PerGpu,
            share: Share::default(),
            keep_alive: 10,
            engine: Engine::default(),
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            nvenc: false,
            outputs: [true, true, false, true],
            exr: None,
            cutout_linear: false,
            despill: 0.5,
            hint_quality: 0.75,
            carry: Some(0.95),
            window: None,
            frame_colour: 0,
            fps: 24.0,
            frames: 0,
            scale: 0,
            refine: true,
            out_parent: None,
            model: String::new(),
            refiner: String::new(),
            engine: None,
            log: String::new(),
            brightness: 1.0,
            contrast: 1.0,
            crf: 12,
            preset: 5,
            threads: 8,
            share: None,
            gpus: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Layout {
    pub left_width: f32,
    pub right_width: f32,
    pub log_height: f32,
    pub log: bool,
    pub stats: bool,
    pub view_b: i32,
    /// Where the add dialogues last picked from.
    pub last_dir: Option<PathBuf>,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            left_width: 340.0,
            right_width: 380.0,
            log_height: 150.0,
            log: true,
            stats: false,
            view_b: -1,
            last_dir: None,
        }
    }
}

/// A finished (or running) job's numbers, with phase timings as durations.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Summary {
    pub clip: String,
    pub status: String,
    pub progress: Progress,
    pub elapsed: f32,
    pub phases: Vec<(String, f32)>,
    pub keying: Option<f32>,
    pub done: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Saved {
    pub settings: Settings,
    pub prefs: Prefs,
    pub layout: Layout,
    pub selected: Option<usize>,
    pub last: Option<Summary>,
    pub queue: Vec<Clip>,
}

fn file() -> Option<PathBuf> {
    Some(dirs::home_dir()?.join(".ck_gui").join("state.toml"))
}

impl Saved {
    #[must_use]
    pub fn load() -> Self {
        file()
            .and_then(|f| std::fs::read_to_string(f).ok())
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Writes the state, returning the text written.
    pub fn save(&self) -> Option<String> {
        let (f, text) = (file()?, toml::to_string_pretty(self).ok()?);
        if let Some(dir) = f.parent() {
            drop(std::fs::create_dir_all(dir));
        }
        std::fs::write(f, &text).ok()?;
        Some(text)
    }

    /// The state on disk with its text, when the file reads and parses.
    #[must_use]
    pub fn read() -> Option<(String, Self)> {
        let text = std::fs::read_to_string(file()?).ok()?;
        let saved = toml::from_str(&text).ok()?;
        Some((text, saved))
    }
}

#[cfg(test)]
mod tests {
    use super::{Saved, Share, Summary};

    #[test]
    fn share_maps_to_nice_and_threads() {
        let at = |priority, cpu| Share { priority, vram: 90, cpu };
        assert_eq!([at(100, 100), at(50, 100), at(1, 100)].map(Share::nice), [0, 10, 19]);
        assert_eq!(at(50, 100).threads(), None);
        assert!(at(50, 1).threads().is_some_and(|n| n >= 1));
    }

    #[test]
    fn round_trip() {
        let mut s = Saved::default();
        s.settings.window = Some((64, 32));
        s.prefs.gpus_off = vec![1];
        s.last = Some(Summary {
            clip: "shot".into(),
            phases: vec![("Keying".into(), 1.5)],
            ..Summary::default()
        });
        let back: Saved = toml::from_str(&toml::to_string_pretty(&s).unwrap()).unwrap();
        assert_eq!((s.settings, s.prefs, s.last), (back.settings, back.prefs, back.last));
    }
}
