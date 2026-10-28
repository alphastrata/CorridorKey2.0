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
    pub gpus_off: Vec<u32>,
    pub placement: Placement,
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
            gpus_off: vec![],
            placement: Placement::PerGpu,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Layout {
    pub left_width: f32,
    pub right_width: f32,
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

    pub fn save(&self) {
        let (Some(f), Ok(text)) = (file(), toml::to_string_pretty(self)) else {
            return;
        };
        if let Some(dir) = f.parent() {
            drop(std::fs::create_dir_all(dir));
        }
        drop(std::fs::write(f, text));
    }
}

#[cfg(test)]
mod tests {
    use super::{Saved, Summary};

    #[test]
    fn round_trip() {
        let mut s = Saved::default();
        s.settings.window = Some((64, 32));
        s.settings.gpus_off = vec![1];
        s.last = Some(Summary {
            clip: "shot".into(),
            phases: vec![("Keying".into(), 1.5)],
            ..Summary::default()
        });
        let back: Saved = toml::from_str(&toml::to_string_pretty(&s).unwrap()).unwrap();
        assert_eq!((s.settings, s.last), (back.settings, back.last));
    }
}
