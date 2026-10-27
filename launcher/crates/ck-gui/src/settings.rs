use std::path::PathBuf;

use rkyv::{Archive, Deserialize, Serialize, rancor};

#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq)]
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
    pub out_parent: Option<String>,
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
            out_parent: None,
        }
    }
}

fn file() -> Option<PathBuf> {
    Some(dirs::config_dir()?.join("corridorkey").join("gui.rkyv"))
}

impl Settings {
    #[must_use]
    pub fn load() -> Self {
        file()
            .and_then(|f| std::fs::read(f).ok())
            .and_then(|b| rkyv::from_bytes::<Self, rancor::Error>(&b).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let (Some(f), Ok(bytes)) = (file(), rkyv::to_bytes::<rancor::Error>(self)) else { return };
        if let Some(dir) = f.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(f, bytes);
    }
}
