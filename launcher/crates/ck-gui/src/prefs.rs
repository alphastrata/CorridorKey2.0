use slint::{ComponentHandle, Model, ModelRc, VecModel};

use crate::settings::{Engine, Placement, Prefs, Share};
use crate::{AppWindow, Prefs as Ui};

pub fn share_to_ui<R>(s: Share, set: impl FnOnce(f32, f32, f32) -> R) -> R {
    set(f32::from(s.priority), f32::from(s.vram), f32::from(s.cpu))
}

pub fn share((priority, vram, cpu): (f32, f32, f32)) -> Share {
    let pct = |v: f32| (1u8..=100).find(|&p| f32::from(p) >= v.round()).unwrap_or(100);
    Share { priority: pct(priority), vram: pct(vram), cpu: pct(cpu) }
}

pub fn to_ui(p: &Prefs, gpus: &[String], ui: &AppWindow) {
    let u = ui.global::<Ui>();
    u.set_gpu_names(ModelRc::new(VecModel::from(gpus.iter().map(Into::into).collect::<Vec<_>>())));
    let on: Vec<bool> = (0..gpus.len())
        .map(|i| u32::try_from(i).is_ok_and(|i| !p.gpus_off.contains(&i)))
        .collect();
    u.set_gpu_on(ModelRc::new(VecModel::from(on)));
    u.set_parallel(p.placement == Placement::PerGpu);
    share_to_ui(p.share, |a, b, c| (u.set_priority(a), u.set_vram(b), u.set_cpu(c)));
    u.set_keep_alive(i32::try_from(p.keep_alive).unwrap_or(10));
    u.set_quantum(i32::try_from(p.engine.quantum).unwrap_or(128));
    u.set_vfm_batch(i32::try_from(p.engine.vfm_batch).unwrap_or(8));
}

pub fn from_ui(ui: &AppWindow) -> Prefs {
    let u = ui.global::<Ui>();
    Prefs {
        gpus_off: u.get_gpu_on().iter().zip(0u32..).filter(|(on, _)| !on).map(|(_, i)| i).collect(),
        placement: if u.get_parallel() { Placement::PerGpu } else { Placement::Together },
        share: share((u.get_priority(), u.get_vram(), u.get_cpu())),
        keep_alive: u.get_keep_alive().max(0).cast_unsigned(),
        engine: engine((u.get_quantum(), u.get_vfm_batch())),
    }
}

/// From the Preferences sheet's spin values.
#[must_use]
pub fn engine((quantum, vfm_batch): (i32, i32)) -> Engine {
    Engine {
        quantum: quantum.max(1).cast_unsigned().clamp(16, 1024),
        vfm_batch: vfm_batch.max(1).cast_unsigned().clamp(1, 64),
    }
}
