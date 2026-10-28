use slint::{ComponentHandle, Model, ModelRc, VecModel};

use crate::settings::{Placement, Prefs};
use crate::{AppWindow, Prefs as Ui};

pub fn to_ui(p: &Prefs, gpus: &[String], ui: &AppWindow) {
    let u = ui.global::<Ui>();
    u.set_gpu_names(ModelRc::new(VecModel::from(gpus.iter().map(Into::into).collect::<Vec<_>>())));
    let on: Vec<bool> = (0..gpus.len())
        .map(|i| u32::try_from(i).is_ok_and(|i| !p.gpus_off.contains(&i)))
        .collect();
    u.set_gpu_on(ModelRc::new(VecModel::from(on)));
    u.set_parallel(p.placement == Placement::PerGpu);
    u.set_nice(p.nice);
    u.set_vram_cap(p.vram_cap);
    u.set_threads(i32::try_from(p.threads).unwrap_or_default());
    u.set_keep_alive(i32::try_from(p.keep_alive).unwrap_or(10));
}

pub fn from_ui(ui: &AppWindow) -> Prefs {
    let u = ui.global::<Ui>();
    Prefs {
        gpus_off: u.get_gpu_on().iter().zip(0u32..).filter(|(on, _)| !on).map(|(_, i)| i).collect(),
        placement: if u.get_parallel() { Placement::PerGpu } else { Placement::Together },
        nice: u.get_nice().clamp(0, 19),
        vram_cap: u.get_vram_cap().max(0.0),
        threads: u.get_threads().max(0).cast_unsigned(),
        keep_alive: u.get_keep_alive().max(0).cast_unsigned(),
    }
}
