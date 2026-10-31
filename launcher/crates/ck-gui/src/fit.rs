//! Whether a clip's settings can fit the GPUs it would run on, using `ck`'s own planner.

use ck_plan::{QUANTUM, encoder_gib, estimate, model_size, window};

use crate::settings::{Settings, Share};
use crate::telemetry::Device;

/// Why the clip cannot run on `gpu`, or `None` when it fits.
#[must_use]
pub fn problem(s: &Settings, share: Share, gpu: &Device, (w, h, frames): (u32, u32, usize)) -> Option<String> {
    let size = model_size((h, w), u32::from(s.scale), QUANTUM);
    let streams = u32::from(s.outputs[1]) + u32::from(s.outputs[2]);
    let nvenc = if s.nvenc && s.exr.is_none() { encoder_gib(streams, (w, h)) } else { 0.0 };
    let budget = gpu.total * f64::from(share.vram) / 100.0 - nvenc;
    let total = u32::try_from(frames).unwrap_or(u32::MAX).max(1);
    let at = format!("{}, {budget:.1} GiB at {}% VRAM", gpu.name, share.vram);

    if let Some((chunk, _)) = s.window {
        let need = estimate(size, chunk.min(total), gpu.sm);
        let fits = window(size, total, budget, gpu.sm).0;
        (need > budget && chunk > fits)
            .then(|| format!("A {chunk}-frame window needs about {need:.1} GiB ({at}). Use {fits} or fewer, or Auto."))
    } else {
        let need = estimate(size, 1, gpu.sm);
        (need > budget).then(|| format!("Even one frame needs about {need:.1} GiB ({at}). Lower the model scale."))
    }
}

/// The first problem among `gpus`, if any of them cannot take the clip.
#[must_use]
pub fn any(s: &Settings, share: Share, devices: &[Device], gpus: &[usize], dims: (u32, u32, usize)) -> Option<String> {
    gpus.iter().filter_map(|&g| devices.get(g)).find_map(|d| problem(s, share, d, dims))
}
