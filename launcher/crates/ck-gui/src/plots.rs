use std::collections::VecDeque;
use std::fmt::Write as _;

use ahash::AHashMap;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::Clock;
use sysinfo::System;

use crate::Plot;
use crate::convert::{Float, Metric};

pub const TICK_MS: u64 = 250;
const SAMPLES: usize = 21;
const W: f32 = 110.0;
const H: f32 = 22.0;

/// Rolling samples per series, as percentages, rendered to sparkline paths in a W×H box.
#[derive(Default)]
pub struct History(AHashMap<String, VecDeque<f32>>);

fn pct(v: u32, max: u32) -> f32 {
    if max == 0 { 0.0 } else { 100.0 * v.f32() / max.f32() }
}

impl History {
    fn push(&mut self, (gpu, label): (i32, String), pct: f32, value: String) -> Plot {
        let q = self.0.entry(label.clone()).or_default();
        q.push_back(pct.clamp(0.0, 100.0));
        while q.len() > SAMPLES {
            q.pop_front();
        }
        let step = W / (SAMPLES - 1).f32();
        let x0 = W - step * (q.len() - 1).f32();
        let line = q.iter().enumerate().fold(String::new(), |mut s, (i, v)| {
            let y = H - 1.0 - v / 100.0 * (H - 2.0);
            let _ = write!(s, "{} {:.1} {y:.1} ", if i == 0 { "M" } else { "L" }, x0 + step * i.f32());
            s
        });
        Plot {
            gpu,
            tint: -1,
            label: label.into(),
            value: value.into(),
            area: format!("{line}L {W} {H} L {x0:.1} {H} Z").into(),
            line: line.into(),
        }
    }

    pub fn sample(&mut self, nvml: Option<&Nvml>, sys: &mut System) -> Vec<Plot> {
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        let gib = |b: u64| b.f64() / f64::from(1u32 << 30);
        let (used, total) = (sys.used_memory(), sys.total_memory());
        let mut out = vec![
            self.push((-1, "CPU".into()), sys.global_cpu_usage(), format!("{:.0}%", sys.global_cpu_usage())),
            self.push(
                (-1, "RAM".into()),
                if total == 0 { 0.0 } else { (100.0 * used.f64() / total.f64()).f32() },
                format!("{:.1} / {:.0} GB", gib(used), gib(total)),
            ),
        ];
        let Some(nvml) = nvml else { return out };
        (0..nvml.device_count().unwrap_or(0))
            .filter_map(|i| Some((i, nvml.device_by_index(i).ok()?)))
            .for_each(|(i, d)| {
                let g = i32::try_from(i).unwrap_or(-1);
                let util = d.utilization_rates().map_or(0, |u| u.gpu);
                let mem = d.memory_info().ok();
                let (sm, sm_max) = (d.clock_info(Clock::SM).unwrap_or(0), d.max_clock_info(Clock::SM).unwrap_or(0));
                let (mc, mc_max) = (d.clock_info(Clock::Memory).unwrap_or(0), d.max_clock_info(Clock::Memory).unwrap_or(0));
                let (w, cap) = (d.power_usage().unwrap_or(0) / 1000, d.enforced_power_limit().unwrap_or(0) / 1000);
                out.extend([
                    self.push((g, format!("GPU{i} util")), util.f32(), format!("{util}%")),
                    self.push(
                        (g, format!("GPU{i} VRAM")),
                        mem.as_ref().map_or(0.0, |m| (100.0 * m.used.f64() / m.total.max(1).f64()).f32()),
                        mem.as_ref().map_or_default(|m| format!("{:.1} GB", gib(m.used))),
                    ),
                    self.push((g, format!("GPU{i} core")), pct(sm, sm_max), format!("{sm} MHz")),
                    self.push((g, format!("GPU{i} mem")), pct(mc, mc_max), format!("{mc} MHz")),
                    self.push((g, format!("GPU{i} power")), pct(w, cap), format!("{w} W")),
                ]);
            });
        out
    }
}
