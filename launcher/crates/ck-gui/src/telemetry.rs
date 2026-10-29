use crate::plots::{History, TICK_MS};
use crate::{Plot, Stat};
use crate::convert::Metric;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use flume::Sender;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::UsedGpuMemory;
use slint::SharedString;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Clone, Default)]
pub struct Snapshot {
    pub gpus: Vec<Stat>,
    pub process: Vec<Stat>,
    pub plots: Vec<Plot>,
}

pub fn kv(key: impl Into<SharedString>, value: impl Into<SharedString>) -> Stat {
    Stat {
        key: key.into(),
        value: value.into(),
        head: false,
        gpu: -1,
        tint: -1,
    }
}

pub fn head(title: impl Into<SharedString>) -> Stat {
    Stat {
        key: title.into(),
        value: SharedString::new(),
        head: true,
        gpu: -1,
        tint: -1,
    }
}

#[derive(Clone)]
pub struct Probe {
    pub pid: Arc<AtomicU32>,
    pub detailed: Arc<AtomicBool>,
}

fn short(name: &str) -> String {
    name.replace("NVIDIA ", "").replace("GeForce ", "")
}

fn tree(sys: &System, root: u32) -> AHashSet<u32> {
    let kids: AHashMap<u32, Vec<u32>> = sys
        .processes()
        .iter()
        .filter_map(|(pid, p)| p.parent().map(|parent| (parent.as_u32(), pid.as_u32())))
        .fold(AHashMap::new(), |mut m, (parent, pid)| {
            m.entry(parent).or_default().push(pid);
            m
        });

    let mut all = AHashSet::from_iter([root]);
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        kids.get(&p)
            .into_iter()
            .flatten()
            .filter_map(|&c| all.insert(c).then_some(c))
            .for_each(|c| stack.push(c));
    }
    all
}

fn process(sys: &System, pids: &AHashSet<u32>, root: u32) -> Vec<Stat> {
    let (mut rss, mut cpu, mut threads) = (0u64, 0f32, 0usize);
    let mut kids: AHashMap<String, usize> = AHashMap::new();
    pids.iter()
        .filter_map(|&p| Some((p, sys.process(Pid::from_u32(p))?)))
        .for_each(|(pid, p)| {
            rss += p.memory();
            cpu += p.cpu_usage();
            threads += p.tasks().map_or(1, |t| t.len().max(1));
            if pid != root {
                *kids.entry(p.name().to_string_lossy().into_owned()).or_default() += 1;
            }
        });

    let kids = kids
        .iter()
        .map(|(n, c)| if *c > 1 { format!("{n} ×{c}") } else { n.clone() })
        .collect::<Vec<_>>()
        .join(", ");
    vec![
        kv("pid", root.to_string()),
        kv("RSS", format!("{:.1} GB", rss.f64() / f64::from(1u32 << 30))),
        kv("CPU", format!("{cpu:.0}%")),
        kv("threads", threads.to_string()),
        kv("children", if kids.is_empty() { "none".into() } else { kids }),
    ]
}

fn gpus(nvml: &Nvml, mine: &AHashSet<u32>) -> Vec<Stat> {
    let gib = |b: u64| b.f64() / f64::from(1u32 << 30);

    (0..nvml.device_count().unwrap_or(0))
        .filter_map(|i| Some((i, nvml.device_by_index(i).ok()?)))
        .flat_map(|(i, d)| {
            let name = short(&d.name().unwrap_or_default());
            let job: u64 = d
                .running_compute_processes()
                .unwrap_or_default()
                .iter()
                .filter(|p| mine.contains(&p.pid))
                .map(|p| if let UsedGpuMemory::Used(b) = p.used_gpu_memory { b } else { 0 })
                .sum();
            let gpu = i32::try_from(i).unwrap_or(-1);

            let mut rows = vec![head(format!("GPU{i} · {name}"))];
            rows.extend((job > 0).then(|| kv("job VRAM", format!("{:.1} GB", gib(job)))));
            rows.extend([
                kv("temp", format!("{} °C", d.temperature(TemperatureSensor::Gpu).unwrap_or(0))),
                kv("PCIe", format!("gen {} ×{}", d.current_pcie_link_gen().unwrap_or(0), d.current_pcie_link_width().unwrap_or(0))),
            ]);

            rows.extend(
                d.encoder_stats()
                    .ok()
                    .filter(|e| e.session_count > 0)
                    .map(|e| kv("NVENC", format!("{} sessions · {} fps", e.session_count, e.average_fps))),
            );
            rows.iter_mut().for_each(|r| r.gpu = gpu);
            rows
        })
        .collect()
}

/// A GPU's label, compute capability as `major * 10 + minor`, and total memory in GiB.
#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub sm: u32,
    pub total: f64,
}

#[must_use]
pub fn devices() -> Vec<Device> {
    let Ok(nvml) = Nvml::init() else { return vec![] };
    (0..nvml.device_count().unwrap_or(0))
        .map(|i| {
            let d = nvml.device_by_index(i).ok();
            let cc = d.as_ref().and_then(|d| d.cuda_compute_capability().ok());
            Device {
                name: format!("GPU{i} · {}", d.as_ref().and_then(|d| d.name().ok()).map_or_else(|| "?".into(), |n| short(&n))),
                sm: cc.map_or(0, |c| u32::try_from(c.major * 10 + c.minor).unwrap_or(0)),
                total: d.and_then(|d| d.memory_info().ok()).map_or(0.0, |m| m.total.f64() / f64::from(1u32 << 30)),
            }
        })
        .collect()
}

/// Samples GPUs every few seconds; while `detailed` is set, also the job's process tree each second and plots at 4 Hz.
pub fn start(tx: Sender<Snapshot>) -> Probe {
    let probe = Probe {
        pid: Arc::default(),
        detailed: Arc::default(),
    };
    let p = probe.clone();
    thread::Builder::new()
        .name("telemetry".into())
        .spawn(move || {
            let nvml = Nvml::init().ok();
            let mut sys = System::new();
            let (mut history, mut tick, mut last) = (History::default(), 0u64, Snapshot::default());

            loop {
                let detailed = p.detailed.load(Ordering::Acquire);
                let root = p.pid.load(Ordering::Acquire);

                if !detailed || tick.is_multiple_of(1000 / TICK_MS) {
                    let mine = if detailed && root != 0 {
                        sys.refresh_processes_specifics(
                            ProcessesToUpdate::All,
                            true,
                            ProcessRefreshKind::nothing().with_memory().with_cpu().with_tasks(),
                        );
                        tree(&sys, root)
                    } else {
                        AHashSet::new()
                    };
                    last.gpus = if detailed { nvml.as_ref().map_or_default(|n| gpus(n, &mine)) } else { vec![] };
                    last.process = if detailed && root != 0 { process(&sys, &mine, root) } else { vec![] };
                }

                last.plots = if detailed { history.sample(nvml.as_ref(), &mut sys) } else { vec![] };
                if tx.send(last.clone()).is_err() {
                    return;
                }
                tick += 1;
                thread::sleep(Duration::from_millis(if detailed { TICK_MS } else { 3000 }));
            }
        })
        .ok();
    probe
}
