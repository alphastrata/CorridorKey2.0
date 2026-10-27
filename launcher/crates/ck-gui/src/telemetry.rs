use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use flume::Sender;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::enums::device::UsedGpuMemory;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Clone, Default)]
pub struct Snapshot {
    pub label: String,
    pub gpus: Vec<String>,
    pub process: String,
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
        kids.get(&p).into_iter().flatten().filter_map(|&c| all.insert(c).then_some(c)).for_each(|c| stack.push(c));
    }
    all
}

fn process(sys: &System, pids: &AHashSet<u32>, root: u32) -> String {
    let (mut rss, mut cpu, mut threads) = (0u64, 0f32, 0usize);
    let mut kids: AHashMap<String, usize> = AHashMap::new();
    pids.iter().filter_map(|&p| Some((p, sys.process(Pid::from_u32(p))?))).for_each(|(pid, p)| {
        rss += p.memory();
        cpu += p.cpu_usage();
        threads += p.tasks().map_or(1, |t| t.len().max(1));
        if pid != root {
            *kids.entry(p.name().to_string_lossy().into_owned()).or_default() += 1;
        }
    });
    let kids = kids.iter().map(|(n, c)| if *c > 1 { format!("{n}×{c}") } else { n.clone() }).collect::<Vec<_>>().join(", ");
    format!(
        "pid {root} · RSS {:.1} GB · CPU {cpu:.0}% · {threads} threads · children {}",
        rss as f64 / f64::from(1u32 << 30),
        if kids.is_empty() { "none" } else { &kids }
    )
}

fn gpus(nvml: &Nvml, mine: &AHashSet<u32>, detailed: bool) -> (String, Vec<String>) {
    let count = nvml.device_count().unwrap_or(0);
    let mut label = String::new();
    let mut lines = Vec::new();
    (0..count).filter_map(|i| Some((i, nvml.device_by_index(i).ok()?))).for_each(|(i, d)| {
        let name = short(&d.name().unwrap_or_default());
        let mem = d.memory_info().ok();
        let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
        if i == 0 {
            label = mem.as_ref().map_or_else(|| name.clone(), |m| format!("{name} · {:.1} of {:.0} GB free", gib(m.free), gib(m.total)));
        }
        if !detailed {
            return;
        }
        let job: u64 = d
            .running_compute_processes()
            .unwrap_or_default()
            .iter()
            .filter(|p| mine.contains(&p.pid))
            .map(|p| if let UsedGpuMemory::Used(b) = p.used_gpu_memory { b } else { 0 })
            .sum();
        let util = d.utilization_rates().map_or(0, |u| u.gpu);
        let watts = d.power_usage().unwrap_or(0) / 1000;
        let cap = d.enforced_power_limit().unwrap_or(0) / 1000;
        let temp = d.temperature(TemperatureSensor::Gpu).unwrap_or(0);
        let (sm, sm_max) = (d.clock_info(Clock::SM).unwrap_or(0), d.max_clock_info(Clock::SM).unwrap_or(0));
        let (gen_, width) = (d.current_pcie_link_gen().unwrap_or(0), d.current_pcie_link_width().unwrap_or(0));
        let enc = d.encoder_stats().ok().filter(|e| e.session_count > 0).map_or(String::new(), |e| format!(" · NVENC {}×{} fps", e.session_count, e.average_fps));
        let (used, total) = mem.map_or((0.0, 0.0), |m| (gib(m.used), gib(m.total)));
        let job = if job > 0 { format!(" (job {:.1})", gib(job)) } else { String::new() };
        lines.push(format!(
            "GPU{i} {name}\n  util {util}% · mem {used:.1}/{total:.1} GB{job} · {watts}/{cap} W · {temp}°C · SM {sm}/{sm_max} MHz · PCIe {gen_}×{width}{enc}"
        ));
    });
    (label, lines)
}

/// Samples GPUs every second and the job's process tree while `detailed` is set.
pub fn start(tx: Sender<Snapshot>) -> Probe {
    let probe = Probe { pid: Arc::default(), detailed: Arc::default() };
    let p = probe.clone();
    thread::Builder::new()
        .name("telemetry".into())
        .spawn(move || {
            let nvml = Nvml::init().ok();
            let mut sys = System::new();
            loop {
                let detailed = p.detailed.load(Ordering::Acquire);
                let root = p.pid.load(Ordering::Acquire);
                let mine = if detailed && root != 0 {
                    sys.refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::nothing().with_memory().with_cpu().with_tasks());
                    tree(&sys, root)
                } else {
                    AHashSet::new()
                };
                let (label, gpus) = nvml.as_ref().map_or_else(|| ("no NVML".into(), vec![]), |n| gpus(n, &mine, detailed));
                let process = if root == 0 { "idle".into() } else if detailed { process(&sys, &mine, root) } else { String::new() };
                if tx.send(Snapshot { label, gpus, process }).is_err() {
                    return;
                }
                thread::sleep(Duration::from_secs(if detailed { 1 } else { 3 }));
            }
        })
        .ok();
    probe
}
