use crate::convert::{Metric, Narrow};
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::Scope;
use std::time::{Duration, Instant};

use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::{Result, bail, ensure, eyre};
use flume::RecvTimeoutError;
use tracing::{info, instrument, warn};

use crate::clip::{Clip, upload};
use crate::frame::{self, Encoding, Geometry, Input, Output};
use crate::io::Source;
use crate::preview::Preview;
use crate::run::{Config, Hints, Models, clip_names};
use crate::stage2::Baton;
use crate::vram;
use crate::write::{Exr, Writer};

const AHEAD: usize = 8;
const QUEUE_BYTES: usize = 3 << 30;
const TELL: Duration = Duration::from_secs(2);

pub struct Meta {
    pub name: String,
    pub g: Geometry,
    pub pix: Dtype,
    pub done: flume::Sender<Done>,
    pub written: Arc<AtomicUsize>,
}

pub struct Job<B = Vec<u8>> {
    pub meta: Arc<Meta>,
    pub frame: usize,
    pub baton: B,
    pub scale: f32,
    pub raw: Vec<u8>,
}

impl<B> Job<B> {
    fn with<C>(self, baton: C) -> Job<C> {
        Job {
            meta: self.meta,
            frame: self.frame,
            baton,
            scale: self.scale,
            raw: self.raw,
        }
    }
}

impl Job {
    fn bytes(&self) -> usize {
        self.baton.len() + self.raw.len()
    }
}

pub struct Done {
    frame: usize,
    images: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct Aborted;

impl std::fmt::Display for Aborted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("aborted")
    }
}

impl std::error::Error for Aborted {}

pub struct Assets {
    pub weights: PathBuf,
    pub vfms: PathBuf,
    pub ptx: PathBuf,
}

/// GPUs and models kept loaded between runs by a long-lived `ck --serve` worker.
#[derive(Default)]
pub struct Resident(Mutex<ahash::AHashMap<usize, Loaded>>);

struct Loaded {
    gpu: Gpu,
    m: Models,
    /// GiB the models took when loaded, credited back to free memory on reuse.
    held: f64,
}

impl Resident {
    fn take(&self, device: usize) -> Option<Loaded> {
        self.0.lock().ok()?.remove(&device)
    }

    fn put(&self, device: usize, l: Loaded) {
        if let Ok(mut slots) = self.0.lock() {
            slots.insert(device, l);
        }
    }
}

/// Per-run hooks for a resident worker: its model cache, and a flag that aborts the run.
#[derive(Clone, Copy)]
pub struct Session<'a> {
    pub resident: Option<&'a Resident>,
    pub abort: &'a AtomicBool,
}

struct Shared<'a> {
    assets: &'a Assets,
    resident: Option<&'a Resident>,
    cfg: &'a Config,
    hints: &'a Hints,
    out_dir: &'a Path,
    clips: Mutex<VecDeque<(PathBuf, String)>>,
    jobs: (flume::Sender<Job>, flume::Receiver<Job>),
    queued: AtomicUsize,

    leading: Vec<AtomicBool>,
    alive: Vec<AtomicBool>,
    active: AtomicUsize,
    abort: &'a AtomicBool,
    loaded: AtomicBool,
    depth: Vec<usize>,
    sinks: flume::Sender<Result<()>>,
}

impl Shared<'_> {
    fn clips(&self) -> Result<std::sync::MutexGuard<'_, VecDeque<(PathBuf, String)>>> {
        self.clips.lock().map_err(|_| eyre!("clip queue poisoned"))
    }

    fn may_lead(&self, k: usize) -> bool {
        (0..k).all(|j| self.leading[j].load(Ordering::Acquire) || !self.alive[j].load(Ordering::Acquire))
    }

    fn take(&self, k: usize, wait: Duration) -> Option<Job> {
        if self.jobs.1.len() < self.depth[k] {
            std::thread::sleep(wait);
            return None;
        }
        let job = if wait.is_zero() {
            self.jobs.1.try_recv().ok()
        } else {
            self.jobs.1.recv_timeout(wait).ok()
        }?;
        self.queued.fetch_sub(job.bytes(), Ordering::AcqRel);
        Some(job)
    }

    fn idle(&self) -> Result<bool> {
        Ok(self.clips()?.is_empty() && self.active.load(Ordering::Acquire) == 0 && self.jobs.1.is_empty())
    }
}

thread_local! {
    static KEYED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn queued(gpu: &Gpu, m: &Models, cfg: &Config, job: Job) -> Result<()> {
    let g = job.meta.g;
    let acc = gpu.upload(&job.baton, &[1, g.h / 16, g.w / 16, 512], Dtype::F32)?;
    stage2(gpu, m, cfg, &job.with(acc))
}

fn stage2(gpu: &Gpu, m: &Models, cfg: &Config, job: &Job<Tensor>) -> Result<()> {
    while job.frame >= job.meta.written.load(Ordering::Acquire) + AHEAD {
        ensure!(!job.meta.done.is_disconnected(), "{}: writer gone", job.meta.name);
        std::thread::sleep(Duration::from_millis(2));
    }
    KEYED.set(KEYED.get() + 1);
    let g = job.meta.g;
    let acc = &job.baton;
    let rgb = gpu.empty(&[1, 3, g.h, g.w], Dtype::F32)?;
    let raw = upload(gpu, &[&job.raw], &[1, 3 * g.hn * g.wn], job.meta.pix)?;
    let linear = job.meta.pix == Dtype::F32 && cfg.linear_input;
    let planes = match &m.refiner {
        Some(_) if !g.native() => Some(frame::planes(gpu, &raw, g)?),

        _ => None,
    };
    match &planes {
        Some((_, low)) => {
            let input = Input { linear, ..cfg.input() };
            frame::prep_planes(gpu, low, &rgb, g, 3 * g.h * g.w, Encoding::Model, input)?;
        }
        None => frame::prep(gpu, &raw, &rgb, g, 3 * g.h * g.w, Encoding::Model, cfg.input())?,
    }
    let logits = m.s2.forward(gpu, &m.cache, &rgb, Baton::Nhwc(acc, job.scale))?;
    let (logits, g) = match (&m.refiner, &planes) {
        (Some(r), Some((native, low))) => (
            r.native(

                gpu,
                &logits,
                (native, low),
                g,
                Input {
                    linear,
                    adjust: cfg.look.adjust,
                },
            )?,
            g.output(),
        ),
        _ => (logits, g),
    };

    let rendered = cfg
        .outputs
        .iter()
        .map(|&o| match cfg.exr {
            Some(t) => frame::render_exr(gpu, &logits, g, o, cfg.look, t == Exr::Half),
            None => frame::render(gpu, &logits, g, o, cfg.look),
        })
        .collect::<Result<Vec<_>>>()?;
    let images = rendered
        .iter()
        .map(|img| gpu.download(img))
        .collect::<Result<Vec<_>>>()?;

    job.meta
        .done
        .send(Done {
            frame: job.frame,
            images,
        })
        .map_err(|_| eyre!("{}: writer gone", job.meta.name))
}

fn sink(
    rx: &flume::Receiver<Done>,
    (writers, mut preview): (Vec<(Output, Writer)>, Option<Preview>),
    (total, written): (usize, &AtomicUsize),
    abort: &AtomicBool,
) -> Result<()> {
    let mut pending = BTreeMap::new();
    let mut next = 0;

    let (start, mut told) = (Instant::now(), Instant::now());
    while next < total {
        ensure!(!abort.load(Ordering::Acquire), Aborted);
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(d) => pending.insert(d.frame, d.images),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => bail!("{} frames never arrived", total - next),
        };

        while let Some(images) = pending.remove(&next) {
            if let Some(p) = preview.as_mut() {
                p.offer(next, writers.iter().map(|(o, _)| *o).zip(images.iter().map(Vec::as_slice)))?;
            }
            writers.iter().zip(images).try_for_each(|((_, w), img)| w.write(img))?;
            next += 1;
            written.store(next, Ordering::Release);
            if told.elapsed() >= TELL || next == total {
                told = Instant::now();

                info!(
                    "{next}/{total} frames written · {:.2} fps",
                    next.f64() / start.elapsed().as_secs_f64()
                );
            }
        }
    }
    writers.into_iter().try_for_each(|(_, w)| w.finish())
}

fn submit((gpu, m, k): (&Gpu, &Models, usize), sh: &Shared, job: Job<Tensor>) -> Result<()> {
    let bytes = job.baton.bytes() + job.raw.len();
    let free = sh
        .alive
        .iter()
        .zip(&sh.leading)
        .any(|(a, l)| a.load(Ordering::Acquire) && !l.load(Ordering::Acquire));
    loop {
        ensure!(!sh.abort.load(Ordering::Acquire), Aborted);

        if free && sh.queued.load(Ordering::Acquire) + bytes <= QUEUE_BYTES {
            sh.queued.fetch_add(bytes, Ordering::AcqRel);
            let baton = gpu.download(&job.baton)?;
            return sh.jobs.0.send(job.with(baton)).map_err(|_| eyre!("job queue closed"));
        }
        match sh.take(k, Duration::ZERO) {
            Some(other) => queued(gpu, m, sh.cfg, other)?,
            None => return stage2(gpu, m, sh.cfg, &job),
        }
    }
}

#[allow(clippy::many_single_char_names)] // tensor axes: (b, c, h, w)
#[instrument(name = "clip", skip_all, fields(input = %input.display()))]
fn lead<'s, 'e>(
    (gpu, m, k): (&Gpu, &Models, usize),
    sh: &'e Shared<'e>,
    s: &'s Scope<'s, 'e>,
    (input, name): (&Path, String),
    free: f64,
) -> Result<()> {
    let src = Source::open(input, sh.cfg.dir_fps)?;
    let total = sh.cfg.frames.map_or(src.frames, |n| n.min(src.frames));
    ensure!(total > 0, "{}: no frames", input.display());
    let g = Geometry::scaled(src.h, src.w, sh.cfg.scale, sh.cfg.quantum);

    let mut cfg = sh.cfg.clone();
    let free = free.min(gib(gpu)?.1 * cfg.vram_share / 100.0) - cfg.encoder_gib((src.w, src.h));
    if !cfg.auto {
        let need = vram::estimate(g, cfg.chunk.min(total), gpu.sm);
        let fits = vram::window(g, total, free, gpu.sm).0;
        ensure!(
            need <= free || cfg.chunk <= fits,
            "a {}-frame window needs about {need:.1} GiB at {}x{} but {free:.1} GiB is free; use a chunk of {fits} or less, or Auto",
            cfg.chunk,
            g.w,
            g.h,
        );
    }
    if cfg.auto {
        let (chunk, stride, est) = vram::window(g, total, free, gpu.sm);
        info!("{free:.1} GiB free → window {chunk}, stride {stride} (estimated peak {est:.1} GiB)");
        if est > free {
            warn!("even one frame may not fit at this resolution");
        }
        (cfg.chunk, cfg.stride) = (chunk, stride);
    }
    if src.float() && !cfg.linear_input {
        warn!("float frames read as display-coded and clipped to [0, 1]; pass --linear for scene-linear");
    }

    std::fs::create_dir_all(sh.out_dir)?;
    let writers = cfg
        .outputs
        .iter()
        .map(|&o| Ok((o, cfg.writer(sh.out_dir, &name, &src, o)?)))
        .collect::<Result<Vec<_>>>()?;
    let preview = cfg
        .preview
        .as_ref()
        .map(|to| Preview::new(to, (src.w, src.h), cfg.exr.is_some()))
        .transpose()?;

    let (done, rx) = flume::unbounded();
    let written = Arc::new(AtomicUsize::new(0));
    let sunk = written.clone();
    s.spawn(move || {
        let r = sink(&rx, (writers, preview), (total, &sunk), sh.abort);
        if r.is_err() {
            sh.abort.store(true, Ordering::Release);
        }

        drop(sh.sinks.send(r));
    });
    info!(
        "{total} frames {}x{} → {}x{}, window {}, stride {}",
        src.w, src.h, g.w, g.h, cfg.chunk, cfg.stride
    );
    let meta = Arc::new(Meta {
        name,
        g,
        pix: src.dtype(),
        done,
        written,
    });
    Clip::open((gpu, m, &cfg), &src, total, sh.hints, meta)?.run(total, |job| submit((gpu, m, k), sh, job))
}

/// Free and total GiB.
fn gib(gpu: &Gpu) -> Result<(f64, f64)> {
    let (free, total) = gpu.ctx.mem_get_info()?;
    let g = |b: usize| b.f64() / f64::from(1u32 << 30);
    Ok((g(free), g(total)))
}

fn load(device: usize, lead: bool, sh: &Shared) -> Result<Loaded> {
    let gpu = Gpu::open(device, &sh.assets.ptx)?;
    let before = gib(&gpu)?.0;
    let m = Models::load(&gpu, &sh.assets.weights, &sh.assets.vfms, sh.cfg.refine.as_deref())?;
    if lead {
        m.front(&gpu)?;
    }
    let held = (before - gib(&gpu)?.0).max(0.0);
    Ok(Loaded { gpu, m, held })
}

#[instrument(name = "gpu", skip_all, fields(id = device))]
fn worker<'s, 'e>(k: usize, device: usize, sh: &'e Shared<'e>, s: &'s Scope<'s, 'e>) -> Result<()> {
    while k > 0 && !sh.loaded.load(Ordering::Acquire) && !sh.abort.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(10));
    }
    if sh.abort.load(Ordering::Acquire) {
        return Ok(());
    }

    let l = sh
        .resident
        .and_then(|r| r.take(device))
        .map_or_else(|| load(device, k == 0, sh), Ok)?;
    if k == 0 {
        l.m.front(&l.gpu)?;
    }
    let free = gib(&l.gpu)?.0 + l.held;
    sh.loaded.store(true, Ordering::Release);
    info!("sm_{}, {free:.1} GiB free", l.gpu.sm);
    loop {
        if matches!(tick((&l.gpu, &l.m, k), sh, s, free)?, Tick::Finish) {
            break;
        }
    }
    if let Some(r) = sh.resident {
        r.put(device, l);
    }
    Ok(())
}

enum Tick {
    Finish,
    CarryOn,
}

fn tick<'s, 'e>(
    (gpu, m, key): (&Gpu, &Models, usize),
    sh: &'e Shared<'e>,
    s: &'s Scope<'s, 'e>,
    free: f64,
) -> Result<Tick> {
    if sh.abort.load(Ordering::Acquire) {
        return Ok(Tick::Finish);
    }

    if let Some(job) = sh.take(key, Duration::ZERO) {
        queued(gpu, m, sh.cfg, job)?;
        return Ok(Tick::CarryOn);
    }
    let next = if sh.may_lead(key) {
        sh.clips()?.pop_front()
    } else {
        None
    };

    if let Some((input, name)) = next {
        sh.leading[key].store(true, Ordering::Release);
        sh.active.fetch_add(1, Ordering::AcqRel);
        let r = lead((gpu, m, key), sh, s, (&input, name), free);
        sh.leading[key].store(false, Ordering::Release);
        sh.active.fetch_sub(1, Ordering::AcqRel);
        r?;
        return Ok(Tick::CarryOn);
    }

    if sh.idle()? {
        info!("{} frames through stage 2", KEYED.get());
        return Ok(Tick::Finish);
    }
    if let Some(job) = sh.take(key, Duration::from_millis(5)) {
        queued(gpu, m, sh.cfg, job)?;
    }
    Ok(Tick::CarryOn)
}

pub fn run(
    assets: &Assets,
    devices: &[usize],
    inputs: &[PathBuf],
    out_dir: &Path,
    cfg: &Config,
    hints: &Hints,
) -> Result<()> {
    let abort = AtomicBool::new(false);
    run_in(Session { resident: None, abort: &abort }, assets, devices, inputs, out_dir, cfg, hints)
}

/// `run`, reusing a worker's loaded models and honouring its abort flag.
pub fn run_in(
    session: Session<'_>,
    assets: &Assets,
    devices: &[usize],
    inputs: &[PathBuf],
    out_dir: &Path,
    cfg: &Config,
    hints: &Hints,
) -> Result<()> {
    ensure!(!devices.is_empty(), "no CUDA devices");

    let mut ranked = devices
        .iter()
        .map(|&d| Ok((Gpu::speed(d)?, d)))
        .collect::<Result<Vec<_>>>()?;
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let depth: Vec<usize> = ranked
        .iter()
        .map(|r| (ranked[0].0 / r.0).ceil().max(0.0).usize())
        .collect();
    let devices: Vec<usize> = ranked.into_iter().map(|(_, d)| d).collect();
    let (sinks, results) = flume::unbounded();

    let sh = Shared {
        assets,
        resident: session.resident,
        cfg,
        hints,
        out_dir,
        clips: Mutex::new(inputs.iter().cloned().zip(clip_names(inputs)).collect()),
        jobs: flume::unbounded(),
        queued: AtomicUsize::new(0),

        leading: devices.iter().map(|_| AtomicBool::new(false)).collect(),
        alive: devices.iter().map(|_| AtomicBool::new(true)).collect(),
        active: AtomicUsize::new(0),
        abort: session.abort,
        loaded: AtomicBool::new(false),
        depth,
        sinks,
    };

    let workers: Vec<Result<()>> = std::thread::scope(|s| {
        let sh = &sh;
        let handles: Vec<_> = devices
            .iter()
            .enumerate()
            .map(|(k, &dev)| {
                s.spawn(move || {
                    let r = worker(k, dev, sh, s).map_err(|e| e.wrap_err(format!("GPU {dev}")));

                    if r.is_err() {
                        sh.abort.store(true, Ordering::Release);
                    }
                    sh.alive[k].store(false, Ordering::Release);
                    r
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| eyre!("worker panicked"))?)
            .collect()
    });
    let errors: Vec<eyre::Report> = workers
        .into_iter()
        .chain(results.drain())
        .filter_map(Result::err)
        .collect();
    let knock_on = |e: &eyre::Report| e.chain().any(<dyn std::error::Error>::is::<Aborted>);
    errors.into_iter().min_by_key(knock_on).map_or(Ok(()), Err)
}
