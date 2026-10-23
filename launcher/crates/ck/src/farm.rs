//! Many GPUs, many clips. One worker per GPU loads every model. A clip's stage 1 is serial, so one worker leads it
//! (a GPU leads only once every faster one does), turning each finished frame into a stage-2 job on a shared queue.
//! Every worker drains that queue first; a lead finding it full runs jobs itself, so no GPU waits on a slower one.
//! Each clip's sink writes its frames in order.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::Scope;
use std::time::Duration;

use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::{Result, bail, ensure, eyre};
use flume::RecvTimeoutError;

use crate::clip::{Clip, upload};
use crate::frame::{self, Encoding, Geometry, Output};
use crate::io::Source;
use crate::run::{Config, Hints, Models, clip_names};
use crate::stage2::Baton;
use crate::vram;
use crate::write::{Exr, Writer};

/// Frames past the sink's next one that stage 2 may start, so finished frames cannot outrun the encoders.
const AHEAD: usize = 8;
/// Host bytes of stage-2 jobs that may wait in the queue.
const QUEUE_BYTES: usize = 3 << 30;

/// What every stage-2 job of a clip shares.
pub struct Meta {
    pub name: String,
    pub g: Geometry,
    pub pix: Dtype,
    pub done: flume::Sender<Done>,
    /// Frames the sink has handed to the encoders.
    pub written: Arc<AtomicUsize>,
}

/// One frame's stage 2: its blended baton (f32 NHWC, to be scaled by `scale`) and raw pixels. The baton stays on
/// the lead's GPU (`Tensor`) unless the job is queued for another (`Vec<u8>`).
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

/// A rendered frame, one image per requested output.
pub struct Done {
    frame: usize,
    images: Vec<Vec<u8>>,
}

/// Another thread failed first; its error is the one reported.
#[derive(Debug)]
struct Aborted;

impl std::fmt::Display for Aborted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("aborted")
    }
}

impl std::error::Error for Aborted {}

/// Model weights and PTX.
pub struct Assets {
    pub weights: PathBuf,
    pub vfms: PathBuf,
    pub ptx: PathBuf,
}

struct Shared<'a> {
    assets: &'a Assets,
    cfg: &'a Config,
    hints: &'a Hints,
    out_dir: &'a Path,
    clips: Mutex<VecDeque<(PathBuf, String)>>,
    jobs: (flume::Sender<Job>, flume::Receiver<Job>),
    queued: AtomicUsize,
    leading: Vec<AtomicBool>,
    alive: Vec<AtomicBool>,
    active: AtomicUsize,
    abort: AtomicBool,
    sinks: flume::Sender<Result<()>>,
}

impl Shared<'_> {
    fn clips(&self) -> Result<std::sync::MutexGuard<'_, VecDeque<(PathBuf, String)>>> {
        self.clips.lock().map_err(|_| eyre!("clip queue poisoned"))
    }

    /// Worker `k` may lead once every faster worker is leading or gone.
    fn may_lead(&self, k: usize) -> bool {
        (0..k).all(|j| self.leading[j].load(Ordering::Acquire) || !self.alive[j].load(Ordering::Acquire))
    }

    /// A queued job, waiting up to `wait` for one.
    fn take(&self, wait: Duration) -> Option<Job> {
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

/// A queued job on this GPU.
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
    frame::prep(gpu, &raw, &rgb, g, 3 * g.h * g.w, Encoding::Model, cfg.input())?;
    let logits = m.s2.forward(gpu, &m.cache, &rgb, Baton::Nhwc(acc, job.scale))?;
    let rendered = cfg
        .outputs
        .iter()
        .map(|&o| match cfg.exr {
            Some(t) => frame::render_exr(gpu, &logits, g, o, cfg.look, t == Exr::Half),
            None => frame::render(gpu, &logits, g, o, cfg.look),
        })
        .collect::<Result<Vec<_>>>()?;
    let images = rendered.iter().map(|img| gpu.download(img)).collect::<Result<Vec<_>>>()?;
    job.meta
        .done
        .send(Done {
            frame: job.frame,
            images,
        })
        .map_err(|_| eyre!("{}: writer gone", job.meta.name))
}

/// Writes `total` frames in order as they arrive, then closes the encoders.
fn sink(
    rx: &flume::Receiver<Done>,
    writers: Vec<(Output, Writer)>,
    (total, written): (usize, &AtomicUsize),
    abort: &AtomicBool,
) -> Result<()> {
    let mut pending = BTreeMap::new();
    let mut next = 0;
    while next < total {
        ensure!(!abort.load(Ordering::Acquire), Aborted);
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(d) => pending.insert(d.frame, d.images),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => bail!("{} frames never arrived", total - next),
        };
        while let Some(images) = pending.remove(&next) {
            writers.iter().zip(images).try_for_each(|((_, w), img)| w.write(img))?;
            next += 1;
            written.store(next, Ordering::Release);
        }
    }
    writers.into_iter().try_for_each(|(_, w)| w.finish())
}

/// Queues `job` while some other worker is free to drain the queue and it holds under [`QUEUE_BYTES`]; otherwise the
/// oldest queued job, or `job` itself, runs here, so a lone GPU keys each frame as it is blended.
fn submit(gpu: &Gpu, m: &Models, sh: &Shared, job: Job<Tensor>) -> Result<()> {
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
        match sh.take(Duration::ZERO) {
            Some(other) => queued(gpu, m, sh.cfg, other)?,
            None => return stage2(gpu, m, sh.cfg, &job),
        }
    }
}

/// Keys `input` with stage 1 on `gpu`, its window from `free` GiB when automatic.
fn lead<'s, 'e>(
    (gpu, m): (&Gpu, &Models),
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
    if cfg.auto {
        let free = free - cfg.encoder_gib((src.w, src.h));
        let (chunk, stride, est) = vram::window(g, total, free);
        println!("{name}: {free:.1} GiB free → window {chunk}, stride {stride} (estimated peak {est:.1} GiB)");
        if est > free {
            println!("warning: even one frame may not fit at this resolution");
        }
        (cfg.chunk, cfg.stride) = (chunk, stride);
    }
    if src.float() && !cfg.linear_input {
        println!("warning: float frames read as display-coded and clipped to [0, 1]; pass --linear for scene-linear");
    }
    std::fs::create_dir_all(sh.out_dir)?;
    let writers = cfg
        .outputs
        .iter()
        .map(|&o| Ok((o, cfg.writer(sh.out_dir, &name, &src, o)?)))
        .collect::<Result<Vec<_>>>()?;
    let (done, rx) = flume::unbounded();
    let written = Arc::new(AtomicUsize::new(0));
    let sunk = written.clone();
    s.spawn(move || {
        let r = sink(&rx, writers, (total, &sunk), &sh.abort);
        if r.is_err() {
            sh.abort.store(true, Ordering::Release);
        }
        let _ = sh.sinks.send(r);
    });
    println!(
        "{name}: {total} frames {}x{} → {}x{}, window {}, stride {}",
        src.w, src.h, g.w, g.h, cfg.chunk, cfg.stride
    );
    let meta = Arc::new(Meta {
        name,
        g,
        pix: src.dtype(),
        done,
        written,
    });
    Clip::open((gpu, m, &cfg), &src, total, sh.hints, meta)?.run(total, |job| submit(gpu, m, sh, job))
}

fn worker<'s, 'e>(k: usize, device: usize, sh: &'e Shared<'e>, s: &'s Scope<'s, 'e>) -> Result<()> {
    let gpu = Gpu::open(device, &sh.assets.ptx)?;
    let free = gpu.ctx.mem_get_info()?.0 as f64 / f64::from(1u32 << 30);
    let m = Models::load(&gpu, &sh.assets.weights, &sh.assets.vfms)?;
    println!("GPU {device}: sm_{}, {free:.1} GiB free", gpu.sm);
    loop {
        if sh.abort.load(Ordering::Acquire) {
            return Ok(());
        }
        if let Some(job) = sh.take(Duration::ZERO) {
            queued(&gpu, &m, sh.cfg, job)?;
            continue;
        }
        let next = if sh.may_lead(k) { sh.clips()?.pop_front() } else { None };
        if let Some((input, name)) = next {
            sh.leading[k].store(true, Ordering::Release);
            sh.active.fetch_add(1, Ordering::AcqRel);
            let r = lead((&gpu, &m), sh, s, (&input, name), free);
            sh.leading[k].store(false, Ordering::Release);
            sh.active.fetch_sub(1, Ordering::AcqRel);
            r?;
            continue;
        }
        if sh.idle()? {
            println!("GPU {device}: {} frames through stage 2", KEYED.get());
            return Ok(());
        }
        if let Some(job) = sh.take(Duration::from_millis(5)) {
            queued(&gpu, &m, sh.cfg, job)?;
        }
    }
}

/// Keys every input across `devices`, writing each clip's outputs into `out_dir`.
///
/// # Errors
/// The first worker or writer failure.
pub fn run(
    assets: &Assets,
    devices: &[usize],
    inputs: &[PathBuf],
    out_dir: &Path,
    cfg: &Config,
    hints: &Hints,
) -> Result<()> {
    ensure!(!devices.is_empty(), "no CUDA devices");
    let (sinks, results) = flume::unbounded();
    let sh = Shared {
        assets,
        cfg,
        hints,
        out_dir,
        clips: Mutex::new(inputs.iter().cloned().zip(clip_names(inputs)).collect()),
        jobs: flume::unbounded(),
        queued: AtomicUsize::new(0),
        leading: devices.iter().map(|_| AtomicBool::new(false)).collect(),
        alive: devices.iter().map(|_| AtomicBool::new(true)).collect(),
        active: AtomicUsize::new(0),
        abort: AtomicBool::new(false),
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
    let errors: Vec<eyre::Report> = workers.into_iter().chain(results.drain()).filter_map(Result::err).collect();
    let knock_on = |e: &eyre::Report| e.chain().any(<dyn std::error::Error>::is::<Aborted>);
    errors
        .into_iter()
        .min_by_key(knock_on)
        .map_or(Ok(()), Err)
}
