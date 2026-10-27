use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime};

use eyre::Context as _;
use flume::{Receiver, Sender};
use slint::winit_030::{EventResult, WinitWindowAccessor, winit};
use slint::{ComponentHandle, Image, Model, ModelRc, SharedString, Timer, TimerMode, VecModel};

use crate::clips::{self, Clip, Hint, Status};
use crate::command::{self, Catalogue};
use crate::frames::{self, Frames, Shown};
use crate::runner::{self, Event, Job as Running, Progress};
use crate::settings::{Placement, Settings};
use crate::telemetry::{self, Probe, Snapshot};
use crate::{AppWindow, ClipRow, Job, Queue, View};

const LOG_LINES: usize = 5000;
const CLIP_FILTER: [&str; 5] = clips::VIDEO;

type ShownMsg = Result<Shown, eyre::Report>;
type StateRef = Rc<RefCell<State>>;

struct Run {
    index: usize,
    gpus: Vec<usize>,
    preview: PathBuf,
    job: Running,
    progress: Progress,
    started: Instant,
    phases: Vec<(String, Instant)>,
    last: String,
}

type Finished = (String, Progress, Instant, Vec<(String, Instant)>, Instant);

#[derive(PartialEq)]
enum Mode {
    Idle,
    Running,
    Stopping,
}

struct State {
    ui: slint::Weak<AppWindow>,
    clips: Vec<Clip>,
    settings: Settings,
    runs: Vec<Run>,
    catalogue: Catalogue,
    live: Option<SystemTime>,
    launches: usize,
    mode: Mode,
    counts: [usize; 3],
    finished: Option<Finished>,
    log: Rc<VecModel<SharedString>>,
    rows: Rc<VecModel<ClipRow>>,
    frames: Frames,
    views: [Option<PathBuf>; 5],
    awaiting: bool,
    detail: bool,
    snapshot: Snapshot,
    probe: Probe,
    events: (Sender<Event>, Receiver<Event>),
}

fn row(c: &Clip) -> ClipRow {
    ClipRow {
        name: c.name().into(),
        hint: c.hint_label().into(),
        status: c.status.label().into(),
        progress: c.progress,
        detail: c.detail.clone().into(),
    }
}

fn strings(v: impl IntoIterator<Item = String>) -> ModelRc<SharedString> {
    ModelRc::new(VecModel::from(v.into_iter().map(SharedString::from).collect::<Vec<_>>()))
}

fn settings_to_ui(s: &Settings, j: &Job, cat: &Catalogue) {
    j.set_models(strings(cat.models.iter().map(PathBuf::as_path).map(Catalogue::label)));
    j.set_model(i32::try_from(Catalogue::index(&cat.models, &s.model, "corridorkey_v2.safetensors")).unwrap_or_default());
    j.set_refiners(strings(cat.refiners.iter().map(PathBuf::as_path).map(Catalogue::label)));
    j.set_refiner(i32::try_from(Catalogue::index(&cat.refiners, &s.refiner, "corridorkey_edge_refiner_v10w.safetensors")).unwrap_or_default());
    j.set_gpu_names(strings(cat.gpus.iter().cloned()));
    let on: Vec<bool> = (0..cat.gpus.len()).map(|i| u32::try_from(i).is_ok_and(|i| !s.gpus_off.contains(&i))).collect();
    j.set_gpu_on(ModelRc::new(VecModel::from(on)));
    j.set_parallel(s.placement == Placement::PerGpu);
    j.set_codec(i32::from(s.nvenc));
    j.set_out_alpha(s.outputs[0]);
    j.set_out_fg(s.outputs[1]);
    j.set_out_checker(s.outputs[2]);
    j.set_out_cutout(s.outputs[3]);
    j.set_format(i32::from(s.exr.is_some()));
    j.set_exr_type(i32::from(s.exr == Some(true)));
    j.set_cutout_colour(i32::from(s.cutout_linear));
    j.set_despill(s.despill);
    j.set_hint_quality(s.hint_quality);
    j.set_carry(s.carry.is_some());
    j.set_carry_quality(s.carry.unwrap_or(0.95));
    j.set_vram_mode(i32::from(s.window.is_some()));
    let (chunk, stride) = s.window.unwrap_or((48, 24));
    j.set_chunk(chunk.cast_signed());
    j.set_stride(stride.cast_signed());
    j.set_frame_colour(i32::from(s.frame_colour));
    j.set_fps(s.fps.to_string().into());
    j.set_num_frames(s.frames.cast_signed());
    j.set_scale(i32::from(s.scale));
    j.set_refine(s.refine);
    j.set_out_mode(i32::from(s.out_parent.is_some()));
    j.set_out_parent(s.out_parent.clone().unwrap_or_default().into());
}

fn settings_from_ui(j: &Job, cat: &Catalogue) -> Settings {
    let mut outputs = [j.get_out_alpha(), j.get_out_fg(), j.get_out_checker(), j.get_out_cutout()];
    if !outputs.contains(&true) {
        outputs[0] = true;
    }
    Settings {
        nvenc: j.get_codec() == 1,
        outputs,
        exr: (j.get_format() == 1).then_some(j.get_exr_type() == 1),
        cutout_linear: j.get_cutout_colour() == 1,
        despill: (j.get_despill() * 1000.0).round() / 1000.0,
        hint_quality: (j.get_hint_quality() * 1000.0).round() / 1000.0,
        carry: j.get_carry().then(|| (j.get_carry_quality() * 1000.0).round() / 1000.0),
        window: (j.get_vram_mode() == 1)
            .then(|| (j.get_chunk().max(1).cast_unsigned(), j.get_stride().clamp(1, j.get_chunk().max(1)).cast_unsigned())),
        frame_colour: u8::try_from(j.get_frame_colour().clamp(0, 2)).unwrap_or_default(),
        fps: j.get_fps().parse().ok().filter(|f: &f32| *f > 0.0).unwrap_or(24.0),
        frames: j.get_num_frames().max(0).cast_unsigned(),
        scale: u8::try_from(j.get_scale().clamp(0, 3)).unwrap_or_default(),
        refine: j.get_refine(),
        out_parent: (j.get_out_mode() == 1).then(|| j.get_out_parent().into()),
        model: Catalogue::name(&cat.models, j.get_model()),
        refiner: Catalogue::name(&cat.refiners, j.get_refiner()),
        gpus_off: j.get_gpu_on().iter().zip(0u32..).filter(|(on, _)| !on).map(|(_, i)| i).collect(),
        placement: if j.get_parallel() { Placement::PerGpu } else { Placement::Together },
    }
}

fn clock(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60) } else { format!("{}:{:02}", s / 60, s % 60) }
}

impl State {
    fn ui(&self) -> Option<AppWindow> {
        self.ui.upgrade()
    }

    fn selected(&self) -> Option<usize> {
        let ui = self.ui()?;
        usize::try_from(ui.global::<Queue>().get_selected()).ok().filter(|&i| i < self.clips.len())
    }

    fn log(&self, line: impl Into<SharedString>) {
        let line: SharedString = line.into();
        eprintln!("LOG {line}");
        self.log.push(line);
        if self.log.row_count() > LOG_LINES {
            self.log.remove(0);
        }
        if let Some(ui) = self.ui() {
            ui.set_log_count(i32::try_from(self.log.row_count()).unwrap_or(i32::MAX));
        }
    }

    fn refresh_row(&self, i: usize) {
        if let Some(c) = self.clips.get(i) {
            self.rows.set_row_data(i, row(c));
        }
    }

    fn add(&mut self, paths: &[PathBuf]) {
        let found: Vec<Clip> = paths.iter().flat_map(|p| clips::discover(p)).collect();
        let fresh: Vec<Clip> = found.iter().filter(|c| !self.clips.iter().any(|o| o.source == c.source)).cloned().collect();
        let status = match (fresh.len(), found.len()) {
            (0, 0) => format!("No clips found in {}.", paths.first().map_or(String::new(), |p| p.display().to_string())),
            (0, _) => "Those clips are already in the queue.".into(),
            (n, _) => format!("Added {n} clip{}.", if n == 1 { "" } else { "s" }),
        };
        let first = self.clips.len();
        self.rows.extend(fresh.iter().map(row));
        self.clips.extend(fresh);
        if let Some(ui) = self.ui() {
            ui.set_status(status.into());
            if self.selected().is_none() && self.clips.len() > first {
                ui.global::<Queue>().set_selected(i32::try_from(first).unwrap_or(i32::MAX));
            }
        }
        self.show_selection();
    }

    fn remove(&mut self) {
        let Some(i) = self.selected() else { return };
        if self.clips[i].status == Status::Running {
            return;
        }
        self.clips.remove(i);
        self.rows.remove(i);
        self.runs.iter_mut().filter(|r| r.index > i).for_each(|r| r.index -= 1);
        if let Some(ui) = self.ui() {
            let next = if self.clips.is_empty() { -1 } else { i32::try_from(i.min(self.clips.len() - 1)).unwrap_or(i32::MAX) };
            ui.global::<Queue>().set_selected(next);
        }
        self.show_selection();
    }

    fn output_dir(&self, c: &Clip) -> PathBuf {
        c.output_dir.clone().unwrap_or_else(|| c.output_dir_for(self.settings.out_parent.as_deref().unwrap_or("")))
    }

    fn show_selection(&mut self) {
        let Some(ui) = self.ui() else { return };
        let (q, v) = (ui.global::<Queue>(), ui.global::<View>());
        let Some(i) = self.selected() else {
            q.set_clip_name("No clip selected".into());
            q.set_clip_where("Drop videos or frame folders anywhere in the window.".into());
            q.set_editable(false);
            q.set_hint_kind(0);
            q.set_hint_path(SharedString::new());
            self.views = Default::default();
            v.set_has(ModelRc::new(VecModel::from(vec![false; 5])));
            v.set_src_w(0);
            v.set_count(0);
            v.set_playing(false);
            return;
        };
        let c = &self.clips[i];
        q.set_clip_name(c.name().into());
        q.set_clip_where(format!("{}{}", if c.frames { "Frame sequence · " } else { "Video · " }, c.source.display()).into());
        q.set_editable(c.status != Status::Running);
        q.set_hint_kind(match c.hint {
            Hint::None => 0,
            Hint::FirstFrame => 1,
            Hint::Video => 2,
        });
        q.set_hint_path(c.hint_path.as_ref().map_or_else(SharedString::new, |p| p.display().to_string().into()));
        let outs = match self.runs.iter().find(|r| r.index == i) {
            Some(r) => clips::OUTPUTS.iter().zip(self.settings.outputs).map(|((name, _), on)| on.then(|| r.preview.join(format!("{name}.ppm")))).collect::<Vec<_>>(),
            None => c.outputs(&self.output_dir(c)).to_vec(),
        };
        self.views = [Some(c.source.clone()), outs[0].clone(), outs[1].clone(), outs[2].clone(), outs[3].clone()];
        v.set_has(ModelRc::new(VecModel::from(self.views.iter().map(Option::is_some).collect::<Vec<_>>())));
        let mut view = usize::try_from(v.get_view()).unwrap_or(0);
        if self.views[view].is_none() {
            view = 0;
        }
        if matches!(c.status, Status::Done | Status::Running) {
            view = [3, 1, 2].into_iter().find(|&k| self.views[k].is_some()).unwrap_or(view);
        }
        v.set_view(i32::try_from(view).unwrap_or_default());
        v.set_src_w(0);
        v.set_count(0);
        v.set_index(0);
        v.set_playing(false);
        self.request(view, 0);
    }

    fn linear(&self, view: usize) -> bool {
        view != 1 && (view != 0 || self.settings.frame_colour != 2)
    }

    fn request(&mut self, view: usize, index: usize) {
        let Some(path) = self.views.get(view).cloned().flatten() else { return };
        if path.extension().is_some_and(|e| e == "ppm") {
            self.live = None;
            self.live();
            return;
        }
        self.awaiting = true;
        if let Some(ui) = self.ui() {
            ui.global::<View>().set_busy(true);
        }
        self.frames.show(&path, index, self.linear(view), self.detail);
    }

    fn live(&mut self) {
        if let Some(ui) = self.ui() { let v = ui.global::<View>(); eprintln!("LIVE view={} src_w={} live={:?}", v.get_view(), v.get_src_w(), self.live); }
        let Some(ui) = self.ui() else { return };
        let v = ui.global::<View>();
        let current = usize::try_from(v.get_view()).ok().and_then(|k| self.views.get(k).cloned().flatten());
        let Some(path) = current.filter(|p| p.extension().is_some_and(|e| e == "ppm")) else { return };
        let stamp = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        if stamp.is_none() {
            v.set_message("Waiting for the first keyed frame…".into());
            v.set_src_w(0);
        }
        if stamp.is_none() || stamp == self.live {
            return;
        }
        if let Ok(px) = frames::ppm(&path) {
            self.live = stamp;
            v.set_src_h(px.height().cast_signed());
            v.set_src_w(px.width().cast_signed());
            v.set_frame(Image::from_rgb8(px));
            v.set_count(1);
            v.set_index(0);
        }
    }

    fn shown(&mut self, s: ShownMsg) {
        self.awaiting = false;
        let Some(ui) = self.ui() else { return };
        let v = ui.global::<View>();
        v.set_busy(false);
        match s {
            Err(e) => {
                v.set_message(e.to_string().into());
                v.set_src_w(0);
            }
            Ok(s) => {
                let current = usize::try_from(v.get_view()).ok().and_then(|k| self.views.get(k).cloned().flatten());
                if current.as_deref() != Some(s.path.as_path()) {
                    return;
                }
                v.set_frame(Image::from_rgb8(s.pixels));
                v.set_src_h(s.source.1.cast_signed());
                v.set_src_w(s.source.0.cast_signed());
                v.set_count(i32::try_from(s.count).unwrap_or(i32::MAX));
                v.set_index(i32::try_from(s.index).unwrap_or(i32::MAX));
                v.set_fps(s.fps);
            }
        }
    }

    fn start(&mut self) {
        if self.mode != Mode::Idle {
            return;
        }
        self.settings = self.ui().map_or_default(|ui| settings_from_ui(&ui.global::<Job>(), &self.catalogue));
        self.settings.save();
        if self.slots().iter().all(Vec::is_empty) && !self.catalogue.gpus.is_empty() {
            if let Some(ui) = self.ui() {
                ui.set_status("Every GPU is switched off. Enable one under GPUs.".into());
            }
            return;
        }
        if !self.clips.iter().any(|c| c.status == Status::Queued) {
            let Some(i) = self.selected() else {
                if let Some(ui) = self.ui() {
                    ui.set_status("Nothing queued. Add clips, or select finished ones to run again.".into());
                }
                return;
            };
            self.clips[i].status = Status::Queued;
            self.clips[i].progress = 0.0;
            self.clips[i].detail.clear();
            self.refresh_row(i);
        }
        self.counts = [0; 3];
        self.mode = Mode::Running;
        self.frames.release();
        if let Some(ui) = self.ui() {
            ui.set_running(true);
        }
        self.fill();
    }

    fn slots(&self) -> Vec<Vec<usize>> {
        let on: Vec<usize> = (0..self.catalogue.gpus.len()).filter(|&i| u32::try_from(i).is_ok_and(|i| !self.settings.gpus_off.contains(&i))).collect();
        if self.settings.placement == Placement::PerGpu && on.len() > 1 { on.into_iter().map(|g| vec![g]).collect() } else { vec![on] }
    }

    fn vacancy(&self) -> Option<(usize, Vec<usize>)> {
        let i = self.clips.iter().position(|c| c.status == Status::Queued)?;
        let slot = self.slots().into_iter().find(|s| !self.runs.iter().any(|r| r.gpus == *s))?;
        Some((i, slot))
    }

    fn fill(&mut self) {
        while let Some((i, gpus)) = self.vacancy().filter(|_| self.mode == Mode::Running) {
            self.launch(i, gpus);
        }
        if self.runs.is_empty() {
            self.finish();
        }
    }

    fn watch(&self) {
        self.probe.pid.store(self.runs.first().map_or(0, |r| r.job.pid), Ordering::Release);
    }

    fn launch(&mut self, i: usize, gpus: Vec<usize>) {
        self.launches += 1;
        let preview = std::env::temp_dir().join(format!("corridorkey-{}", std::process::id())).join(self.launches.to_string());
        let out = self.clips[i].output_dir_for(self.settings.out_parent.as_deref().unwrap_or(""));
        self.clips[i].output_dir = Some(out.clone());
        let cwd = command::repo().unwrap_or_else(|| PathBuf::from("."));
        let spawned = command::build(&self.clips[i], &self.settings, &out, (&gpus, &preview)).and_then(|cmd| {
            self.log(format!("=== {} ===", self.clips[i].name()));
            self.log(std::iter::once(cmd.program.display().to_string()).chain(cmd.args.iter().cloned()).collect::<Vec<_>>().join(" "));
            runner::spawn(&cmd, &cwd, &self.events.0).wrap_err_with(|| format!("could not start {}", cmd.program.display()))
        });
        let c = &mut self.clips[i];
        match spawned {
            Err(e) => {
                let why = e.to_string();
                (c.status, c.detail) = (Status::Failed, why.clone());
                self.counts[1] += 1;
                self.log(why);
                self.refresh_row(i);
            }
            Ok(job) => {
                (c.status, c.progress) = (Status::Running, 0.0);
                c.detail = if gpus.is_empty() { "Starting ck".into() } else { format!("Starting ck on GPU {}", gpus.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")) };
                let now = Instant::now();
                self.runs.push(Run { index: i, gpus, preview, job, progress: Progress::default(), started: now, phases: vec![("Starting".into(), now)], last: String::new() });
                self.watch();
                self.finished = None;
                self.refresh_row(i);
                self.update_bar(i);
                if self.selected() == Some(i) {
                    self.show_selection();
                }
            }
        }
    }

    fn update_bar(&self, i: usize) {
        let (Some(ui), Some(c)) = (self.ui(), self.clips.get(i)) else { return };
        ui.set_progress(c.progress);
        ui.set_status(format!("{} · {}", c.name(), c.detail).into());
    }

    fn line(&mut self, pid: u32, text: &str) {
        let tag = self.runs.iter().filter(|_| self.runs.len() > 1).find(|r| r.job.pid == pid).map(|r| format!("[{}] ", self.clips[r.index].name()));
        self.log(format!("{}{text}", tag.unwrap_or_default()));
        let Some(run) = self.runs.iter_mut().find(|r| r.job.pid == pid) else { return };
        if !text.trim().is_empty() {
            run.last = text.trim().chars().rev().take(110).collect::<Vec<_>>().into_iter().rev().collect();
        }
        if run.progress.feed(text) {
            let phase = if run.progress.detail.starts_with(|c: char| c.is_ascii_digit()) {
                "Keying".to_owned()
            } else {
                run.progress.detail.split(|c: char| c.is_ascii_digit()).next().unwrap_or("").trim().to_owned()
            };
            if run.phases.last().is_none_or(|(p, _)| *p != phase) {
                run.phases.push((phase, Instant::now()));
            }
            let i = run.index;
            let (value, detail) = (run.progress.value, run.progress.detail.clone());
            let c = &mut self.clips[i];
            (c.progress, c.detail) = (value, detail);
            self.refresh_row(i);
            self.update_bar(i);
        }
    }

    fn exited(&mut self, pid: u32, status: &Result<i32, eyre::Report>) {
        let Some(k) = self.runs.iter().position(|r| r.job.pid == pid) else { return };
        let run = self.runs.remove(k);
        self.watch();
        drop(std::fs::remove_dir_all(&run.preview));
        let i = run.index;
        let c = &mut self.clips[i];
        let stopping = self.mode == Mode::Stopping;
        match (status, stopping) {
            (_, true) => (c.status, c.progress, c.detail) = (Status::Stopped, 0.0, "Stopped".into()),
            (Ok(0), _) => (c.status, c.progress, c.detail) = (Status::Done, 1.0, "Done".into()),
            (Ok(code), _) => (c.status, c.detail) = (Status::Failed, if run.last.is_empty() { format!("exited with code {code}") } else { run.last.clone() }),
            (Err(sig), _) => (c.status, c.detail) = (Status::Failed, format!("killed by {sig}")),
        }
        let k = match c.status {
            Status::Done => 0,
            Status::Failed => 1,
            _ => 2,
        };
        self.counts[k] += 1;
        let name = c.name();
        let failed = (c.status == Status::Failed).then(|| c.detail.clone());
        let mut progress = run.progress;
        if c.status == Status::Done {
            progress.keyed = progress.frames;
        }
        self.finished = Some((c.status.label().into(), progress, run.started, run.phases, Instant::now()));
        self.refresh_row(i);
        if let (Some(ui), Some(why)) = (self.ui(), failed) {
            ui.set_status(format!("{name} failed: {why}").into());
            ui.set_log_visible(true);
        }
        if self.selected() == Some(i) {
            self.show_selection();
        }
        if !stopping {
            self.fill();
        } else if self.runs.is_empty() {
            self.finish();
        }
    }

    fn finish(&mut self) {
        self.mode = Mode::Idle;
        let Some(ui) = self.ui() else { return };
        ui.set_running(false);
        ui.set_progress(0.0);
        let parts: Vec<String> = ["done", "failed", "stopped"].iter().zip(self.counts).filter(|(_, n)| *n > 0).map(|(s, n)| format!("{n} {s}")).collect();
        ui.set_status(if parts.is_empty() { "Run finished.".into() } else { format!("Run finished: {}.", parts.join(", ")).into() });
        let q = ui.global::<Queue>();
        q.set_editable(self.selected().is_some());
    }

    fn stop(&mut self) {
        if !self.runs.is_empty() {
            self.mode = Mode::Stopping;
            self.runs.iter().for_each(|r| runner::stop(&r.job));
        }
    }

    fn stats(&self) -> String {
        let mut l = vec!["Stats for nerds".to_owned(), String::new()];
        let live = self.runs.first().map(|r| (self.clips[r.index].status.label().to_owned(), r.progress.clone(), r.started, r.phases.clone(), None, r.index));
        let done = self.finished.clone().map(|(s, p, t, ph, end)| (s, p, t, ph, Some(end), usize::MAX));
        match live.or(done) {
            None => l.push("clip       none yet · start the queue".into()),
            Some((status, p, started, phases, end, i)) => {
                let now = end.unwrap_or_else(Instant::now);
                let name = self.clips.get(i).map_or(String::new(), Clip::name);
                let frames = if p.frames > 0 { format!(" · {} frames {}", p.frames, p.size) } else { String::new() };
                l.push(format!("clip       {name}{}{status}{frames}", if name.is_empty() { "" } else { " · " }));
                if !p.window.is_empty() {
                    let at = if p.at.1 > 0 { format!("window {}/{} · ", p.at.0, p.at.1) } else { String::new() };
                    l.push(format!("window     {at}{}", p.window));
                }
                l.push(format!("progress   {:.0}% · {}", p.value * 100.0, p.detail));
                let spent = phases
                    .iter()
                    .enumerate()
                    .filter(|(_, (name, _))| !name.is_empty())
                    .fold(Vec::new(), |mut spent: Vec<(String, Duration)>, (k, (name, t))| {
                        let until = phases.get(k + 1).map_or(now, |n| n.1);
                        match spent.iter_mut().find(|(n, _)| n == name) {
                            Some((_, d)) => *d += until - *t,
                            None => spent.push((name.clone(), until - *t)),
                        }
                        spent
                    });
                let spent = spent.iter().map(|(n, d)| format!("{} {:.1}s", n.to_lowercase(), d.as_secs_f32())).collect::<Vec<_>>().join(" · ");
                l.push(format!("elapsed    {} · {spent}", clock(now - started)));
                if let Some((_, k)) = phases.iter().find(|(n, _)| n == "Keying").filter(|_| p.frames > 0) {
                    let keyed = if p.keyed > 0 { p.keyed } else { p.frames.checked_mul(p.at.0).and_then(|n| n.checked_div(p.at.1)).unwrap_or_default() };
                    let rate = keyed as f32 / (now - *k).as_secs_f32().max(1e-3);
                    let overall = end.map_or(String::new(), |_| format!(" · {:.2} fps overall", p.frames as f32 / (now - started).as_secs_f32().max(1e-3)));
                    l.push(format!("throughput {keyed} frames keyed · {rate:.2} fps{overall}"));
                }
                l.push(format!("process    {}", if self.runs.is_empty() { "idle" } else { self.snapshot.process.as_str() }));
            }
        }
        l.extend(self.snapshot.gpus.iter().cloned());
        if self.runs.len() > 1 {
            l.push(format!("running    {} clips · {}", self.runs.len(), self.runs.iter().map(|r| format!("{} on GPU {:?}", self.clips[r.index].name(), r.gpus)).collect::<Vec<_>>().join(" · ")));
        }
        if let Some(r) = self.runs.first() {
            l.push(format!("last       {}", r.last));
        }
        l.join("\n")
    }
}

fn later(task: impl std::future::Future<Output = ()> + 'static) {
    drop(slint::spawn_local(task));
}

fn dialog_dir() -> Option<PathBuf> {
    dirs::home_dir()
}

#[must_use]
fn wire_timers(state: &StateRef, shown_rx: Receiver<ShownMsg>, snap_rx: Receiver<Snapshot>) -> [Timer; 3] {
    let s = state.clone();
    let pump = Timer::default();
    pump.start(TimerMode::Repeated, Duration::from_millis(15), move || {
        let mut st = s.borrow_mut();
        let events: Vec<Event> = st.events.1.try_iter().take(500).collect();
        for e in events {
            match e {
                Event::Line(pid, l) => st.line(pid, &l),
                Event::Exit(pid, code) => st.exited(pid, &code),
            }
        }
        if let Some(last) = shown_rx.try_iter().last() {
            st.shown(last);
        }
        if let Some(snap) = snap_rx.try_iter().last() {
            if let Some(ui) = st.ui() {
                ui.set_gpu(snap.label.clone().into());
            }
            st.snapshot = snap;
        }
    });

    let s = state.clone();
    let refresh = Timer::default();
    refresh.start(TimerMode::Repeated, Duration::from_millis(250), move || {
        let Ok(mut st) = s.try_borrow_mut() else { return };
        st.live();
        if let Some(ui) = st.ui().filter(AppWindow::get_stats_visible) {
            ui.set_stats(st.stats().into());
        }
    });

    let s = state.clone();
    let play = Timer::default();
    play.start(TimerMode::Repeated, Duration::from_millis(1000 / 60), move || {
        let Ok(mut st) = s.try_borrow_mut() else { return };
        let Some(ui) = st.ui() else { return };
        let v = ui.global::<View>();
        if !v.get_playing() || st.awaiting {
            return;
        }
        let (i, n) = (v.get_index(), v.get_count());
        if n <= 1 {
            v.set_playing(false);
            return;
        }
        let view = usize::try_from(v.get_view()).unwrap_or(0);
        st.request(view, usize::try_from((i + 1) % n).unwrap_or(0));
    });
    [pump, refresh, play]
}

fn wire_actions(ui: &AppWindow, state: &StateRef) {
    let s = state.clone();
    ui.on_run(move || s.borrow_mut().start());
    let s = state.clone();
    ui.on_stop(move || s.borrow_mut().stop());
    let s = state.clone();
    ui.on_stats_toggled(move |on| {
        let st = s.borrow();
        st.probe.detailed.store(on, Ordering::Release);
        if let Some(ui) = st.ui() {
            ui.set_stats(st.stats().into());
        }
    });
}

fn wire_queue(ui: &AppWindow, state: &StateRef) {
    let q = ui.global::<Queue>();
    let s = state.clone();
    q.on_select(move |i| {
        let mut st = s.borrow_mut();
        if let Some(ui) = st.ui() {
            ui.global::<Queue>().set_selected(i);
        }
        st.show_selection();
    });
    let s = state.clone();
    q.on_remove(move || s.borrow_mut().remove());
    let s = state.clone();
    q.on_add_clips(move || {
        let s = s.clone();
        let picked = rfd::AsyncFileDialog::new().set_title("Add clips").add_filter("Clips", &CLIP_FILTER).set_directory(dialog_dir().unwrap_or_default()).pick_files();
        later(async move {
            if let Some(files) = picked.await {
                s.borrow_mut().add(&files.iter().map(|f| f.path().to_owned()).collect::<Vec<_>>());
            }
        });
    });
    let s = state.clone();
    q.on_add_folder(move || {
        let s = s.clone();
        let picked = rfd::AsyncFileDialog::new().set_title("Add a frame folder or a folder of clips").set_directory(dialog_dir().unwrap_or_default()).pick_folder();
        later(async move {
            if let Some(dir) = picked.await {
                s.borrow_mut().add(&[dir.path().to_owned()]);
            }
        });
    });
    let s = state.clone();
    q.on_hint_edited(move || {
        let mut st = s.borrow_mut();
        let (Some(ui), Some(i)) = (st.ui(), st.selected()) else { return };
        let q = ui.global::<Queue>();
        let c = &mut st.clips[i];
        c.hint = match q.get_hint_kind() {
            1 => Hint::FirstFrame,
            2 => Hint::Video,
            _ => Hint::None,
        };
        let text = q.get_hint_path();
        c.hint_path = (c.hint != Hint::None && !text.is_empty()).then(|| PathBuf::from(text.as_str()));
        st.refresh_row(i);
    });
    let w = ui.as_weak();
    q.on_choose_hint(move || {
        let w = w.clone();
        let picked = rfd::AsyncFileDialog::new()
            .set_title("Choose a hint")
            .add_filter("Hints", &["png", "jpg", "jpeg", "exr", "mp4", "mov", "avi", "mkv", "webm"])
            .pick_file();
        later(async move {
            if let (Some(f), Some(ui)) = (picked.await, w.upgrade()) {
                let q = ui.global::<Queue>();
                q.set_hint_path(f.path().display().to_string().into());
                q.invoke_hint_edited();
            }
        });
    });
}

fn wire_view(ui: &AppWindow, state: &StateRef) {
    let v = ui.global::<View>();
    let s = state.clone();
    v.on_show(move |view, index| {
        let mut st = s.borrow_mut();
        let (view, index) = (usize::try_from(view).unwrap_or(0), usize::try_from(index).unwrap_or(0));
        if let Some(ui) = st.ui() {
            ui.global::<View>().set_index(i32::try_from(index).unwrap_or(i32::MAX));
        }
        st.request(view, index);
    });
    let s = state.clone();
    v.on_detail(move |on| {
        let mut st = s.borrow_mut();
        if st.detail != on {
            st.detail = on;
            if let Some(ui) = st.ui() {
                let v = ui.global::<View>();
                let (view, index) = (usize::try_from(v.get_view()).unwrap_or(0), usize::try_from(v.get_index()).unwrap_or(0));
                st.request(view, index);
            }
        }
    });
    let s = state.clone();
    v.on_open_folder(move || {
        let st = s.borrow();
        let Some(i) = st.selected() else { return };
        let dir = st.output_dir(&st.clips[i]);
        if dir.is_dir() {
            let opener = if cfg!(windows) { "explorer" } else if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
            drop(std::process::Command::new(opener).arg(&dir).spawn());
        } else if let Some(ui) = st.ui() {
            ui.set_status(format!("{} has no output folder yet.", st.clips[i].name()).into());
        }
    });
}

fn wire_job(ui: &AppWindow, state: &StateRef) {
    let j = ui.global::<Job>();
    let s = state.clone();
    j.on_changed(move || {
        let mut st = s.borrow_mut();
        if let Some(ui) = st.ui() {
            st.settings = settings_from_ui(&ui.global::<Job>(), &st.catalogue);
            st.settings.save();
        }
    });
    let s = state.clone();
    j.on_gpu_toggled(move |i, on| {
        let mut st = s.borrow_mut();
        let Some(ui) = st.ui() else { return };
        let j = ui.global::<Job>();
        j.get_gpu_on().set_row_data(usize::try_from(i).unwrap_or_default(), on);
        st.settings = settings_from_ui(&j, &st.catalogue);
        st.settings.save();
    });
    let w = ui.as_weak();
    j.on_browse_out_parent(move || {
        let w = w.clone();
        let picked = rfd::AsyncFileDialog::new().set_title("Output folder").pick_folder();
        later(async move {
            if let (Some(d), Some(ui)) = (picked.await, w.upgrade()) {
                ui.global::<Job>().set_out_parent(d.path().display().to_string().into());
                ui.global::<Job>().invoke_changed();
            }
        });
    });
}

fn wire_window(ui: &AppWindow, state: &StateRef) {
    let s = state.clone();
    let dropped: Rc<RefCell<Vec<PathBuf>>> = Rc::default();
    let batch = Rc::new(Timer::default());
    ui.window().on_winit_window_event(move |_, event| {
        if let winit::event::WindowEvent::DroppedFile(path) = event {
            dropped.borrow_mut().push(path.clone());
            let (d, s) = (dropped.clone(), s.clone());
            batch.start(TimerMode::SingleShot, Duration::from_millis(50), move || {
                let paths: Vec<PathBuf> = d.borrow_mut().drain(..).collect();
                s.borrow_mut().add(&paths);
            });
            return EventResult::PreventDefault;
        }
        EventResult::Propagate
    });

    let s = state.clone();
    ui.window().on_close_requested(move || {
        s.borrow_mut().stop();
        slint::CloseRequestResponse::HideWindow
    });
}

pub fn run(paths: &[PathBuf]) -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;
    let (shown_tx, shown_rx) = flume::unbounded::<ShownMsg>();
    let (snap_tx, snap_rx) = flume::unbounded::<Snapshot>();
    let frames = Frames::new(move |s| {
        drop(shown_tx.send(s));
    });
    let probe = telemetry::start(snap_tx);
    let log = Rc::new(VecModel::default());
    let rows = Rc::new(VecModel::default());
    ui.set_log(ModelRc::from(log.clone()));
    ui.global::<Queue>().set_rows(ModelRc::from(rows.clone()));
    let settings = Settings::load();
    let catalogue = Catalogue::scan(telemetry::devices());
    settings_to_ui(&settings, &ui.global::<Job>(), &catalogue);
    ui.global::<Job>().set_ck_built(command::ck_binary().is_some());
    let state = StateRef::new(RefCell::new(State {
        ui: ui.as_weak(),
        clips: vec![],
        settings,
        runs: vec![],
        catalogue,
        live: None,
        launches: 0,
        mode: Mode::Idle,
        counts: [0; 3],
        finished: None,
        log,
        rows,
        frames,
        views: Default::default(),
        awaiting: false,
        detail: false,
        snapshot: Snapshot::default(),
        probe,
        events: flume::unbounded(),
    }));

    drop(wire_timers(&state, shown_rx, snap_rx));
    wire_actions(&ui, &state);
    wire_queue(&ui, &state);
    wire_view(&ui, &state);
    wire_job(&ui, &state);
    wire_window(&ui, &state);
    if !paths.is_empty() {
        state.borrow_mut().add(paths);
        state.borrow_mut().start();
    }
    ui.run()
}