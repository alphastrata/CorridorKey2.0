use std::collections::VecDeque;
use std::os::unix::net::UnixStream;
use std::thread;
use std::time::{Duration, Instant};

use flume::{Receiver, Sender};

use crate::process::{self, Spec};
use crate::{CONTROL_FD, Event, Exit, Frame, JobInfo, PREVIEW_FD, Work, send};

/// A resident worker prints this, then the exit code, on stderr when a job ends.
pub const DONE: &str = "\u{1e}ck:done ";
const KEEP_LINES: usize = 4000;
const KEEP_DONE: usize = 64;
const GRACE: Duration = Duration::from_secs(10);
/// Flags whose values decide what a resident worker has loaded.
const LOADS: [&str; 7] = ["--device", "--weights", "--vfms", "--refiner", "--ptx", "--upscale", "--model-scale"];

/// A job's record and its buffered `(ms, line)` output.
pub type Snapshot = (JobInfo, Vec<(u64, String)>);
pub type Started = std::io::Result<(String, u32)>;

pub enum Hub {
    Join(u64, Sender<Event>),
    Leave(u64),
    List(Sender<Vec<JobInfo>>),
    Attach(String, Sender<Option<Snapshot>>),
    Submit { spec: Spec, preview: bool, resident: bool, keep_alive: Duration, reply: Sender<Started> },
    Stop(u32),
    Expire(u32, String),
    Line(u32, String),
    Frame(u32, Frame),
    Gone(u32, Exit),
    Tick,
}

struct Record {
    info: JobInfo,
    started: Instant,
    lines: VecDeque<(u64, String)>,
}

impl Record {
    fn ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn info(&self) -> JobInfo {
        let mut info = self.info.clone();
        if info.exit.is_none() {
            info.elapsed_ms = self.ms();
        }
        info
    }
}

/// A long-lived `ck --serve` holding models for one GPU set and weights.
struct Worker {
    key: Vec<(String, String)>,
    pid: u32,
    ctl: UnixStream,
    queue: VecDeque<(u64, Vec<String>)>,
    /// The job running now, if any.
    busy: Option<u64>,
    idle: Instant,
    keep_alive: Duration,
}

impl Worker {
    fn devices(&self) -> Option<&str> {
        self.key.iter().find(|(k, _)| k == "--device").map(|(_, v)| v.as_str())
    }

    fn overlaps(&self, other: Option<&str>) -> bool {
        let set = |d: &str| d.split(',').map(str::to_owned).collect::<Vec<_>>();
        match (self.devices(), other) {
            (Some(a), Some(b)) => set(a).iter().any(|g| set(b).contains(g)),
            _ => true,
        }
    }

    fn run(&mut self, job: u64, args: Vec<String>) {
        self.busy = send(&mut self.ctl, &Work::Run { job, args }).is_ok().then_some(job);
    }

    /// Starts the next queued job, or goes idle.
    fn advance(&mut self) {
        if let Some((job, args)) = self.queue.pop_front() {
            self.run(job, args);
        } else {
            (self.busy, self.idle) = (None, Instant::now());
        }
    }
}

fn key(spec: &Spec) -> Vec<(String, String)> {
    let mut k: Vec<(String, String)> = spec
        .args
        .windows(2)
        .filter(|w| LOADS.contains(&w[0].as_str()))
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect();
    k.extend([
        ("program".into(), spec.program.clone()),
        ("cwd".into(), spec.cwd.clone()),
        ("nice".into(), spec.nice.to_string()),
    ]);
    k.extend(spec.env.iter().cloned());
    k
}

struct State {
    tx: Sender<Hub>,
    sessions: Vec<(u64, Sender<Event>)>,
    jobs: Vec<Record>,
    workers: Vec<Worker>,
    next: u64,
}

impl State {
    fn emit(&mut self, e: &Event) {
        self.sessions.retain(|(_, s)| s.send(e.clone()).is_ok());
    }

    fn running(&mut self, pid: u32) -> Option<&mut Record> {
        self.jobs.iter_mut().find(|r| r.info.pid == pid && r.info.exit.is_none())
    }

    fn record(&mut self, pid: u32, args: Vec<String>) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        let info = JobInfo { id: id.clone(), pid, args, elapsed_ms: 0, lines: 0, exit: None };
        self.jobs.push(Record { info, started: Instant::now(), lines: VecDeque::new() });
        id
    }

    fn finish(&mut self, pid: u32, exit: Exit) {
        let Some(r) = self.running(pid) else { return };
        r.info.elapsed_ms = r.ms();
        r.info.exit = Some(exit.clone());
        let id = r.info.id.clone();
        if self.jobs.iter().filter(|r| r.info.exit.is_some()).count() > KEEP_DONE
            && let Some(k) = self.jobs.iter().position(|r| r.info.exit.is_some())
        {
            self.jobs.remove(k);
        }
        self.emit(&Event::Exit { id, pid, exit });
    }

    fn worker(&mut self, spec: &Spec, keep_alive: Duration) -> std::io::Result<usize> {
        let key = key(spec);

        if let Some(k) = self.workers.iter().position(|w| w.key == key) {
            return Ok(k);
        }

        let devices = key.iter().find(|(k, _)| k == "--device").map(|(_, v)| v.clone());
        self.workers
            .iter()
            .filter(|w| w.busy.is_none() && w.overlaps(devices.as_deref()))
            .for_each(|w| process::kill(w.pid));

        let ((frames, theirs_f), (ctl, theirs_c)) = (UnixStream::pair()?, UnixStream::pair()?);
        let serve = Spec { args: vec![], env: spec.env.clone(), cwd: spec.cwd.clone(), program: spec.program.clone(), nice: spec.nice };
        let child = process::spawn(&serve, &["--serve"], &[(&theirs_f, PREVIEW_FD), (&theirs_c, CONTROL_FD)])?;
        let pid = process::watch(child, Some(frames), &self.tx);
        self.workers.push(Worker { key, pid, ctl, queue: VecDeque::new(), busy: None, idle: Instant::now(), keep_alive });
        Ok(self.workers.len() - 1)
    }

    fn submit(&mut self, spec: Spec, (preview, resident, keep_alive): (bool, bool, Duration)) -> Started {
        let mut args = spec.args.clone();

        if preview {
            args.extend(["--preview-fd".into(), PREVIEW_FD.to_string()]);
        }

        if resident {
            let k = self.worker(&spec, keep_alive)?;
            let pid = self.workers[k].pid;
            let id = self.record(pid, spec.args);

            let job = self.next;
            self.next += 1;

            let w = &mut self.workers[k];
            if w.busy.is_some() {
                w.queue.push_back((job, args));
            } else {
                w.run(job, args);
            }

            return Ok((id, pid));
        }

        let pair = preview.then(UnixStream::pair).transpose()?;
        let fds: Vec<(&UnixStream, i32)> = pair.iter().map(|(_, t)| (t, PREVIEW_FD)).collect();
        let extra: Vec<&str> = args[spec.args.len()..].iter().map(String::as_str).collect();
        let child = process::spawn(&spec, &extra, &fds)?;
        let pid = process::watch(child, pair.map(|(ours, _)| ours), &self.tx);
        Ok((self.record(pid, spec.args), pid))
    }

    fn line(&mut self, pid: u32, text: String) {
        if let Some(code) = text.strip_prefix(DONE) {
            self.finish(pid, Exit { code: code.trim().parse().ok(), signal: None });
            return self.workers.iter_mut().filter(|w| w.pid == pid).for_each(Worker::advance);
        }
        let Some(r) = self.running(pid) else { return };
        let (ms, seq) = (r.ms(), r.info.lines);
        r.info.lines += 1;
        r.lines.push_back((ms, text.clone()));
        if r.lines.len() > KEEP_LINES {
            r.lines.pop_front();
        }
        let id = r.info.id.clone();
        self.emit(&Event::Line { id, pid, seq, ms, text });
    }

    fn gone(&mut self, pid: u32, exit: &Exit) {
        let queued = self.workers.iter().position(|w| w.pid == pid).map_or(0, |k| self.workers.remove(k).queue.len());
        (0..=queued).for_each(|_| self.finish(pid, exit.clone()));
    }

    fn stop(&mut self, pid: u32) {
        let Some(w) = self.workers.iter_mut().find(|w| w.pid == pid) else {
            return process::kill(pid);
        };
        let Some(job) = w.busy else { return };
        if send(&mut w.ctl, &Work::Abort(job)).is_err() {
            return process::kill(pid);
        }
        let Some(id) = self.running(pid).map(|r| r.info.id.clone()) else { return };
        let tx = self.tx.clone();
        thread::spawn(move || {
            thread::sleep(GRACE);
            drop(tx.send(Hub::Expire(pid, id)));
        });
    }

    fn handle(&mut self, m: Hub) {
        match m {
            Hub::Join(id, tx) => self.sessions.push((id, tx)),
            Hub::Leave(id) => self.sessions.retain(|(s, _)| *s != id),
            Hub::List(tx) => drop(tx.send(self.jobs.iter().map(Record::info).collect())),
            Hub::Attach(id, tx) => {
                let found = self.jobs.iter().find(|r| r.info.id == id);
                drop(tx.send(found.map(|r| (r.info(), r.lines.iter().cloned().collect()))));
            }

            Hub::Submit { spec, preview, resident, keep_alive, reply } => {
                drop(reply.send(self.submit(spec, (preview, resident, keep_alive))));
            }

            Hub::Stop(pid) => self.stop(pid),

            Hub::Expire(pid, id) => {
                if self.running(pid).is_some_and(|r| r.info.id == id) {
                    process::kill(pid);
                }
            }

            Hub::Line(pid, text) => self.line(pid, text),

            Hub::Frame(pid, frame) => {
                let Some(id) = self.running(pid).map(|r| r.info.id.clone()) else { return };

                self.emit(&Event::Frame { id, pid, frame });
            }

            Hub::Gone(pid, exit) => self.gone(pid, &exit),

            Hub::Tick => self
                .workers
                .iter()
                .filter(|w| w.busy.is_none() && w.idle.elapsed() > w.keep_alive)
                .for_each(|w| process::kill(w.pid)),
        }
    }
}

/// Owns sessions, job records and resident workers; stamps output with job time and fans events out.
pub fn run(tx: Sender<Hub>, rx: &Receiver<Hub>) {
    let ticker = tx.clone();
    thread::spawn(move || {
        while ticker.send(Hub::Tick).is_ok() {
            thread::sleep(Duration::from_secs(30));
        }
    });
    let mut s = State { tx, sessions: vec![], jobs: vec![], workers: vec![], next: 0 };
    rx.iter().for_each(|m| s.handle(m));
}
