use std::io::Read;
use std::path::Path;
use std::process::{Child, Stdio};
use std::thread;
use std::time::Duration;

use flume::Sender;

use crate::command::Command;

pub enum Event {
    Line(String),
    Exit(Result<i32, String>),
}

pub struct Job {
    pub pid: u32,
}

fn pump(mut from: impl Read + Send + 'static, tx: Sender<Event>) {
    thread::spawn(move || {
        let (mut buf, mut line) = ([0u8; 8192], Vec::new());
        while let Ok(n) = from.read(&mut buf) {
            if n == 0 {
                break;
            }
            buf[..n].iter().for_each(|&b| {
                if b == b'\n' || b == b'\r' {
                    let text = String::from_utf8_lossy(&line).into_owned();
                    line.clear();
                    if !text.trim().is_empty() {
                        let _ = tx.send(Event::Line(text));
                    }
                } else {
                    line.push(b);
                }
            });
        }
        if !line.is_empty() {
            let _ = tx.send(Event::Line(String::from_utf8_lossy(&line).into_owned()));
        }
    });
}

fn describe(status: std::process::ExitStatus) -> Result<i32, String> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return Err(nix::sys::signal::Signal::try_from(sig).map_or_else(|_| format!("signal {sig}"), |s| s.to_string()));
        }
    }
    Ok(status.code().unwrap_or(-1))
}

pub fn spawn(cmd: &Command, cwd: &Path, tx: &Sender<Event>) -> std::io::Result<Job> {
    let mut c = std::process::Command::new(&cmd.program);
    c.args(&cmd.args).envs(cmd.env.iter().copied()).current_dir(cwd).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut c, 0);
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut c, 0x0800_0000);
    let mut child: Child = c.spawn()?;
    let pid = child.id();
    if let Some(out) = child.stdout.take() {
        pump(out, tx.clone());
    }
    if let Some(err) = child.stderr.take() {
        pump(err, tx.clone());
    }
    let tx = tx.clone();
    thread::spawn(move || {
        let status = child.wait().map_or_else(|e| Err(e.to_string()), describe);
        let _ = tx.send(Event::Exit(status));
    });
    Ok(Job { pid })
}

pub fn stop(job: &Job) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, killpg};
        use nix::unistd::Pid;
        let group = Pid::from_raw(job.pid.cast_signed());
        let _ = killpg(group, Signal::SIGTERM);
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(3));
            let _ = killpg(group, Signal::SIGKILL);
        });
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill").args(["/PID", &job.pid.to_string(), "/T", "/F"]).output();
    }
}

#[derive(Default, Clone)]
pub struct Progress {
    pub value: f32,
    pub detail: String,
    pub low_vram: bool,
    pub frames: usize,
    pub size: String,
    pub window: String,
    pub at: (usize, usize),
    pub keyed: usize,
}

const KEY_START: f32 = 0.15;
const KEY_END: f32 = 0.97;
const KEY_END_LOW: f32 = 0.65;

fn nums(s: &str) -> Vec<usize> {
    s.split(|c: char| !c.is_ascii_digit()).filter(|t| !t.is_empty()).filter_map(|t| t.parse().ok()).collect()
}

fn after<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    s.find(key).map(|i| &s[i + key.len()..])
}

impl Progress {
    /// Updates from one line of ck or infer.py output; true if progress or detail moved.
    pub fn feed(&mut self, line: &str) -> bool {
        let t = line.trim();
        let before = (self.value, self.detail.clone());
        if t.ends_with("frames through stage 2") {
            let n = nums(t).into_iter().rev().nth(1).unwrap_or(0);
            if n > self.keyed {
                self.keyed = n;
                self.detail = format!("{n} frames keyed");
            }
        } else if let Some(rest) = after(t, "window ").filter(|r| r.contains('/')) {
            let n = nums(rest);
            if let [done, total, ..] = n[..]
                && total > 0 && (t.starts_with("window") || rest.contains(" [")) {
                    self.at = (done, total);
                    let end = if self.low_vram { KEY_END_LOW } else { KEY_END };
                    self.value = KEY_START + (end - KEY_START) * done as f32 / total as f32;
                    self.detail = if done == total && self.low_vram { "Writing outputs".into() } else { format!("Keying {done}/{total}") };
                }
        } else if t.contains(" frames ") && t.contains(" → ") && t.contains(", window ") {
            let n = nums(t);
            if let [.., frames, w, h, mw, mh, chunk, stride] = n[..] {
                self.frames = frames;
                self.size = format!("{w}x{h} → model {mw}x{mh}");
                self.window = format!("chunk {chunk} · stride {stride}");
            }
            self.value = KEY_START;
            self.detail = "Keying".into();
        } else if let Some(rest) = after(t, "Hann stage1: ") {
            if let [chunk, stride, windows, ..] = nums(rest)[..] {
                self.window = format!("chunk {chunk} · stride {stride} · {windows} windows");
            }
            self.value = KEY_START;
            self.detail = "Keying".into();
        } else if let Some(rest) = after(t, "Auto VRAM: ") {
            self.low_vram = rest.contains("low_vram=True");
        } else if t.contains("frozen VFM prepass") {
            (self.value, self.detail) = (0.12, "Feature prepass".into());
        } else if t.contains("RVM continuous prepass") {
            (self.value, self.detail) = (0.08, "Motion prepass".into());
        } else if let Some(rest) = after(t, "probed ") {
            self.frames = nums(rest).first().copied().unwrap_or(0);
            if let Some([mw, mh, w, h, ..]) = after(rest, "@ ").map(nums).as_deref() {
                self.size = format!("{w}x{h} → model {mw}x{mh}");
            }
            (self.value, self.detail) = (0.06, "Reading clip".into());
        } else if t.starts_with("Loading CorridorKey v2 runtime weights")
            || (t.contains("GiB free") && t.contains("sm_") && !t.contains('→') && self.value < KEY_START)
        {
            (self.value, self.detail) = (0.02, "Loading model".into());
        }
        (self.value, self.detail.as_str()) != (before.0, before.1.as_str())
    }
}
