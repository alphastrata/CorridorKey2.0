#[cfg(unix)]
use crate::convert::Float;
use nix::sys::signal::{Signal, killpg};
#[cfg(unix)]
use nix::unistd::Pid;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::{Child, Stdio};
use std::thread;
use std::time::Duration;

use flume::Sender;

use crate::command::Command;

pub enum Event {
    Line(u32, String),
    Exit(u32, eyre::Result<i32>),
}

pub struct Job {
    pub pid: u32,
}

fn pump(mut from: impl Read + Send + 'static, pid: u32, tx: Sender<Event>) {
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

                        drop(tx.send(Event::Line(pid, text)));
                    }
                } else {
                    line.push(b);
                }
            });
        }
        if !line.is_empty() {
            drop(tx.send(Event::Line(pid, String::from_utf8_lossy(&line).into_owned())));
        }
    });
}

fn describe(status: std::process::ExitStatus) -> eyre::Result<i32> {
    #[cfg(unix)]
    if let Some(sig) = status.signal() {
        return Err(eyre::eyre!(
            "{}",
            nix::sys::signal::Signal::try_from(sig).map_or_else(|_| format!("signal {sig}"), |s| s.to_string())
        ));
    }
    Ok(status.code().unwrap_or(-1))
}

pub fn spawn(cmd: &Command, cwd: &Path, tx: &Sender<Event>) -> std::io::Result<Job> {
    let mut c = std::process::Command::new(&cmd.program);
    c.args(&cmd.args)
        .envs(cmd.env.iter().copied())
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut c, 0);
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut c, 0x0800_0000);
    let mut child: Child = c.spawn()?;
    let pid = child.id();

    if let Some(out) = child.stdout.take() {
        pump(out, pid, tx.clone());
    }
    if let Some(err) = child.stderr.take() {
        pump(err, pid, tx.clone());
    }
    let tx = tx.clone();
    thread::spawn(move || {
        let status = child.wait().map_or_else(|e| Err(e.into()), describe);
        drop(tx.send(Event::Exit(pid, status)));
    });
    Ok(Job { pid })
}

pub fn stop(job: &Job) {
    #[cfg(unix)]
    {
        let group = Pid::from_raw(job.pid.cast_signed());
        killpg(group, Signal::SIGTERM).ok();
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(3));
            killpg(group, Signal::SIGKILL).ok();
        });
    }

    #[cfg(windows)]
    {
        drop(
            std::process::Command::new("taskkill")
                .args(["/PID", &job.pid.to_string(), "/T", "/F"])
                .output(),
        );
    }
}

#[derive(Default, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Progress {
    pub value: f32,
    pub detail: String,
    pub frames: usize,
    pub size: String,
    pub window: String,
    pub at: (usize, usize),
    pub keyed: usize,
}

const KEY_START: f32 = 0.15;
const KEY_END: f32 = 0.97;

fn nums(s: &str) -> Vec<usize> {
    s.split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse().ok())
        .collect()
}

fn after<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    s.find(key).map(|i| &s[i + key.len()..])
}

impl Progress {
    /// Updates from one line of ck output; true if progress or detail moved.
    pub fn feed(&mut self, line: &str) -> bool {
        let t = line.trim();
        let before = (self.value, self.detail.clone());
        if let Some(head) = t
            .split(" frames written")
            .next()
            .filter(|_| t.contains(" frames written"))
        {
            if let [n, total] = nums(head.rsplit(' ').next().unwrap_or_default())[..] {

                self.keyed = self.keyed.max(n);
                self.detail = format!("{n}/{total} frames written");
            }
        } else if t.ends_with("frames through stage 2") {
            let n = nums(t).into_iter().rev().nth(1).unwrap_or(0);
            if n > self.keyed {
                self.keyed = n;
                self.detail = format!("{n} frames keyed");
            }
        } else if let Some(rest) = after(t, "window ").filter(|r| r.contains('/')) {
            let n = nums(rest);

            if let [done, total, ..] = n[..]
                && total > 0
                && (t.starts_with("window") || rest.contains(" ["))
            {
                self.at = (done, total);
                self.value = KEY_START + (KEY_END - KEY_START) * done.f32() / total.f32();
                self.detail = format!("Keying {done}/{total}");
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
        } else if t.contains("GiB free") && t.contains("sm_") && !t.contains('→') && self.value < KEY_START {
            (self.value, self.detail) = (0.02, "Loading model".into());
        }
        (self.value, self.detail.as_str()) != (before.0, before.1.as_str())
    }
}
