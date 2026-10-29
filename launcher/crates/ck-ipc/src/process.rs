use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Child, Command, Stdio};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use flume::Sender;
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;

use crate::hub::Hub;
use crate::{Exit, Frame, recv};

/// What a job runs: program, arguments, environment and working directory.
pub struct Spec {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: String,
    pub nice: i32,
}

/// Spawns `spec` in its own process group at lower CPU priority, handing each socket down on its paired descriptor.
pub fn spawn(spec: &Spec, extra: &[&str], fds: &[(&UnixStream, i32)]) -> std::io::Result<Child> {
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args)
        .args(extra)
        .envs(spec.env.iter().map(|(k, v)| (k, v)))
        .current_dir(&spec.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);

    let nice = spec.nice;
    let pairs: Vec<(i32, i32)> = fds.iter().map(|(s, to)| (s.as_raw_fd(), *to)).collect();
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        cmd.pre_exec(move || {
            let batch = libc::sched_param { sched_priority: 0 };
            libc::sched_setscheduler(0, libc::SCHED_BATCH, &raw const batch);
            libc::nice(nice);
            pairs.iter().try_for_each(|&(from, to)| {
                let moved = if from == to { libc::fcntl(from, libc::F_SETFD, 0) } else { libc::dup2(from, to) };
                if moved < 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) }
            })
        });
    }

    cmd.spawn()
}

fn lines(mut from: impl Read + Send + 'static, pid: u32, hub: Sender<Hub>) -> JoinHandle<()> {
    thread::spawn(move || {
        let (mut buf, mut line) = ([0u8; 8192], Vec::new());
        let emit = |line: &mut Vec<u8>| {
            let text = String::from_utf8_lossy(line).into_owned();
            line.clear();
            if !text.trim().is_empty() {
                drop(hub.send(Hub::Line(pid, text)));
            }
        };

        while let Ok(n @ 1..) = from.by_ref().read(&mut buf) {
            buf[..n].iter().for_each(|&b| match b {
                b'\n' | b'\r' => emit(&mut line),
                _ => line.push(b),
            });
        }
        emit(&mut line);
    })
}

/// Pumps the child's output and preview frames into the hub, then reports its exit once output has drained.
pub fn watch(mut child: Child, frames: Option<UnixStream>, hub: &Sender<Hub>) -> u32 {
    let pid = child.id();
    if let Some(mut s) = frames {
        let hub = hub.clone();
        thread::spawn(move || {
            while let Ok(f) = recv::<Frame>(&mut s) {
                drop(hub.send(Hub::Frame(pid, f)));
            }
        });
    }

    let readers: Vec<_> = [
        child.stdout.take().map(|o| lines(o, pid, hub.clone())),
        child.stderr.take().map(|e| lines(e, pid, hub.clone())),
    ]
    .into_iter()
    .flatten()
    .collect();

    let hub = hub.clone();
    thread::spawn(move || {
        let status = child.wait().ok();
        readers.into_iter().map(JoinHandle::join).for_each(drop);
        let signal = status
            .and_then(|s| s.signal())
            .map(|s| Signal::try_from(s).map_or_else(|_| format!("signal {s}"), |s| s.to_string()));
        drop(hub.send(Hub::Gone(pid, Exit { code: status.and_then(|s| s.code()), signal })));
    });
    pid
}

/// SIGTERM to the process group, then SIGKILL after three seconds.
pub fn kill(pid: u32) {
    let group = Pid::from_raw(pid.cast_signed());
    killpg(group, Signal::SIGTERM).ok();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(3));
        killpg(group, Signal::SIGKILL).ok();
    });
}
