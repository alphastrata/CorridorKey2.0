use std::io::Read;
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use flume::{Receiver, Sender};
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;
use tachyon::{Bus, TachyonError};

use crate::hub::{self, Hub};
use crate::{Error, Event, Exit, Reply, Request, Result, encode, recv, send, socket};

/// Runs the daemon until the listener fails; refuses to start beside a live one.
pub fn serve() -> Result<()> {
    let path = socket()?;
    if UnixStream::connect(&path).is_ok() {
        return Err(Error::Daemon(format!("already running on {}", path.display())));
    }
    drop(std::fs::remove_file(&path));
    let listener = UnixListener::bind(&path)?;
    let (tx, rx) = flume::unbounded();
    thread::spawn(move || hub::run(&rx));
    listener.incoming().flatten().for_each(|s| {
        let tx = tx.clone();
        thread::spawn(move || session(s, &tx));
    });
    Ok(())
}

fn session(stream: UnixStream, hub: &Sender<Hub>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let (Ok(mut r), mut w) = (stream.try_clone(), stream) else { return };
    while let Ok(req) = recv::<Request>(&mut r) {
        let reply = match req {
            Request::Hello { events } => {
                let (tx, rx) = flume::unbounded();
                drop(hub.send(Hub::Join(id, tx)));
                thread::spawn(move || publish(&events, &rx));
                Reply::Ok
            }
            Request::Submit { program, args, env, cwd } => match spawn(&program, &args, &env, &cwd, hub) {
                Ok((id, pid)) => Reply::Started { id, pid },
                Err(e) => Reply::Error(format!("could not start {program}: {e}")),
            },
            Request::Stop { pid } => {
                stop(pid);
                Reply::Ok
            }
            Request::List => {
                let (tx, rx) = flume::bounded(1);
                drop(hub.send(Hub::List(tx)));
                Reply::Jobs(rx.recv().unwrap_or_default())
            }
            Request::Attach { id } => {
                let (tx, rx) = flume::bounded(1);
                drop(hub.send(Hub::Attach(id.clone(), tx)));
                rx.recv()
                    .ok()
                    .flatten()
                    .map_or_else(|| Reply::Error(format!("no job {id}")), |(info, lines)| Reply::Job { info, lines })
            }
        };
        if send(&mut w, &reply).is_err() {
            break;
        }
    }
    drop(hub.send(Hub::Leave(id)));
}

/// Producer side of a client's event ring; retries while the client binds and while the ring is full.
fn publish(path: &str, rx: &Receiver<Event>) {
    let start = Instant::now();
    let bus = loop {
        match Bus::connect(path) {
            Ok(b) => break b,
            Err(_) if start.elapsed() < Duration::from_secs(3) => thread::sleep(Duration::from_millis(10)),
            Err(_) => return,
        }
    };
    rx.iter().flat_map(|e| encode(&e)).try_for_each(|bytes| loop {
        match bus.send(&bytes, 1) {
            Err(TachyonError::BufferFull) => thread::sleep(Duration::from_micros(200)),
            r => return r,
        }
    })
    .ok();
}

fn lines(mut from: impl Read + Send + 'static, pid: u32, hub: Sender<Hub>) -> thread::JoinHandle<()> {
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

fn spawn(
    program: &str,
    args: &[String],
    env: &[(String, String)],
    cwd: &str,
    hub: &Sender<Hub>,
) -> std::io::Result<(String, u32)> {
    let mut child = Command::new(program)
        .args(args)
        .envs(env.iter().map(|(k, v)| (k, v)))
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()?;
    let pid = child.id();
    let id = uuid::Uuid::new_v4().to_string();
    drop(hub.send(Hub::Started { id: id.clone(), pid, args: args.to_vec() }));
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
        readers.into_iter().map(thread::JoinHandle::join).for_each(drop);
        let signal = status.and_then(|s| s.signal()).map(|s| {
            Signal::try_from(s).map_or_else(|_| format!("signal {s}"), |s| s.to_string())
        });
        drop(hub.send(Hub::Exit(pid, Exit { code: status.and_then(|s| s.code()), signal })));
    });
    Ok((id, pid))
}

/// SIGTERM to the job's process group, then SIGKILL after three seconds.
pub fn stop(pid: u32) {
    let group = Pid::from_raw(pid.cast_signed());
    killpg(group, Signal::SIGTERM).ok();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(3));
        killpg(group, Signal::SIGKILL).ok();
    });
}
