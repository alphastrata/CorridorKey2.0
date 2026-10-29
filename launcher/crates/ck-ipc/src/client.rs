use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use flume::Receiver;
use tachyon::Bus;

use crate::{BUS_BYTES, Error, Event, Reply, Request, Result, SPIN, decode, dir, recv, send, socket};

pub struct Client {
    stream: UnixStream,
}

fn dial(daemon: Option<&Path>) -> Result<UnixStream> {
    let path = socket()?;
    if let Ok(s) = UnixStream::connect(&path) {
        return Ok(s);
    }
    let exe = daemon.ok_or_else(|| Error::Daemon("ckd is not running".into()))?;
    Command::new(exe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()?;
    let start = Instant::now();
    loop {
        match UnixStream::connect(&path) {
            Ok(s) => return Ok(s),
            Err(_) if start.elapsed() < Duration::from_secs(5) => thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(e.into()),
        }
    }
}

/// Listens on a tachyon bus at `path`, forwarding each message `read` accepts.
pub(crate) fn listen<T: Send + 'static>(path: String, read: fn(&[u8]) -> Option<T>) -> Receiver<T> {
    let (tx, rx) = flume::unbounded();
    thread::spawn(move || {
        let Ok(bus) = Bus::listen(&path, BUS_BYTES) else { return };
        drop(std::fs::remove_file(&path));
        while let Ok(msg) = bus.acquire_rx(SPIN) {
            let item = read(msg.data());
            if msg.commit().is_err() || item.is_some_and(|e| tx.send(e).is_err()) {
                return;
            }
        }
    });
    rx
}

impl Client {
    /// Connects to `ckd`, starting `daemon` detached if none is running, and subscribes to its events.
    pub fn connect(daemon: Option<&Path>) -> Result<(Self, Receiver<Event>)> {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let mut c = Self { stream: dial(daemon)? };
        let events = dir()?
            .join(format!("ev-{}-{}.sock", std::process::id(), NEXT.fetch_add(1, Ordering::AcqRel)))
            .to_string_lossy()
            .into_owned();
        let rx = listen(events.clone(), |b| decode::<Event>(b).ok());
        c.call(&Request::Hello { events })?;
        Ok((c, rx))
    }

    pub fn call(&mut self, req: &Request) -> Result<Reply> {
        send(&mut self.stream, req)?;
        match recv::<Reply>(&mut self.stream)? {
            Reply::Error(e) => Err(Error::Daemon(e)),
            r => Ok(r),
        }
    }
}
