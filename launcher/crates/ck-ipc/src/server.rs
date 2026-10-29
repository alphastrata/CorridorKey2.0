use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use flume::{Receiver, Sender};
use tachyon::{Bus, TachyonError};

use crate::hub::{self, Hub};
use crate::process::Spec;
use crate::{Error, Event, Reply, Request, Result, encode, recv, send, socket};

/// Runs the daemon until the listener fails; refuses to start beside a live one.
pub fn serve() -> Result<()> {
    let path = socket()?;
    if UnixStream::connect(&path).is_ok() {
        return Err(Error::Daemon(format!("already running on {}", path.display())));
    }
    drop(std::fs::remove_file(&path));
    let listener = UnixListener::bind(&path)?;
    let (tx, rx) = flume::unbounded();
    let hub_tx = tx.clone();
    thread::spawn(move || hub::run(hub_tx, &rx));
    listener.incoming().flatten().for_each(|s| {
        let tx = tx.clone();
        thread::spawn(move || session(s, &tx));
    });
    Ok(())
}

fn session(stream: UnixStream, hub: &Sender<Hub>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::AcqRel);
    let (Ok(mut r), mut w) = (stream.try_clone(), stream) else { return };

    while let Ok(req) = recv::<Request>(&mut r) {
        let reply = match req {
            Request::Hello { events } => {
                let (tx, rx) = flume::unbounded();
                drop(hub.send(Hub::Join(id, tx)));
                thread::spawn(move || publish(&events, &rx));
                Reply::Ok
            }
            Request::Submit { program, args, env, cwd, preview, resident, nice, keep_alive_secs } => {
                let (tx, rx) = flume::bounded(1);
                let spec = Spec { program: program.clone(), args, env, cwd, nice };
                let keep_alive = Duration::from_secs(keep_alive_secs.into());
                drop(hub.send(Hub::Submit { spec, preview, resident, keep_alive, reply: tx }));
                match rx.recv() {
                    Ok(Ok((id, pid))) => Reply::Started { id, pid },
                    Ok(Err(e)) => Reply::Error(format!("could not start {program}: {e}")),
                    Err(_) => Reply::Error("ckd is shutting down".into()),
                }
            }
            Request::Stop { pid } => {
                drop(hub.send(Hub::Stop(pid)));
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

/// Connects to a bus the other side is binding, waiting up to three seconds.
#[must_use]
pub fn dial_bus(path: &str) -> Option<Bus> {
    let start = Instant::now();
    loop {
        match Bus::connect(path) {
            Ok(b) => return Some(b),
            Err(_) if start.elapsed() < Duration::from_secs(3) => thread::sleep(Duration::from_millis(10)),
            Err(_) => return None,
        }
    }
}

/// Producer side of a client's event ring; retries while the client binds and while the ring is full.
fn publish(path: &str, rx: &Receiver<Event>) {
    let Some(bus) = dial_bus(path) else { return };
    rx.iter()
        .filter_map(|e| Some((matches!(e, Event::Frame { .. }), encode(&e).ok()?)))
        .try_for_each(|(lossy, bytes)| loop {
            match bus.send(&bytes, 1) {
                Err(TachyonError::BufferFull) if lossy => return Ok(()),
                Err(TachyonError::BufferFull) => thread::sleep(Duration::from_micros(200)),
                r => return r,
            }
        })
        .ok();
}
