use std::path::Path;
use std::thread;

use flume::Sender;

use crate::command::Command;
use crate::runner::{Event, Job};

pub fn status(exit: ck_ipc::Exit) -> eyre::Result<i32> {
    exit.signal.map_or(Ok(exit.code.unwrap_or(-1)), |s| Err(eyre::eyre!(s)))
}

/// A connection to `ckd`; jobs submitted through it outlive this process's pipes.
pub struct Daemon(ck_ipc::Client);

/// Connects to `ckd`, starting it when built, and forwards its events into `tx`.
pub fn connect(tx: &Sender<Event>) -> Option<Daemon> {
    let (client, rx) = ck_ipc::Client::connect(crate::command::ckd_binary().as_deref()).ok()?;
    let tx = tx.clone();
    thread::spawn(move || {
        rx.iter()
            .map(|e| match e {
                ck_ipc::Event::Line { pid, text, seq, .. } => Event::Line(pid, text, Some(seq)),
                ck_ipc::Event::Exit { pid, exit, .. } => Event::Exit(pid, status(exit)),
                ck_ipc::Event::Frame { pid, frame, .. } => Event::Frame(
                    pid,
                    frame.output,
                    slint::SharedPixelBuffer::clone_from_slice(&frame.rgb, frame.w, frame.h),
                ),
            })
            .try_for_each(|e| tx.send(e))
            .ok();
    });
    Some(Daemon(client))
}

impl Daemon {
    pub fn submit(&mut self, cmd: &Command, cwd: &Path) -> eyre::Result<Job> {
        let req = ck_ipc::Request::Submit {
            program: cmd.program.to_string_lossy().into_owned(),
            args: cmd.args.clone(),
            env: cmd.env.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect(),
            cwd: cwd.to_string_lossy().into_owned(),
            preview: true,
        };
        match self.0.call(&req)? {
            ck_ipc::Reply::Started { id, pid } => Ok(Job { pid, id: Some(id) }),
            r => Err(eyre::eyre!("ckd replied {r:?}")),
        }
    }

    /// The job's record and buffered output; `None` when `ckd` no longer knows it.
    pub fn attach(&mut self, id: &str) -> eyre::Result<Option<ck_ipc::Snapshot>> {
        match self.0.call(&ck_ipc::Request::Attach { id: id.to_owned() }) {
            Ok(ck_ipc::Reply::Job { info, lines }) => Ok(Some((info, lines))),
            Ok(r) => Err(eyre::eyre!("ckd replied {r:?}")),
            Err(ck_ipc::Error::Daemon(_)) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}
