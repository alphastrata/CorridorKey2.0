use std::path::Path;
use std::thread;

use flume::Sender;

use crate::command::Command;
use crate::runner::{Event, Job};

/// A connection to `ckd`; jobs submitted through it outlive this process's pipes.
pub struct Daemon(ck_ipc::Client);

/// Connects to `ckd`, starting it when built, and forwards its events into `tx`.
pub fn connect(tx: &Sender<Event>) -> Option<Daemon> {
    let (client, rx) = ck_ipc::Client::connect(crate::command::ckd_binary().as_deref()).ok()?;
    let tx = tx.clone();
    thread::spawn(move || {
        rx.iter()
            .map(|e| match e {
                ck_ipc::Event::Line { pid, text } => Event::Line(pid, text),
                ck_ipc::Event::Exit { pid, code, signal } => {
                    Event::Exit(pid, signal.map_or(Ok(code.unwrap_or(-1)), |s| Err(eyre::eyre!(s))))
                }
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
        };
        match self.0.call(&req)? {
            ck_ipc::Reply::Started { pid } => Ok(Job { pid }),
            r => Err(eyre::eyre!("ckd replied {r:?}")),
        }
    }
}
