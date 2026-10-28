use std::collections::VecDeque;
use std::time::Instant;

use flume::{Receiver, Sender};

use crate::{Event, Exit, JobInfo};

const KEEP_LINES: usize = 4000;
const KEEP_DONE: usize = 64;

/// A job's record and its buffered `(ms, line)` output.
pub type Snapshot = (JobInfo, Vec<(u64, String)>);

pub enum Hub {
    Join(u64, Sender<Event>),
    Leave(u64),
    Started { id: String, pid: u32, args: Vec<String> },
    Line(u32, String),
    Exit(u32, Exit),
    List(Sender<Vec<JobInfo>>),
    Attach(String, Sender<Option<Snapshot>>),
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

/// Owns sessions and job records; stamps output with job time and fans events out.
pub fn run(rx: &Receiver<Hub>) {
    let (mut sessions, mut jobs): (Vec<(u64, Sender<Event>)>, Vec<Record>) = (vec![], vec![]);
    rx.iter().for_each(|m| {
        let event = match m {
            Hub::Join(id, tx) => return sessions.push((id, tx)),
            Hub::Leave(id) => return sessions.retain(|(s, _)| *s != id),
            Hub::List(tx) => return drop(tx.send(jobs.iter().map(Record::info).collect())),
            Hub::Attach(id, tx) => {
                let found = jobs.iter().find(|r| r.info.id == id);
                return drop(tx.send(found.map(|r| (r.info(), r.lines.iter().cloned().collect()))));
            }
            Hub::Started { id, pid, args } => {
                let info = JobInfo { id, pid, args, elapsed_ms: 0, lines: 0, exit: None };
                return jobs.push(Record { info, started: Instant::now(), lines: VecDeque::new() });
            }
            Hub::Line(pid, text) => {
                let Some(r) = jobs.iter_mut().find(|r| r.info.pid == pid && r.info.exit.is_none()) else { return };
                let (ms, seq) = (r.ms(), r.info.lines);
                r.info.lines += 1;
                r.lines.push_back((ms, text.clone()));
                if r.lines.len() > KEEP_LINES {
                    r.lines.pop_front();
                }
                Event::Line { id: r.info.id.clone(), pid, seq, ms, text }
            }
            Hub::Exit(pid, exit) => {
                let Some(r) = jobs.iter_mut().find(|r| r.info.pid == pid && r.info.exit.is_none()) else { return };
                r.info.elapsed_ms = r.ms();
                r.info.exit = Some(exit.clone());
                let id = r.info.id.clone();
                let done = jobs.iter().filter(|r| r.info.exit.is_some()).count();
                if let Some(k) = jobs.iter().position(|r| r.info.exit.is_some()).filter(|_| done > KEEP_DONE) {
                    jobs.remove(k);
                }
                Event::Exit { id, pid, exit }
            }
        };
        sessions.retain(|(_, s)| s.send(event.clone()).is_ok());
    });
}
