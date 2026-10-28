use std::os::fd::BorrowedFd;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ck::farm::{Resident, Session};
use ck_ipc::{CONTROL_FD, Done, Work};
use argh::FromArgs;
use eyre::{Result, eyre};

use crate::{Args, job};

/// Resident worker: runs `Work::Run` jobs one at a time with models kept loaded; exits when `ckd` hangs up.
pub fn run() -> Result<()> {
    // SAFETY: `ckd` hands fd 4 down open for our lifetime; we only duplicate it.
    let ctl = UnixStream::from(unsafe { BorrowedFd::borrow_raw(CONTROL_FD) }.try_clone_to_owned()?);
    let (mut rx, mut tx) = (ctl.try_clone()?, ctl);
    let abort = Arc::new(AtomicBool::new(false));
    let (jobs, queue) = flume::unbounded();
    let flag = abort.clone();
    std::thread::spawn(move || {
        while let Ok(w) = ck_ipc::recv::<Work>(&mut rx) {
            match w {
                Work::Run(argv) => drop(jobs.send(argv)),
                Work::Abort => flag.store(true, Ordering::Release),
            }
        }
        std::process::exit(0);
    });
    let resident = Resident::default();
    queue.iter().try_for_each(|argv| {
        let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
        let r = Args::from_args(&["ck"], &argv)
            .map_err(|e| eyre!("{}", e.output.trim()))
            .and_then(|a| job(a, Session { resident: Some(&resident), abort: &abort }));
        if let Err(e) = &r {
            eprintln!("Error: {e:?}");
        }
        ck_ipc::send(&mut tx, &Done { code: i32::from(r.is_err()) }).map_err(|e| eyre!(e))
    })
}
