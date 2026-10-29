use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::{Work, recv};

/// Worker side of `ck --serve`: runs each `Work::Run` in turn with an abort flag only its own `Abort` sets.
pub fn serve(ctl: UnixStream, mut run: impl FnMut(&[String], &AtomicBool)) {
    let abort = Arc::new(AtomicBool::new(false));
    let (current, cancel) = (Arc::new(AtomicU64::new(u64::MAX)), Arc::new(AtomicU64::new(u64::MAX)));
    let (jobs, queue) = flume::unbounded();
    let (flag, now, kill) = (abort.clone(), current.clone(), cancel.clone());
    let mut rx = ctl;

    std::thread::spawn(move || {
        while let Ok(w) = recv::<Work>(&mut rx) {
            match w {
                Work::Run { job, args } => drop(jobs.send((job, args))),
                Work::Abort(job) => {
                    kill.store(job, Ordering::Release);

                    if now.load(Ordering::Acquire) == job {
                        flag.store(true, Ordering::Release);
                    }
                }
            }
        }

        std::process::exit(0);
    });

    queue.iter().for_each(|(job, args)| {
        current.store(job, Ordering::Release);
        abort.store(cancel.load(Ordering::Acquire) == job, Ordering::Release);
        run(&args, &abort);
        current.store(u64::MAX, Ordering::Release);
    });
}
