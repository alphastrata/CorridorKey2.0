//! Stands in for `ck --serve` in tests: echoes each job's arguments, sleeps on `slow`, honours `Abort`.
use std::os::fd::BorrowedFd;
use std::os::unix::net::UnixStream;
use std::time::Duration;

use ck_ipc::{CONTROL_FD, DONE, worker};

fn main() {
    // SAFETY: ckd hands fd 4 down open for our lifetime; we only duplicate it.
    let ctl = UnixStream::from(unsafe { BorrowedFd::borrow_raw(CONTROL_FD) }.try_clone_to_owned().unwrap());
    worker::serve(ctl, |args, abort| {
        eprintln!("job {}", args.join(" "));
        let slow = args.iter().any(|a| a == "slow");
        let aborted = (0..if slow { 200 } else { 0 }).any(|_| {
            std::thread::sleep(Duration::from_millis(10));
            abort.load(std::sync::atomic::Ordering::Acquire)
        });
        eprintln!("{DONE}{}", i32::from(aborted));
    });
}
