use std::os::fd::BorrowedFd;
use std::os::unix::net::UnixStream;

use argh::FromArgs;
use ck::farm::{Resident, Session};
use ck_ipc::{CONTROL_FD, DONE};
use eyre::{Result, eyre};

use crate::{Args, job};

/// Resident worker: runs jobs from `ckd` one at a time with models kept loaded; exits when `ckd` hangs up.
pub fn run() -> Result<()> {
    // SAFETY: `ckd` hands fd 4 down open for our lifetime; we only duplicate it.
    let ctl = UnixStream::from(unsafe { BorrowedFd::borrow_raw(CONTROL_FD) }.try_clone_to_owned()?);
    let resident = Resident::default();
    ck_ipc::worker::serve(ctl, |args, abort| {
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let r = Args::from_args(&["ck"], &argv)
            .map_err(|e| eyre!("{}", e.output.trim()))
            .and_then(|a| job(a, Session { resident: Some(&resident), abort }));
        if let Err(e) = &r {
            eprintln!("Error: {e:?}");
        }
        eprintln!("{DONE}{}", i32::from(r.is_err()));
    });
    Ok(())
}
