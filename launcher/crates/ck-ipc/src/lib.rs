//! `ckd`: a per-user daemon that runs `ck` jobs. Requests travel over a Unix socket;
//! events stream back over a tachyon shared-memory ring per client.

pub mod client;
pub mod proto;
pub mod server;

use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use rkyv::api::high::{HighDeserializer, HighSerializer, HighValidator};
use rkyv::bytecheck::CheckBytes;
use rkyv::rancor;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;

pub use client::Client;
pub use proto::{Event, JobInfo, Reply, Request};

pub const BUS_BYTES: usize = 1 << 26;
const SPIN: u32 = 2000;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Codec(#[from] rancor::Error),
    #[error("tachyon: {0}")]
    Bus(#[from] tachyon::TachyonError),
    #[error("{0}")]
    Daemon(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Per-user runtime directory, `$XDG_RUNTIME_DIR/ck` or `/tmp/ck-<uid>`, mode 0700.
pub fn dir() -> Result<PathBuf> {
    let d = std::env::var_os("XDG_RUNTIME_DIR").map_or_else(
        || std::env::temp_dir().join(format!("ck-{}", nix::unistd::getuid())),
        |r| PathBuf::from(r).join("ck"),
    );
    std::fs::create_dir_all(&d)?;
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700))?;
    Ok(d)
}

pub fn socket() -> Result<PathBuf> {
    Ok(dir()?.join("ck.sock"))
}

pub fn encode<T>(v: &T) -> Result<AlignedVec>
where
    T: for<'a> rkyv::Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
{
    Ok(rkyv::to_bytes::<rancor::Error>(v)?)
}

pub fn decode<T>(bytes: &[u8]) -> Result<T>
where
    T: rkyv::Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, rancor::Error>> + rkyv::Deserialize<T, HighDeserializer<rancor::Error>>,
{
    let mut aligned = AlignedVec::<16>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    Ok(rkyv::from_bytes::<T, rancor::Error>(&aligned)?)
}

fn send<T>(w: &mut impl Write, v: &T) -> Result<()>
where
    T: for<'a> rkyv::Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
{
    let bytes = encode(v)?;
    let len = u32::try_from(bytes.len()).map_err(|_| Error::Daemon("message too large".into()))?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&bytes)?;
    Ok(w.flush()?)
}

fn recv<T>(r: &mut impl Read) -> Result<T>
where
    T: rkyv::Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, rancor::Error>> + rkyv::Deserialize<T, HighDeserializer<rancor::Error>>,
{
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let mut buf = vec![0u8; u32::from_le_bytes(len) as usize];
    r.read_exact(&mut buf)?;
    decode(&buf)
}
