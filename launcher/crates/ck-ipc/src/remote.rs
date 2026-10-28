//! Remote control of `ck-gui`: actions over a Unix socket, snapshots back over a tachyon ring.

use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use flume::Receiver;
use rkyv::{Archive, Deserialize, Serialize};

use crate::{Error, Result, client, dir, recv, send};

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Act {
    /// Stream snapshots to a tachyon bus the controller is listening on.
    Hello { shots: String },
    Hover { x: f32, y: f32 },
    Click { x: f32, y: f32 },
    /// Negative `dy` scrolls down.
    Scroll { x: f32, y: f32, dy: f32 },
    /// Keys pressed in order and released in reverse: `["ctrl", ","]`, `["esc"]`, `["a"]`.
    Keys(Vec<String>),
    /// RGBA pixels of the window, sent on the bus.
    Snap,
    /// The last `n` lines of the log panel, as `Answer::Text`.
    Log(u32),
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Answer {
    Ok,
    /// Physical size; logical coordinates are these divided by `scale`.
    Shot { w: u32, h: u32, scale: f32 },
    Text(String),
    Error(String),
}

pub fn socket() -> Result<PathBuf> {
    Ok(dir()?.join("gui.sock"))
}

pub struct Remote {
    stream: UnixStream,
    shots: Receiver<Vec<u8>>,
}

impl Remote {
    pub fn connect() -> Result<Self> {
        let mut stream = UnixStream::connect(socket()?)?;
        let path = dir()?.join(format!("shots-{}.sock", std::process::id())).to_string_lossy().into_owned();
        let shots = client::listen(path.clone(), |b| Some(b.to_vec()));
        call(&mut stream, &Act::Hello { shots: path })?;
        Ok(Self { stream, shots })
    }

    pub fn act(&mut self, a: &Act) -> Result<Answer> {
        call(&mut self.stream, a)
    }

    /// Window pixels as RGBA with their physical size and scale factor.
    pub fn snap(&mut self) -> Result<(Vec<u8>, u32, u32, f32)> {
        let Answer::Shot { w, h, scale } = self.act(&Act::Snap)? else {
            return Err(Error::Daemon("ck-gui sent no snapshot".into()));
        };
        let px = self.shots.recv_timeout(Duration::from_secs(5)).map_err(|e| Error::Daemon(e.to_string()))?;
        Ok((px, w, h, scale))
    }
}

fn call(s: &mut UnixStream, a: &Act) -> Result<Answer> {
    send(s, a)?;
    match recv::<Answer>(s)? {
        Answer::Error(e) => Err(Error::Daemon(e)),
        r => Ok(r),
    }
}
