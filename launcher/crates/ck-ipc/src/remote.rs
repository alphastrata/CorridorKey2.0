//! Remote control of `ck-gui`: actions over a Unix socket, snapshots back over a tachyon ring.

use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
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
    /// The status bar's text, as `Answer::Text`.
    Status,
    /// Elements matching a query, as `Answer::Found`: `id:Component::name`, `type:CheckBox`, or an accessible label.
    Find(String),
    /// Clicks the centre of the `n`th match.
    Press { query: String, n: u32 },
    /// Invokes the `n`th match's accessible default action: toggles a check box, presses a button.
    Activate { query: String, n: u32 },
    /// Sets the `n`th match's accessible value: spin boxes, sliders, text fields.
    Set { query: String, n: u32, value: String },
}

/// One element as the accessibility tree sees it, in logical window coordinates.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Element {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub value: String,
    pub checked: Option<bool>,
    pub enabled: Option<bool>,
    pub visible: bool,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Answer {
    Ok,
    /// Physical size; logical coordinates are these divided by `scale`.
    Shot { w: u32, h: u32, scale: f32 },
    Text(String),
    Found(Vec<Element>),
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
        Self::connect_in(&dir()?)
    }

    /// Connects to the GUI whose runtime directory is `dir`.
    pub fn connect_in(dir: &Path) -> Result<Self> {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let mut stream = UnixStream::connect(dir.join("gui.sock"))?;
        let n = NEXT.fetch_add(1, Ordering::AcqRel);
        let path = dir.join(format!("shots-{}-{n}.sock", std::process::id())).to_string_lossy().into_owned();
        let shots = client::listen(path.clone(), |b| Some(b.to_vec()));
        call(&mut stream, &Act::Hello { shots: path })?;
        Ok(Self { stream, shots })
    }

    pub fn act(&mut self, a: &Act) -> Result<Answer> {
        call(&mut self.stream, a)
    }

    pub fn text(&mut self, a: &Act) -> Result<String> {
        match self.act(a)? {
            Answer::Text(t) => Ok(t),
            r => Err(Error::Daemon(format!("expected text, got {r:?}"))),
        }
    }

    pub fn find(&mut self, query: &str) -> Result<Vec<Element>> {
        match self.act(&Act::Find(query.into()))? {
            Answer::Found(e) => Ok(e),
            r => Err(Error::Daemon(format!("expected elements, got {r:?}"))),
        }
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
