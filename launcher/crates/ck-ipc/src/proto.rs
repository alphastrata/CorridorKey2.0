use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Request {
    /// Subscribe to events on a tachyon bus the client is listening on.
    Hello { events: String },
    Submit {
        program: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        cwd: String,
        /// Hand the job a preview socket on fd `PREVIEW_FD` and relay its frames as `Event::Frame`.
        preview: bool,
        /// Run in a resident `ck --serve` worker that keeps models loaded between jobs.
        resident: bool,
    },
    Stop { pid: u32 },
    List,
    /// A job's record and buffered output, for a client rehydrating after a restart.
    Attach { id: String },
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Reply {
    Ok,
    Started { id: String, pid: u32 },
    Jobs(Vec<JobInfo>),
    Job { info: JobInfo, lines: Vec<(u64, String)> },
    Error(String),
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Exit {
    pub code: Option<i32>,
    pub signal: Option<String>,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct JobInfo {
    pub id: String,
    pub pid: u32,
    pub args: Vec<String>,
    /// Milliseconds since start, or the run's length once it has exited.
    pub elapsed_ms: u64,
    /// Lines emitted so far; the next `Line` event carries this `seq`.
    pub lines: u64,
    pub exit: Option<Exit>,
}

/// `ckd` → a resident `ck --serve` worker, on fd `CONTROL_FD`; the worker answers with `hub::DONE` on stderr.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Work {
    /// Run job `job` with these `ck` arguments.
    Run { job: u64, args: Vec<String> },
    /// Abort job `job`, whether running or still waiting.
    Abort(u64),
}

/// A downscaled 8-bit RGB snapshot of one output.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Frame {
    pub output: String,
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Event {
    /// `seq` numbers the job's lines from zero; `ms` counts from its start.
    Line { id: String, pid: u32, seq: u64, ms: u64, text: String },
    Exit { id: String, pid: u32, exit: Exit },
    /// Lossy: dropped when a client's ring is full.
    Frame { id: String, pid: u32, frame: Frame },
}
