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
    pub exit: Option<Exit>,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Event {
    /// `ms` counts from the job's start.
    Line { id: String, pid: u32, ms: u64, text: String },
    Exit { id: String, pid: u32, exit: Exit },
}
