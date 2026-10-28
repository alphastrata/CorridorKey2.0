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
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Reply {
    Ok,
    Started { pid: u32 },
    Jobs(Vec<JobInfo>),
    Error(String),
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct JobInfo {
    pub pid: u32,
    pub args: Vec<String>,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Event {
    Line { pid: u32, text: String },
    Exit { pid: u32, code: Option<i32>, signal: Option<String> },
}
