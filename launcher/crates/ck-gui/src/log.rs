use slint::SharedString;

use crate::LogLine;

#[derive(Clone, Copy)]
pub enum Kind {
    Plain,
    Head,
    Command,
    Debug,
    Info,
    Warn,
    Error,
}

impl Kind {
    fn level(level: &str) -> Option<Self> {
        Some(match level {
            "TRACE" | "DEBUG" => Self::Debug,
            "INFO" => Self::Info,
            "WARN" => Self::Warn,
            "ERROR" => Self::Error,
            _ => return None,
        })
    }
}

#[must_use]
pub fn line(kind: Kind, text: impl Into<SharedString>) -> LogLine {
    LogLine {
        time: SharedString::new(),
        level: SharedString::new(),
        text: text.into(),
        kind: kind as i32,
    }
}

fn tracing(t: &str) -> Option<(&str, &str, Kind, &str)> {
    let (time, rest) = t.trim_start().split_once(' ')?;
    time.strip_suffix('s')?.parse::<f64>().ok()?;
    let rest = rest.trim_start();
    let (level, msg) = rest.split_once(' ').unwrap_or((rest, ""));
    Some((time, level, Kind::level(level)?, msg.trim_start()))
}

/// Splits one line of `ck` output into timestamp, level and message.
#[must_use]
pub fn parse(tag: &str, text: &str) -> LogLine {
    if let Some((time, level, kind, msg)) = tracing(text) {
        return LogLine {
            time: time.into(),
            level: level.into(),
            text: format!("{tag}{msg}").into(),
            kind: kind as i32,
        };
    }

    let t = text.trim_start().to_lowercase();
    let kind = if t.starts_with("error") || t.starts_with("location:") {
        Kind::Error
    } else if t.starts_with("warn") {
        Kind::Warn
    } else {
        Kind::Plain
    };
    line(kind, format!("{tag}{text}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracing_line() {
        let l = parse("", "    1.25s  WARN low memory free=2.1");
        assert_eq!((l.time.as_str(), l.level.as_str(), l.text.as_str()), ("1.25s", "WARN", "low memory free=2.1"));
        assert_eq!(l.kind, Kind::Warn as i32);
    }

    #[test]
    fn plain_line() {
        let l = parse("[a] ", "Error: boom");
        assert_eq!((l.text.as_str(), l.kind), ("[a] Error: boom", Kind::Error as i32));
        assert_eq!(parse("", "frame 3").kind, Kind::Plain as i32);
    }
}
