use std::time::Duration;

use ck_ipc::{Client, Event, Reply, Request};

#[test]
fn submit_streams_lines_and_exit() {
    let dir = std::env::temp_dir().join(format!("ck-ipc-test-{}", std::process::id()));
    // SAFETY: set before any thread is spawned.
    unsafe { std::env::set_var("XDG_RUNTIME_DIR", &dir) };
    std::thread::spawn(|| ck_ipc::server::serve().unwrap());
    let (mut c, events) = (0..100)
        .find_map(|_| {
            std::thread::sleep(Duration::from_millis(20));
            Client::connect(None).ok()
        })
        .expect("daemon up");
    let Reply::Started { pid } = c
        .call(&Request::Submit {
            program: "sh".into(),
            args: vec!["-c".into(), "echo hi; exit 3".into()],
            env: vec![],
            cwd: ".".into(),
        })
        .unwrap()
    else {
        panic!("not started")
    };
    let got: Vec<Event> = events.iter().take(2).collect();
    assert_eq!(got[0], Event::Line { pid, text: "hi".into() });
    assert_eq!(got[1], Event::Exit { pid, code: Some(3), signal: None });
    assert_eq!(c.call(&Request::List).unwrap(), Reply::Jobs(vec![]));
    drop(std::fs::remove_dir_all(dir));
}
