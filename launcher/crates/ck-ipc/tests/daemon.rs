use std::time::Duration;

use ck_ipc::{Client, Event, Exit, Frame, Reply, Request};

fn submit(c: &mut Client, script: &str, preview: bool) -> (String, u32) {
    let req = Request::Submit {
        program: "sh".into(),
        args: vec!["-c".into(), script.into()],
        env: vec![],
        cwd: ".".into(),
        preview,
    };
    let Reply::Started { id, pid } = c.call(&req).unwrap() else { panic!("not started") };
    (id, pid)
}

#[test]
fn daemon_round_trip() {
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

    let (id, pid) = submit(&mut c, "echo hi; exit 3", false);
    let got: Vec<Event> = events.iter().take(2).collect();
    assert!(matches!(&got[0], Event::Line { id: i, pid: p, text, .. } if *i == id && *p == pid && text == "hi"));
    let exit = Exit { code: Some(3), signal: None };
    assert_eq!(got[1], Event::Exit { id: id.clone(), pid, exit: exit.clone() });
    let Reply::Job { info, lines } = c.call(&Request::Attach { id }).unwrap() else { panic!("no record") };
    assert_eq!((info.exit, lines.len()), (Some(exit), 1));
    assert!(c.call(&Request::Attach { id: "nope".into() }).is_err());

    let frame = Frame { output: "alpha".into(), w: 2, h: 1, rgb: vec![1, 2, 3, 4, 5, 6] };
    let mut framed = Vec::new();
    ck_ipc::send(&mut framed, &frame).unwrap();
    let file = dir.join("frame.bin");
    std::fs::write(&file, framed).unwrap();
    let (_, pid) = submit(&mut c, &format!("cat {} >&3", file.display()), true);
    let relayed = events
        .iter()
        .take(3)
        .find_map(|e| match e {
            Event::Frame { pid: p, frame, .. } if p == pid => Some(frame),
            _ => None,
        });
    assert_eq!(relayed, Some(frame));
    drop(std::fs::remove_dir_all(dir));
}
