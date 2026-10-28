//! Serves `ck_ipc::remote`: drives this window's input and snapshots it for an outside controller.

use std::fmt::Write;
use std::os::unix::net::{UnixListener, UnixStream};
use std::thread;

use ck_ipc::remote::{Act, Answer, socket};
use ck_ipc::{recv, send};
use slint::platform::{Key, PointerEventButton, WindowEvent};
use slint::{ComponentHandle, LogicalPosition, Model, SharedString};

use crate::AppWindow;

/// Listens for controllers unless another live window already owns the socket.
pub fn serve(ui: &AppWindow) {
    let Ok(path) = socket() else { return };
    if UnixStream::connect(&path).is_ok() {
        return;
    }
    drop(std::fs::remove_file(&path));
    let Ok(listener) = UnixListener::bind(&path) else { return };
    let ui = ui.as_weak();
    thread::spawn(move || {
        listener.incoming().flatten().for_each(|s| {
            let ui = ui.clone();
            thread::spawn(move || session(s, &ui));
        });
    });
}

fn session(stream: UnixStream, ui: &slint::Weak<AppWindow>) {
    let (Ok(mut rx), mut tx) = (stream.try_clone(), stream) else { return };
    let mut bus = None;
    while let Ok(act) = recv::<Act>(&mut rx) {
        let answer = match act {
            Act::Hello { shots } => {
                bus = ck_ipc::server::dial_bus(&shots);
                if bus.is_some() { Answer::Ok } else { Answer::Error("could not reach the snapshot bus".into()) }
            }
            Act::Snap => match (on_ui(ui, snap), &bus) {
                (Some(Ok((px, w, h, scale))), Some(b)) => match b.send(&px, 1) {
                    Ok(()) => Answer::Shot { w, h, scale },
                    Err(e) => Answer::Error(e.to_string()),
                },
                (Some(Err(e)), _) => Answer::Error(e),
                (_, None) => Answer::Error("say Hello first".into()),
                (None, _) => Answer::Error("the window has closed".into()),
            },
            Act::Log(n) => on_ui(ui, move |u| log(u, n)).map_or(Answer::Error("the window has closed".into()), Answer::Text),
            a => on_ui(ui, move |u| input(u, &a)).map_or(Answer::Error("the window has closed".into()), |()| Answer::Ok),
        };
        if send(&mut tx, &answer).is_err() {
            break;
        }
    }
}

/// Runs `f` on the event loop and waits for its result.
fn on_ui<T: Send + 'static>(ui: &slint::Weak<AppWindow>, f: impl FnOnce(&AppWindow) -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = flume::bounded(1);
    ui.upgrade_in_event_loop(move |u| drop(tx.send(f(&u)))).ok()?;
    rx.recv().ok()
}

fn snap(u: &AppWindow) -> Result<(Vec<u8>, u32, u32, f32), String> {
    let w = u.window();
    w.request_redraw();
    let px = w.take_snapshot().map_err(|e| e.to_string())?;
    Ok((px.as_bytes().to_vec(), px.width(), px.height(), w.scale_factor()))
}

fn log(u: &AppWindow, n: u32) -> String {
    let lines = u.get_log();
    let skip = lines.row_count().saturating_sub(usize::try_from(n).unwrap_or(usize::MAX));
    lines.iter().skip(skip).fold(String::new(), |mut out, l| {
        let _ = writeln!(out, "{} {} {}", l.time, l.level, l.text);
        out
    })
}

fn key(name: &str) -> SharedString {
    match name {
        "esc" => Key::Escape.into(),
        "enter" => Key::Return.into(),
        "tab" => Key::Tab.into(),
        "space" => Key::Space.into(),
        "backspace" => Key::Backspace.into(),
        "delete" => Key::Delete.into(),
        "up" => Key::UpArrow.into(),
        "down" => Key::DownArrow.into(),
        "left" => Key::LeftArrow.into(),
        "right" => Key::RightArrow.into(),
        "home" => Key::Home.into(),
        "end" => Key::End.into(),
        "ctrl" => Key::Control.into(),
        "shift" => Key::Shift.into(),
        "alt" => Key::Alt.into(),
        s => s.into(),
    }
}

fn input(u: &AppWindow, a: &Act) {
    let w = u.window();
    let at = |x: f32, y: f32| LogicalPosition::new(x, y);
    match a {
        Act::Hover { x, y } => w.dispatch_event(WindowEvent::PointerMoved { position: at(*x, *y) }),
        Act::Click { x, y } => [
            WindowEvent::PointerMoved { position: at(*x, *y) },
            WindowEvent::PointerPressed { position: at(*x, *y), button: PointerEventButton::Left },
            WindowEvent::PointerReleased { position: at(*x, *y), button: PointerEventButton::Left },
        ]
        .into_iter()
        .for_each(|e| w.dispatch_event(e)),
        Act::Scroll { x, y, dy } => {
            w.dispatch_event(WindowEvent::PointerMoved { position: at(*x, *y) });
            w.dispatch_event(WindowEvent::PointerScrolled { position: at(*x, *y), delta_x: 0.0, delta_y: *dy });
        }
        Act::Keys(keys) => {
            let keys: Vec<SharedString> = keys.iter().map(|k| key(k)).collect();
            keys.iter().for_each(|k| w.dispatch_event(WindowEvent::KeyPressed { text: k.clone() }));
            keys.iter().rev().for_each(|k| w.dispatch_event(WindowEvent::KeyReleased { text: k.clone() }));
        }
        Act::Hello { .. } | Act::Snap | Act::Log(_) => {}
    }
}
