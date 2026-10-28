//! Drives a running ck-gui: `ckr 1300,500 s:1300,500,-120 k:ctrl+, h:20,30 w:300 p:out.png`.

use std::time::Duration;

use ck_ipc::remote::{Act, Answer, Remote};
use eyre::{WrapErr, eyre};

#[derive(argh::FromArgs)]
/// Steps: `x,y` click, `h:x,y` hover, `s:x,y,dy` scroll (dy < 0 scrolls down), `k:ctrl+,` keys, `w:ms` wait, `l:n` last n log lines, `p:file.png` snapshot.
struct Args {
    #[argh(positional)]
    steps: Vec<String>,
}

fn coords(s: &str) -> eyre::Result<Vec<f32>> {
    s.split(',').map(|v| v.trim().parse::<f32>().wrap_err_with(|| format!("bad number in {s}"))).collect()
}

fn step(r: &mut Remote, s: &str) -> eyre::Result<()> {
    let (kind, rest) = s.split_once(':').unwrap_or(("", s));
    match (kind, coords(rest).as_deref()) {
        ("w", _) => std::thread::sleep(Duration::from_millis(rest.parse()?)),
        ("p", _) => {
            let (px, w, h, scale) = r.snap()?;
            image::RgbaImage::from_raw(w, h, px).ok_or_else(|| eyre!("short snapshot"))?.save(rest)?;
            println!("{rest}: {w}x{h} at scale {scale}");
        }
        ("l", _) => {
            if let Answer::Text(t) = r.act(&Act::Log(rest.parse()?))? {
                print!("{t}");
            }
        }
        ("k", _) => {
            let keys = match rest.strip_suffix('+') {
                Some(head) => head.split('+').chain(["+"]).map(String::from).collect(),
                None => rest.split('+').map(String::from).collect(),
            };
            r.act(&Act::Keys(keys))?;
        }
        ("h", Ok([x, y])) => drop(r.act(&Act::Hover { x: *x, y: *y })?),
        ("s", Ok([x, y, dy])) => drop(r.act(&Act::Scroll { x: *x, y: *y, dy: *dy })?),
        ("", Ok([x, y])) => drop(r.act(&Act::Click { x: *x, y: *y })?),
        _ => return Err(eyre!("unrecognised step {s}")),
    }
    Ok(())
}

fn main() -> eyre::Result<()> {
    let args: Args = argh::from_env();
    let mut r = Remote::connect().wrap_err("no ck-gui is listening")?;
    args.steps.iter().try_for_each(|s| step(&mut r, s).wrap_err_with(|| format!("step {s}")))
}
