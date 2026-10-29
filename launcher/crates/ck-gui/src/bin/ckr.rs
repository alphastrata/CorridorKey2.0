//! Drives a running ck-gui: `ckr 1300,500 s:1300,500,-120 k:ctrl+, h:20,30 w:300 p:out.png`.

use std::time::Duration;

use ck_ipc::remote::{Act, Answer, Remote};
use eyre::{WrapErr, eyre};

#[derive(argh::FromArgs)]
/// Steps: `x,y` click, `h:x,y` hover, `s:x,y,dy` scroll (dy < 0 scrolls down), `k:ctrl+,` keys, `w:ms` wait, `l:n` last n log lines, `t:` status, `p:file.png` snapshot; by element (`id:Comp::name`, `type:CheckBox` or a label, `#n` for the nth): `f:` find, `c:` click, `a:` activate, `v:query=value` set.
struct Args {
    #[argh(positional)]
    steps: Vec<String>,
}

fn coords(s: &str) -> eyre::Result<Vec<f32>> {
    s.split(',').map(|v| v.trim().parse::<f32>().wrap_err_with(|| format!("bad number in {s}"))).collect()
}

/// `query#n` picks the `n`th match; plain `query` the first.
fn nth(s: &str) -> (String, u32) {
    s.rsplit_once('#').and_then(|(q, n)| Some((q.to_owned(), n.parse().ok()?))).unwrap_or_else(|| (s.to_owned(), 0))
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
        ("f", _) => r.find(rest)?.iter().for_each(|e| {
            println!(
                "{} {:?} label={:?} value={:?} checked={:?} enabled={:?} visible={} at {:.0},{:.0} {:.0}x{:.0}",
                e.kind, e.id, e.label, e.value, e.checked, e.enabled, e.visible, e.x, e.y, e.w, e.h
            );
        }),

        ("c" | "a", _) => {
            let (query, n) = nth(rest);
            drop(r.act(&if kind == "c" { Act::Press { query, n } } else { Act::Activate { query, n } })?);
        }
        ("v", _) => {
            let (target, value) = rest.rsplit_once('=').ok_or_else(|| eyre!("v: wants query=value"))?;
            let (query, n) = nth(target);
            drop(r.act(&Act::Set { query, n, value: value.into() })?);
        }
        ("t", _) => println!("{}", r.text(&Act::Status)?),
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
