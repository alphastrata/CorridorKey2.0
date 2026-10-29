//! A private ck-gui per test: its own Xvfb display, HOME and runtime directory, driven over `ck_ipc::remote`.

use std::io::{BufRead, BufReader};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ck_ipc::remote::{Act, Element, Remote};

pub struct Gui {
    pub r: Remote,
    pub root: PathBuf,
    app: Child,
    xvfb: Child,
}

fn have(tool: &str) -> bool {
    Command::new("which").arg(tool).stdout(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// Polls `f` every 50 ms until it yields, or panics after `secs` with `what`.
pub fn wait<T>(secs: u64, what: &str, mut f: impl FnMut() -> Option<T>) -> T {
    let start = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(start.elapsed() < Duration::from_secs(secs), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The release `ck`, if built; tests that key real clips skip without it.
pub fn ck() -> Option<PathBuf> {
    std::env::var_os("CK_BIN")
        .map(PathBuf::from)
        .or_else(|| Some(Path::new(env!("CARGO_BIN_EXE_ck-gui")).parent()?.parent()?.join("release/ck")))
        .filter(|p| p.is_file())
}

/// GPUs NVML reports, for tests that need one.
pub fn gpus() -> usize {
    nvml_wrapper::Nvml::init().ok().and_then(|n| n.device_count().ok()).map_or(0, |c| c as usize)
}

/// A short test-pattern video, made once per size and frame count.
pub fn clip(w: u32, h: u32, frames: u32) -> PathBuf {
    let dir = std::env::temp_dir().join("ck-gui-tests");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("footage dir: {e}"));
    let out = dir.join(format!("pattern_{w}x{h}_{frames}.mp4"));
    if !out.is_file() {
        let part = dir.join(format!("{}.{}.part.mp4", out.display(), std::process::id()));
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg(format!("testsrc2=size={w}x{h}:rate=24"))
            .args(["-frames:v", &frames.to_string(), "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast"])
            .arg(&part)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "ffmpeg could not make {}", out.display());

        std::fs::rename(&part, &out).unwrap_or_else(|e| panic!("footage rename: {e}"));
    }
    out
}

impl Gui {
    /// Starts a GUI seeded with `state` (TOML) and `clips`, or `None` when Xvfb or ffmpeg is missing.
    pub fn start(state: Option<&str>, clips: &[PathBuf]) -> Option<Self> {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        if !have("Xvfb") || !have("ffmpeg") {
            eprintln!("skipped: Xvfb and ffmpeg are needed for GUI tests");
            return None;
        }
        let n = NEXT.fetch_add(1, Ordering::AcqRel);
        let root = std::env::temp_dir().join(format!("ck-gui-test-{}-{n}", std::process::id()));
        drop(std::fs::remove_dir_all(&root));
        let (home, rt, bin) = (root.join("home"), root.join("rt"), root.join("bin"));

        [&home.join(".ck_gui"), &rt, &bin]
            .iter()
            .for_each(|d| std::fs::create_dir_all(d).unwrap_or_else(|e| panic!("test dirs: {e}")));
        std::fs::set_permissions(&rt, std::fs::Permissions::from_mode(0o700)).unwrap_or_else(|e| panic!("rt mode: {e}"));
        if let Some(s) = state {
            std::fs::write(home.join(".ck_gui/state.toml"), s).unwrap_or_else(|e| panic!("seed state: {e}"));
        }
        if let Some(ck) = ck() {
            std::os::unix::fs::symlink(ck, bin.join("ck")).unwrap_or_else(|e| panic!("ck link: {e}"));
        }

        let mut xvfb = Command::new("Xvfb")
            .args(["-displayfd", "1", "-screen", "0", "1600x1000x24", "-nolisten", "tcp"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("Xvfb: {e}"));
        let mut display = String::new();
        BufReader::new(xvfb.stdout.take().unwrap_or_else(|| panic!("Xvfb stdout")))
            .read_line(&mut display)
            .unwrap_or_else(|e| panic!("Xvfb display: {e}"));

        let display = display.trim().to_owned();
        let gui = Command::new(env!("CARGO_BIN_EXE_ck-gui"))
            .args(clips)
            .env("HOME", &home)
            .env("XDG_RUNTIME_DIR", &rt)
            .env("DISPLAY", format!(":{display}"))
            .env_remove("WAYLAND_DISPLAY")
            .env("CK_BIN", bin.join("ck"))
            .env("CK_REPO", Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.."))
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(root.join("gui.log")).unwrap_or_else(|e| panic!("gui log: {e}")))
            .spawn()
            .unwrap_or_else(|e| panic!("ck-gui: {e}"));

        let r = wait(30, "the GUI's remote socket", || Remote::connect_in(&rt.join("ck")).ok());
        let g = Self { r, root, app: gui, xvfb };
        Self::settle();
        Some(g)
    }

    /// Lets timers and layout catch up after an action.
    pub fn settle() {
        std::thread::sleep(Duration::from_millis(300));
    }

    pub fn find(&mut self, q: &str) -> Vec<Element> {
        self.r.find(q).unwrap_or_else(|e| panic!("find {q}: {e:?}"))
    }

    /// Index and element of the first visible `kind` matching `q`.
    fn first(&mut self, q: &str, kind: &str) -> Option<(u32, Element)> {
        let e = self.find(q).into_iter().enumerate().find(|(_, e)| e.kind == kind && e.visible)?;
        Some((u32::try_from(e.0).ok()?, e.1))
    }

    /// Scrolls the inspector, top down, until a visible `kind` matching `q` shows.
    pub fn reveal(&mut self, q: &str, kind: &str) -> Option<(u32, Element)> {
        if let Some(found) = self.first(q, kind) {
            return Some(found);
        }

        let head = self.first("INSPECTOR", "PanelTitle").unwrap_or_else(|| panic!("inspector")).1;
        let (x, y) = (head.x + head.w / 2.0, head.y + 300.0);
        self.act(&Act::Scroll { x, y, dy: 10_000.0 });
        (0..30).find_map(|_| {
            self.first(q, kind).or_else(|| {
                self.act(&Act::Scroll { x, y, dy: -120.0 });
                None
            })
        })
    }

    fn must(&mut self, q: &str, kind: &str) -> (u32, Element) {
        self.reveal(q, kind).unwrap_or_else(|| panic!("no {kind} matching {q}, even scrolled"))
    }

    /// The first `kind` matching `q`, scrolled into view.
    pub fn one(&mut self, q: &str, kind: &str) -> Element {
        self.must(q, kind).1
    }

    pub fn act(&mut self, a: &Act) {
        self.r.act(a).unwrap_or_else(|e| panic!("{a:?}: {e}"));
        Self::settle();
    }

    pub fn activate(&mut self, q: &str, kind: &str) {
        let n = self.must(q, kind).0;
        self.act(&Act::Activate { query: q.into(), n });
    }

    pub fn press(&mut self, q: &str, kind: &str) {
        let n = self.must(q, kind).0;
        self.act(&Act::Press { query: q.into(), n });
    }

    /// Picks `item` from the combo box matching `q` by keyboard, as popup items cannot be clicked by position.
    pub fn choose(&mut self, q: &str, item: &str) {
        self.press(q, "ComboBox");
        let items: Vec<String> = self.find("type:ListItem").into_iter().filter(|e| e.visible).map(|e| e.label).collect();
        let at = items.iter().position(|l| l == item).unwrap_or_else(|| panic!("{item} not among {items:?}"));
        (0..items.len()).for_each(|_| self.keys(&["up"]));
        (0..at).for_each(|_| self.keys(&["down"]));
        self.keys(&["enter"]);
    }

    pub fn keys(&mut self, keys: &[&str]) {
        self.act(&Act::Keys(keys.iter().map(|&k| k.into()).collect()));
    }

    /// Sets the value of the first `kind` labelled `q`.
    pub fn set(&mut self, q: &str, kind: &str, value: &str) {
        let n = self.must(q, kind).0;
        self.act(&Act::Set { query: q.into(), n, value: value.into() });
    }

    pub fn status(&mut self) -> String {
        self.r.text(&Act::Status).unwrap_or_else(|e| panic!("status: {e:?}"))
    }

    pub fn log(&mut self, n: u32) -> String {
        self.r.text(&Act::Log(n)).unwrap_or_else(|e| panic!("log {n}: {e:?}"))
    }

    pub fn state_path(&self) -> PathBuf {
        self.root.join("home/.ck_gui/state.toml")
    }

    pub fn state(&self) -> toml::Table {
        std::fs::read_to_string(self.state_path()).ok().and_then(|s| s.parse().ok()).unwrap_or_default()
    }

    /// A value from the state file by dotted path, numbers indexing arrays: `queue.0.settings.share.vram`.
    pub fn saved(&self, path: &str) -> Option<toml::Value> {
        path.split('.').try_fold(toml::Value::Table(self.state()), |v, k| match v {
            toml::Value::Table(mut t) => t.remove(k),
            toml::Value::Array(mut a) => k.parse().ok().filter(|&i| i < a.len()).map(|i| a.swap_remove(i)),
            _ => None,
        })
    }

    /// Saves a snapshot beside the test's files, for a look when one fails.
    pub fn snap(&mut self, name: &str) -> PathBuf {
        let (px, w, h, _) = self.r.snap().unwrap_or_else(|e| panic!("snapshot: {e:?}"));
        let path = self.root.join(format!("{name}.png"));

        image::RgbaImage::from_raw(w, h, px)
            .unwrap_or_else(|| panic!("snapshot size"))
            .save(&path)
            .unwrap_or_else(|e| panic!("snapshot save: {e}"));
        path
    }
}

impl Drop for Gui {
    fn drop(&mut self) {
        drop(self.app.kill());
        drop(self.app.wait());
        drop(self.xvfb.kill());
        drop(self.xvfb.wait());
        if !std::thread::panicking() {
            drop(std::fs::remove_dir_all(&self.root));
        }
    }
}
