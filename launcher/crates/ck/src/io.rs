
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ck_gpu::Dtype;
use eyre::{OptionExt, Result, WrapErr, ensure};

#[derive(Clone, Debug)]
pub struct Source {
    pub args: Vec<String>,
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    pub frames: usize,
    pub pix_fmt: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plane {
    Rgb,
    Grey,
    Alpha,
}

fn ffprobe(args: &[&str]) -> Result<String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error"])
        .args(args)
        .output()
        .wrap_err("ffprobe")?;
    ensure!(
        out.status.success(),
        "ffprobe: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(String::from_utf8(out.stdout)?)
}

fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
}

fn size(path: &str) -> Result<(usize, usize, String)> {
    let text = ffprobe(&[
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,pix_fmt",
        "-of",
        "default=nw=1",
        path,
    ])?;
    let num = |k: &str| -> Result<usize> { Ok(field(&text, k).ok_or_eyre(format!("{path}: no {k}"))?.parse()?) };
    Ok((num("width")?, num("height")?, text))
}

impl Source {
    pub fn open(path: &Path, dir_fps: f64) -> Result<Self> {
        if path.is_dir() {
            let ext = ["png", "jpg", "jpeg", "exr"]
                .into_iter()
                .find_map(|e| {
                    let files = frame_files(path, e);
                    (!files.is_empty()).then_some((e, files))
                })
                .ok_or_eyre(format!("{}: no frames", path.display()))?;
            let (w, h, text) = size(&ext.1[0].to_string_lossy())?;
            let pattern = path.join(format!("*.{}", ext.0)).to_string_lossy().into_owned();
            let args = [
                "-framerate",
                &dir_fps.to_string(),
                "-pattern_type",
                "glob",
                "-i",
                &pattern,
            ];
            return Ok(Self {
                args: args.map(String::from).to_vec(),
                w,
                h,
                fps: dir_fps,
                frames: ext.1.len(),
                pix_fmt: field(&text, "pix_fmt").unwrap_or_default().into(),
            });
        }
        let name = path.to_string_lossy().into_owned();
        let (w, h, text) = size(&name)?;
        let fps = field(&text, "r_frame_rate")
            .and_then(|r| r.split_once('/'))
            .and_then(|(a, b)| Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?))
            .unwrap_or(24.0);
        let frames = if let Some(n) = field(&text, "nb_frames").and_then(|n| n.parse().ok()) {
            n
        } else {
            let text = ffprobe(&[
                "-count_packets",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "default=nw=1",
                &name,
            ])?;
            field(&text, "nb_read_packets").ok_or_eyre("no frame count")?.parse()?
        };
        Ok(Self {
            args: vec!["-i".into(), name],
            w,
            h,
            fps,
            frames,
            pix_fmt: field(&text, "pix_fmt").unwrap_or_default().into(),
        })
    }

    #[must_use]
    pub fn deep(&self) -> bool {
        ["9", "10", "12", "14", "16", "48", "64", "f16", "f32"]
            .iter()
            .any(|d| self.pix_fmt.contains(d))
    }

    #[must_use]
    pub fn float(&self) -> bool {
        self.pix_fmt.contains("f16") || self.pix_fmt.contains("f32")
    }

    #[must_use]
    pub fn alpha(&self) -> bool {
        ["rgba", "bgra", "argb", "abgr", "ya", "gbrap"]
            .iter()
            .any(|p| self.pix_fmt.starts_with(p))
    }

    #[must_use]
    pub fn dtype(&self) -> Dtype {
        if self.float() { Dtype::F32 } else { Dtype::U16 }
    }

    #[must_use]
    pub fn decode(&self, plane: Plane, limit: usize) -> flume::Receiver<Result<Vec<u8>>> {
        let (tx, rx) = flume::bounded(4);
        let (pix, size, widen) = match (plane, self.float(), self.deep()) {
            (Plane::Rgb, true, _) => ("gbrpf32le", 12, false),
            (Plane::Rgb, _, true) => ("rgb48le", 6, false),
            (Plane::Rgb, _, false) => ("rgb24", 3, true),
            (_, _, true) => ("gray16le", 2, false),
            _ => ("gray", 1, true),
        };
        let len = self.w * self.h * size;
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-nostdin"]).args(&self.args);
        if plane == Plane::Alpha {
            cmd.args(["-vf", "alphaextract"]);
        }
        cmd.args([
            "-frames:v",
            &limit.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "pipe:1",
        ])
        .stdout(Stdio::piped());
        std::thread::spawn(move || {
            let mut run = || -> Result<()> {
                let mut child = cmd.spawn().wrap_err("ffmpeg")?;
                let mut out = child.stdout.take().ok_or_eyre("ffmpeg stdout")?;
                let mut buf = vec![0u8; len];
                (0..limit).try_for_each(|_| {
                    out.read_exact(&mut buf).wrap_err("ffmpeg ended early")?;
                    let frame = if widen {
                        buf.iter().flat_map(|&b| (u16::from(b) * 257).to_le_bytes()).collect()
                    } else {
                        buf.clone()
                    };
                    tx.send(Ok(frame)).map_err(|_| eyre::eyre!("decoder dropped"))
                })?;
                drop(out);
                child.wait()?;
                Ok(())
            };
            if let Err(e) = run() {
                let _ = tx.send(Err(e));
            }
        });
        rx
    }
}

#[must_use]
pub fn frame_files(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case(ext)))
        .collect();
    files.sort();
    files
}
