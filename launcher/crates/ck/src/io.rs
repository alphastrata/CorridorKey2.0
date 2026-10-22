//! ffmpeg pipes: probing a video or frame directory, decoding to u16 frames, encoding u16 BGR(A) frames.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::JoinHandle;

use eyre::{OptionExt, Result, WrapErr, bail, ensure};

/// A clip ffmpeg can read: its `-i` arguments, size, rate and length.
#[derive(Clone, Debug)]
pub struct Source {
    pub args: Vec<String>,
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    pub frames: usize,
    pub deep: bool,
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

/// Whether a pixel format holds more than 8 bits per channel.
fn deep(text: &str) -> bool {
    field(text, "pix_fmt")
        .is_some_and(|f| ["9", "10", "12", "14", "16", "48", "64", "f16", "f32"].iter().any(|d| f.contains(d)))
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
    /// # Errors
    /// When ffprobe fails or the directory holds no frames.
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
                deep: deep(&text),
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
            deep: deep(&text),
        })
    }

    /// Frames as u16 rows of `channels`, the first `limit`, decoded on a thread; 8-bit sources decode at 8 bits
    /// and widen exactly (`v · 257`), as swscale's 16-bit path rounds differently.
    #[must_use]
    pub fn decode(&self, channels: usize, limit: usize) -> flume::Receiver<Result<Vec<u16>>> {
        let (tx, rx) = flume::bounded(4);
        let deep = self.deep;
        let pix = match (channels, deep) {
            (3, true) => "rgb48le",
            (3, false) => "rgb24",
            (_, true) => "gray16le",
            _ => "gray",
        };
        let len = self.w * self.h * channels * (1 + usize::from(deep));
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-nostdin"])
            .args(&self.args)
            .args([
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
                    let frame = if deep {
                        bytemuck::pod_collect_to_vec(&buf)
                    } else {
                        buf.iter().map(|&b| u16::from(b) * 257).collect()
                    };
                    tx.send(Ok(frame))
                        .map_err(|_| eyre::eyre!("decoder dropped"))
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

/// Image files with extension `ext` in `dir`, sorted.
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

/// Encoder settings for the 3-channel outputs (4-channel ones are `ProRes` 4444).
#[derive(Clone, Debug)]
pub struct Encoder {
    pub codec: String,
    pub crf: u32,
    pub preset: String,
    pub pix_fmt: String,
    pub threads: usize,
}

/// An ffmpeg encoder fed u16 BGR(A) frames on a thread.
pub struct Writer {
    tx: Option<flume::Sender<Vec<u8>>>,
    join: Option<JoinHandle<Result<()>>>,
}

impl Writer {
    /// # Errors
    /// When ffmpeg cannot start.
    pub fn spawn(
        path: &Path,
        fps: f64,
        (w, h): (usize, usize),
        channels: usize,
        linear: bool,
        enc: &Encoder,
    ) -> Result<Self> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-pix_fmt",
        ])
        .arg(if channels == 4 { "bgra64le" } else { "bgr48le" })
        .args([
            "-s:v",
            &format!("{w}x{h}"),
            "-r",
            &format!("{fps:.8}"),
            "-i",
            "pipe:0",
            "-an",
        ]);
        if channels == 4 {
            cmd.args(["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]);
        } else {
            cmd.args([
                "-c:v",
                &enc.codec,
                "-preset",
                &enc.preset,
                "-pix_fmt",
                &enc.pix_fmt,
                "-crf",
                &enc.crf.to_string(),
            ]);
        }
        cmd.args([
            "-threads",
            &enc.threads.max(1).to_string(),
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
        ])
        .args([
            "-color_trc",
            if linear { "linear" } else { "iec61966-2-1" },
            "-movflags",
            "+write_colr",
        ])
        .arg(path)
        .stdin(Stdio::piped());
        let mut child = cmd.spawn().wrap_err("ffmpeg")?;
        let mut stdin = child.stdin.take().ok_or_eyre("ffmpeg stdin")?;
        let (tx, rx) = flume::bounded::<Vec<u8>>(2);
        let name = path.display().to_string();
        let join = std::thread::spawn(move || {
            rx.iter().try_for_each(|f| stdin.write_all(&f))?;
            drop(stdin);
            let status = child.wait()?;
            if !status.success() {
                bail!("ffmpeg failed ({status}): {name}");
            }
            Ok(())
        });
        Ok(Self {
            tx: Some(tx),
            join: Some(join),
        })
    }

    /// # Errors
    /// When the encoder has stopped.
    pub fn write(&self, frame: Vec<u8>) -> Result<()> {
        self.tx
            .as_ref()
            .ok_or_eyre("writer closed")?
            .send(frame)
            .map_err(|_| eyre::eyre!("encoder stopped"))
    }

    /// Closes the pipe and waits for ffmpeg.
    ///
    /// # Errors
    /// When ffmpeg failed.
    pub fn finish(mut self) -> Result<()> {
        drop(self.tx.take());
        self.join
            .take()
            .ok_or_eyre("writer joined")?
            .join()
            .map_err(|_| eyre::eyre!("writer panicked"))?
    }
}
