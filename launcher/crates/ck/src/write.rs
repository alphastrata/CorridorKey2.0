use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread::JoinHandle;

use eyre::{OptionExt, Result, WrapErr, bail};

#[derive(Clone, Debug)]
pub struct Encoder {
    pub codec: String,
    pub crf: u32,
    pub preset: String,
    pub pix_fmt: String,
    pub threads: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exr {
    Half,
    Float,
}

#[derive(Clone, Copy, Debug)]
pub enum Format<'a> {
    Video { enc: &'a Encoder, linear: bool },
    Exr(Exr),
}

fn tags(cmd: &mut Command, linear: bool) {
    cmd.args(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc"])
        .arg(if linear { "linear" } else { "iec61966-2-1" })
        .args(["-movflags", "+write_colr"]);
}

pub struct Writer {
    tx: Option<flume::Sender<Vec<u8>>>,
    join: Option<JoinHandle<Result<()>>>,
}

impl Writer {
    pub fn spawn(path: &Path, fps: f64, (w, h): (usize, usize), channels: usize, format: Format) -> Result<Self> {
        let pix = match (format, channels) {
            (Format::Video { .. }, 4) => "bgra64le",
            (Format::Video { .. }, _) => "bgr48le",
            (Format::Exr(_), 1) => "grayf32le",
            (Format::Exr(_), 4) => "gbrapf32le",
            (Format::Exr(_), _) => "gbrpf32le",
        };
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
            pix,

        ])
        .args([
            "-s:v",
            &format!("{w}x{h}"),
            "-r",
            &format!("{fps:.8}"),
            "-i",
            "pipe:0",
            "-an",
        ]);

        match format {
            Format::Exr(t) => {
                let fmt = if t == Exr::Half { "half" } else { "float" };
                cmd.args([
                    "-c:v",
                    "exr",
                    "-compression",
                    "zip16",
                    "-format",
                    fmt,
                    "-start_number",
                    "0",

                ]);
            }
            Format::Video { linear, .. } if channels == 4 => {
                cmd.args(["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]);
                tags(&mut cmd, linear);
            }
            Format::Video { enc, linear } => {
                let pix = match (enc.codec.as_str(), enc.pix_fmt.as_str()) {
                    ("hevc_nvenc", "yuv444p10le") => "yuv444p16le",
                    (_, p) => p,
                };

                cmd.args(["-c:v", &enc.codec, "-preset", &enc.preset, "-pix_fmt", pix]);
                if enc.codec.ends_with("_nvenc") {
                    cmd.args(["-rc", "vbr", "-cq", &enc.crf.to_string()]);
                } else {
                    cmd.args(["-crf", &enc.crf.to_string(), "-threads", &enc.threads.to_string()]);
                }
                tags(&mut cmd, linear);
            }
        }

        cmd.arg(path).stdin(Stdio::piped());
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

    pub fn write(&self, frame: Vec<u8>) -> Result<()> {
        self.tx
            .as_ref()
            .ok_or_eyre("writer closed")?
            .send(frame)
            .map_err(|_| eyre::eyre!("encoder stopped"))
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.tx.take());
        self.join
            .take()
            .ok_or_eyre("writer joined")?
            .join()
            .map_err(|_| eyre::eyre!("writer panicked"))?
    }
}
