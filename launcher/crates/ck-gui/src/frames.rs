use crate::convert::{Float, Narrow, Quant};
use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread;

use ahash::AHashMap;
use eyre::eyre;
use flume::{Receiver, Sender, TryRecvError};
use image::DynamicImage;
use slint::{Rgb8Pixel, SharedPixelBuffer};

use crate::clips::frame_files;

const PREVIEW: (u32, u32) = (1920, 1080);
const CACHE_BYTES: usize = 1_500 << 20;
const AHEAD: usize = 48;

pub type Pixels = SharedPixelBuffer<Rgb8Pixel>;

#[derive(Clone)]
pub struct Shown {
    pub path: PathBuf,
    pub index: usize,
    pub count: usize,
    pub fps: f32,
    pub source: (u32, u32),
    pub pixels: Pixels,
    pub b: Option<(PathBuf, Pixels)>,
}

enum Req {
    Show {
        path: PathBuf,
        index: usize,
        linear: bool,
        full: bool,
        b: Option<(PathBuf, bool)>,
    },
    Release,
}

#[derive(Clone)]
pub struct Frames(Sender<Req>);

impl Frames {
    pub fn new(shown: impl Fn(eyre::Result<Shown>) + Send + 'static) -> Self {
        let (tx, rx) = flume::unbounded();
        thread::Builder::new()
            .name("frames".into())
            .spawn(move || Worker::default().run(&rx, &shown))
            .ok();
        Self(tx)
    }

    /// Decodes `path` at `index`, plus the same frame of `b` for an A|B wipe.
    pub fn show(&self, path: &Path, index: usize, linear: bool, full: bool, b: Option<(PathBuf, bool)>) {
        drop(self.0.send(Req::Show {
            path: path.to_owned(),
            index,
            linear,
            full,
            b,
        }));
    }

    pub fn release(&self) {
        drop(self.0.send(Req::Release));
    }
}

struct Decoder {
    child: Child,
    out: ChildStdout,
    next: usize,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        drop(self.child.kill());
        drop(self.child.wait());
    }
}

enum Kind {
    Video { fps: f32, decoder: Option<Decoder> },
    Sequence(Vec<PathBuf>),
}

struct Source {
    kind: Kind,
    count: usize,
    size: (u32, u32),
    preview: (u32, u32),
}

fn fit(size: (u32, u32), bound: (u32, u32)) -> (u32, u32) {
    let s = (f64::from(bound.0) / f64::from(size.0))
        .min(f64::from(bound.1) / f64::from(size.1))
        .min(1.0);
    let even = |v: f64| (v.round().max(0.0).u32() & !1).max(2);
    (even(f64::from(size.0) * s), even(f64::from(size.1) * s))
}

fn probe(path: &Path) -> eyre::Result<(u32, u32, f32, usize)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",

            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "default=nw=1",
        ])
        .arg(path)
        .output()
        .map_err(|e| eyre!("ffprobe: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);

    let field = |k: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .unwrap_or("")
            .trim()
            .to_owned()
    };
    let (w, h) = (
        field("width").parse().unwrap_or(0),
        field("height").parse().unwrap_or(0),
    );
    let fps = field("avg_frame_rate")
        .split_once('/')
        .and_then(|(a, b)| Some(a.parse::<f32>().ok()? / b.parse::<f32>().ok().filter(|&d| d > 0.0)?))
        .unwrap_or(24.0);

    let count = field("nb_frames").parse().ok().unwrap_or_else(|| {
        (field("duration").parse::<f32>().unwrap_or(0.0) * fps)
            .round()
            .max(0.0)
            .usize()
    });
    if w == 0 || h == 0 {
        return Err(eyre!("cannot read {}", path.display()));
    }
    Ok((w, h, fps, count.max(1)))
}

fn srgb(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let v = if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (v * 255.0 + 0.5).u8()
}

/// A binary PPM as ck's `--preview` writes it.
pub fn ppm(path: &Path) -> eyre::Result<Pixels> {
    let bytes = std::fs::read(path)?;
    let mut f = bytes.splitn(5, u8::is_ascii_whitespace);
    eyre::ensure!(f.next() == Some(b"P6".as_slice()), "{}: not a PPM", path.display());
    let mut num = || -> eyre::Result<u32> { Ok(std::str::from_utf8(f.next().unwrap_or_default())?.parse()?) };
    let (w, h) = (num().ok(), num().ok());
    let (Some(w), Some(h), Some(_), Some(data)) = (w, h, f.next(), f.next()) else {
        return Err(eyre!("{}: not a PPM", path.display()));

    };
    eyre::ensure!(
        data.len() == 3 * w as usize * h as usize,
        "{}: truncated",
        path.display()
    );
    Ok(SharedPixelBuffer::clone_from_slice(data, w, h))
}

fn decode_file(path: &Path, linear: bool, size: (u32, u32)) -> eyre::Result<Pixels> {
    let img = image::open(path).map_err(|e| eyre!("{}: {e}", path.display()))?;
    let float = matches!(img, DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_));
    let rgb = if linear && float {
        let f = img.to_rgb32f();
        image::RgbImage::from_fn(f.width(), f.height(), |x, y| image::Rgb(f.get_pixel(x, y).0.map(srgb)))
    } else {
        img.to_rgb8()
    };

    let rgb = if (rgb.width(), rgb.height()) == size {
        rgb
    } else {
        image::imageops::resize(&rgb, size.0, size.1, image::imageops::FilterType::Triangle)
    };
    Ok(SharedPixelBuffer::clone_from_slice(rgb.as_raw(), size.0, size.1))
}

/// A clip's source size and frame count.
pub fn dims(path: &Path) -> Option<(u32, u32, usize)> {
    Source::open(path).ok().map(|s| (s.size.0, s.size.1, s.count))
}

impl Source {
    fn open(path: &Path) -> eyre::Result<Self> {
        if path.is_dir() {
            let files = frame_files(path);
            let first = files.first().ok_or_else(|| eyre!("no frames in {}", path.display()))?;
            let size = image::image_dimensions(first).map_err(|e| eyre!("{}: {e}", first.display()))?;
            return Ok(Self {
                count: files.len(),
                kind: Kind::Sequence(files),
                size,
                preview: fit(size, PREVIEW),

            });
        }
        let (w, h, fps, count) = probe(path)?;
        Ok(Self {
            kind: Kind::Video { fps, decoder: None },
            count,
            size: (w, h),
            preview: fit((w, h), PREVIEW),
        })
    }

    fn fps(&self) -> f32 {
        match self.kind {
            Kind::Video { fps, .. } => fps,
            Kind::Sequence(_) => 24.0,
        }
    }

    fn decode(&mut self, path: &Path, index: usize, linear: bool, size: (u32, u32)) -> eyre::Result<Pixels> {
        match &mut self.kind {
            Kind::Sequence(files) => decode_file(&files[index], linear, size),
            Kind::Video { fps, decoder } => {
                let streaming = size == self.preview;
                let reusable = streaming
                    && decoder
                        .as_ref()
                        .is_some_and(|d| d.next <= index && index < d.next + AHEAD);
                if !reusable {
                    *decoder = None;

                    let mut child = Command::new("ffmpeg")
                        .args([
                            "-v",
                            "error",
                            "-nostdin",
                            "-ss",
                            &format!("{:.6}", index.f32() / *fps),
                            "-i",
                        ])
                        .arg(path)
                        .args([
                            "-an",

                            "-vf",
                            &format!("scale={}:{}:flags=bilinear", size.0, size.1),
                            "-pix_fmt",
                            "rgb24",
                            "-f",
                            "rawvideo",
                            "-",
                        ])
                        .stdout(Stdio::piped())
                        .stderr(Stdio::null())
                        .spawn()
                        .map_err(|e| eyre!("ffmpeg: {e}"))?;

                    let out = child.stdout.take().ok_or_else(|| eyre!("ffmpeg: no stdout"))?;
                    *decoder = Some(Decoder {
                        child,
                        out,
                        next: index,
                    });
                }
                let d = decoder.as_mut().ok_or_else(|| eyre!("ffmpeg: no decoder"))?;
                let mut buf = vec![0u8; size.0 as usize * size.1 as usize * 3];
                while d.next <= index {
                    d.out
                        .read_exact(&mut buf)
                        .map_err(|_| eyre!("frame {} is past the end", index + 1))?;
                    d.next += 1;
                }

                if !streaming {
                    *decoder = None;
                }
                Ok(SharedPixelBuffer::clone_from_slice(&buf, size.0, size.1))
            }
        }
    }
}

#[derive(Default)]
struct Worker {
    sources: AHashMap<PathBuf, Source>,
    cache: AHashMap<(PathBuf, usize), Pixels>,
    order: VecDeque<(PathBuf, usize)>,
    bytes: usize,
}

impl Worker {
    fn keep(&mut self, key: (PathBuf, usize), px: Pixels) {
        self.bytes += px.as_bytes().len();
        self.cache.insert(key.clone(), px);
        self.order.push_back(key);
        while self.bytes > CACHE_BYTES {
            let Some(old) = self.order.pop_front() else { break };
            if let Some(px) = self.cache.remove(&old) {
                self.bytes -= px.as_bytes().len();
            }
        }
    }

    fn frame(&mut self, path: &Path, index: usize, linear: bool, full: bool) -> eyre::Result<Shown> {
        if !self.sources.contains_key(path) {
            let src = Source::open(path)?;
            self.sources.insert(path.to_owned(), src);
        }
        let src = self.sources.get_mut(path).ok_or_else(|| eyre!("source vanished"))?;
        let index = index.min(src.count - 1);
        let (count, fps, size) = (src.count, src.fps(), src.size);
        let full = full && src.preview != size;

        let pixels = match self.cache.get(&(path.to_owned(), index)) {
            Some(px) if !full => px.clone(),
            _ => {
                let target = if full { size } else { src.preview };
                let px = src.decode(path, index, linear, target)?;
                if !full {
                    self.keep((path.to_owned(), index), px.clone());
                }
                px
            }

        };
        Ok(Shown {
            path: path.to_owned(),
            index,
            count,
            fps,
            source: size,
            pixels,
            b: None,
        })
    }

    fn prefetch(&mut self, path: &Path, from: usize, linear: bool, rx: &Receiver<Req>) -> Option<Req> {
        let count = self.sources.get(path)?.count;
        let drained: Result<(), Option<Req>> =
            (from + 1..(from + 1 + AHEAD).min(count)).try_fold((), |(), i| match rx.try_recv() {
                Ok(req) => Err(Some(req)),
                Err(TryRecvError::Disconnected) => Err(None),
                Err(TryRecvError::Empty) if !self.cache.contains_key(&(path.to_owned(), i)) => {
                    let Some(src) = self.sources.get_mut(path) else {
                        return Err(None);
                    };
                    let preview = src.preview;
                    let Ok(px) = src.decode(path, i, linear, preview) else {

                        return Err(None);
                    };
                    self.keep((path.to_owned(), i), px);
                    Ok(())
                }
                Err(TryRecvError::Empty) => Ok(()),
            });
        match drained {
            Err(req) => req,
            Ok(()) => None,
        }
    }

    fn run(mut self, rx: &Receiver<Req>, shown: &dyn Fn(eyre::Result<Shown>)) {
        let mut pending = None;
        loop {
            let Some(mut req) = pending.take().or_else(|| rx.recv().ok()) else {
                return;
            };
            while let Ok(newer) = rx.try_recv() {
                req = match (req, newer) {
                    (Req::Release, n) | (_, n @ Req::Release) => {
                        self.release();
                        n
                    }

                    (_, n) => n,
                };
            }
            match req {
                Req::Release => self.release(),
                Req::Show {
                    path,
                    index,
                    linear,
                    full,
                    b,
                } => {
                    let result = self.frame(&path, index, linear, full).map(|mut s| {
                        s.b = b.as_ref().and_then(|(p, lin)| {
                            Some((p.clone(), self.frame(p, s.index, *lin, full).ok()?.pixels))
                        });
                        s
                    });
                    let next = result.as_ref().ok().map(|s| s.index);
                    shown(result);
                    if let Some(i) = next {
                        pending = self.prefetch(&path, i, linear, rx).or_else(|| {
                            b.as_ref().and_then(|(p, lin)| self.prefetch(p, i, *lin, rx))
                        });
                    }
                }
            }
        }
    }

    fn release(&mut self) {
        self.sources.clear();
        self.cache.clear();
        self.order.clear();
        self.bytes = 0;
    }
}
