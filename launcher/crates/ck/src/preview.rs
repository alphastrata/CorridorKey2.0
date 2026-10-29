use crate::convert::Quant;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eyre::Result;

use crate::frame::Output;
use crate::run::PreviewTo;

const WIDTH: usize = 960;
const EVERY: Duration = Duration::from_millis(250);

enum Sink {
    Dir(PathBuf),
    #[cfg(unix)]
    Socket(UnixStream),
    Gone,
}

/// Downscaled 8-bit snapshots of the latest written frame, per output, for a live view.
pub struct Preview {
    sink: Sink,
    size: (usize, usize),
    exr: bool,
    last: Option<Instant>,
}

fn srgb(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

impl Preview {
    pub fn new(to: &PreviewTo, size: (usize, usize), exr: bool) -> Result<Self> {
        let sink = match to {
            PreviewTo::Dir(d) => {
                std::fs::create_dir_all(d)?;
                Sink::Dir(d.clone())
            }

            #[cfg(unix)]
            PreviewTo::Fd(fd) => {
                let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(*fd) }.try_clone_to_owned()?;
                Sink::Socket(UnixStream::from(fd))
            }
            #[cfg(not(unix))]
            PreviewTo::Fd(_) => Sink::Gone,
        };
        Ok(Self {
            sink,
            size,
            exr,
            last: None,
        })
    }

    /// Snapshots frame `index`'s outputs, at most every `EVERY`.
    pub fn offer<'a>(&mut self, index: usize, images: impl IntoIterator<Item = (Output, &'a [u8])>) -> Result<()> {
        if self.last.is_some_and(|t| t.elapsed() < EVERY) {
            return Ok(());
        }
        self.last = Some(Instant::now());
        images.into_iter().try_for_each(|(o, img)| self.write(index, o, img))
    }

    #[allow(clippy::many_single_char_names)] // tensor axes: (b, c, h, w)
    fn rgba(&self, o: Output, img: &[u8], i: usize) -> [f32; 4] {
        let n = self.size.0 * self.size.1;
        if self.exr {
            let at = |p: usize| {
                let k = 4 * (p * n + i);
                img.get(k..k + 4)
                    .map_or(0.0, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            };
            if o.planes() == 1 {
                let a = at(0);

                return [a, a, a, 1.0];
            }
            let [g, b, r] = [at(0), at(1), at(2)].map(|v| srgb(v.clamp(0.0, 1.0)));
            return [r, g, b, if o.planes() == 4 { at(3) } else { 1.0 }];
        }
        let c = o.channels();
        let at = |k: usize| {
            img.get(2 * (c * i + k)..2 * (c * i + k) + 2)
                .map_or(0.0, |b| f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0)
        };
        [at(2), at(1), at(0), if c == 4 { at(3) } else { 1.0 }]
    }

    #[allow(clippy::many_single_char_names)] // tensor axes: (b, c, h, w)
    fn write(&mut self, index: usize, o: Output, img: &[u8]) -> Result<()> {
        let (w, h) = self.size;
        let step = w.div_ceil(WIDTH).max(1);
        let (pw, ph) = (w / step, h / step);
        let rgb: Vec<u8> = (0..ph)
            .flat_map(|y| (0..pw).map(move |x| (x, y)))
            .flat_map(|(x, y)| {
                let [r, g, b, a] = self.rgba(o, img, y * step * w + x * step);
                let under = if (x / 8 + y / 8) % 2 == 0 { 0.4 } else { 0.6 };
                [r, g, b].map(|v| ((v * a + under * (1.0 - a)).clamp(0.0, 1.0) * 255.0 + 0.5).u8())
            })
            .collect();

        match &mut self.sink {
            Sink::Dir(dir) => {
                let (tmp, out) = (dir.join(format!(".{}.ppm", o.name())), dir.join(format!("{}.ppm", o.name())));
                let mut ppm = format!("P6\n{pw} {ph}\n255\n").into_bytes();
                ppm.extend(rgb);
                std::fs::write(&tmp, ppm)?;
                std::fs::rename(tmp, out)?;
            }

            #[cfg(unix)]
            Sink::Socket(s) => {
                let frame = ck_ipc::Frame {
                    output: o.name().into(),
                    index: u32::try_from(index)?,
                    w: u32::try_from(pw)?,
                    h: u32::try_from(ph)?,
                    rgb,
                };
                if ck_ipc::send(s, &frame).is_err() {
                    self.sink = Sink::Gone;
                }
            }

            Sink::Gone => {}
        }
        Ok(())
    }
}
