use crate::convert::Quant;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eyre::Result;

use crate::frame::Output;

const WIDTH: usize = 960;
const EVERY: Duration = Duration::from_millis(250);

/// Downscaled 8-bit snapshots of the latest written frame, one `<output>.ppm` per output, for a live view.
pub struct Preview {
    dir: PathBuf,
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
    pub fn new(dir: &Path, size: (usize, usize), exr: bool) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_owned(),
            size,
            exr,
            last: None,
        })
    }

    pub fn offer<'a>(&mut self, images: impl IntoIterator<Item = (Output, &'a [u8])>) -> Result<()> {
        if self.last.is_some_and(|t| t.elapsed() < EVERY) {
            return Ok(());
        }
        self.last = Some(Instant::now());
        images.into_iter().try_for_each(|(o, img)| self.write(o, img))
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
    fn write(&self, o: Output, img: &[u8]) -> Result<()> {
        let (w, h) = self.size;
        let step = w.div_ceil(WIDTH).max(1);
        let (pw, ph) = (w / step, h / step);
        let mut ppm = format!("P6\n{pw} {ph}\n255\n").into_bytes();
        ppm.extend((0..ph).flat_map(|y| (0..pw).map(move |x| (x, y))).flat_map(|(x, y)| {
            let [r, g, b, a] = self.rgba(o, img, y * step * w + x * step);

            let under = if (x / 8 + y / 8) % 2 == 0 { 0.4 } else { 0.6 };
            [r, g, b].map(|v| ((v * a + under * (1.0 - a)).clamp(0.0, 1.0) * 255.0 + 0.5).u8())
        }));
        let (tmp, out) = (
            self.dir.join(format!(".{}.ppm", o.name())),
            self.dir.join(format!("{}.ppm", o.name())),
        );
        std::fs::write(&tmp, ppm)?;
        Ok(std::fs::rename(tmp, out)?)
    }
}
