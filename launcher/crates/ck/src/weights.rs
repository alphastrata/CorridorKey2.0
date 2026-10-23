use std::path::Path;

use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::{OptionExt, Result, WrapErr, bail};
use half::{bf16, f16};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype as St, Metadata};

pub struct Weights {
    map: Mmap,
    start: usize,
    meta: Metadata,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).wrap_err_with(|| path.display().to_string())?;
        let map = unsafe { Mmap::map(&file) }?;
        let (header, meta) = SafeTensors::read_metadata(&map)?;
        Ok(Self {
            map,
            start: header + 8,
            meta,
        })
    }

    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.meta.info(name).is_some()
    }

    pub fn view(&self, name: &str) -> Result<(Vec<usize>, St, &[u8])> {
        let info = self.meta.info(name).ok_or_eyre(format!("no weight {name}"))?;
        let bytes = &self.map[self.start + info.data_offsets.0..self.start + info.data_offsets.1];
        Ok((info.shape.clone(), info.dtype, bytes))
    }

    pub fn get(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (shape, dtype, bytes) = self.view(name)?;
        let mut v = vec![0.0f32; bytes.len() * 8 / dtype.bitsize().max(8)];
        fill(bytes, dtype, &mut v, |_, x| x).wrap_err_with(|| name.to_owned())?;
        Ok((shape, v))
    }
}

fn fill<T: Send>(src: &[u8], dtype: St, dst: &mut [T], to: impl Fn(usize, f32) -> T + Sync) -> Result<()> {
    let read: fn(&[u8]) -> f32 = match dtype {
        St::F32 => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        St::BF16 => |b| f32::from(bf16::from_le_bytes([b[0], b[1]])),
        St::F16 => |b| f32::from(f16::from_le_bytes([b[0], b[1]])),
        d => bail!("unsupported dtype {d:?}"),
    };
    let size = dtype.bitsize() / 8;
    dst.par_chunks_mut(1 << 14)
        .zip(src.par_chunks(size << 14))
        .enumerate()
        .for_each(|(c, (out, bytes))| {
            out.iter_mut()
                .zip(bytes.chunks_exact(size))
                .enumerate()
                .for_each(|(j, (o, b))| *o = to((c << 14) + j, read(b)));
        });
    Ok(())
}

pub const TAIL: usize = 8;

#[derive(Clone, Copy)]
pub struct Loader<'a> {
    pub gpu: &'a Gpu,
    pub weights: &'a Weights,
    pub prefix: &'a str,
    pub half: Dtype,
}

impl Loader<'_> {
    fn name(&self, n: &str) -> String {
        format!("{}{n}", self.prefix)
    }

    #[must_use]
    pub fn has(&self, n: &str) -> bool {
        self.weights.has(&self.name(n))
    }

    pub fn get(&self, n: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        self.weights.get(&self.name(n))
    }

    pub fn f32(&self, n: &str) -> Result<Tensor> {
        let (shape, v) = self.get(n)?;
        self.gpu.upload_f32(&v, &shape, Dtype::F32)
    }

    pub fn mat(&self, n: &str) -> Result<Tensor> {
        self.rows(&[n])
    }

    pub fn biased(&self, w: &str, b: &str) -> Result<Tensor> {
        let ((shape, w), (_, b)) = (self.get(w)?, self.get(b)?);
        self.tail(&shape, &w, &b)
    }

    pub fn tail(&self, shape: &[usize], w: &[f32], b: &[f32]) -> Result<Tensor> {
        let inner = shape[1..].iter().product::<usize>();
        let round: fn(f32) -> f32 = if self.half == Dtype::F16 {
            |v| f16::from_f32(v).to_f32()
        } else {
            |v| bf16::from_f32(v).to_f32()
        };
        let v: Vec<f32> = w
            .chunks(inner)
            .zip(b)
            .flat_map(|(row, &bias)| {
                let hi = round(bias);
                row.iter().copied().chain([hi, bias - hi]).chain([0.0; TAIL - 2])
            })
            .collect();
        self.gpu.upload_f32(&v, &[shape[0], inner + TAIL], self.half)
    }

    pub fn rows(&self, names: &[&str]) -> Result<Tensor> {
        self.stack(names, None)
    }

    pub fn scaled(&self, names: &[&str], gamma: &[f32]) -> Result<Tensor> {
        self.stack(names, Some(gamma))
    }

    fn stack(&self, names: &[&str], gamma: Option<&[f32]>) -> Result<Tensor> {
        let parts = names
            .iter()
            .map(|n| self.weights.view(&self.name(n)))
            .collect::<Result<Vec<_>>>()?;
        let width: usize = parts[0].0[1..].iter().product();
        let total: usize = parts.iter().map(|(s, _, _)| s.iter().product::<usize>()).sum();
        let mut out = vec![0u16; total];
        let to: fn(f32) -> u16 = if self.half == Dtype::F16 {
            |v| f16::from_f32(v).to_bits()
        } else {
            |v| bf16::from_f32(v).to_bits()
        };
        parts
            .iter()
            .try_fold((out.as_mut_slice(), 0), |(rest, row), (shape, dtype, bytes)| {
                let n: usize = shape.iter().product();
                let (head, tail) = rest.split_at_mut(n);
                match gamma {
                    Some(g) => fill(bytes, *dtype, head, |i, v| to(v * g[row + i / width]))?,
                    None => fill(bytes, *dtype, head, |_, v| to(v))?,
                }
                Ok::<_, eyre::Report>((tail, row + n / width))
            })?;
        self.gpu
            .upload(bytemuck::cast_slice(&out), &[total / width, width], self.half)
    }

    pub fn cat(&self, names: &[&str]) -> Result<Tensor> {
        let v: Vec<f32> = names
            .iter()
            .map(|n| self.get(n).map(|(_, v)| v))
            .collect::<Result<Vec<_>>>()?
            .concat();
        self.gpu.upload_f32(&v, &[v.len()], Dtype::F32)
    }

    pub fn deconv(&self, n: &str) -> Result<Tensor> {
        let (shape, v) = self.get(n)?;
        let (rows, cols) = (shape[0], shape[1..].iter().product::<usize>());
        let t: Vec<f32> = (0..cols * rows).map(|i| v[i % rows * cols + i / rows]).collect();
        self.gpu.upload_f32(&t, &[cols, rows], self.half)
    }

    pub fn taps(&self, names: &[&str]) -> Result<Tensor> {
        let parts = names.iter().map(|n| self.get(n)).collect::<Result<Vec<_>>>()?;
        let c = parts[0].0[0];
        let v: Vec<f32> = parts
            .iter()
            .flat_map(|(_, w)| (0..49 * c).map(move |i| w[i % c * 49 + i / c]))
            .collect();
        self.gpu.upload_f32(&v, &[names.len(), 49, c], Dtype::F32)
    }
}
