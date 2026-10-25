use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use ahash::AHashMap;
use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::{OptionExt, Result, WrapErr, bail};
use half::{bf16, f16};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype as St;

use crate::sources;

pub enum Data {
    Map(Arc<Mmap>, Range<usize>),
    Owned(Vec<u8>),
}

pub struct Entry {
    pub shape: Vec<usize>,
    pub dtype: St,
    pub data: Data,
}

#[derive(Default)]
pub struct Weights(pub AHashMap<String, Entry>);

pub fn map(path: &Path) -> Result<Arc<Mmap>> {
    let file = std::fs::File::open(path).wrap_err_with(|| path.display().to_string())?;
    Ok(Arc::new(unsafe { Mmap::map(&file) }?))
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        Self::renamed(path, "", "")
    }

    pub fn vfms(dir: &Path, stats: &Path) -> Result<Self> {
        let mut w = Self::renamed(&dir.join("cradio/C-RADIOv4-SO400M/model.safetensors"), "radio_model.", "cradio.")?;
        w.0.extend(sources::torch(&dir.join("moge/model.pt"), "model.encoder.", "moge.")?);
        w.0.extend(sources::rvm(&dir.join("rvm/pretrain_rvm_small16_256_204031069.npz"))?);
        w.0.extend(Self::renamed(stats, "", "stats.")?.0);
        Ok(w)
    }

    fn renamed(path: &Path, from: &str, to: &str) -> Result<Self> {
        let map = map(path)?;
        let (header, meta) = SafeTensors::read_metadata(&map)?;
        let start = header + 8;
        Ok(Self(
            meta.tensors()
                .into_iter()
                .filter_map(|(name, info)| {
                    let entry = Entry {
                        shape: info.shape.clone(),
                        dtype: info.dtype,
                        data: Data::Map(map.clone(), start + info.data_offsets.0..start + info.data_offsets.1),
                    };
                    Some((format!("{to}{}", name.strip_prefix(from)?), entry))
                })
                .collect(),
        ))
    }

    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    pub fn view(&self, name: &str) -> Result<(Vec<usize>, St, &[u8])> {
        let e = self.0.get(name).ok_or_eyre(format!("no weight {name}"))?;
        let bytes = match &e.data {
            Data::Map(map, range) => &map[range.clone()],
            Data::Owned(v) => v.as_slice(),
        };
        Ok((e.shape.clone(), e.dtype, bytes))
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

    pub fn nhwc(&self, n: &str) -> Result<Tensor> {
        let (shape, v) = self.get(n)?;
        let [o, c, k, _] = shape[..] else {
            eyre::bail!("{n}: want [out, c, k, k], got {shape:?}")
        };
        let kk = k * k;
        let t: Vec<f32> = (0..o * c * kk)
            .map(|i| {
                let (row, col) = (i / (c * kk), i % (c * kk));
                v[row * c * kk + col % c * kk + col / c]
            })
            .collect();
        self.gpu.upload_f32(&t, &[o, c * kk], self.half)
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
