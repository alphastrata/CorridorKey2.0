//! Safetensors files mapped once; tensors uploaded in the layouts the kernels want.

use std::path::Path;

use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::{OptionExt, Result, WrapErr, bail};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype as St, Metadata};

pub struct Weights {
    map: Mmap,
    start: usize,
    meta: Metadata,
}

impl Weights {
    /// # Errors
    /// When the file cannot be mapped or its header parsed.
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

    /// Shape and values (as f32) of `name`.
    ///
    /// # Errors
    /// When `name` is missing or not a float tensor.
    pub fn get(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let info = self.meta.info(name).ok_or_eyre(format!("no weight {name}"))?;
        let bytes = &self.map[self.start + info.data_offsets.0..self.start + info.data_offsets.1];
        let values = match info.dtype {
            St::F32 => bytemuck::pod_collect_to_vec(bytes),
            St::BF16 => bytemuck::pod_collect_to_vec::<u8, bf16>(bytes)
                .into_iter()
                .map(f32::from)
                .collect(),
            St::F16 => bytemuck::pod_collect_to_vec::<u8, f16>(bytes)
                .into_iter()
                .map(f32::from)
                .collect(),
            d => bail!("{name}: unsupported dtype {d:?}"),
        };
        Ok((info.shape.clone(), values))
    }
}

/// Uploads weights under `prefix` to one GPU; GEMM operands in `half`, everything else f32.
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

    /// # Errors
    /// When the weight is missing.
    pub fn get(&self, n: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        self.weights.get(&self.name(n))
    }

    /// # Errors
    /// When the weight is missing or the upload fails.
    pub fn f32(&self, n: &str) -> Result<Tensor> {
        let (shape, v) = self.get(n)?;
        self.gpu.upload_f32(&v, &shape, Dtype::F32)
    }

    /// A linear or convolution weight as a GEMM operand `[out, in · kh · kw]`.
    ///
    /// # Errors
    /// When the weight is missing or the upload fails.
    pub fn mat(&self, n: &str) -> Result<Tensor> {
        self.rows(&[n])
    }

    /// Several weights stacked along `out`.
    ///
    /// # Errors
    /// When a weight is missing or widths differ.
    pub fn rows(&self, names: &[&str]) -> Result<Tensor> {
        let parts = names.iter().map(|n| self.get(n)).collect::<Result<Vec<_>>>()?;
        let width: usize = parts[0].0[1..].iter().product();
        let v: Vec<f32> = parts.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        self.gpu.upload_f32(&v, &[v.len() / width, width], self.half)
    }

    /// Biases (or any vectors) concatenated, f32.
    ///
    /// # Errors
    /// When a weight is missing.
    pub fn cat(&self, names: &[&str]) -> Result<Tensor> {
        let v: Vec<f32> = names
            .iter()
            .map(|n| self.get(n).map(|(_, v)| v))
            .collect::<Result<Vec<_>>>()?
            .concat();
        self.gpu.upload_f32(&v, &[v.len()], Dtype::F32)
    }

    /// A transposed convolution's `[in, out, k, k]` as the GEMM operand `[out · k · k, in]`.
    ///
    /// # Errors
    /// When the weight is missing.
    pub fn deconv(&self, n: &str) -> Result<Tensor> {
        let (shape, v) = self.get(n)?;
        let (rows, cols) = (shape[0], shape[1..].iter().product::<usize>());
        let t: Vec<f32> = (0..cols * rows).map(|i| v[i % rows * cols + i / rows]).collect();
        self.gpu.upload_f32(&t, &[cols, rows], self.half)
    }

    /// Depthwise 7×7 filters `[c, 1, 7, 7]` as f32 taps `[filters, 49, c]`.
    ///
    /// # Errors
    /// When a weight is missing.
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
