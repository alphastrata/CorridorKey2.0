use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::driver::sys::{self, CUresult};
use cudarc::driver::{CudaSlice, DevicePtr, DriverError};
use eyre::{Result, WrapErr, ensure};
use half::{bf16, f16};
use rayon::prelude::*;

use crate::Gpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    Bf16,
    F16,
    I32,
    U16,
}

impl Dtype {
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 | Self::U16 => 2,
        }
    }
}

#[derive(Default, Debug)]
pub struct Meter {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl Meter {
    #[must_use]
    pub fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.peak.store(self.live(), Ordering::Relaxed);
    }

    fn add(&self, n: usize) {
        let live = self.live.fetch_add(n, Ordering::Relaxed) + n;
        self.peak.fetch_max(live, Ordering::Relaxed);
    }
}

struct Buf {
    slice: CudaSlice<u8>,
    meter: Arc<Meter>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        self.meter.live.fetch_sub(self.slice.len(), Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct Tensor {
    _alloc: Arc<Buf>,
    base: u64,
    off: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl Tensor {
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    #[must_use]
    pub fn bytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    #[must_use]
    pub fn ptr(&self) -> u64 {
        self.base + self.off as u64
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        ensure!(
            shape.iter().product::<usize>() == self.numel(),
            "reshape {:?} → {shape:?}",
            self.shape
        );
        Ok(Self {
            shape: shape.to_vec(),
            ..self.clone()
        })
    }

    pub fn into_shape(self, shape: &[usize]) -> Result<Self> {
        ensure!(
            shape.iter().product::<usize>() == self.numel(),
            "reshape {:?} → {shape:?}",
            self.shape
        );
        Ok(Self {
            shape: shape.to_vec(),
            ..self
        })
    }

    pub fn rows(&self, start: usize, len: usize) -> Result<Self> {
        ensure!(start + len <= self.shape[0], "rows {start}+{len} of {:?}", self.shape);
        let row = self.numel() / self.shape[0] * self.dtype.size();
        let mut shape = self.shape.clone();
        shape[0] = len;
        Ok(Self {
            off: self.off + start * row,
            shape,
            ..self.clone()
        })
    }

    pub fn at(&self, elems: usize, shape: &[usize]) -> Result<Self> {
        ensure!(
            elems + shape.iter().product::<usize>() <= self.numel(),
            "view {elems}+{shape:?} of {:?}",
            self.shape
        );
        Ok(Self {
            off: self.off + elems * self.dtype.size(),
            shape: shape.to_vec(),
            ..self.clone()
        })
    }

    pub fn bitcast(&self, dtype: Dtype) -> Result<Self> {
        let mut shape = self.shape.clone();
        let last = shape.last_mut().ok_or_else(|| eyre::eyre!("scalar bitcast"))?;
        let bytes = *last * self.dtype.size();
        ensure!(
            bytes.is_multiple_of(dtype.size()),
            "bitcast {:?} → {dtype:?}",
            self.dtype
        );
        *last = bytes / dtype.size();
        Ok(Self {
            dtype,
            shape,
            ..self.clone()
        })
    }
}

impl Gpu {
    fn retry<T>(&self, alloc: impl Fn() -> Result<T, DriverError>) -> Result<T> {
        match alloc() {
            Err(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY)) => {
                self.stream.synchronize()?;
                let mut pool = std::ptr::null_mut();
                unsafe {
                    sys::cuDeviceGetDefaultMemPool(&raw mut pool, self.ctx.cu_device()).result()?;
                    sys::cuMemPoolTrimTo(pool, 0).result()?;
                }
                Ok(alloc()?)
            }
            other => Ok(other?),
        }
    }

    fn oom(&self, n: usize) -> String {
        let mib = |b: usize| b >> 20;
        format!("allocating {} MiB with {} MiB live", mib(n), mib(self.meter.live()))
    }

    pub fn empty(&self, shape: &[usize], dtype: Dtype) -> Result<Tensor> {
        let n = shape.iter().product::<usize>() * dtype.size();
        let buf = self
            .retry(|| unsafe { self.stream.alloc::<u8>(n.max(16)) })
            .wrap_err_with(|| self.oom(n))?;
        Ok(wrap(self, buf, dtype, shape))
    }

    pub fn zeros(&self, shape: &[usize], dtype: Dtype) -> Result<Tensor> {
        let n = shape.iter().product::<usize>() * dtype.size();
        let buf = self
            .retry(|| self.stream.alloc_zeros::<u8>(n.max(16)))
            .wrap_err_with(|| self.oom(n))?;
        Ok(wrap(self, buf, dtype, shape))
    }

    pub fn copy(&self, t: &Tensor) -> Result<Tensor> {
        let out = self.empty(&t.shape, t.dtype)?;
        unsafe { cudarc::driver::result::memcpy_dtod_async(out.ptr(), t.ptr(), t.bytes(), self.stream.cu_stream()) }?;
        Ok(out)
    }

    pub fn copy_into(&self, src: &Tensor, dst: &Tensor) -> Result<()> {
        ensure!(src.bytes() == dst.bytes(), "copy {:?} into {:?}", src.shape, dst.shape);
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst.ptr(), src.ptr(), src.bytes(), self.stream.cu_stream())
        }?;
        Ok(())
    }

    pub fn upload(&self, bytes: &[u8], shape: &[usize], dtype: Dtype) -> Result<Tensor> {
        self.upload_parts(&[bytes], shape, dtype)
    }

    pub fn upload_f32(&self, v: &[f32], shape: &[usize], dtype: Dtype) -> Result<Tensor> {
        let bytes: Vec<u8> = match dtype {
            Dtype::F32 => bytemuck::cast_slice(v).to_vec(),
            Dtype::Bf16 => {
                bytemuck::cast_slice(&v.par_iter().copied().map(bf16::from_f32).collect::<Vec<_>>()).to_vec()
            }
            Dtype::F16 => bytemuck::cast_slice(&v.par_iter().copied().map(f16::from_f32).collect::<Vec<_>>()).to_vec(),
            Dtype::I32 | Dtype::U16 => eyre::bail!("upload_f32 to an integer type"),
        };
        self.upload(&bytes, shape, dtype)
    }

    pub fn download_f32(&self, t: &Tensor) -> Result<Vec<f32>> {
        let bytes = self.download(t)?;
        Ok(match t.dtype {
            Dtype::F32 => bytemuck::pod_collect_to_vec(&bytes),
            Dtype::Bf16 => bytemuck::pod_collect_to_vec::<u8, bf16>(&bytes)
                .into_iter()
                .map(f32::from)
                .collect(),
            Dtype::F16 => bytemuck::pod_collect_to_vec::<u8, f16>(&bytes)
                .into_iter()
                .map(f32::from)
                .collect(),
            Dtype::I32 | Dtype::U16 => eyre::bail!("download_f32 of an integer type"),
        })
    }
}

fn wrap(gpu: &Gpu, buf: CudaSlice<u8>, dtype: Dtype, shape: &[usize]) -> Tensor {
    let (base, _) = buf.device_ptr(&gpu.stream);
    gpu.meter.add(buf.len());
    Tensor {
        _alloc: Arc::new(Buf {
            slice: buf,
            meter: gpu.meter.clone(),
        }),
        base,
        off: 0,
        dtype,
        shape: shape.to_vec(),
    }
}
