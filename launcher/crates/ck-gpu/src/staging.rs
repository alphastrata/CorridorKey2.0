use cudarc::driver::result::{memcpy_dtoh_async, memcpy_htod_async};
use cudarc::driver::{CudaEvent, PinnedHostSlice};
use eyre::{Result, ensure};
use rayon::prelude::*;

use crate::{Dtype, Gpu, Tensor};

const PIECE: usize = 32 << 20;
const GRAIN: usize = 1 << 20;

#[derive(Default)]
pub(crate) struct Staging(Vec<(PinnedHostSlice<u8>, Option<CudaEvent>)>);

fn pieces(lens: impl Iterator<Item = usize>) -> impl Iterator<Item = (usize, usize, usize)> {
    lens.enumerate()
        .flat_map(|(p, len)| (0..len).step_by(PIECE).map(move |o| (p, o, PIECE.min(len - o))))
}

impl Gpu {
    fn slot<'a>(&self, staging: &'a mut Staging, i: usize) -> Result<&'a mut (PinnedHostSlice<u8>, Option<CudaEvent>)> {
        while staging.0.len() < 2 {
            staging
                .0
                .push((unsafe { self.ctx.alloc_pinned_with_flags::<u8>(PIECE, 0) }?, None));
        }
        let slot = &mut staging.0[i % 2];
        if let Some(e) = slot.1.take() {
            e.synchronize()?;
        }
        Ok(slot)
    }

    pub fn upload_parts(&self, parts: &[&[u8]], shape: &[usize], dtype: Dtype) -> Result<Tensor> {
        let len: usize = parts.iter().map(|p| p.len()).sum();
        ensure!(
            len == shape.iter().product::<usize>() * dtype.size(),
            "upload size {shape:?}"
        );
        let out = self.empty(shape, dtype)?;
        let mut staging = self.staging.borrow_mut();
        let starts: Vec<usize> = parts
            .iter()
            .scan(0, |at, p| Some(std::mem::replace(at, *at + p.len())))
            .collect();
        pieces(parts.iter().map(|p| p.len()))
            .enumerate()
            .try_for_each(|(i, (p, o, n))| -> Result<()> {
                let (host, done) = self.slot(&mut staging, i)?;
                let host = &mut host.as_mut_slice()?[..n];
                host.par_chunks_mut(GRAIN)
                    .zip(parts[p][o..o + n].par_chunks(GRAIN))
                    .for_each(|(d, s)| d.copy_from_slice(s));
                let dst = out.ptr() + (starts[p] + o) as u64;
                unsafe { memcpy_htod_async(dst, &*host, self.stream.cu_stream()) }?;
                *done = Some(self.stream.record_event(None)?);
                Ok(())
            })?;
        Ok(out)
    }

    pub fn download(&self, t: &Tensor) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; t.bytes()];
        let mut staging = self.staging.borrow_mut();
        let all: Vec<_> = pieces(std::iter::once(t.bytes())).map(|(_, o, n)| (o, n)).collect();
        let issue = |staging: &mut Staging, i: usize| -> Result<()> {
            let (o, n) = all[i];
            let (host, done) = self.slot(staging, i)?;
            let host = &mut host.as_mut_slice()?[..n];
            unsafe { memcpy_dtoh_async(host, t.ptr() + o as u64, self.stream.cu_stream()) }?;
            *done = Some(self.stream.record_event(None)?);
            Ok(())
        };
        if !all.is_empty() {
            issue(&mut staging, 0)?;
        }
        (0..all.len()).try_for_each(|i| -> Result<()> {
            if i + 1 < all.len() {
                issue(&mut staging, i + 1)?;
            }
            let (o, n) = all[i];
            let (host, _) = self.slot(&mut staging, i)?;
            bytes[o..o + n]
                .par_chunks_mut(GRAIN)
                .zip(host.as_slice()?[..n].par_chunks(GRAIN))
                .for_each(|(d, s)| d.copy_from_slice(s));
            Ok(())
        })?;
        Ok(bytes)
    }
}
