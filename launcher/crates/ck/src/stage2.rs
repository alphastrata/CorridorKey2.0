//! Stage 2: per-frame refinement of the baton plus RGB at /16, depatchified to /1 logits.

use ck_gpu::{Dtype, Gpu, Tensor};
use eyre::Result;

use crate::attn::Cache;
use crate::block::{Block, Kind, load_blocks};
use crate::ops::{self, Conv, Gate};
use crate::weights::Loader;

const PATCH: Conv = Conv { k: 32, s: 16, p: 8 };

pub struct Stage2 {
    patch1: (Tensor, Tensor),
    patch2: (Tensor, Tensor),
    gate: (Tensor, Tensor),
    pre: Vec<Block>,
    last: Vec<Block>,
    depatch: (Tensor, Tensor),
    baton: usize,
    dim: usize,
}

impl Stage2 {
    /// # Errors
    /// When a weight is missing.
    pub fn load(ld: &Loader) -> Result<Self> {
        let (w, hw) = (16, 8);
        let swin = |shift| Kind::Swin { window: w, shift };
        let pre = [
            Kind::Full,
            swin((hw, hw)),
            Kind::CSwin,
            swin((hw, 0)),
            Kind::Full,
            Kind::CSwin,
            swin((0, 0)),
            swin((hw, hw)),
        ];
        let last = [swin((hw, 0)), swin((0, hw))];
        let pair = |n: &str, mat: bool| -> Result<(Tensor, Tensor)> {
            let w = if mat {
                ld.mat(&format!("{n}.weight"))?
            } else {
                ld.deconv(&format!("{n}.weight"))?
            };
            Ok((w, ld.f32(&format!("{n}.bias"))?))
        };
        let patch1 = pair("patch_1.conv", true)?;
        Ok(Self {
            baton: ld.get("depatchify.deconv.weight")?.0[0] - patch1.0.shape[0],
            dim: ld.get("depatchify.deconv.weight")?.0[0],
            patch1,
            patch2: pair("patch_2.conv", true)?,
            gate: pair("patch_2_gate", true)?,
            pre: load_blocks(ld, "pre_blocks", &pre, false, 16)?,
            last: load_blocks(ld, "final_blocks", &last, false, 16)?,
            depatch: pair("depatchify.deconv", false)?,
        })
    }

    /// The blocks in order.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        self.pre.iter().chain(&self.last)
    }

    /// `rgb` f32 `[n, 3, H, W]`, `baton` f32 `[n, 512, H / 16, W / 16]` → logits f32 `[n, 4, H, W]`.
    ///
    /// # Errors
    /// On a launch or GEMM failure.
    pub fn forward(&self, gpu: &Gpu, cache: &Cache, rgb: &Tensor, baton: &Tensor) -> Result<Tensor> {
        let [n, _, hh, ww] = rgb.shape[..] else {
            eyre::bail!("rgb wants [n, 3, H, W]")
        };
        let (h, w, d) = (hh / 16, ww / 16, self.dim);
        let x = gpu.empty(&[n, 1, h, w, d], Dtype::F32)?;
        ops::to_nhwc(gpu, baton, &x, (d, 0))?;
        let cols = ops::im2col(gpu, rgb, false, PATCH)?;
        gpu.linear_into(&cols, &self.patch1.0, Some(&self.patch1.1), &x, (d, self.baton), false)?;
        let x = self.pre.iter().try_fold(x, |x, b| b.forward(gpu, cache, &x, None))?;

        let rgb_dim = d - self.baton;
        let gate = gpu.empty(&[n * h * w, rgb_dim], Dtype::F32)?;
        gpu.linear(&ops::to_half(gpu, &x)?, &self.gate.0, Some(&self.gate.1), &gate, false)?;
        let skip = gpu.empty(&[n * h * w, rgb_dim], Dtype::F32)?;
        gpu.linear(&cols, &self.patch2.0, Some(&self.patch2.1), &skip, false)?;
        ops::add_gated(gpu, &x, &skip, Gate::Silu(&gate), (d, self.baton))?;
        let x = self.last.iter().try_fold(x, |x, b| b.forward(gpu, cache, &x, None))?;

        let cols = gpu.empty(&[n * h * w, self.depatch.0.shape[0]], Dtype::F32)?;
        gpu.linear(&ops::to_half(gpu, &x)?, &self.depatch.0, None, &cols, false)?;
        ops::col2im(gpu, &cols, &self.depatch.1, (n, 4, h, w), PATCH)
    }
}
