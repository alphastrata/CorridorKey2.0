
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Arc;

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys::{
    self, cublasComputeType_t, cublasLtEpilogue_t, cublasLtMatmulDescAttributes_t as Attr, cudaDataType_t,
};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use eyre::{Result, ensure};

use crate::{Arg, Dtype, Gpu, Tensor};

const WORKSPACE: usize = 32 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Epilogue {
    None,
    Bias,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Gemm {
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    pub ld: usize,
    pub x: Dtype,
    pub w: Dtype,
    pub y: Dtype,
    pub epilogue: Epilogue,
    pub accumulate: bool,
}

struct Plan {
    desc: sys::cublasLtMatmulDesc_t,
    layouts: [sys::cublasLtMatrixLayout_t; 3],
    algo: sys::cublasLtMatmulAlgo_t,
}

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            self.layouts.iter().for_each(|l| _ = lt::destroy_matrix_layout(*l));
            _ = lt::destroy_matmul_desc(self.desc);
        }
    }
}

pub(crate) struct Lt {
    handle: sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    work: u64,
    stream: Arc<CudaStream>,
    plans: RefCell<HashMap<Gemm, Plan>>,
}

impl Drop for Lt {
    fn drop(&mut self) {
        self.plans.borrow_mut().clear();
        unsafe { _ = lt::destroy_handle(self.handle) };
    }
}

fn cuda(d: Dtype) -> cudaDataType_t {
    match d {
        Dtype::F32 => cudaDataType_t::CUDA_R_32F,
        Dtype::Bf16 => cudaDataType_t::CUDA_R_16BF,
        Dtype::F16 => cudaDataType_t::CUDA_R_16F,
        Dtype::I32 => cudaDataType_t::CUDA_R_32I,
        Dtype::U16 => cudaDataType_t::CUDA_R_16U,
    }
}

unsafe fn set<T>(desc: sys::cublasLtMatmulDesc_t, attr: Attr, v: &T) -> Result<()> {
    unsafe { lt::set_matmul_desc_attribute(desc, attr, std::ptr::from_ref(v).cast(), size_of::<T>()) }?;
    Ok(())
}

impl Lt {
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let handle = lt::create_handle()?;
        let workspace = stream.alloc_zeros::<u8>(WORKSPACE)?;
        let (work, _) = workspace.device_ptr(stream);
        Ok(Self {
            handle,
            workspace,
            work,
            stream: stream.clone(),
            plans: RefCell::default(),
        })
    }

    fn plan(&self, g: Gemm) -> Result<Plan> {
        let desc = lt::create_matmul_desc(cublasComputeType_t::CUBLAS_COMPUTE_32F, cudaDataType_t::CUDA_R_32F)?;
        let layouts = [
            lt::create_matrix_layout(cuda(g.w), g.k as u64, g.n as u64, g.k as i64)?,
            lt::create_matrix_layout(cuda(g.x), g.k as u64, g.rows as u64, g.k as i64)?,
            lt::create_matrix_layout(cuda(g.y), g.n as u64, g.rows as u64, g.ld as i64)?,
        ];
        let mut plan = Plan {
            desc,
            layouts,
            algo: unsafe { std::mem::zeroed() },
        };
        unsafe {
            set(desc, Attr::CUBLASLT_MATMUL_DESC_TRANSA, &1i32)?;
            set(desc, Attr::CUBLASLT_MATMUL_DESC_TRANSB, &0i32)?;
            if g.epilogue == Epilogue::Bias {
                set(
                    desc,
                    Attr::CUBLASLT_MATMUL_DESC_EPILOGUE,
                    &cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
                )?;
            }
            let pref = lt::create_matmul_pref()?;
            lt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                std::ptr::from_ref(&WORKSPACE).cast(),
                size_of::<usize>(),
            )?;
            let found =
                lt::get_matmul_algo_heuristic(self.handle, desc, layouts[0], layouts[1], layouts[2], layouts[2], pref);
            lt::destroy_matmul_pref(pref)?;
            plan.algo = found
                .map_err(|e| eyre::eyre!("no cuBLASLt algorithm for {g:?}: {e:?}"))?
                .algo;
            Ok(plan)
        }
    }

    fn run(&self, gemm: Gemm, input: u64, weight: u64, bias: Option<u64>, out: u64) -> Result<()> {
        if !self.plans.borrow().contains_key(&gemm) {
            let plan = self.plan(gemm)?;
            self.plans.borrow_mut().insert(gemm, plan);
        }
        let plans = self.plans.borrow();
        let p = &plans[&gemm];
        let (alpha, beta) = (1.0f32, if gemm.accumulate { 1.0f32 } else { 0.0 });
        unsafe {
            if let Some(b) = bias {
                set(p.desc, Attr::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &b)?;
            }
            lt::matmul(
                self.handle,
                p.desc,
                std::ptr::from_ref(&alpha).cast(),
                std::ptr::from_ref(&beta).cast(),
                weight as *const c_void,
                p.layouts[0],
                input as *const c_void,
                p.layouts[1],
                out as *const c_void,
                p.layouts[2],
                out as *mut c_void,
                p.layouts[2],
                &raw const p.algo,
                self.work as *mut c_void,
                WORKSPACE,
                self.stream.cu_stream().cast(),
            )?;
        }
        _ = &self.workspace;
        Ok(())
    }
}

impl Gpu {
    pub fn linear(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, y: &Tensor, accumulate: bool) -> Result<()> {
        self.linear_into(x, w, bias, y, (w.shape[0], 0), accumulate)
    }

    pub fn linear_into(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        y: &Tensor,
        (ld, off): (usize, usize),
        accumulate: bool,
    ) -> Result<()> {
        let (out, inner) = (w.shape[0], w.shape[1]);
        let rows = x.numel() / inner;
        if x.dtype == Dtype::Bf16 && w.dtype == Dtype::F16 {
            return self.linear_f16(x, w, bias, y, (ld, off), accumulate);
        }
        ensure!(
            x.numel() == rows * inner && y.numel() == rows * ld && off + out <= ld && x.dtype == w.dtype,
            "linear {:?} · {:?} → {:?} [{off}..+{out} of {ld}]",
            x.shape,
            w.shape,
            y.shape
        );
        ensure!(
            bias.is_none_or(|b| b.dtype == Dtype::F32 && b.numel() == out),
            "bias must be f32 [{out}]"
        );
        let gemm = Gemm {
            rows,
            n: out,
            k: inner,
            ld,
            x: x.dtype,
            w: w.dtype,
            y: y.dtype,
            epilogue: if bias.is_some() { Epilogue::Bias } else { Epilogue::None },
            accumulate,
        };
        let at = y.ptr() + (off * y.dtype.size()) as u64;
        let bias = bias.map(|b| self.bias(b, y.dtype)).transpose()?;
        self.lt.run(gemm, x.ptr(), w.ptr(), bias, at)
    }

    fn linear_f16(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        y: &Tensor,
        (ld, off): (usize, usize),
        accumulate: bool,
    ) -> Result<()> {
        let grid = |n: usize| (n.div_ceil(256) as u32, 1, 1);
        let xh = self.empty(&x.shape, Dtype::F16)?;
        self.launch(
            "ck_ops",
            "bf16_f16",
            grid(x.numel()),
            256,
            0,
            &[x.into(), (&xh).into(), x.numel().into()],
        )?;
        if y.dtype != Dtype::Bf16 {
            return self.linear_into(&xh, w, bias, y, (ld, off), accumulate);
        }
        ensure!(!accumulate, "no bf16 accumulation on sm_{}", self.sm);
        let (rows, n) = (x.numel() / w.shape[1], w.shape[0]);
        let tmp = self.empty(&[rows, n], Dtype::F32)?;
        self.linear_into(&xh, w, bias, &tmp, (n, 0), false)?;
        let dims = [rows, n, ld, off].map(Arg::from);
        let args = [[(&tmp).into(), y.into()].as_slice(), &dims].concat();
        self.launch("ck_ops", "f32_bf16", grid(tmp.numel()), 256, 0, &args)
    }

    fn bias(&self, b: &Tensor, dtype: Dtype) -> Result<u64> {
        if dtype == Dtype::F32 {
            return Ok(b.ptr());
        }
        let key = (b.ptr(), dtype);
        if let Some(hit) = self.biases.borrow().get(&key) {
            return Ok(hit.ptr());
        }
        let half = self.upload_f32(&self.download_f32(b)?, &b.shape, dtype)?;
        let ptr = half.ptr();
        self.biases.borrow_mut().insert(key, half);
        Ok(ptr)
    }
}
