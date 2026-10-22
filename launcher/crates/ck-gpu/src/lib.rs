//! One GPU: context, stream, our PTX modules for its architecture, cached cuBLASLt plans and tensors.

mod gemm;
mod tensor;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use eyre::{Context, OptionExt, Result};

pub use gemm::{Epilogue, Gemm};
pub use tensor::{Dtype, Meter, Tensor};

/// One kernel argument as the PTX entries take them: a buffer (pointer, element count), a `u32` or an `f32`.
#[derive(Clone, Copy)]
pub enum Arg {
    Buf(u64, usize),
    U(u32),
    F(f32),
}

impl From<&Tensor> for Arg {
    fn from(t: &Tensor) -> Self {
        Self::Buf(t.ptr(), t.numel())
    }
}

impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Self::U(v as u32)
    }
}

impl From<f32> for Arg {
    fn from(v: f32) -> Self {
        Self::F(v)
    }
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub sm: u32,
    modules: HashMap<String, Arc<CudaModule>>,
    functions: RefCell<HashMap<(String, String), CudaFunction>>,
    biases: RefCell<HashMap<(u64, Dtype), Tensor>>,
    lt: gemm::Lt,
    pub meter: Arc<Meter>,
}

impl Gpu {
    /// Opens device `index` and loads every crate under `bin/sm_XX` for the highest `XX` it can run.
    ///
    /// # Errors
    /// When the context, a PTX module or cuBLASLt cannot be set up.
    pub fn open(index: usize, bin: &Path) -> Result<Self> {
        let ctx = CudaContext::new(index).wrap_err_with(|| format!("CUDA device {index}"))?;
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx.compute_capability()?;
        let sm = (major * 10 + minor) as u32;
        let stream = ctx.new_stream()?;
        let modules = std::fs::read_dir(best(bin, sm)?)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "ptx"))
            .map(|p| {
                let name = p.file_stem().ok_or_eyre("ptx name")?.to_string_lossy().into_owned();
                let src = std::fs::read_to_string(&p)?;
                Ok((name, ctx.load_module(Ptx::from_src(src))?))
            })
            .collect::<Result<_>>()?;
        let lt = gemm::Lt::new(&stream)?;
        Ok(Self {
            ctx,
            stream,
            sm,
            modules,
            functions: RefCell::default(),
            biases: RefCell::default(),
            lt,
            meter: Arc::default(),
        })
    }

    fn function(&self, module: &str, name: &str, shared: u32) -> Result<CudaFunction> {
        let key = (module.to_owned(), name.to_owned());
        if let Some(f) = self.functions.borrow().get(&key) {
            return Ok(f.clone());
        }
        let f = self
            .modules
            .get(module)
            .ok_or_eyre(format!("no PTX module {module}"))?
            .load_function(name)
            .wrap_err_with(|| format!("{module}::{name}"))?;
        f.set_attribute(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared.max(48 * 1024) as i32,
        )?;
        self.functions.borrow_mut().insert(key, f.clone());
        Ok(f)
    }

    /// Launches `module::name` on this GPU's stream.
    ///
    /// # Errors
    /// When the function is missing or the launch fails.
    pub fn launch(
        &self,
        module: &str,
        name: &str,
        grid: (u32, u32, u32),
        block: u32,
        shared: u32,
        args: &[Arg],
    ) -> Result<()> {
        let f = self.function(module, name, shared)?;
        let wide: Vec<[u64; 2]> = args
            .iter()
            .map(|a| match *a {
                Arg::Buf(p, n) => [p, n as u64],
                Arg::U(_) | Arg::F(_) => [0, 0],
            })
            .collect();
        let narrow: Vec<u32> = args
            .iter()
            .map(|a| match *a {
                Arg::U(u) => u,
                Arg::F(x) => x.to_bits(),
                Arg::Buf(..) => 0,
            })
            .collect();
        let mut lb = self.stream.launch_builder(&f);
        args.iter().zip(&wide).zip(&narrow).for_each(|((a, w), n)| match a {
            Arg::Buf(..) => _ = lb.arg(&w[0]).arg(&w[1]),
            Arg::U(_) | Arg::F(_) => _ = lb.arg(n),
        });
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        unsafe { lb.launch(cfg) }.wrap_err_with(|| format!("{module}::{name}"))?;
        Ok(())
    }

    /// The half type GEMM weights take here: bf16 with bf16 tensor cores (`sm_80` on), else f16.
    #[must_use]
    pub fn half(&self) -> Dtype {
        if self.sm >= 80 { Dtype::Bf16 } else { Dtype::F16 }
    }

    /// Visible CUDA devices.
    ///
    /// # Errors
    /// When the driver cannot be initialised.
    pub fn count() -> Result<usize> {
        Ok(usize::try_from(CudaContext::device_count()?)?)
    }

    /// Waits for all work queued on this GPU's stream.
    ///
    /// # Errors
    /// On an asynchronous CUDA error.
    pub fn sync(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }
}

fn best(bin: &Path, sm: u32) -> Result<PathBuf> {
    std::fs::read_dir(bin)
        .wrap_err_with(|| bin.display().to_string())?
        .filter_map(Result::ok)
        .filter_map(|e| {
            Some((
                e.file_name().to_str()?.strip_prefix("sm_")?.parse::<u32>().ok()?,
                e.path(),
            ))
        })
        .filter(|(s, _)| *s <= sm)
        .max_by_key(|(s, _)| *s)
        .map(|(_, p)| p)
        .ok_or_eyre(format!("no PTX for sm_{sm} under {}", bin.display()))
}
