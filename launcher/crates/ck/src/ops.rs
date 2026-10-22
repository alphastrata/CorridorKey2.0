//! Typed launches of the PTX kernels (`ck_ops`, `ck_glue`, `ck_dw`).

use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::{Result, ensure};

fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(256) as u32, 1, 1)
}

fn ops(gpu: &Gpu, name: &str, n: usize, args: &[Arg]) -> Result<()> {
    gpu.launch("ck_ops", name, grid(n), 256, 0, args)
}

/// A `bf16` tensor as `u32` pairs, as `ck_glue` and `ck_attn` take them.
fn pairs(t: &Tensor) -> Arg {
    Arg::Buf(t.ptr(), t.numel() / 2)
}

/// Swin's shifted grid: `(hp, wp, sh, sw)` around an `h × w` image.
#[derive(Clone, Copy)]
pub struct Shift {
    pub hp: usize,
    pub wp: usize,
    pub sh: usize,
    pub sw: usize,
}

impl Shift {
    #[must_use]
    pub fn none(h: usize, w: usize) -> Self {
        Self {
            hp: h,
            wp: w,
            sh: 0,
            sw: 0,
        }
    }

    #[must_use]
    pub fn moved(self, h: usize, w: usize) -> bool {
        self.sh != 0 || self.sw != 0 || self.hp != h || self.wp != w
    }
}

/// Convolution geometry: kernel, stride, padding.
#[derive(Clone, Copy)]
pub struct Conv {
    pub k: usize,
    pub s: usize,
    pub p: usize,
}

impl Conv {
    #[must_use]
    pub fn out(self, n: usize) -> usize {
        (n + 2 * self.p - self.k) / self.s + 1
    }
}

/// `x` f32 `[n, c, h, w]` (or `[n, h, w, c]` with `nhwc`) → bf16 patches `[n · oh · ow, c · k · k]`.
///
/// # Errors
/// On a launch failure.
pub fn im2col(gpu: &Gpu, x: &Tensor, nhwc: bool, conv: Conv) -> Result<Tensor> {
    let [n, a, b, c] = x.shape[..] else {
        eyre::bail!("im2col wants 4-d, got {:?}", x.shape)
    };
    let (c, h, w) = if nhwc { (c, a, b) } else { (a, b, c) };
    let (oh, ow) = (conv.out(h), conv.out(w));
    let cols = gpu.empty(&[n * oh * ow, c * conv.k * conv.k], Dtype::Bf16)?;
    let dims = [n, c, h, w, conv.k, conv.s, conv.p, oh, ow, usize::from(nhwc)].map(Arg::from);
    ops(
        gpu,
        "im2col",
        cols.numel(),
        &[[x.into(), (&cols).into()].as_slice(), &dims].concat(),
    )?;
    Ok(cols)
}

/// Overlap-add of f32 `cols [n · th · tw, c · k · k]` → f32 `[n, c, oh, ow]` plus `bias`.
///
/// # Errors
/// On a launch failure.
pub fn col2im(
    gpu: &Gpu,
    cols: &Tensor,
    bias: &Tensor,
    (n, c, th, tw): (usize, usize, usize, usize),
    conv: Conv,
) -> Result<Tensor> {
    let (oh, ow) = (
        (th - 1) * conv.s + conv.k - 2 * conv.p,
        (tw - 1) * conv.s + conv.k - 2 * conv.p,
    );
    let out = gpu.empty(&[n, c, oh, ow], Dtype::F32)?;
    let dims = [n, c, th, tw, conv.k, conv.s, conv.p, oh, ow].map(Arg::from);
    ops(
        gpu,
        "col2im",
        out.numel(),
        &[[cols.into(), bias.into(), (&out).into()].as_slice(), &dims].concat(),
    )?;
    Ok(out)
}

/// NHWC pixel shuffle `[n, h, w, c · r²]` → `[n, h · r, w · r, c]`, f32.
///
/// # Errors
/// On a launch failure.
pub fn shuffle(gpu: &Gpu, x: &Tensor, r: usize) -> Result<Tensor> {
    let [n, h, w, cr] = x.shape[..] else {
        eyre::bail!("shuffle wants 4-d")
    };
    let c = cr / (r * r);
    let out = gpu.empty(&[n, h * r, w * r, c], Dtype::F32)?;
    ops(
        gpu,
        "shuffle",
        out.numel(),
        &[
            x.into(),
            (&out).into(),
            n.into(),
            h.into(),
            w.into(),
            c.into(),
            r.into(),
        ],
    )?;
    Ok(out)
}

/// f32 → bf16.
///
/// # Errors
/// On a launch failure.
pub fn to_half(gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
    let out = gpu.empty(&x.shape, Dtype::Bf16)?;
    ops(gpu, "to_bf16", x.numel(), &[x.into(), (&out).into(), x.numel().into()])?;
    Ok(out)
}

#[derive(Clone, Copy)]
pub enum Act {
    Silu = 0,
    Gelu = 1,
    GeluTanh = 2,
}

/// `act(x)` of f32 `x` into `dtype`.
///
/// # Errors
/// On a launch failure.
pub fn act(gpu: &Gpu, x: &Tensor, kind: Act, dtype: Dtype) -> Result<Tensor> {
    let out = gpu.empty(&x.shape, dtype)?;
    let name = match (x.dtype, dtype) {
        (Dtype::F32, Dtype::F32) => "act_f32",
        (Dtype::F32, _) => "act_bf16",
        _ => "act_bf16_bf16",
    };
    ops(
        gpu,
        name,
        x.numel(),
        &[x.into(), (&out).into(), x.numel().into(), Arg::U(kind as u32)],
    )?;
    Ok(out)
}

/// f32 `[n, c, h, w]` into channels `off..off + c` of the NHWC f32 grid `out [n, h, w, ld]`.
///
/// # Errors
/// On a launch failure.
pub fn to_nhwc(gpu: &Gpu, x: &Tensor, out: &Tensor, (ld, off): (usize, usize)) -> Result<()> {
    let [n, c, h, w] = x.shape[..] else {
        eyre::bail!("to_nhwc wants 4-d")
    };
    ops(
        gpu,
        "to_nhwc",
        x.numel(),
        &[
            x.into(),
            out.into(),
            n.into(),
            c.into(),
            (h * w).into(),
            ld.into(),
            off.into(),
        ],
    )
}

/// `x += scale[c] · y`, f32.
///
/// # Errors
/// On a launch failure.
pub fn add_scaled(gpu: &Gpu, x: &Tensor, y: &Tensor, scale: &Tensor) -> Result<()> {
    ensure!(x.numel() == y.numel(), "add_scaled sizes");
    ops(
        gpu,
        "add_scaled",
        x.numel(),
        &[x.into(), y.into(), scale.into(), x.numel().into(), scale.numel().into()],
    )
}

/// How [`add_gated`] gates: `silu(gate)` elementwise, or one gate row (`ld` apart) per frame of `per` pixels.
#[derive(Clone, Copy)]
pub enum Gate<'a> {
    Silu(&'a Tensor),
    Frame { gate: &'a Tensor, per: usize, ld: usize },
}

/// `x[r, off + j] += y[r, j] · gate`, f32; `x` rows `ld` wide.
///
/// # Errors
/// On a launch failure.
pub fn add_gated(gpu: &Gpu, x: &Tensor, y: &Tensor, gate: Gate, (ld, off): (usize, usize)) -> Result<()> {
    let c = *y.shape.last().unwrap_or(&1);
    let rows = y.numel() / c;
    let (g, per, gld) = match gate {
        Gate::Silu(g) => (g, 0, 0),
        Gate::Frame { gate, per, ld } => (gate, per, ld),
    };
    let dims = [rows, c, ld, off, per, gld].map(Arg::from);
    ops(
        gpu,
        "add_gated",
        y.numel(),
        &[[x.into(), y.into(), g.into()].as_slice(), &dims].concat(),
    )
}

/// `rms(x) · w · (1 + scale) + shift` to bf16 with `AdaLN` rows `params [frames, ld]` (`shift`, `scale` offsets),
/// `per` pixels a frame.
///
/// # Errors
/// On a launch failure.
pub fn norm_mod(
    gpu: &Gpu,
    x: &Tensor,
    w: &Tensor,
    params: &Tensor,
    per: usize,
    (shift, scale): (usize, usize),
) -> Result<Tensor> {
    let c = w.numel();
    let pixels = x.numel() / c;
    let out = gpu.empty(&x.shape, Dtype::Bf16)?;
    let ld = params.shape[1];
    let args = [
        x.into(),
        w.into(),
        params.into(),
        (&out).into(),
        pixels.into(),
        c.into(),
        Arg::F(1e-6),
        per.into(),
        ld.into(),
        shift.into(),
        scale.into(),
    ];
    gpu.launch("ck_ops", "norm_mod", (pixels.div_ceil(8) as u32, 1, 1), 256, 0, &args)?;
    Ok(out)
}

/// `rms_norm(x) · w` of f32 NHWC `x [lead, h, w, c]` into the bf16 grid `[lead, hp, wp, c]` rolled by `-shift`.
///
/// # Errors
/// On a launch failure.
pub fn norm(gpu: &Gpu, x: &Tensor, w: &Tensor, s: Shift) -> Result<Tensor> {
    let [lead, h, wd, c] = four(x)?;
    let out = gpu.empty(&[lead, s.hp, s.wp, c], Dtype::Bf16)?;
    let pixels = lead * s.hp * s.wp;
    let dims = [pixels, c].map(Arg::from);
    let geo = [h, wd, s.hp, s.wp, s.sh, s.sw].map(Arg::from);
    let args = [
        [x.into(), w.into(), pairs(&out)].as_slice(),
        &dims,
        &[Arg::F(1e-6)],
        &geo,
    ]
    .concat();
    gpu.launch("ck_glue", "norm_bf16", (pixels.div_ceil(8) as u32, 1, 1), 256, 0, &args)?;
    Ok(out)
}

/// `x += roll(a, shift)[.., :h, :w]` with bf16 `a` on the shifted grid, in place.
///
/// # Errors
/// On a launch failure.
pub fn unshift_add(gpu: &Gpu, x: &Tensor, a: &Tensor, s: Shift) -> Result<()> {
    let [lead, h, w, c] = four(x)?;
    let pixels = lead * h * w;
    let geo = [h, w, s.hp, s.wp, s.sh, s.sw].map(Arg::from);
    let args = [[x.into(), pairs(a), x.into(), pixels.into(), c.into()].as_slice(), &geo].concat();
    gpu.launch("ck_glue", "unshift_add", grid(pixels * c / 8), 256, 0, &args)
}

/// `silu(g) · u` of bf16 rows `[g | u]`.
///
/// # Errors
/// On a launch failure.
pub fn swiglu(gpu: &Gpu, gu: &Tensor) -> Result<Tensor> {
    let width = *gu.shape.last().unwrap_or(&0);
    let rows = gu.numel() / width;
    let out = gpu.empty(&[rows, width / 2], Dtype::Bf16)?;
    let args = [pairs(gu), pairs(&out), rows.into(), (width / 2).into()];
    gpu.launch("ck_glue", "swiglu_bf16", grid(rows * width / 2 / 8), 256, 0, &args)?;
    Ok(out)
}

const TILE: u32 = (8 + 6) * (16 + 6) * 32 * 4;

fn dw_grid(lead: usize, h: usize, w: usize, c: usize) -> (u32, u32, u32) {
    (w.div_ceil(16) as u32, h.div_ceil(8) as u32, (lead * c / 32) as u32)
}

/// `DepthwiseSwiGLU2D`'s weights.
pub struct Mix {
    pub norm: Tensor,
    pub taps: Tensor,
    pub bias: Tensor,
    pub gamma: Tensor,
}

/// `x + γ · v · silu(g)` over `rms_norm(x)` with 7×7 depthwise `v`, `g`, f32 NHWC.
///
/// # Errors
/// On a launch failure.
pub fn mix_add(gpu: &Gpu, x: &Tensor, m: &Mix) -> Result<Tensor> {
    let [lead, h, w, c] = four(x)?;
    let r = gpu.empty(&[lead * h * w], Dtype::F32)?;
    let pixels = r.numel();
    gpu.launch(
        "ck_dw",
        "rms_f32",
        (pixels.div_ceil(8) as u32, 1, 1),
        256,
        0,
        &[x.into(), (&r).into(), pixels.into(), c.into(), Arg::F(1e-6)],
    )?;
    let out = gpu.empty(&x.shape, Dtype::F32)?;
    let reflect = usize::from(h > 3 && w > 3);
    let args = [
        x.into(),
        (&r).into(),
        (&m.norm).into(),
        (&m.taps).into(),
        (&m.bias).into(),
        (&m.gamma).into(),
        (&out).into(),
        h.into(),
        w.into(),
        c.into(),
        reflect.into(),
        Arg::U(1),
    ];
    gpu.launch("ck_dw", "mix_f32", dw_grid(lead, h, w, c), 256, TILE, &args)?;
    Ok(out)
}

/// 7×7 depthwise conv of the bf16 NHWC qkv grid, reflect-padded.
///
/// # Errors
/// On a launch failure.
pub fn dw_qkv(gpu: &Gpu, y: &Tensor, taps: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let [lead, h, w, c] = four(y)?;
    let out = gpu.empty(&y.shape, Dtype::Bf16)?;
    let reflect = usize::from(h > 3 && w > 3);
    let args = [
        y.into(),
        taps.into(),
        bias.into(),
        (&out).into(),
        h.into(),
        w.into(),
        c.into(),
        reflect.into(),
    ];
    gpu.launch(
        "ck_dw",
        "dw_bf16",
        dw_grid(lead, h, w, c),
        256,
        TILE + 49 * 32 * 4,
        &args,
    )?;
    Ok(out)
}

fn four(x: &Tensor) -> Result<[usize; 4]> {
    let c = *x.shape.last().unwrap_or(&0);
    let s = &x.shape;
    ensure!(s.len() >= 3, "want [.., h, w, c], got {s:?}");
    let (h, w) = (s[s.len() - 3], s[s.len() - 2]);
    Ok([x.numel() / (h * w * c), h, w, c])
}

/// Where a GEMM writes: `out` rows `ld` wide from column `off`, overwritten or accumulated.
#[derive(Clone, Copy)]
pub struct Into<'a> {
    pub out: &'a Tensor,
    pub ld: usize,
    pub off: usize,
    pub accumulate: bool,
}

const COLS: usize = 1 << 26;

/// Convolution of f32 `x` (NCHW, or NHWC with `nhwc`) by `w [o, c · k · k]` into NHWC rows of `into`, as im2col
/// and GEMM over as many images at a time as keep the patches under 64 Mi elements.
///
/// # Errors
/// On a launch or GEMM failure.
pub fn conv(gpu: &Gpu, x: &Tensor, nhwc: bool, geo: Conv, (w, bias): (&Tensor, &Tensor), into: Into) -> Result<()> {
    let [n, a, b, c] = x.shape[..] else {
        eyre::bail!("conv wants 4-d, got {:?}", x.shape)
    };
    let (h, wd) = if nhwc { (a, b) } else { (b, c) };
    let per = geo.out(h) * geo.out(wd);
    let step = (COLS / (per * w.shape[1])).clamp(1, n);
    let rows = into.out.numel() / into.ld;
    let out = into.out.reshape(&[rows, into.ld])?;
    (0..n).step_by(step).try_for_each(|i| {
        let m = step.min(n - i);
        let cols = im2col(gpu, &x.rows(i, m)?, nhwc, geo)?;
        let dst = out.rows(i * per, m * per)?;
        gpu.linear_into(&cols, w, Some(bias), &dst, (into.ld, into.off), into.accumulate)
    })
}
