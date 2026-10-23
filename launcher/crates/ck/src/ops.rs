use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::{Result, ensure};

use crate::weights::TAIL;

fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(256) as u32, 1, 1)
}

fn ops(gpu: &Gpu, name: &str, n: usize, args: &[Arg]) -> Result<()> {
    gpu.launch("ck_ops", name, grid(n), 256, 0, args)
}

fn pairs(t: &Tensor) -> Arg {
    Arg::Buf(t.ptr(), t.numel() / 2)
}

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

    #[must_use]
    pub fn inverse(self, th: usize, tw: usize) -> (usize, usize) {
        (
            (th - 1) * self.s + self.k - 2 * self.p,
            (tw - 1) * self.s + self.k - 2 * self.p,
        )
    }
}

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

pub fn col2im(
    gpu: &Gpu,
    cols: &Tensor,
    bias: &Tensor,
    (n, c, th, tw): (usize, usize, usize, usize),
    conv: Conv,
) -> Result<Tensor> {
    let (oh, ow) = conv.inverse(th, tw);
    let out = gpu.empty(&[n, c, oh, ow], Dtype::F32)?;
    col2im_into(gpu, cols, bias, conv, &out)?;
    Ok(out)
}

pub fn col2im_into(gpu: &Gpu, cols: &Tensor, bias: &Tensor, conv: Conv, out: &Tensor) -> Result<()> {
    let [n, c, oh, ow] = out.shape[..] else {
        eyre::bail!("col2im wants [n, c, oh, ow]")
    };
    let (th, tw) = (conv.out(oh), conv.out(ow));
    let dims = [n, c, th, tw, conv.k, conv.s, conv.p, oh, ow].map(Arg::from);
    ops(
        gpu,
        "col2im",
        out.numel(),
        &[[cols.into(), bias.into(), out.into()].as_slice(), &dims].concat(),
    )
}

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

pub fn act(gpu: &Gpu, x: &Tensor, kind: Act, dtype: Dtype) -> Result<Tensor> {
    if (x.dtype, dtype) == (Dtype::F32, Dtype::Bf16) {
        return act_ld(gpu, x, kind, 0);
    }
    let out = gpu.empty(&x.shape, dtype)?;
    let n = x.numel();
    let name = if x.dtype == Dtype::F32 {
        "act_f32"
    } else {
        "act_bf16_bf16"
    };
    ops(gpu, name, n, &[x.into(), (&out).into(), n.into(), Arg::U(kind as u32)])?;
    Ok(out)
}

pub fn act_tail(gpu: &Gpu, x: &Tensor, kind: Act) -> Result<Tensor> {
    act_ld(gpu, x, kind, TAIL)
}

fn act_ld(gpu: &Gpu, x: &Tensor, kind: Act, tail: usize) -> Result<Tensor> {
    let (n, c) = (x.numel(), *x.shape.last().unwrap_or(&1));
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap_or(&mut 0) += tail;
    let out = gpu.empty(&shape, Dtype::Bf16)?;
    let args = [
        x.into(),
        (&out).into(),
        n.into(),
        Arg::U(kind as u32),
        c.into(),
        (c + tail).into(),
    ];
    ops(gpu, "act_bf16", n, &args)?;
    Ok(out)
}

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

pub fn place(gpu: &Gpu, x: &Tensor, out: &Tensor, (ld, off): (usize, usize), s: f32) -> Result<()> {
    let c = *x.shape.last().unwrap_or(&1);
    let rows = x.numel() / c;
    let dims = [rows, c, ld, off].map(Arg::from);
    ops(
        gpu,
        "place",
        x.numel(),
        &[[x.into(), out.into()].as_slice(), &dims, &[Arg::F(s)]].concat(),
    )
}

pub fn axpy(gpu: &Gpu, x: &Tensor, y: &Tensor, a: f32) -> Result<()> {
    ensure!(x.numel() == y.numel(), "axpy sizes");
    ops(
        gpu,
        "axpy",
        x.numel(),
        &[x.into(), y.into(), Arg::F(a), x.numel().into()],
    )
}

pub fn add_scaled(gpu: &Gpu, x: &Tensor, y: &Tensor, scale: &Tensor) -> Result<()> {
    ensure!(x.numel() == y.numel(), "add_scaled sizes");
    ops(
        gpu,
        "add_scaled",
        x.numel(),
        &[x.into(), y.into(), scale.into(), x.numel().into(), scale.numel().into()],
    )
}

#[derive(Clone, Copy)]
pub enum Gate<'a> {
    Silu(&'a Tensor),
    Frame { gate: &'a Tensor, per: usize, ld: usize },
}

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

pub fn unshift_add(gpu: &Gpu, x: &Tensor, a: &Tensor, s: Shift) -> Result<()> {
    let [lead, h, w, c] = four(x)?;
    let pixels = lead * h * w;
    let geo = [h, w, s.hp, s.wp, s.sh, s.sw].map(Arg::from);
    let args = [[x.into(), pairs(a), x.into(), pixels.into(), c.into()].as_slice(), &geo].concat();
    gpu.launch("ck_glue", "unshift_add", grid(pixels * c / 8), 256, 0, &args)
}

pub fn swiglu(gpu: &Gpu, gu: &Tensor) -> Result<Tensor> {
    let width = *gu.shape.last().unwrap_or(&0);
    let rows = gu.numel() / width;
    let ld = width / 2 + TAIL;
    let out = gpu.empty(&[rows, ld], Dtype::Bf16)?;
    let args = [pairs(gu), pairs(&out), rows.into(), (width / 2).into(), ld.into()];
    gpu.launch("ck_glue", "swiglu_bf16", grid(rows * width / 2 / 8), 256, 0, &args)?;
    Ok(out)
}

const TILE: u32 = (8 + 6) * (16 + 6) * 32 * 4;

fn dw_grid(lead: usize, h: usize, w: usize, c: usize) -> (u32, u32, u32) {
    (w.div_ceil(16) as u32, h.div_ceil(8) as u32, (lead * c / 32) as u32)
}

pub struct Mix {
    pub norm: Tensor,
    pub taps: Tensor,
    pub bias: Tensor,
    pub gamma: Tensor,
}

const MIX_BYTES: usize = 256 << 20;

pub fn mix_add(gpu: &Gpu, x: &Tensor, m: &Mix) -> Result<()> {
    let [lead, h, w, c] = four(x)?;
    let step = (MIX_BYTES / (h * w * c * 4)).clamp(1, lead);
    (0..lead).step_by(step).try_for_each(|f| {
        let x = x.rows(f, step.min(lead - f))?;
        let n = x.shape[0];
        let r = gpu.empty(&[n * h * w], Dtype::F32)?;
        let pixels = r.numel();
        gpu.launch(
            "ck_dw",
            "rms_f32",
            (pixels.div_ceil(8) as u32, 1, 1),
            256,
            0,
            &[(&x).into(), (&r).into(), pixels.into(), c.into(), Arg::F(1e-6)],
        )?;
        let out = gpu.empty(&x.shape, Dtype::F32)?;
        let reflect = usize::from(h > 3 && w > 3);
        let args = [
            (&x).into(),
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
        gpu.launch("ck_dw", "mix_f32", dw_grid(n, h, w, c), 256, TILE, &args)?;
        gpu.copy_into(&out, &x)
    })
}

pub fn dw_qkv(gpu: &Gpu, y: &Tensor, (taps, bias): (&Tensor, &Tensor), out: &Tensor) -> Result<()> {
    let [lead, h, w, c] = four(y)?;
    let reflect = usize::from(h > 3 && w > 3);
    let args = [
        y.into(),
        taps.into(),
        bias.into(),
        out.into(),
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
    )
}

fn four(x: &Tensor) -> Result<[usize; 4]> {
    let c = *x.shape.last().unwrap_or(&0);
    let s = &x.shape;
    ensure!(s.len() >= 3, "want [.., h, w, c], got {s:?}");
    let (h, w) = (s[s.len() - 3], s[s.len() - 2]);
    Ok([x.numel() / (h * w * c), h, w, c])
}

#[derive(Clone, Copy)]
pub struct Into<'a> {
    pub out: &'a Tensor,
    pub ld: usize,
    pub off: usize,
    pub accumulate: bool,
}

const COLS: usize = 1 << 26;

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

thread_local! {
    static LAST: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
}

pub fn probe(gpu: &Gpu, what: &str) {
    if std::env::var_os("CK_MEM").is_some() {
        let gib = |b: usize| b as f64 / f64::from(1u32 << 30);
        println!(
            "    mem {what:>10}: live {:6.2} GiB, peak {:6.2} GiB",
            gib(gpu.meter.live()),
            gib(gpu.meter.peak())
        );
        gpu.meter.reset();
    }
    if std::env::var_os("CK_TIME").is_some() && gpu.sync().is_ok() {
        let now = std::time::Instant::now();
        if let Some(then) = LAST.replace(Some(now)) {
            println!("    time {what:>9}: {:8.1} ms", (now - then).as_secs_f64() * 1e3);
        }
    }
}
