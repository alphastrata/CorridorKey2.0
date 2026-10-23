
use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::{OptionExt, Result};

use crate::ops::{Act, act};
use crate::weights::Loader;

fn ops(gpu: &Gpu, name: &str, n: usize, args: &[Arg]) -> Result<()> {
    gpu.launch("ck_ops", name, (n.div_ceil(256) as u32, 1, 1), 256, 0, args)
}

fn warps(gpu: &Gpu, name: &str, rows: usize, args: &[Arg]) -> Result<()> {
    gpu.launch("ck_ops", name, (rows.div_ceil(8) as u32, 1, 1), 256, 0, args)
}

pub struct Norm {
    pub w: Tensor,
    pub b: Tensor,
    pub eps: f32,
}

impl Norm {
    pub fn load(ld: &Loader, name: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            w: ld.f32(&format!("{name}.weight"))?,
            b: ld.f32(&format!("{name}.bias"))?,
            eps,
        })
    }

    pub fn apply(&self, gpu: &Gpu, x: &Tensor, dtype: Dtype) -> Result<Tensor> {
        let c = self.w.numel();
        let rows = x.numel() / c;
        let out = gpu.empty(&[rows, c], dtype)?;
        let name = if dtype == Dtype::F32 {
            "layer_norm_f32"
        } else {
            "layer_norm"
        };
        let args = [
            x.into(),
            (&self.w).into(),
            (&self.b).into(),
            (&out).into(),
            rows.into(),
            c.into(),
            Arg::F(self.eps),
        ];
        warps(gpu, name, rows, &args)?;
        Ok(out)
    }
}

pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
}

impl Linear {
    pub fn load(ld: &Loader, names: &[&str], gamma: Option<&str>) -> Result<Self> {
        let gamma = gamma.map(|g| ld.get(g).map(|(_, g)| g)).transpose()?;
        let weights: Vec<String> = names.iter().map(|n| format!("{n}.weight")).collect();
        let weights: Vec<&str> = weights.iter().map(String::as_str).collect();
        let mut b = names
            .iter()
            .map(|n| Ok(ld.get(&format!("{n}.bias"))?.1))
            .collect::<Result<Vec<_>>>()?
            .concat();
        let w = match &gamma {
            Some(g) => {
                b.iter_mut().zip(g).for_each(|(v, s)| *v *= s);
                ld.scaled(&weights, g)?
            }
            None => ld.rows(&weights)?,
        };
        Ok(Self {
            w,
            b: ld.gpu.upload_f32(&b, &[b.len()], Dtype::F32)?,
        })
    }

    pub fn run(&self, gpu: &Gpu, x: &Tensor, y: &Tensor, accumulate: bool) -> Result<()> {
        gpu.linear(x, &self.w, Some(&self.b), y, accumulate)
    }

    fn out(&self) -> usize {
        self.w.shape[0]
    }
}

pub fn attention(
    gpu: &Gpu,
    qkv: &Tensor,
    (b, n): (usize, usize),
    (heads, d): (usize, usize),
    ones: &Tensor,
) -> Result<Tensor> {
    let dp = [64, 80, 96]
        .into_iter()
        .find(|&p| p >= d)
        .ok_or_eyre("head size above 96")?;
    let shape = [b, heads, n, dp];
    let [q, k, v, o] = [(); 4].map(|()| gpu.empty(&shape, Dtype::Bf16));
    let [q, k, v, o] = [q?, k?, v?, o?];
    let dims = [b, n, heads, d, dp].map(Arg::from);
    let split = [[qkv, &q, &k, &v].map(Arg::from).as_slice(), &dims].concat();
    ops(gpu, "heads", 3 * q.numel(), &split)?;
    let pairs = |t: &Tensor| Arg::Buf(t.ptr(), t.numel() / 2);
    let stages = if gpu.sm >= 80 { 2 } else { 1 };
    let shared = stages * (2 * 64 * (dp + 8) * 2 + 4 * 64);
    let args = [
        pairs(&q),
        pairs(&k),
        pairs(&v),
        ones.into(),
        ones.into(),
        ones.into(),
        pairs(&o),
        n.into(),
        heads.into(),
        Arg::U(0),
        Arg::F(std::f32::consts::LOG2_E / (d as f32).sqrt()),
    ];
    gpu.launch(
        "ck_attn",
        &format!("attn{dp}"),
        (n.div_ceil(128) as u32, (b * heads) as u32, 1),
        256,
        shared as u32,
        &args,
    )?;
    let out = gpu.empty(&[b * n, heads * d], Dtype::Bf16)?;
    ops(
        gpu,
        "unheads",
        out.numel(),
        &[[&o, &out].map(Arg::from).as_slice(), &dims].concat(),
    )?;
    Ok(out)
}

pub struct Attn {
    pub norm: Norm,
    pub qkv: Linear,
    pub out: Linear,
    pub heads: usize,
}

impl Attn {
    pub fn forward(
        &self,
        gpu: &Gpu,
        x: &Tensor,
        (b, n): (usize, usize),
        kv: Option<&Tensor>,
        ones: &Tensor,
    ) -> Result<()> {
        let h = self.norm.apply(gpu, x, Dtype::Bf16)?;
        let wide = self.qkv.out();
        let inner = wide / 3;
        let qkv = gpu.empty(&[b * n, wide], Dtype::Bf16)?;
        match kv {
            None => self.qkv.run(gpu, &h, &qkv, false)?,
            Some(kv) => [(&h, 0, inner), (kv, inner, 2 * inner)]
                .into_iter()
                .try_for_each(|(src, off, len)| {
                    let w = self.qkv.w.rows(off, len)?;
                    let bias = self.qkv.b.rows(off, len)?;
                    gpu.linear_into(src, &w, Some(&bias), &qkv, (wide, off), false)
                })?,
        }
        let o = attention(gpu, &qkv, (b, n), (self.heads, inner / self.heads), ones)?;
        self.out.run(gpu, &o, x, true)
    }
}

pub struct Mlp {
    pub norm: Norm,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl Mlp {
    pub fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<()> {
        let h = self.norm.apply(gpu, x, Dtype::Bf16)?;
        let f = gpu.empty(&[h.shape[0], self.fc1.out()], Dtype::F32)?;
        self.fc1.run(gpu, &h, &f, false)?;
        self.fc2.run(gpu, &act(gpu, &f, Act::Gelu, Dtype::Bf16)?, x, true)
    }
}

pub struct Block {
    pub attn: Attn,
    pub mlp: Mlp,
}

impl Block {
    pub fn timm(ld: &Loader, p: &str, heads: usize, eps: f32) -> Result<Self> {
        let ls = |i: u8| {
            let name = format!("{p}.ls{i}.gamma");
            ld.has(&name).then_some(name)
        };
        Ok(Self {
            attn: Attn {
                norm: Norm::load(ld, &format!("{p}.norm1"), eps)?,
                qkv: Linear::load(ld, &[&format!("{p}.attn.qkv")], None)?,
                out: Linear::load(ld, &[&format!("{p}.attn.proj")], ls(1).as_deref())?,
                heads,
            },
            mlp: Mlp {
                norm: Norm::load(ld, &format!("{p}.norm2"), eps)?,
                fc1: Linear::load(ld, &[&format!("{p}.mlp.fc1")], None)?,
                fc2: Linear::load(ld, &[&format!("{p}.mlp.fc2")], ls(2).as_deref())?,
            },
        })
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor, bn: (usize, usize), ones: &Tensor) -> Result<()> {
        self.attn.forward(gpu, x, bn, None, ones)?;
        self.mlp.forward(gpu, x)
    }
}

pub fn rvm_attn(ld: &Loader, norm: &str, p: &str, heads: usize) -> Result<Attn> {
    let qkv = ["query", "key", "value"].map(|s| format!("{p}.{s}"));
    Ok(Attn {
        norm: Norm::load(ld, norm, 1e-5)?,
        qkv: Linear::load(ld, &qkv.each_ref().map(String::as_str), None)?,
        out: Linear::load(ld, &[&format!("{p}.out")], None)?,
        heads,
    })
}

pub fn rvm_mlp(ld: &Loader, norm: &str, p: &str) -> Result<Mlp> {
    Ok(Mlp {
        norm: Norm::load(ld, norm, 1e-5)?,
        fc1: Linear::load(ld, &[&format!("{p}.dense_in")], None)?,
        fc2: Linear::load(ld, &[&format!("{p}.dense_out")], None)?,
    })
}

pub fn embed(gpu: &Gpu, patches: &Tensor, prefix: &Tensor, pos: &Tensor, b: usize) -> Result<Tensor> {
    let [p, c] = prefix.shape[..] else {
        eyre::bail!("prefix wants [p, c]")
    };
    let hw = patches.numel() / (b * c);
    let out = gpu.empty(&[b, p + hw, c], Dtype::F32)?;
    let dims = [b, p, hw, c].map(Arg::from);
    ops(
        gpu,
        "embed",
        out.numel(),
        &[[patches, prefix, pos, &out].map(Arg::from).as_slice(), &dims].concat(),
    )?;
    Ok(out)
}

pub fn standardise(
    gpu: &Gpu,
    x: &Tensor,
    (mean, std): (&Tensor, &Tensor),
    (b, skip, h, w): (usize, usize, usize, usize),
    half: bool,
) -> Result<Tensor> {
    let c = mean.numel();
    let out = gpu.empty(&[b, c, h, w], Dtype::F32)?;
    let dims = [b, skip, h * w, c, usize::from(half)].map(Arg::from);
    ops(
        gpu,
        "standardise",
        out.numel(),
        &[[x, mean, std, &out].map(Arg::from).as_slice(), &dims].concat(),
    )?;
    Ok(out)
}

#[derive(Clone, Copy)]
pub enum Resize {
    Area = 0,
    Bilinear = 1,
    Bicubic = 2,
    Lanczos = 3,
}

pub fn resize(
    gpu: &Gpu,
    x: &Tensor,
    (oh, ow): (usize, usize),
    (sy, sx): (f32, f32),
    mode: Resize,
    (mean, std): (&Tensor, &Tensor),
) -> Result<Tensor> {
    let [n, c, h, w] = x.shape[..] else {
        eyre::bail!("resize wants 4-d")
    };
    let out = gpu.empty(&[n, c, oh, ow], Dtype::F32)?;
    let dims = [n * c, mean.numel(), h, w, oh, ow].map(Arg::from);
    let args = [
        [x, mean, std, &out].map(Arg::from).as_slice(),
        &dims,
        &[
            Arg::F(sy),
            Arg::F(sx),
            Arg::U(mode as u32),
            Arg::F(f32::MIN),
            Arg::F(f32::MAX),
        ],
    ]
    .concat();
    ops(gpu, "resize", out.numel(), &args)?;
    Ok(out)
}

pub fn gru_kv(gpu: &Gpu, gates: &Tensor, s: &Tensor, w: &Tensor) -> Result<Tensor> {
    let c = w.numel();
    let rows = s.numel() / c;
    let out = gpu.empty(&[rows, c], Dtype::Bf16)?;
    let args = [
        [gates, s, w, &out].map(Arg::from).as_slice(),
        &[rows.into(), c.into(), Arg::F(1e-4)],
    ]
    .concat();
    warps(gpu, "gru_kv", rows, &args)?;
    Ok(out)
}

pub fn gru_mix(gpu: &Gpu, gates: &Tensor, h: &Tensor, s: &Tensor, out: &Tensor) -> Result<()> {
    let c = *s.shape.last().unwrap_or(&1);
    let rows = s.numel() / c;
    ops(
        gpu,
        "gru_mix",
        s.numel(),
        &[[gates, h, s, out].map(Arg::from).as_slice(), &[rows.into(), c.into()]].concat(),
    )
}
