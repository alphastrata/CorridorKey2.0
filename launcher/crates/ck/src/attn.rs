use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ck_gpu::{Arg, Dtype, Gpu, Tensor};
use eyre::Result;

use crate::rope;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Layout {
    pub b: usize,
    pub t: usize,
    pub gh: usize,
    pub gw: usize,
    pub ph: usize,
    pub pw: usize,
    pub wh: usize,
    pub ww: usize,
}

impl Layout {
    fn windows(self) -> usize {
        self.b * (self.ph / self.wh) * (self.pw / self.ww)
    }

    fn tokens(self) -> usize {
        self.t * self.wh * self.ww
    }
}

fn regions(l: Layout, (sh, sw): (usize, usize), (h, w): (usize, usize)) -> Option<Vec<i32>> {
    let (hp, wp, k) = (l.ph, l.pw, l.wh);
    if sh == 0 && sw == 0 && hp == h && wp == w {
        return None;
    }
    let bands = |n: usize, s: usize| -> Vec<std::ops::Range<usize>> {
        if s == 0 {
            std::iter::once(0..n).collect()
        } else {
            vec![0..n - k, n - k..n - s, n - s..n]
        }
    };
    let mut grid = vec![-1i32; hp * wp];
    bands(hp, sh)
        .into_iter()
        .flat_map(|ys| bands(wp, sw).into_iter().map(move |xs| (ys.clone(), xs)))
        .enumerate()
        .for_each(|(id, (ys, xs))| {
            ys.flat_map(|y| xs.clone().map(move |x| (y, x)))
                .filter(|&(y, x)| y < h && x < w)
                .for_each(|(y, x)| grid[y * wp + x] = id as i32);
        });
    let rolled = |y: usize, x: usize| grid[(y + sh) % hp * wp + (x + sw) % wp];
    let (nh, nw) = (hp / k, wp / l.ww);
    Some(
        (0..nh * nw)
            .flat_map(|win| (0..l.tokens()).map(move |tok| (win, tok % (k * l.ww))))
            .map(|(win, tok)| rolled(win / nw * k + tok / l.ww, win % nw * l.ww + tok % l.ww))
            .collect(),
    )
}

fn spans(rid: &[i32], n: usize) -> Vec<i32> {
    rid.chunks(n)
        .flat_map(|row| {
            (0..n.div_ceil(64)).flat_map(move |t| {
                let tile = (t * 64..t * 64 + 64).map(|i| row[i.min(n - 1)]);
                [tile.clone().min().unwrap_or(0), tile.max().unwrap_or(0)]
            })
        })
        .collect()
}

type Key = (usize, usize, usize, usize, bool);
type MaskKey = (Layout, usize, usize, usize, usize);
type Pair = Rc<(Tensor, Tensor)>;

pub struct Mask {
    rid: Tensor,
    spans: Tensor,
    order: Tensor,
}

fn sorted(rid: &[i32], n: usize) -> (Vec<i32>, Vec<u32>) {
    let perms: Vec<Vec<usize>> = rid
        .chunks(n)
        .map(|row| {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by_key(|&j| (row[j] < 0, row[j]));
            idx
        })
        .collect();
    let rid = perms
        .iter()
        .zip(rid.chunks(n))
        .flat_map(|(p, row)| p.iter().map(|&j| row[j]))
        .collect();
    let forward = perms.iter().flatten().map(|&j| j as u32);
    let inverse = perms.iter().flat_map(|p| {
        let mut inv = vec![0u32; n];
        p.iter().enumerate().for_each(|(at, &j)| inv[j] = at as u32);
        inv
    });
    (rid, forward.chain(inverse).collect())
}

#[derive(Default)]
pub struct Cache {
    tables: RefCell<HashMap<Key, Pair>>,
    masks: RefCell<HashMap<MaskKey, Option<Rc<Mask>>>>,
    ones: RefCell<Option<Tensor>>,
    zeros: RefCell<Option<Tensor>>,
}

impl Cache {
    fn tables(
        &self,
        gpu: &Gpu,
        head: usize,
        (t, h, w): (usize, usize, usize),
        spatial: bool,
    ) -> Result<Rc<(Tensor, Tensor)>> {
        let key = (head, t, h, w, spatial);
        if let Some(hit) = self.tables.borrow().get(&key) {
            return Ok(hit.clone());
        }
        let (cos, sin) = rope::tables(head, (t, h, w), spatial);
        let shape = [t * h * w, head / 2];
        let pair = Rc::new((
            gpu.upload_f32(&cos, &shape, Dtype::F32)?,
            gpu.upload_f32(&sin, &shape, Dtype::F32)?,
        ));
        self.tables.borrow_mut().insert(key, pair.clone());
        Ok(pair)
    }

    fn mask(&self, gpu: &Gpu, l: Layout, shift: (usize, usize), real: (usize, usize)) -> Result<Option<Rc<Mask>>> {
        let key = (l, shift.0, shift.1, real.0, real.1);
        if let Some(hit) = self.masks.borrow().get(&key) {
            return Ok(hit.clone());
        }
        let made = regions(l, shift, real)
            .map(|rid| -> Result<_> {
                let n = l.tokens();
                let (rid, order) = sorted(&rid, n);
                let sp = spans(&rid, n);
                let up = |v: &[u8], shape: &[usize]| gpu.upload(v, shape, Dtype::I32);
                Ok(Rc::new(Mask {
                    rid: up(bytemuck::cast_slice(&rid), &[rid.len() / n, n])?,
                    spans: up(bytemuck::cast_slice(&sp), &[sp.len()])?,
                    order: up(bytemuck::cast_slice(&order), &[order.len()])?,
                }))
            })
            .transpose()?;
        self.masks.borrow_mut().insert(key, made.clone());
        Ok(made)
    }
}

fn constant(gpu: &Gpu, slot: &RefCell<Option<Tensor>>, v: &[f32]) -> Result<Tensor> {
    if slot.borrow().is_none() {
        *slot.borrow_mut() = Some(gpu.upload_f32(v, &[v.len()], Dtype::F32)?);
    }
    slot.borrow().clone().ok_or_else(|| eyre::eyre!("constant"))
}

#[derive(Clone, Copy)]
pub struct Group {
    pub head0: usize,
    pub heads: usize,
    pub layout: Layout,
    pub rope: (usize, usize, usize),
    pub spatial: bool,
    pub shift: (usize, usize),
    pub real: (usize, usize),
}

pub struct Qk<'a> {
    pub q_norm: &'a Tensor,
    pub k_norm: &'a Tensor,
    pub heads: usize,
    pub head: usize,
}

const HEAD_BYTES: usize = 512 << 20;

pub fn launch(gpu: &Gpu, d: usize) -> (u32, u32) {
    if gpu.sm < 70 {
        return ((64 * d * 8) as u32, 256);
    }
    let (stages, threads) = if gpu.sm >= 120 { (2, 128) } else { (1, 256) };
    ((stages * (2 * 64 * (d + 8) * 2 + 4 * 64)) as u32, threads)
}

pub fn attend(gpu: &Gpu, cache: &Cache, qkv: &Tensor, gate: &Tensor, out: &Tensor, qk: &Qk, g: Group) -> Result<()> {
    let per_head = g.layout.windows() * g.layout.tokens() * qk.head * 2 * 4;
    let step = (HEAD_BYTES / per_head).clamp(1, g.heads);
    (0..g.heads).step_by(step).try_for_each(|h| {
        let part = Group {
            head0: g.head0 + h,
            heads: step.min(g.heads - h),
            ..g
        };
        heads(gpu, cache, qkv, gate, out, qk, part)
    })
}

fn heads(gpu: &Gpu, cache: &Cache, qkv: &Tensor, gate: &Tensor, out: &Tensor, qk: &Qk, g: Group) -> Result<()> {
    let l = g.layout;
    let (windows, n, d) = (l.windows(), l.tokens(), qk.head);
    let shape = [windows, g.heads, n, d];
    let [q, k, v, o] = [(); 4].map(|()| gpu.empty(&shape, Dtype::Bf16));
    let [q, k, v, o] = [q?, k?, v?, o?];
    let rope = cache.tables(gpu, d, g.rope, g.spatial)?;
    let mask = cache.mask(gpu, l, g.shift, g.real)?;
    let zeros = constant(gpu, &cache.zeros, &[0.0, 0.0])?;
    let order = mask.as_ref().map_or(&zeros, |m| &m.order);
    let geo = [l.t, l.gh, l.gw, l.ph, l.pw, l.wh, l.ww, qk.heads, d, g.head0, g.heads].map(Arg::from);
    let split = [
        [qkv, qk.q_norm, qk.k_norm, &rope.0, &rope.1, &q, &k, &v, order]
            .map(Arg::from)
            .as_slice(),
        &[windows.into()],
        &geo,
        &[Arg::F(1e-6), Arg::U(3)],
    ]
    .concat();
    gpu.launch(
        "ck_ops",
        "split",
        ((q.numel() / d).div_ceil(8 * (32 / (d / 8))) as u32, 1, 1),
        256,
        0,
        &split,
    )?;

    let ones = constant(gpu, &cache.ones, &[1.0, 1.0, 1.0])?;
    let pairs = |t: &Tensor| Arg::Buf(t.ptr(), t.numel() / 2);
    let (rid, sp, rows) = mask.as_ref().map_or((Arg::from(&zeros), Arg::from(&zeros), 0), |m| {
        (Arg::from(&m.rid), Arg::from(&m.spans), m.rid.shape[0])
    });
    let (shared, threads) = launch(gpu, d);
    let name = format!("attn{d}{}", if mask.is_some() { "_masked" } else { "" });
    let scale = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let args = [
        pairs(&q),
        pairs(&k),
        pairs(&v),
        rid,
        sp,
        (&ones).into(),
        pairs(&o),
        n.into(),
        g.heads.into(),
        rows.into(),
        Arg::F(scale),
    ];
    gpu.launch(
        "ck_attn",
        &name,
        (n.div_ceil(128) as u32, (windows * g.heads) as u32, 1),
        threads,
        shared,
        &args,
    )?;

    let tokens = l.b * l.t * l.gh * l.gw;
    let ld = *out.shape.last().unwrap_or(&0);
    let merge = [
        [&o, gate, out, order].map(Arg::from).as_slice(),
        &[tokens.into(), ld.into()],
        &geo,
    ]
    .concat();
    gpu.launch(
        "ck_ops",
        "merge",
        ((tokens * g.heads * d / 8).div_ceil(256) as u32, 1, 1),
        256,
        0,
        &merge,
    )
}
