use crate::{gid, rn};
use ptx::kernel;
use ptx::slice::GpuSlice;

#[derive(Clone, Copy)]
struct Geo {
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    p: usize,
    oh: usize,
    ow: usize,
    nhwc: bool,
}

impl Geo {
    fn locate(&self, i: usize) -> [usize; 4] {
        let g = *self;
        let width = g.c * g.k * g.k;
        let (row, col) = (i / width, i % width);
        let (ch, ky, kx) = if g.nhwc {
            (col % g.c, col / (g.k * g.c), col / g.c % g.k)
        } else {
            (col / (g.k * g.k), col / g.k % g.k, col % g.k)
        };
        let (b, oy, ox) = (row / (g.oh * g.ow), row / g.ow % g.oh, row % g.ow);
        [
            b,
            ch,
            (oy * g.s + ky).wrapping_sub(g.p),
            (ox * g.s + kx).wrapping_sub(g.p),
        ]
    }

    fn read(&self, x: GpuSlice<f32>, [b, ch, iy, ix]: [usize; 4]) -> f32 {
        let g = *self;
        if iy < g.h && ix < g.w {
            x[if g.nhwc {
                ((b * g.h + iy) * g.w + ix) * g.c + ch
            } else {
                ((b * g.c + ch) * g.h + iy) * g.w + ix
            }]
        } else {
            0.0
        }
    }
}

#[kernel]
pub fn im2col(
    x: &[f32],
    cols: &mut [u16],
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    k: u32,
    s: u32,
    p: u32,
    oh: u32,
    ow: u32,
    nhwc: u32,
) {
    let [n, c, h, w, k, s, p, oh, ow] = [n, c, h, w, k, s, p, oh, ow].map(|v| v as usize);
    let g = Geo {
        c,
        h,
        w,
        k,
        s,
        p,
        oh,
        ow,
        nhwc: nhwc != 0,
    };
    let (i, total) = (gid() * 8, n * oh * ow * c * k * k);
    if i >= total {
        return;
    }
    if (if g.nhwc { c } else { k }) % 8 == 0 {
        let [b, ch, iy, ix] = g.locate(i);
        let tap = |e: usize| {
            let at = if g.nhwc {
                [b, ch + e, iy, ix]
            } else {
                [b, ch, iy, ix.wrapping_add(e)]
            };
            g.read(x, at)
        };
        crate::heads::store(&mut cols, i, core::array::from_fn(tap));
    } else {
        (i..(i + 8).min(total)).for_each(|j| cols.set(j, rn(g.read(x, g.locate(j)))));
    }
}

#[kernel]
pub fn col2im(
    cols: &[f32],
    bias: &[f32],
    out: &mut [f32],
    n: u32,
    c: u32,
    th: u32,
    tw: u32,
    k: u32,
    s: u32,
    p: u32,
    oh: u32,
    ow: u32,
) {
    let i = gid();
    let [n, c, th, tw, k, s, p, oh, ow] = [n, c, th, tw, k, s, p, oh, ow].map(|v| v as usize);
    if i >= n * c * oh * ow {
        return;
    }
    let (b, ch, y, x) = (i / (c * oh * ow), i / (oh * ow) % c, i / ow % oh, i % ow);
    let taps = |o: usize, t: usize| {
        let hi = ((o + p) / s).min(t - 1);
        let lo = (o + p + s).saturating_sub(k) / s;
        lo..=hi
    };
    let sum = taps(y, th).fold(bias[ch], |acc, ty| {
        taps(x, tw).fold(acc, |acc, tx| {
            let (ky, kx) = (y + p - ty * s, x + p - tx * s);
            acc + cols[((b * th + ty) * tw + tx) * c * k * k + (ch * k + ky) * k + kx]
        })
    });
    out.set(i, sum);
}

#[kernel]
pub fn shuffle(x: &[f32], out: &mut [f32], n: u32, h: u32, w: u32, c: u32, r: u32) {
    let i = gid();
    let [n, h, w, c, r] = [n, h, w, c, r].map(|v| v as usize);
    if i >= n * h * r * w * r * c {
        return;
    }
    let (ch, xx, yy, b) = (
        i % c,
        i / c % (w * r),
        i / (c * w * r) % (h * r),
        i / (c * w * r * h * r),
    );
    let src = ((b * h + yy / r) * w + xx / r) * c * r * r + ch * r * r + yy % r * r + xx % r;
    out.set(i, x[src]);
}
