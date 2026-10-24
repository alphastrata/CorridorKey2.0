//! Convolutions as GEMM. `im2col`: f32 `x` (NCHW, or NHWC with `nhwc`) → bf16 `cols [(n, oy, ox), (c, ky, kx)]`,
//! zero padding `p`. `col2im`: transposed convolution's overlap-add of f32 `cols [(n, ty, tx), (c, ky, kx)]` into
//! f32 NCHW `out` plus `bias [c]`. `shuffle`: NHWC pixel shuffle `[n, h, w, c·r²] → [n, h·r, w·r, c]`.

use crate::{gid, rn};
use ptx::kernel;

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
    let i = gid();
    let [n, c, h, w, k, s, p, oh, ow] = [n, c, h, w, k, s, p, oh, ow].map(|v| v as usize);
    let width = c * k * k;
    if i >= n * oh * ow * width {
        return;
    }
    let (row, col) = (i / width, i % width);
    let (ch, ky, kx) = (col / (k * k), col / k % k, col % k);
    let (b, oy, ox) = (row / (oh * ow), row / ow % oh, row % ow);
    let (iy, ix) = (
        (oy * s + ky) as isize - p as isize,
        (ox * s + kx) as isize - p as isize,
    );
    let inside = iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w;
    let v = if inside {
        let (iy, ix) = (iy as usize, ix as usize);
        x[if nhwc == 0 {
            ((b * c + ch) * h + iy) * w + ix
        } else {
            ((b * h + iy) * w + ix) * c + ch
        }]
    } else {
        0.0
    };
    cols.set(i, rn(v));
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
