use core::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

use crate::gid;
use ck_arch::convert::Conv;
use ptx::kernel;
use ptx::num::f32::ptx_f2i_rm;
use ptx::prelude::*;
use ptx::slice::GpuSlice;

const A: f32 = -0.75;

fn cubic(t: f32) -> [f32; 4] {
    let near = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let far = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
}

fn lanczos4(t: f32) -> [f32; 8] {
    const S: f32 = FRAC_1_SQRT_2;
    #[allow(clippy::cast_precision_loss)]
    const CS: [[f32; 2]; 8] = [
        [1.0, 0.0],
        [-S, -S],
        [0.0, 1.0],
        [S, -S],
        [-1.0, 0.0],
        [S, S],
        [0.0, -1.0],
        [-S, S],
    ];
    let y0 = -(t + 3.0) * FRAC_PI_4;

    let (s0, c0) = (y0.sin(), y0.cos());
    let w = [0, 1, 2, 3, 4, 5, 6, 7].map(|i: usize| {
        let d = t + 3.0 - i.f32();
        if d.abs() < 1e-6 {
            1e30
        } else {
            let y = d * FRAC_PI_4;
            (CS[i][0] * s0 + CS[i][1] * c0) / (y * y)
        }
    });
    let sum = w.iter().sum::<f32>();
    w.map(|v| v / sum)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn at(i: i32, n: usize) -> usize {
    i.clamp(0, n as i32 - 1) as usize
}

#[allow(clippy::cast_precision_loss)]
fn area(
    x: GpuSlice<f32>,
    base: usize,
    (h, w): (usize, usize),
    (oy, ox): (usize, usize),
    (oh, ow): (usize, usize),
) -> f32 {
    let (y0, y1) = (oy * h / oh, ((oy + 1) * h).div_ceil(oh));
    let (x0, x1) = (ox * w / ow, ((ox + 1) * w).div_ceil(ow));
    let sum = (y0..y1).fold(0.0f32, |acc, y| {
        (x0..x1).fold(acc, |acc, xx| acc + x[base + y * w + xx])
    });
    sum / ((y1 - y0) * (x1 - x0)) as f32
}

#[allow(clippy::cast_precision_loss)]
fn bilinear(x: GpuSlice<f32>, base: usize, (h, w): (usize, usize), (sy, sx): (f32, f32)) -> f32 {
    let (fy, fx) = (sy.max(0.0), sx.max(0.0));
    let (y0, x0) = (ptx_f2i_rm(fy), ptx_f2i_rm(fx));
    let (ty, tx) = (fy - y0 as f32, fx - x0 as f32);
    let (ya, yb, xa, xb) = (at(y0, h), at(y0 + 1, h), at(x0, w), at(x0 + 1, w));
    let row = |y: usize| x[base + y * w + xa] * (1.0 - tx) + x[base + y * w + xb] * tx;
    row(ya) * (1.0 - ty) + row(yb) * ty
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn bicubic(x: GpuSlice<f32>, base: usize, (h, w): (usize, usize), (sy, sx): (f32, f32)) -> f32 {
    let (y0, x0) = (ptx_f2i_rm(sy), ptx_f2i_rm(sx));
    let (wy, wx) = (cubic(sy - y0 as f32), cubic(sx - x0 as f32));

    (0..4).fold(0.0f32, |acc, i| {
        let y = at(y0 - 1 + i as i32, h);
        let row = (0..4).fold(0.0f32, |r, j| {
            r + wx[j] * x[base + y * w + at(x0 - 1 + j as i32, w)]
        });
        acc + wy[i] * row
    })
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn lanczos(x: GpuSlice<f32>, base: usize, (h, w): (usize, usize), (sy, sx): (f32, f32)) -> f32 {
    let (y0, x0) = (ptx_f2i_rm(sy), ptx_f2i_rm(sx));
    let (wy, wx) = (lanczos4(sy - y0 as f32), lanczos4(sx - x0 as f32));

    (0..8).fold(0.0f32, |acc, i| {
        let y = at(y0 - 3 + i as i32, h);
        let row = (0..8).fold(0.0f32, |r, j| {
            r + wx[j] * x[base + y * w + at(x0 - 3 + j as i32, w)]
        });
        acc + wy[i] * row
    })
}

#[kernel]
#[allow(clippy::cast_precision_loss)]

pub fn resize(
    x: &[f32],
    mean: &[f32],
    std: &[f32],
    out: &mut [f32],
    planes: u32,
    c: u32,
    h: u32,
    w: u32,

    oh: u32,
    ow: u32,
    scale_y: f32,
    scale_x: f32,
    mode: u32,
    lo: f32,
    hi: f32,
) {
    let i = gid();
    let [planes, c, h, w, oh, ow] = [planes, c, h, w, oh, ow].map(|v| v as usize);

    if i >= planes * oh * ow {
        return;
    }
    let (ox, oy, p) = (i % ow, i / ow % oh, i / (oh * ow));
    let base = p * h * w;
    let src = (
        (oy.f32() + 0.5) * scale_y - 0.5,
        (ox.f32() + 0.5) * scale_x - 0.5,
    );

    let v = match mode {
        0 => area(x, base, (h, w), (oy, ox), (oh, ow)),
        1 => bilinear(x, base, (h, w), src),
        2 => bicubic(x, base, (h, w), src),
        _ => lanczos(x, base, (h, w), src),
    };
    out.set(i, ((v - mean[p % c]) / std[p % c]).clamp(lo, hi));
}
