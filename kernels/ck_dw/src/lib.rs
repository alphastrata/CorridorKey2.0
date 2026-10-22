#![no_std]
#![feature(abi_ptx)]
//! 7×7 depthwise convs on NHWC activations with reflect (or replicate) padding read in place, f32 arithmetic.
//! `mix_*` is `DepthwiseSwiGLU2D`: `γ · v · silu(g)` over `rms_norm(x)` (plus `x` when `add`), in x's dtype; `dw_bf16` is
//! `QKVDepthwiseSpatial`. Weights are f32 `[F][49][C]`. Dynamic shared `TILE` bytes, plus `49 · 128` for `dw_bf16`'s staged weights. Grid `(⌈W / 16⌉, ⌈H / 8⌉, N · C / 32)`, block 256.

use ptx::kernel;
use ptx::prelude::*;
use ptx::slice::{GpuSlice, GpuSliceMut};
use ptx::sync::shared::{shared_window, shld_f32, shst_f32};
use ptx::warp_ops::shfl_bfly_sync_b32;

const K: usize = 7;
const TAPS: usize = K * K;
const CG: usize = 32;
const TH: usize = 8;
const TW: usize = 16;
const SH: usize = TH + K - 1;
const SW: usize = TW + K - 1;
const THREADS: usize = CG * TH;
/// Bytes of the f32 input tile; the `[F][49][32]` f32 weights follow it in dynamic shared memory.
pub const TILE: usize = SH * SW * CG * 4;

#[derive(Clone, Copy)]
struct Dims {
    h: usize,
    w: usize,
    c: usize,
    reflect: bool,
}

impl Dims {
    fn new(h: u32, w: u32, c: u32, reflect: u32) -> Self {
        Self {
            h: h as usize,
            w: w as usize,
            c: c as usize,
            reflect: reflect != 0,
        }
    }

    fn edge(self, i: usize, n: usize) -> usize {
        let i = i as isize - (K / 2) as isize;
        let n = n as isize;
        let r = match (self.reflect, i < 0, i >= n) {
            (true, true, _) => -i,
            (true, _, true) => 2 * n - 2 - i,
            _ => i.clamp(0, n - 1),
        };
        r as usize
    }
}

/// `s[i]` unchecked: the host sizes every operand to the launch.
fn ld<T: Copy>(s: &[T], i: usize) -> T {
    unsafe { *s.get_unchecked(i) }
}

fn bf(u: u16) -> f32 {
    f32::from_bits(u32::from(u) << 16)
}

fn rn(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Block origin: `(n, channel base, y0, x0)`.
fn origin(d: Dims) -> (usize, usize, usize, usize) {
    let groups = d.c / CG;
    let z = block_idx_z() as usize;
    (
        z / groups,
        z % groups * CG,
        block_idx_y() as usize * TH,
        block_idx_x() as usize * TW,
    )
}

/// Stage the haloed tile as f32 (`load(pixel, channel)`), then each thread convolves one channel over one
/// output row of `TW` pixels with `F` filters. Returns `(channel, row, acc)`.
#[inline(always)]
fn conv<const F: usize>(
    dims: Dims,
    weights: GpuSlice<f32>,
    bias: GpuSlice<f32>,
    load: impl Fn(usize, usize) -> f32,
) -> (usize, usize, [[f32; TW]; F]) {
    let tile = shared_window();
    let taps = tile + TILE as u32;
    let (img, c0, y0, x0) = origin(dims);
    let tid = thread_idx_x() as usize;
    if F == 1 {
        (tid..TAPS * CG).step_by(THREADS).for_each(|i| unsafe {
            shst_f32(taps + 4 * i as u32, ld(&weights, i / CG * dims.c + c0 + i % CG));
        });
    }
    let (c, line) = (tid % CG, tid / CG);
    let ch = c0 + c;
    ptx::unroll!(2, |r| {
        let py = line + r * TH;
        if py < SH {
            let row = (img * dims.h + dims.edge(y0 + py, dims.h)) * dims.w;
            ptx::unroll!(22, |i| unsafe {
                shst_f32(
                    tile + 4 * ((py * SW + i) * CG + c) as u32,
                    load(row + dims.edge(x0 + i, dims.w), ch),
                );
            });
        }
    });
    sync_threads();
    let mut acc: [[f32; TW]; F] = core::array::from_fn(|f| [ld(&bias, f * dims.c + ch); TW]);
    ptx::unroll!(7, |ky| {
        let mut row = [0.0f32; SW];
        ptx::unroll!(22, |i| row[i] = unsafe {
            shld_f32(tile + 4 * (((line + ky) * SW + i) * CG + c) as u32)
        });
        let wt: [[f32; F]; K] = core::array::from_fn(|kx| {
            core::array::from_fn(|f| match F {
                1 => unsafe { shld_f32(taps + 4 * ((ky * K + kx) * CG + c) as u32) },
                _ => ld(&weights, (f * TAPS + ky * K + kx) * dims.c + ch),
            })
        });
        ptx::unroll!(7, |kx| {
            ptx::unroll!(16, |o| {
                (0..F).for_each(|f| acc[f][o] = wt[kx][f].fma_rn(row[o + kx], acc[f][o]));
            });
        });
    });
    (ch, line, acc)
}

/// Calls `f(o, pixel)` for each in-bounds column `o` of this thread's output row `j`, unrolled so the
/// accumulators stay in registers.
#[allow(clippy::many_single_char_names)]
#[inline(always)]
fn outputs(d: Dims, j: usize, mut f: impl FnMut(usize, usize)) {
    let (n, _, y0, x0) = origin(d);
    let y = y0 + j;
    let base = (n * d.h + y) * d.w + x0;
    ptx::unroll!(16, |o| if y < d.h && x0 + o < d.w {
        f(o, base + o);
    });
}

/// `r[p] = rsqrt(mean(x[p]²) + eps)`: one warp per pixel, block 256.
#[inline(always)]
fn rms(get: impl Fn(usize) -> f32, mut r: GpuSliceMut<f32>, pixels: u32, c: u32, eps: f32) {
    let p = (block_idx_x() * 8 + thread_idx_x() / 32) as usize;
    if p >= pixels as usize {
        return;
    }
    let c = c as usize;
    let lane = thread_idx_x() as usize % 32;
    let s = (lane..c)
        .step_by(32)
        .map(|i| get(p * c + i))
        .fold(0.0f32, |s, v| v.fma_rn(v, s));
    let s = [16, 8, 4, 2, 1].into_iter().fold(s, |s, m| {
        s + f32::from_bits(shfl_bfly_sync_b32(s.to_bits(), m, 0x1f))
    });
    if lane == 0 {
        r.set(p, (s / c as f32 + eps).rsqrt());
    }
}

#[kernel]
pub fn rms_f32(x: &[f32], r: &mut [f32], pixels: u32, c: u32, eps: f32) {
    rms(|i| x[i], r, pixels, c, eps);
}

#[kernel]
pub fn rms_bf16(x: &[u16], r: &mut [f32], pixels: u32, c: u32, eps: f32) {
    rms(|i| bf(x[i]), r, pixels, c, eps);
}

#[inline(always)]
fn mix(
    dims: Dims,
    rms: GpuSlice<f32>,
    norm: GpuSlice<f32>,
    taps: GpuSlice<f32>,
    bias: GpuSlice<f32>,
    gamma: GpuSlice<f32>,
    add: bool,
    get: impl Fn(usize) -> f32,
    mut put: impl FnMut(usize, f32),
) {
    let c = dims.c;
    let (ch, row, [value, gate]) = conv::<2>(dims, taps, bias, |px, k| {
        get(px * c + k) * ld(&rms, px) * ld(&norm, k)
    });
    outputs(dims, row, |o, px| {
        let silu = gate[o] * (1.0 + (-gate[o]).exp()).rcp();
        let branch = ld(&gamma, ch) * value[o] * silu;
        let at = px * c + ch;
        put(at, if add { get(at) + branch } else { branch });
    });
}

#[kernel]
pub fn mix_f32(
    x: &[f32],
    r: &[f32],
    wn: &[f32],
    w: &[f32],
    bias: &[f32],
    gamma: &[f32],
    out: &mut [f32],
    h: u32,
    width: u32,
    c: u32,
    reflect: u32,
    add: u32,
) {
    mix(
        Dims::new(h, width, c, reflect),
        r,
        wn,
        w,
        bias,
        gamma,
        add != 0,
        |i| ld(&x, i),
        |i, v| out.set(i, v),
    );
}

#[kernel]
pub fn mix_bf16(
    x: &[u16],
    r: &[f32],
    wn: &[f32],
    w: &[f32],
    bias: &[f32],
    gamma: &[f32],
    out: &mut [u16],
    h: u32,
    width: u32,
    c: u32,
    reflect: u32,
    add: u32,
) {
    let d = Dims::new(h, width, c, reflect);
    mix(
        d,
        r,
        wn,
        w,
        bias,
        gamma,
        add != 0,
        |i| bf(ld(&x, i)),
        |i, v| out.set(i, rn(v)),
    );
}

#[kernel]
pub fn dw_bf16(
    x: &[u16],
    w: &[f32],
    bias: &[f32],
    out: &mut [u16],
    h: u32,
    width: u32,
    c: u32,
    reflect: u32,
) {
    let d = Dims::new(h, width, c, reflect);
    let (ch, j, [acc]) = conv::<1>(d, w, bias, |p, c| bf(ld(&x, p * d.c + c)));
    outputs(d, j, |o, p| out.set(p * d.c + ch, rn(acc[o])));
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}
