use ck_arch::{bf16x2, commit, mma16, operand, pack, stage16, wait};
use ptx::mma::{ldmatrix_x2_b16_u32, ldmatrix_x4_b16_u32, ldmatrix_x4_trans_b16_u32};
use ptx::num::f32::ptx_exp2_approx;
use ptx::prelude::*;
use ptx::slice::{GpuSlice, GpuSliceMut};
use ptx::sync::shared::{shared_window, shld_u32, shst_u32};
use ptx::warp_ops::shfl_bfly_sync_b32;
use tune::{RB, STAGES};

use crate::{BC, BR, Shape, span};

#[cfg_attr(feature = "sm_86", path = "tune/sm86.rs")]
#[cfg_attr(feature = "sm_89", path = "tune/sm89.rs")]
#[cfg_attr(all(feature = "sm_80", not(any(feature = "sm_86", feature = "sm_89"))), path = "tune/sm120.rs")]
#[cfg_attr(not(feature = "sm_80"), path = "tune/sm75.rs")]
mod tune;

pub const THREADS: usize = BR / (16 * RB) * 32;
const KS: usize = 6;
const DN: usize = 12;

const fn row(d: usize) -> u32 {
    ((d + 8) * 2) as u32
}

const fn tile(d: usize) -> u32 {
    BC as u32 * row(d)
}

const fn slot_bytes(d: usize) -> u32 {
    2 * tile(d) + 4 * BC as u32
}

pub const fn shared(d: usize) -> u32 {
    STAGES as u32 * slot_bytes(d)
}

#[derive(Clone, Copy)]
struct Acc {
    m: [[f32; 2]; RB],
    l: [[f32; 2]; RB],
    o: [[[f32; 4]; DN]; RB],
}

fn quad_max(v: f32) -> f32 {
    [1, 2].into_iter().fold(v, |v, m| {
        v.max(f32::from_bits(shfl_bfly_sync_b32(v.to_bits(), m, 0x1f)))
    })
}

fn quad_sum(v: f32) -> f32 {
    [1, 2].into_iter().fold(v, |v, m| {
        v + f32::from_bits(shfl_bfly_sync_b32(v.to_bits(), m, 0x1f))
    })
}

#[inline(always)]
fn load<const D: usize, const MASK: bool>(
    (keys, values): (usize, usize),
    rid: GpuSlice<u32>,
    shape: Shape,
    (window, kb): (usize, usize),
    slot: u32,
    (kmul, vmul): (f32, f32),
) {
    let s = shape;
    let tid = thread_idx_x() as usize;
    let dn = D / 8;
    ptx::unroll!(12, |it| if it < 2 * BC * dn / THREADS {
        let i = tid + it * THREADS;
        let (which, rem) = (i / (BC * dn), i % (BC * dn));
        let (key, ch) = (rem / dn, rem % dn);
        let token = (kb * BC + key).min(s.n - 1);
        let src = if which == 0 { keys } else { values } + 2 * (token * D + ch * 8);
        stage16(
            slot + which as u32 * tile(D) + key as u32 * row(D) + ch as u32 * 16,
            src,
            if which == 0 { kmul } else { vmul },
        );
    });
    if MASK && tid < BC {
        let token = (kb * BC + tid).min(s.n - 1);
        unsafe {
            shst_u32(
                slot + 2 * tile(D) + 4 * tid as u32,
                rid[window * s.n + token],
            );
        };
    }
}

#[inline(always)]
fn softmax(m: &mut [f32; 2], l: &mut [f32; 2], s: &mut [[f32; 4]; 8], scale: f32) -> [f32; 2] {
    let mut corr = [0.0f32; 2];
    ptx::unroll!(2, |r| {
        let mut mx = f32::NEG_INFINITY;
        ptx::unroll!(8, |j| mx = mx.max(s[j][2 * r]).max(s[j][2 * r + 1]));
        let m_new = m[r].max(quad_max(mx));
        let shift = if m_new == f32::NEG_INFINITY {
            0.0
        } else {
            -m_new * scale
        };
        corr[r] = ptx_exp2_approx(m[r].fma_rn(scale, shift));
        let mut sum = 0.0f32;
        ptx::unroll!(8, |j| {
            s[j][2 * r] = ptx_exp2_approx(s[j][2 * r].fma_rn(scale, shift));
            s[j][2 * r + 1] = ptx_exp2_approx(s[j][2 * r + 1].fma_rn(scale, shift));
            sum += s[j][2 * r] + s[j][2 * r + 1];
        });
        l[r] = l[r].fma_rn(corr[r], sum);
        m[r] = m_new;
    });
    corr
}

#[inline(always)]
fn step<const D: usize, const MASK: bool>(
    mut acc: Acc,
    qa: &[[[u32; 4]; KS]; RB],
    buf: u32,
    (valid, qrid, masked): (usize, [[u32; 2]; RB], bool),
    scale: f32,
) -> Acc {
    let lane = thread_idx_x() % 32;
    let (kbuf, vbuf, rbuf) = (buf, buf + tile(D), buf + 2 * tile(D));
    let mut s = [[[0.0f32; 4]; 8]; RB];
    ptx::unroll!(8, |j| {
        let at = kbuf + (j as u32 * 8 + lane % 8) * row(D);
        let mut kf = [[0u32; 2]; KS];
        ptx::unroll!(3, |p| if p < D / 32 {
            let r = unsafe { ldmatrix_x4_b16_u32(at + (4 * p as u32 + lane / 8) * 16).0 };
            kf[2 * p] = [r[0], r[1]];
            kf[2 * p + 1] = [r[2], r[3]];
        });
        if D % 32 != 0 {
            kf[D / 16 - 1] =
                unsafe { ldmatrix_x2_b16_u32(at + ((D / 8 - 2) as u32 + (lane / 8) % 2) * 16) };
        }
        ptx::unroll!(2, |rb| if rb < RB {
            ptx::unroll!(6, |ks| if ks < D / 16 {
                s[rb][j] = mma16(qa[rb][ks], kf[ks], s[rb][j]);
            });
        });
    });
    if (MASK && masked) || valid < BC {
        let quad = (lane % 4) as usize;
        ptx::unroll!(8, |j| ptx::unroll!(2, |e| {
            let key = j * 8 + quad * 2 + e;
            let kr = if MASK && masked {
                unsafe { shld_u32(rbuf + 4 * key as u32) }
            } else {
                0
            };
            ptx::unroll!(2, |rb| if rb < RB {
                ptx::unroll!(
                    2,
                    |r| if key >= valid || (MASK && masked && kr != qrid[rb][r]) {
                        s[rb][j][2 * r + e] = f32::NEG_INFINITY;
                    }
                );
            });
        }));
    }
    ptx::unroll!(2, |rb| if rb < RB {
        let corr = softmax(&mut acc.m[rb], &mut acc.l[rb], &mut s[rb], scale);
        ptx::unroll!(12, |dn| if dn < D / 8 {
            ptx::unroll!(4, |e| acc.o[rb][dn][e] *= corr[e / 2]);
        });
    });
    let mi = lane / 8;
    ptx::unroll!(4, |kk| {
        let pa: [[u32; 4]; RB] = core::array::from_fn(|rb| {
            let (x, y) = (s[rb][2 * kk], s[rb][2 * kk + 1]);
            [
                pack(x[0], x[1]),
                pack(x[2], x[3]),
                pack(y[0], y[1]),
                pack(y[2], y[3]),
            ]
        });
        let key = kk as u32 * 16 + (mi % 2) * 8 + lane % 8;
        ptx::unroll!(6, |p| if p < D / 16 {
            let v = unsafe {
                ldmatrix_x4_trans_b16_u32(vbuf + key * row(D) + (2 * p as u32 + mi / 2) * 16)
            };
            ptx::unroll!(2, |rb| if rb < RB {
                acc.o[rb][2 * p] = mma16(pa[rb], [v[0], v[1]], acc.o[rb][2 * p]);
                acc.o[rb][2 * p + 1] = mma16(pa[rb], [v[2], v[3]], acc.o[rb][2 * p + 1]);
            });
        });
    });
    acc
}

#[inline(always)]
pub fn attn<const D: usize, const MASK: bool>(
    (query, key, value): (GpuSlice<u32>, GpuSlice<u32>, GpuSlice<u32>),
    (rid, spans, mul): (GpuSlice<u32>, GpuSlice<u32>, GpuSlice<f32>),
    mut out: GpuSliceMut<u32>,
    shape: Shape,
    scale: f32,
) {
    let s = shape;
    let (qmul, kmul, vmul) = (mul[0], mul[1], mul[2]);
    let scale = scale / (qmul * kmul);
    let smem = shared_window();
    let tid = thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (group, quad) = (lane / 4, lane % 4);
    let (qx, bh) = (block_idx_x() as usize, block_idx_y() as usize);
    let w = if MASK { bh / s.heads % s.windows } else { 0 };
    let base = bh * s.n * D;
    let kv = (
        key.addr_at(base / 2) as usize,
        value.addr_at(base / 2) as usize,
    );
    let tiles = s.n.div_ceil(BC);
    let own = if MASK {
        let first = span(spans, tiles, w, 2 * qx);
        let second = span(spans, tiles, w, (2 * qx + 1).min(tiles - 1));
        (first.0.min(second.0), first.1.max(second.1))
    } else {
        (0, 0)
    };
    let seen = |t: usize| {
        let (lo, hi) = span(spans, tiles, w, t);
        own.1 >= 0 && hi >= own.0 && lo <= own.1
    };
    let next = |t: usize| (t..tiles).find(|&t| !MASK || seen(t)).unwrap_or(tiles);
    let slot = |j: usize| smem + u32::from(STAGES == 2 && j % 2 == 1) * slot_bytes(D);
    let ld = |t: usize, j: usize| {
        if t < tiles {
            load::<D, MASK>(kv, rid, s, (w, t), slot(j), (kmul, vmul));
        }
        commit();
    };
    let first = next(0);
    if STAGES == 2 {
        ld(first, 0);
    }
    let rows: [[usize; 2]; RB] = core::array::from_fn(|rb| {
        let top = qx * BR + (warp * RB + rb) * 16 + group;
        [top, top + 8]
    });
    let qp = |rb: usize, r: usize, d: usize| {
        operand(query[(base + rows[rb][r].min(s.n - 1) * D + d) / 2], qmul)
    };
    let mut qa = [[[0u32; 4]; KS]; RB];
    ptx::unroll!(2, |rb| if rb < RB {
        ptx::unroll!(6, |ks| if ks < D / 16 {
            let d = 16 * ks + 2 * quad;
            qa[rb][ks] = [
                qp(rb, 0, d),
                qp(rb, 1, d),
                qp(rb, 0, d + 8),
                qp(rb, 1, d + 8),
            ];
        });
    });
    let qrid: [[u32; 2]; RB] = core::array::from_fn(|rb| {
        core::array::from_fn(|r| {
            if MASK {
                rid[w * s.n + rows[rb][r].min(s.n - 1)]
            } else {
                0
            }
        })
    });
    let init = Acc {
        m: [[f32::NEG_INFINITY; 2]; RB],
        l: [[0.0; 2]; RB],
        o: [[[0.0; 4]; DN]; RB],
    };
    let visits =
        core::iter::successors(Some(first), |&t| Some(next(t + 1))).take_while(|&t| t < tiles);
    let acc = visits.enumerate().fold(init, |acc, (j, t)| {
        if STAGES == 1 {
            ld(t, j);
        } else {
            ld(next(t + 1), j + 1);
        }
        wait::<{ STAGES - 1 }>();
        sync_threads();
        let valid = (s.n - t * BC).min(BC);
        let masked = MASK && {
            let (lo, hi) = span(spans, tiles, w, t);
            !(own.0 == own.1 && lo == hi && lo == own.0)
        };
        let acc = step::<D, MASK>(acc, &qa, slot(j), (valid, qrid, masked), scale);
        sync_threads();
        acc
    });
    write::<D, MASK>(&acc, &rows, &qrid, (base, quad, vmul), &mut out, s.n);
}

#[inline(always)]
fn write<const D: usize, const MASK: bool>(
    acc: &Acc,
    rows: &[[usize; 2]; RB],
    qrid: &[[u32; 2]; RB],
    (base, quad, vmul): (usize, usize, f32),
    out: &mut GpuSliceMut<u32>,
    n: usize,
) {
    ptx::unroll!(2, |rb| if rb < RB {
        ptx::unroll!(2, |r| {
            let outside = MASK && (qrid[rb][r] as i32) < 0;
            let inv = if outside {
                0.0
            } else {
                1.0 / (quad_sum(acc.l[rb][r]) * vmul)
            };
            if rows[rb][r] < n {
                let at = base / 2 + rows[rb][r] * (D / 2) + quad;
                ptx::unroll!(12, |dn| if dn < D / 8 {
                    out.set(
                        at + dn * 4,
                        bf16x2(acc.o[rb][dn][2 * r] * inv, acc.o[rb][dn][2 * r + 1] * inv),
                    );
                });
            }
        });
    });
}
