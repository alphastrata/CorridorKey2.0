use ptx::mem::ld_global_nc_v4_u32;
use ptx::num::b16::cvt_rn_bf16x2_f32;
use ptx::prelude::*;
use ptx::sync::shared::{shld_u16, shld_u32, shst_v4_u32};
use ptx::warp_ops::shfl_idx_sync_b32;

fn lane() -> u32 {
    thread_idx_x() % 32
}

fn lo(pair: u32) -> f32 {
    f32::from_bits(pair << 16)
}

fn hi(pair: u32) -> f32 {
    f32::from_bits(pair & 0xffff_0000)
}

fn dot(acc: f32, a: u32, b: u32) -> f32 {
    hi(a).fma_rn(hi(b), lo(a).fma_rn(lo(b), acc))
}

fn k8([a0, a1, b]: [u32; 3], c: [f32; 4]) -> [f32; 4] {
    let (g, q) = (lane() / 4, lane() % 4);
    (0..4).fold(c, |c, j| {
        let (r0, r1) = (
            shfl_idx_sync_b32(a0, 4 * g + j),
            shfl_idx_sync_b32(a1, 4 * g + j),
        );
        let (c0, c1) = (
            shfl_idx_sync_b32(b, 8 * q + j),
            shfl_idx_sync_b32(b, 8 * q + 4 + j),
        );
        [
            dot(c[0], r0, c0),
            dot(c[1], r0, c1),
            dot(c[2], r1, c0),
            dot(c[3], r1, c1),
        ]
    })
}

#[inline]
#[must_use]
pub fn mma16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    k8([a[2], a[3], b[1]], k8([a[0], a[1], b[0]], c))
}

fn ldsm<const N: usize>(addr: u32) -> [u32; N] {
    core::array::from_fn(|i| unsafe {
        shld_u32(shfl_idx_sync_b32(addr, 8 * i as u32 + lane() / 4) + 4 * (lane() % 4))
    })
}

#[inline]
#[must_use]
pub fn ldsm4(addr: u32) -> [u32; 4] {
    ldsm(addr)
}

#[inline]
#[must_use]
pub fn ldsm2(addr: u32) -> [u32; 2] {
    ldsm(addr)
}

#[inline]
#[must_use]
pub fn ldsm4t(addr: u32) -> [u32; 4] {
    core::array::from_fn(|i| {
        let at = |r: u32| unsafe {
            u32::from(shld_u16(
                shfl_idx_sync_b32(addr, 8 * i as u32 + r) + 2 * (lane() / 4),
            ))
        };
        at(2 * (lane() % 4)) | at(2 * (lane() % 4) + 1) << 16
    })
}

#[inline]
#[must_use]
pub fn bf16x2(lo: f32, hi: f32) -> u32 {
    cvt_rn_bf16x2_f32(lo, hi)
}

#[inline]
#[must_use]
pub fn pack(lo: f32, hi: f32) -> u32 {
    bf16x2(lo, hi)
}

#[inline]
#[must_use]
pub fn operand(pair: u32, _mul: f32) -> u32 {
    pair
}

#[inline]
pub fn stage16(dst: u32, src: usize, _mul: f32) {
    put16(dst, fetch16(dst, src, true), false);
}

#[inline]
pub fn commit() {}

#[inline]
pub fn wait<const N: usize>() {}

#[inline]
#[must_use]
pub fn fetch16(_dst: u32, src: usize, live: bool) -> [u32; 4] {
    if live {
        let (a, b, c, d) = unsafe { ld_global_nc_v4_u32(src as u64) };
        [a, b, c, d]
    } else {
        [0; 4]
    }
}

#[inline]
pub fn put16(dst: u32, w: [u32; 4], _convert: bool) {
    unsafe { shst_v4_u32(dst, (w[0], w[1], w[2], w[3])) };
}

#[inline]
#[must_use]
pub fn widen(half: u16) -> f32 {
    f32::from_bits(u32::from(half) << 16)
}
