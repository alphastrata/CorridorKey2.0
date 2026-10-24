//! Turing (`sm_75`): no bf16 MMA, so bf16 operands are widened to f16 while staging (synchronous, no
//! `cp.async`) and `m16n8k16` runs as two f16 `m16n8k8`. f16 tops out at 65504, so operands are first scaled
//! by an exact power of two `mul` that the caller undoes.

use ptx::mem::ld_global_nc_v4_u32;
use ptx::mma::{FragA16x8F16, FragB8x8F16, FragC16x8F32, mma_m16n8k8_f16f16f32};
use ptx::num::f16::cvt_rn_f16x2_f32;
use ptx::sync::shared::shst_v4_u32;

/// `m16n8k16` with f32 accumulate on operands from [`operand`] / [`pack`], as k = 0..8 then 8..16.
#[inline]
#[must_use]
pub fn mma16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    let c = mma_m16n8k8_f16f16f32(
        FragA16x8F16([a[0], a[1]]),
        FragB8x8F16([b[0]]),
        FragC16x8F32(c),
    );
    mma_m16n8k8_f16f16f32(FragA16x8F16([a[2], a[3]]), FragB8x8F16([b[1]]), c).0
}

/// Two f32 rounded to a bf16x2 (`lo` in the low half), round-to-nearest-even.
#[inline]
#[must_use]
pub fn bf16x2(lo: f32, hi: f32) -> u32 {
    let rn = |x: f32| {
        let b = x.to_bits();
        (b + 0x7fff + ((b >> 16) & 1)) >> 16
    };
    rn(lo) | (rn(hi) << 16)
}

/// Two f32 as an MMA operand pair (f16x2).
#[inline]
#[must_use]
pub fn pack(lo: f32, hi: f32) -> u32 {
    cvt_rn_f16x2_f32(lo, hi)
}

/// A bf16x2 from memory as an MMA operand pair: exact widening to f32, scaling by `mul`, then f16.
#[inline]
#[must_use]
pub fn operand(pair: u32, mul: f32) -> u32 {
    pack(
        f32::from_bits(pair << 16) * mul,
        f32::from_bits(pair & 0xffff_0000) * mul,
    )
}

/// Stage 16 bytes of bf16 from global `src` to shared `dst` as MMA operands scaled by `mul`.
#[inline]
pub fn stage16(dst: u32, src: usize, mul: f32) {
    let (a, b, c, d) = unsafe { ld_global_nc_v4_u32(src as u64) };
    let words = (
        operand(a, mul),
        operand(b, mul),
        operand(c, mul),
        operand(d, mul),
    );
    unsafe { shst_v4_u32(dst, words) };
}

/// No-op: copies complete on issue.
#[inline]
pub fn commit() {}

/// No-op: copies complete on issue.
#[inline]
pub fn wait<const N: usize>() {}
