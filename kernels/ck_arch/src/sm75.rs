
use ptx::mem::ld_global_nc_v4_u32;
use ptx::mma::{FragA16x8F16, FragB8x8F16, FragC16x8F32, mma_m16n8k8_f16f16f32};
use ptx::num::f16::cvt_rn_f16x2_f32;
use ptx::sync::shared::shst_v4_u32;

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

#[inline]
#[must_use]
pub fn bf16x2(lo: f32, hi: f32) -> u32 {
    let rn = |x: f32| {
        let b = x.to_bits();
        (b + 0x7fff + ((b >> 16) & 1)) >> 16
    };
    rn(lo) | (rn(hi) << 16)
}

#[inline]
#[must_use]
pub fn pack(lo: f32, hi: f32) -> u32 {
    cvt_rn_f16x2_f32(lo, hi)
}

#[inline]
#[must_use]
pub fn operand(pair: u32, mul: f32) -> u32 {
    pack(
        f32::from_bits(pair << 16) * mul,
        f32::from_bits(pair & 0xffff_0000) * mul,
    )
}

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

#[inline]
pub fn commit() {}

#[inline]
pub fn wait<const N: usize>() {}
