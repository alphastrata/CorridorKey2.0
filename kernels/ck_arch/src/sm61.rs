use ptx::num::b16::cvt_rn_bf16x2_f32;

#[inline]
#[must_use]
pub fn bf16x2(lo: f32, hi: f32) -> u32 {
    cvt_rn_bf16x2_f32(lo, hi)
}

#[inline]
#[must_use]
pub fn widen(half: u16) -> f32 {
    f32::from_bits(u32::from(half) << 16)
}
