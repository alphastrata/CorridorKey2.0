use ptx::mma::{FragA16x16Bf16, FragB16x8Bf16, FragC16x8F32, mma_m16n8k16_bf16bf16f32};
use ptx::num::b16::cvt_rn_bf16x2_f32;
use ptx::pipe::{cp_async_cg, cp_async_commit_group, cp_async_wait_group};

#[inline]
#[must_use]
pub fn mma16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    mma_m16n8k16_bf16bf16f32(FragA16x16Bf16(a), FragB16x8Bf16(b), FragC16x8F32(c)).0
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
    unsafe { cp_async_cg::<16>(dst as usize, src) };
}

#[inline]
pub fn commit() {
    cp_async_commit_group();
}

#[inline]
pub fn wait<const N: usize>() {
    cp_async_wait_group::<N>();
}
