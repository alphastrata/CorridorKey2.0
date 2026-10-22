//! Blackwell consumer (`sm_120`): bf16 operands stay bf16 for native `m16n8k16`; `cp.async` staging.

use core::arch::asm;
use ptx::mma::{FragA16x16Bf16, FragB16x8Bf16, FragC16x8F32, mma_m16n8k16_bf16bf16f32};
use ptx::pipe::{cp_async_cg, cp_async_commit_group, cp_async_wait_group};

/// `m16n8k16` with f32 accumulate on operands from [`operand`] / [`pack`].
#[inline]
#[must_use]
pub fn mma16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    mma_m16n8k16_bf16bf16f32(FragA16x16Bf16(a), FragB16x8Bf16(b), FragC16x8F32(c)).0
}

/// Two f32 rounded to a bf16x2 (`lo` in the low half).
#[inline]
#[must_use]
pub fn bf16x2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe {
        asm!("cvt.rn.bf16x2.f32 {r}, {hi}, {lo};", r = out(reg32) r, hi = in(reg32) hi, lo = in(reg32) lo,
             options(pure, nomem, nostack));
    }
    r
}

/// Two f32 as an MMA operand pair.
#[inline]
#[must_use]
pub fn pack(lo: f32, hi: f32) -> u32 {
    bf16x2(lo, hi)
}

/// A bf16x2 from memory as an MMA operand pair; bf16 needs no range scaling, so `mul` (always 1) is unused.
#[inline]
#[must_use]
pub fn operand(pair: u32, _mul: f32) -> u32 {
    pair
}

/// Stage 16 bytes of bf16 from global `src` to shared `dst` as MMA operands, asynchronously (`mul` as
/// [`operand`]).
#[inline]
pub fn stage16(dst: u32, src: usize, _mul: f32) {
    unsafe { cp_async_cg::<16>(dst as usize, src) };
}

/// Close the current group of [`stage16`]s.
#[inline]
pub fn commit() {
    cp_async_commit_group();
}

/// Wait until at most `N` committed groups are in flight.
#[inline]
pub fn wait<const N: usize>() {
    cp_async_wait_group::<N>();
}
