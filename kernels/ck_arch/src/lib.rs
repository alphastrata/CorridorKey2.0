#![no_std]

#[cfg(feature = "sm_120")]
mod sm120;
#[cfg(feature = "sm_120")]
pub use sm120::*;

#[cfg(feature = "sm_61")]
mod sm61;
#[cfg(feature = "sm_61")]
pub use sm61::*;

#[cfg(not(any(feature = "sm_61", feature = "sm_120")))]
mod sm75;
#[cfg(not(any(feature = "sm_61", feature = "sm_120")))]
pub use sm75::*;

#[cfg(not(feature = "sm_61"))]
#[inline]
#[must_use]
pub fn ldsm4(addr: u32) -> [u32; 4] {
    unsafe { ptx::mma::ldmatrix_x4_b16_u32(addr).0 }
}

#[cfg(not(feature = "sm_61"))]
#[inline]
#[must_use]
pub fn ldsm2(addr: u32) -> [u32; 2] {
    unsafe { ptx::mma::ldmatrix_x2_b16_u32(addr) }
}

#[cfg(not(feature = "sm_61"))]
#[inline]
#[must_use]
pub fn ldsm4t(addr: u32) -> [u32; 4] {
    unsafe { ptx::mma::ldmatrix_x4_trans_b16_u32(addr) }
}
