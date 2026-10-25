#![no_std]

#[cfg(feature = "sm_120")]
mod sm120;
#[cfg(feature = "sm_120")]
pub use sm120::*;

#[cfg(not(feature = "sm_120"))]
mod sm75;
#[cfg(not(feature = "sm_120"))]
pub use sm75::*;

