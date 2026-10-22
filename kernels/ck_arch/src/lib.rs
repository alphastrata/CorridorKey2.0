#![cfg_attr(feature = "sm_120", feature(asm_experimental_arch))]
#![no_std]
//! Per-architecture primitives shared by the kernel crates; each target's file implements the same API.

#[cfg(feature = "sm_120")]
mod sm120;
#[cfg(feature = "sm_120")]
pub use sm120::*;

#[cfg(not(feature = "sm_120"))]
mod sm75;
#[cfg(not(feature = "sm_120"))]
pub use sm75::*;
