//! Device kernels: the release build sets `#![no_std]`; `--no-default-features` yields a host `rlib` for tests.
#![cfg_attr(target_arch = "nvptx64", no_std)]
#![feature(abi_ptx)]

#[cfg(feature = "device")]
use ptx::prelude::*;

#[cfg(feature = "device")]
mod conv;
#[cfg(feature = "device")]
mod elem;
#[cfg(feature = "device")]
mod frame;
#[cfg(feature = "device")]
mod heads;
#[cfg(feature = "device")]
mod refine;

#[cfg(all(feature = "device", not(feature = "sm_61")))]
mod refine_mma;
#[cfg(feature = "device")]
mod render;
#[cfg(feature = "device")]
mod resize;
#[cfg(feature = "device")]
mod vit;

#[cfg(feature = "device")]
pub(crate) fn gid() -> usize {
    block_idx_x() as usize * 256 + thread_idx_x() as usize
}

#[must_use]
pub fn bf(u: u16) -> f32 {
    f32::from_bits(u32::from(u) << 16)
}

#[must_use]
pub fn rn(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[cfg(feature = "device")]
pub(crate) fn silu(x: f32) -> f32 {
    x * (1.0 + (-x).exp()).rcp()
}

#[cfg(feature = "device")]
pub(crate) fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x).exp()).rcp()
}

#[cfg(feature = "device")]
pub(crate) fn erf(x: f32) -> f32 {
    let a = x.abs();
    let t = (1.0 + 0.327_591_1 * a).rcp();
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    (1.0 - poly * (-a * a).exp()).copysign(x)
}

#[cfg(feature = "device")]
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * core::f32::consts::FRAC_1_SQRT_2))
}

#[cfg(feature = "device")]
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    x * sigmoid(1.595_769_2 * (x + 0.044_715 * x * x * x))
}

#[cfg(all(target_arch = "nvptx64", feature = "device", not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}

#[cfg(test)]
mod tests {
    #![cfg(not(feature = "device"))]

    use super::{bf, rn};

    #[test]
    fn bf_widens_half_precision() {
        assert_eq!(bf(0x3f80), 1.0);
        assert_eq!(bf(0xbf80), -1.0);
        assert_eq!(bf(0), 0.0);
    }

    #[test]
    fn rn_rounds_to_nearest_even_halfway() {
        assert_eq!(rn(1.0), 0x3f80);
        assert_eq!(rn(-2.0), 0xc000);
        assert_eq!(rn(0.0), 0);
        let bits = 0x4049_0fdbu32;
        assert_eq!(
            u32::from(rn(f32::from_bits(bits))),
            (bits + 0x7fff + 1) >> 16
        );
    }
}
