#![no_std]
#![feature(abi_ptx)]

use ptx::prelude::*;

mod conv;
mod elem;
mod frame;
mod heads;
mod render;
mod resize;
mod vit;

pub(crate) fn gid() -> usize {
    block_idx_x() as usize * 256 + thread_idx_x() as usize
}

pub(crate) fn bf(u: u16) -> f32 {
    f32::from_bits(u32::from(u) << 16)
}

pub(crate) fn rn(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub(crate) fn silu(x: f32) -> f32 {
    x * (1.0 + (-x).exp()).rcp()
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x).exp()).rcp()
}

pub(crate) fn erf(x: f32) -> f32 {
    let a = x.abs();
    let t = (1.0 + 0.327_591_1 * a).rcp();
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    (1.0 - poly * (-a * a).exp()).copysign(x)
}

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * core::f32::consts::FRAC_1_SQRT_2))
}

pub(crate) fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}
