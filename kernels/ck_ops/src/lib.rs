#![no_std]
#![feature(abi_ptx)]

use ptx::prelude::*;

mod heads;

pub(crate) fn gid() -> usize {
    block_idx_x() as usize * 256 + thread_idx_x() as usize
}

pub(crate) fn rn(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x).exp()).rcp()
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}
