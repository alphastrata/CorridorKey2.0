#![no_std]
#![feature(abi_ptx)]

use ptx::kernel;
use ptx::mem::{ld_global_nc_v4_u32, st_global_v4_u32};
use ptx::prelude::*;
use ptx::slice::GpuSlice;

mod grid;
mod head;
mod window;

pub(crate) fn low(p: u32) -> f32 {
    f32::from_bits(p << 16)
}

pub(crate) fn high(p: u32) -> f32 {
    f32::from_bits(p & 0xffff_0000)
}

fn rn(x: f32) -> u32 {
    let b = x.to_bits();
    (b + 0x7fff + ((b >> 16) & 1)) >> 16
}

pub(crate) fn pack(a: f32, b: f32) -> u32 {
    rn(a) | (rn(b) << 16)
}

pub(crate) fn f32x8(x: GpuSlice<f32>, i: usize) -> [f32; 8] {
    let first = unsafe { ld_global_nc_v4_u32(x.addr_window::<32>(i)) };
    let second = unsafe { ld_global_nc_v4_u32(x.addr_at(i + 4)) };
    [
        first.0, first.1, first.2, first.3, second.0, second.1, second.2, second.3,
    ]
    .map(f32::from_bits)
}

#[kernel]
pub fn swiglu_bf16(gu: &[u32], out: &mut [u32], rows: u32, h: u32) {
    let i = (block_idx_x() * 256 + thread_idx_x()) as usize;
    let per = h as usize / 8;
    if i >= rows as usize * per {
        return;
    }
    let (r, j) = (i / per, 4 * (i % per));
    let at = |k: usize| unsafe { ld_global_nc_v4_u32(gu.addr_window::<16>(k)) };
    let (g, u) = (
        at(r * h as usize + j),
        at(r * h as usize + h as usize / 2 + j),
    );
    let silu = |g: f32| g * (1.0 + (-g).exp()).rcp();
    let f = |g: u32, u: u32| pack(silu(low(g)) * low(u), silu(high(g)) * high(u));
    unsafe {
        st_global_v4_u32(
            out.addr_at(r * h as usize / 2 + j),
            f(g.0, u.0),
            f(g.1, u.1),
            f(g.2, u.2),
            f(g.3, u.3),
        );
    };
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}
