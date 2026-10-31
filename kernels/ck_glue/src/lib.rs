//! Device kernels: the release build sets `#![no_std]`; `--no-default-features` yields a host `rlib` for tests.
#![cfg_attr(target_arch = "nvptx64", no_std)]
#![feature(abi_ptx)]

#[cfg(feature = "device")]
use ptx::kernel;
#[cfg(feature = "device")]
use ptx::mem::{ld_global_nc_v4_u32, st_global_v4_u32};
#[cfg(feature = "device")]
use ptx::prelude::*;
#[cfg(feature = "device")]
use ptx::slice::GpuSlice;

#[cfg(feature = "device")]
mod grid;
#[cfg(feature = "device")]
mod head;

pub use ck_arch::convert::Conv;
use ck_arch::convert::Host;

pub(crate) const ONES: u32 = 0x3f80_3f80;

#[must_use]
pub fn low(p: u32) -> f32 {
    f32::from_bits(p << 16)
}

#[must_use]
pub fn high(p: u32) -> f32 {
    f32::from_bits(p & 0xffff_0000)
}

#[must_use]
pub fn rn(x: f32) -> u32 {
    let b = x.to_bits();
    (b + 0x7fff + ((b >> 16) & 1)) >> 16
}

#[must_use]
pub fn pack(a: f32, b: f32) -> u32 {
    rn(a) | (rn(b) << 16)
}

#[cfg(feature = "device")]
pub(crate) fn f32x8(x: GpuSlice<f32>, i: usize) -> [f32; 8] {
    let first = unsafe { ld_global_nc_v4_u32(x.addr_window::<32>(i)) };
    let second = unsafe { ld_global_nc_v4_u32(x.addr_at(i + 4)) };
    [
        first.0, first.1, first.2, first.3, second.0, second.1, second.2, second.3,
    ]
    .map(f32::from_bits)
}

#[cfg(feature = "device")]
#[kernel]
pub fn swiglu_bf16(gu: &[u32], out: &mut [u32], rows: u32, h: u32, ld: u32) {
    let i = (block_idx_x() * 256 + thread_idx_x()) as usize;
    let per = h as usize / 8;
    if i >= rows as usize * per {
        return;
    }

    let (r, j) = (i / per, 4 * (i % per));
    let at = |k: usize| unsafe { ld_global_nc_v4_u32(gu.addr_window::<16>(k)) };
    let (g, u) = (
        at(r * Host::usize(h) + j),
        at(r * Host::usize(h) + Host::usize(h) / 2 + j),
    );
    let silu = |g: f32| g * (1.0 + (-g).exp()).rcp();
    let f = |g: u32, u: u32| pack(silu(low(g)) * low(u), silu(high(g)) * high(u));

    let row = r * ld as usize / 2;
    if j == 0 && ld > h {
        unsafe { st_global_v4_u32(out.addr_at(row + h as usize / 2), ONES, 0, 0, 0) };
    }
    unsafe {
        st_global_v4_u32(
            out.addr_at(row + j),
            f(g.0, u.0),
            f(g.1, u.1),
            f(g.2, u.2),
            f(g.3, u.3),
        );
    };
}

#[cfg(all(target_arch = "nvptx64", feature = "device", not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}

#[cfg(test)]
mod tests {
    #![cfg(not(feature = "device"))]

    use super::{high, low, pack, rn};

    #[test]
    fn rn_rounds_to_nearest_even_halfway() {
        let a = f32::from_bits(0x4049_0fdb); // ~3.14
        assert_eq!(rn(a), (a.to_bits() + 0x7fff + 1) >> 16);
        assert_eq!(rn(0.0), 0);
    }

    #[test]
    fn pack_low_high_roundtrip() {
        let v = pack(3.5, -1.25);
        assert_eq!(low(v), 3.5);
        assert_eq!(high(v), -1.25);
    }
}
