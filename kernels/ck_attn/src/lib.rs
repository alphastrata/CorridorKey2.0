//! Device kernels: the release build sets `#![no_std]`; `--no-default-features` yields a host `rlib` for tests.
#![cfg_attr(target_arch = "nvptx64", no_std)]
#![feature(abi_ptx)]

#[cfg(feature = "device")]
use ptx::kernel;
#[cfg(feature = "device")]
use ptx::slice::GpuSlice;

#[cfg(all(feature = "device", not(feature = "sm_61")))]
mod mma;
#[cfg(all(feature = "device", feature = "sm_61"))]
mod simt;
#[cfg(all(feature = "device", not(feature = "sm_61")))]
use mma::{THREADS, attn, shared};
#[cfg(all(feature = "device", feature = "sm_61"))]
use simt::{THREADS, attn, shared};

pub const BR: usize = 128;
pub const BC: usize = 64;

#[derive(Clone, Copy)]
pub struct Shape {
    pub n: usize,
    pub heads: usize,
    pub windows: usize,
}

#[cfg(feature = "device")]
#[allow(clippy::cast_possible_wrap)]
fn span(spans: GpuSlice<u32>, tiles: usize, window: usize, t: usize) -> (i32, i32) {
    let at = 2 * (window * tiles + t);
    (spans[at] as i32, spans[at + 1] as i32)
}

macro_rules! entry {
    ($name:ident, $d:literal, $mask:literal) => {
        #[kernel]
        pub fn $name(
            q: &[u32],
            k: &[u32],
            v: &[u32],
            rid: &[u32],
            span: &[u32],
            mul: &[f32],
            out: &mut [u32],
            n: u32,
            heads: u32,

            windows: u32,
            scale: f32,
        ) {
            let s = Shape {
                n: n as usize,
                heads: heads as usize,
                windows: windows as usize,
            };
            attn::<$d, $mask>((q, k, v), (rid, span, mul), out, s, scale);
        }
    };
}

#[cfg(feature = "device")]
entry!(attn64, 64, false);
#[cfg(feature = "device")]
entry!(attn80, 80, false);
#[cfg(feature = "device")]
entry!(attn96, 96, false);
#[cfg(feature = "device")]
entry!(attn64_masked, 64, true);
#[cfg(feature = "device")]
entry!(attn80_masked, 80, true);
#[cfg(feature = "device")]
entry!(attn96_masked, 96, true);

macro_rules! export {
    ($($name:ident = $v:expr),*) => {
        $(
            #[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation)]
            pub static $name: u32 = $v;
        )*
    };
}

#[cfg(feature = "device")]
export!(
    ATTN_THREADS = THREADS as u32,
    ATTN_SHARED_64 = shared(64),
    ATTN_SHARED_80 = shared(80),
    ATTN_SHARED_96 = shared(96)
);

#[cfg(all(target_arch = "nvptx64", feature = "device", not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}

#[cfg(test)]
mod tests {
    #![cfg(not(feature = "device"))]

    use super::{BC, BR, Shape};

    #[test]
    fn tile_blocks_eight_lane_aligned() {
        assert_eq!(BR % 8, 0);
        assert_eq!(BC % 8, 0);
    }

    #[test]
    fn shape_records_attention_geometry() {
        let s = Shape {
            n: 4,
            heads: 2,
            windows: 3,
        };
        assert_eq!((s.n, s.heads, s.windows), (4, 2, 3));
    }
}
