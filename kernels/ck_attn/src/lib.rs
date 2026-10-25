#![no_std]
#![feature(abi_ptx)]

use ptx::kernel;
use ptx::slice::GpuSlice;

#[cfg(not(feature = "sm_61"))]
mod mma;
#[cfg(feature = "sm_61")]
mod simt;
#[cfg(not(feature = "sm_61"))]
use mma::attn;
#[cfg(feature = "sm_61")]
use simt::attn;

const BR: usize = 128;
const BC: usize = 64;

#[derive(Clone, Copy)]
struct Shape {
    n: usize,
    heads: usize,
    windows: usize,
}

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

entry!(attn64, 64, false);
entry!(attn80, 80, false);
entry!(attn96, 96, false);
entry!(attn64_masked, 64, true);
entry!(attn80_masked, 80, true);
entry!(attn96_masked, 96, true);

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    ptx::intrinsics::trap()
}
