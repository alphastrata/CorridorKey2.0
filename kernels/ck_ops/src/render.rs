use crate::frame::{Adjust, DENOM, KNEE, to_linear, to_srgb};
use crate::gid;
use ptx::kernel;
use ptx::num::f16::{cvt_f32_f16, cvt_rn_f16_f32};
use ptx::prelude::*;

fn despill([r, g, b]: [f32; 3], s: f32) -> [f32; 3] {
    let spill = (g - f32::midpoint(r, b)).max(0.0);
    let full = [r + spill * 0.5, g - spill, b + spill * 0.5];
    if s <= 0.0 {
        [r, g, b]
    } else if s >= 1.0 {
        full
    } else {
        [
            r * (1.0 - s) + full[0] * s,
            g * (1.0 - s) + full[1] * s,
            b * (1.0 - s) + full[2] * s,
        ]
    }
}

fn linear(at: impl Fn(usize) -> f32, adj: Adjust) -> [f32; 3] {
    [1, 2, 3].map(|c| {
        let e = (at(c) * DENOM).exp();
        adj.invert((e - e.rcp()) * 0.5 * KNEE)
    })
}

fn checker(x: usize, y: usize) -> f32 {
    to_linear(if (y / 64 + x / 64) % 2 == 0 {
        0.75
    } else {
        0.5
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn unorm(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0) as u16
}

#[kernel]

pub fn render(
    logits: &[f32],
    out: &mut [u16],
    h0: u32,
    w0: u32,
    h: u32,
    w: u32,
    top: u32,
    left: u32,

    kind: u32,
    strength: f32,
    linear_cutout: u32,
    adjust: u32,
    gain: f32,
    contrast: f32,
    pivot: f32,
) {
    let i = gid();
    let [h0, w0, h, w, top, left] = [h0, w0, h, w, top, left].map(|v| v as usize);
    if i >= h0 * w0 {

        return;
    }
    let (x, y) = (i % w0, i / w0);
    let at = |c: usize| logits[(c * h + y + top) * w + x + left];
    let a = at(0).clamp(0.0, 1.0);
    let adj = Adjust {
        on: adjust != 0,
        gain,
        contrast,
        pivot,
    };

    let lin = linear(at, adj).map(|v| v.max(0.0));
    let srgb = lin.map(|v| to_srgb(v).clamp(0.0, 1.0));
    let (rgb, alpha) = match kind {
        0 => ([a; 3], None),
        1 => (srgb, None),
        2 => (
            despill(if linear_cutout == 0 { srgb } else { lin }, strength),
            Some(a),
        ),

        _ => {
            let bg = checker(x, y);
            (
                despill(srgb, strength)
                    .map(|v| to_srgb((to_linear(v) * a + bg * (1.0 - a)).max(0.0))),
                None,
            )
        }

    };
    let ch = 3 + usize::from(alpha.is_some());
    [rgb[2], rgb[1], rgb[0]]
        .into_iter()
        .chain(alpha)
        .enumerate()
        .for_each(|(c, v)| out.set(i * ch + c, unorm(v)));
}

#[kernel]

pub fn render_exr(
    logits: &[f32],
    out: &mut [f32],
    h0: u32,
    w0: u32,
    h: u32,
    w: u32,
    top: u32,
    left: u32,

    kind: u32,
    strength: f32,
    half: u32,
    adjust: u32,
    gain: f32,
    contrast: f32,
    pivot: f32,
) {
    let i = gid();
    let [h0, w0, h, w, top, left] = [h0, w0, h, w, top, left].map(|v| v as usize);
    if i >= h0 * w0 {

        return;
    }
    let (x, y) = (i % w0, i / w0);
    let at = |c: usize| logits[(c * h + y + top) * w + x + left];
    let a = at(0).clamp(0.0, 1.0);
    let adj = Adjust {
        on: adjust != 0,
        gain,
        contrast,
        pivot,
    };

    let lin = linear(at, adj);
    let rgb = match kind {
        0 => [a; 3],
        1 => lin,
        2 => despill(lin, strength),
        _ => {
            let bg = checker(x, y);
            despill(lin, strength).map(|v| v * a + bg * (1.0 - a))
        }

    };
    let mut store = |p: usize, v: f32| {
        let v = if half == 0 {
            v
        } else {
            cvt_f32_f16(cvt_rn_f16_f32(v.clamp(-65504.0, 65504.0)))
        };

        out.set(p * h0 * w0 + i, v);
    };
    if kind == 0 {
        store(0, a);
        return;
    }
    [rgb[1], rgb[2], rgb[0]]
        .into_iter()
        .enumerate()
        .for_each(|(p, v)| store(p, v));
    if kind == 2 {
        store(3, a);
    }
}
