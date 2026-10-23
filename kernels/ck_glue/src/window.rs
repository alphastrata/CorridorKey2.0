use crate::head::{Head, emit, lane};
use crate::{high, low, pack};
use ptx::kernel;
use ptx::mem::{ld_global_nc_u32, ld_global_nc_v4_u32, st_global_v4_u32};
use ptx::prelude::*;

#[derive(Clone, Copy)]
struct Windows {
    t: usize,
    hp: usize,
    wp: usize,
    wh: usize,
    ww: usize,
}

impl Windows {
    fn new(t: u32, hp: u32, wp: u32, wh: u32, ww: u32) -> Self {
        let [t, hp, wp, wh, ww] = [t, hp, wp, wh, ww].map(|v| v as usize);
        Self { t, hp, wp, wh, ww }
    }

    fn tokens(self) -> usize {
        self.t * self.wh * self.ww
    }

    fn pixel(self, win: usize, n: usize) -> usize {
        let (nh, nw) = (self.hp / self.wh, self.wp / self.ww);
        let (b, iy, ix) = (win / (nh * nw), win / nw % nh, win % nw);
        let (tt, y, x) = (n / (self.wh * self.ww), n / self.ww % self.wh, n % self.ww);
        ((b * self.t + tt) * self.hp + iy * self.wh + y) * self.wp + ix * self.ww + x
    }

    fn slot(self, p: usize) -> (usize, usize) {
        let (nh, nw) = (self.hp / self.wh, self.wp / self.ww);
        let (x, y) = (p % self.wp, p / self.wp % self.hp);
        let (tt, b) = (p / (self.wp * self.hp) % self.t, p / (self.wp * self.hp * self.t));
        let win = (b * nh + y / self.wh) * nw + x / self.ww;
        (win, (tt * self.wh + y % self.wh) * self.ww + x % self.ww)
    }
}

macro_rules! gather {
    ($name:ident, $d:literal) => {
        #[kernel]
        pub fn $name(
            qkv: &[u32],
            wq: &[f32],
            wk: &[f32],
            cos: &[f32],
            sin: &[f32],
            out: &mut [u32],
            rows: u32,
            heads: u32,
            t: u32,
            hp: u32,
            wp: u32,
            wh: u32,
            ww: u32,
            eps: f32,
            mode: u32,
        ) {
            let l = lane::<$d>();
            let live = l.live && l.row < rows as usize;
            let row = if live { l.row } else { 0 };
            let g = Windows::new(t, hp, wp, wh, ww);
            let (heads, n) = (heads as usize, g.tokens());
            let per = rows as usize / 3;
            let (s, r) = (row / per, row % per);
            let (win, h, tok) = (r / (heads * n), r / n % heads, r % n);
            let c = heads * $d;
            let from = g.pixel(win, tok) * 3 * c / 2 + s * c / 2 + h * $d / 2;
            let p = Head {
                weight: if s == 0 { wq } else { wk },
                cos,
                sin,
                eps,
                mode: if s == 2 { 0 } else { mode },
            };
            emit::<$d>(qkv, from, &p, tok, &l, &mut out, row * $d / 2, live);
        }
    };
}

macro_rules! scatter {
    ($name:ident, $d:literal) => {
        #[kernel]
        pub fn $name(
            attn: &[u32],
            gate: &[u32],
            out: &mut [u32],
            items: u32,
            heads: u32,
            t: u32,
            hp: u32,
            wp: u32,
            wh: u32,
            ww: u32,
        ) {
            let l = lane::<$d>();
            let i = l.row;
            if !l.live || i >= items as usize {
                return;
            }
            let g = Windows::new(t, hp, wp, wh, ww);
            let heads = heads as usize;
            let (p, h) = (i / heads, i % heads);
            let (win, tok) = g.slot(p);
            let from = ((win * heads + h) * g.tokens() + tok) * $d / 2 + 4 * l.part;
            let word = unsafe { ld_global_nc_u32(gate.addr_at(i / 2)) };
            let logit = if i % 2 == 0 { low(word) } else { high(word) };
            let gv = low(pack(1.0 / (1.0 + (-logit).exp()), 0.0));
            let q = unsafe { ld_global_nc_v4_u32(attn.addr_window::<16>(from)) };
            let f = |w: u32| pack(low(w) * gv, high(w) * gv);
            unsafe {
                st_global_v4_u32(
                    out.addr_at(i * $d / 2 + 4 * l.part),
                    f(q.0),
                    f(q.1),
                    f(q.2),
                    f(q.3),
                );
            };
        }
    };
}

gather!(gather64, 64);
gather!(gather80, 80);
gather!(gather96, 96);
scatter!(scatter64, 64);
scatter!(scatter80, 80);
scatter!(scatter96, 96);
