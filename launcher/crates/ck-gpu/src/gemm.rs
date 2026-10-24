use eyre::{Result, ensure};

use crate::{Arg, Dtype, Gpu, Tensor};

#[derive(Clone, Copy)]
struct Tile {
    name: &'static str,
    bm: usize,
    bn: usize,
    stages: [u32; 2],
    resident: [usize; 2],
    width: u32,
}

const WIDE: Tile = Tile {
    name: "wide",
    bm: 256,
    bn: 128,
    stages: [2, 4],
    resident: [1, 1],
    width: 64,
};
const SQUARE: Tile = Tile {
    name: "square",
    bm: 128,
    bn: 128,
    stages: [2, 3],
    resident: [2, 2],
    width: 32,
};
const THIN: Tile = Tile {
    name: "thin",
    bm: 128,
    bn: 16,
    stages: [2, 4],
    resident: [2, 3],
    width: 16,
};
const SMALL: Tile = Tile {
    name: "small",
    bm: 64,
    bn: 64,
    stages: [2, 4],
    resident: [2, 3],
    width: 32,
};
const SIMT: Tile = Tile {
    name: "simt",
    bm: 128,
    bn: 128,
    stages: [2, 2],
    resident: [2, 2],
    width: 0,
};
const ANY: usize = 64;
const BK: usize = 32;
const MIN_SPAN: usize = 256;
const TAIL: usize = 4;

fn words(t: &Tensor) -> Arg {
    Arg::Buf(t.ptr(), t.numel() * t.dtype.size() / 4)
}

impl Gpu {
    pub fn linear(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, y: &Tensor, accumulate: bool) -> Result<()> {
        self.linear_into(x, w, bias, y, (w.shape[0], 0), accumulate)
    }

    pub fn linear_into(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        y: &Tensor,
        place: (usize, usize),
        accumulate: bool,
    ) -> Result<()> {
        self.gemm(x, w, bias, y, place, (accumulate, false))
    }

    fn gemm(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        y: &Tensor,
        (ld, off): (usize, usize),
        (accumulate, tail): (bool, bool),
    ) -> Result<()> {
        let (n, k) = (w.shape[0], w.shape[1]);
        let rows = x.numel() / k;
        ensure!(
            x.numel() == rows * k && y.numel() == rows * ld && off + n <= ld,
            "linear {:?} · {:?} → {:?} [{off}..+{n} of {ld}]",
            x.shape,
            w.shape,
            y.shape
        );
        ensure!(
            bias.is_none_or(|b| b.dtype == Dtype::F32 && b.numel() == n),
            "bias must be f32 [{n}]"
        );
        let suffix = format!(
            "{}{}{}",
            if y.dtype == Dtype::F32 { "f32" } else { "bf16" },
            if accumulate { "_acc" } else { "" },
            if bias.is_some() { "_bias" } else { "" }
        );
        let args = [
            words(x),
            words(w),
            bias.map_or(Arg::Buf(w.ptr(), 0), |b| Arg::Buf(b.ptr(), n)),
            words(y),
            rows.into(),
            n.into(),
            k.into(),
            ld.into(),
            off.into(),
        ];
        let lanes = 16 / y.dtype.size();
        if x.dtype == Dtype::Bf16 && w.dtype == self.half() && k % 8 == 0 && ld % lanes == 0 && off % lanes == 0 {
            let (t, tiles, splits, span) = self.plan(rows, n, k, tail);
            if let Some(head) = self.head(t, tiles, splits, n, k) {
                let (x, y) = (x.reshape(&[rows, k])?, y.reshape(&[rows, ld])?);
                let part = |start, len, tail| {
                    self.gemm(
                        &x.rows(start, len)?,
                        w,
                        bias,
                        &y.rows(start, len)?,
                        (ld, off),
                        (accumulate, tail),
                    )
                };
                part(0, head, false)?;
                return part(head, rows - head, true);
            }
            let shared =
                (t.stages[usize::from(self.sm >= 80)] * (t.bm + t.bn) as u32 * 64).max(8 * 16 * (4 * t.width + 16));
            let parts = (splits > 1)
                .then(|| self.empty(&[splits, rows, n], Dtype::F32))
                .transpose()?;
            let mut each = args;
            if let Some(p) = &parts {
                (each[3], each[7], each[8]) = (words(p), n.into(), 0.into());
            }
            let name = format!("gemm_{}_{}", t.name, if parts.is_some() { "f32" } else { &suffix });
            let grid = (tiles as u32, splits as u32, 1);
            self.launch(
                "ck_gemm",
                &name,
                grid,
                256,
                shared,
                &[&each[..], &[span.into()]].concat(),
            )?;
            let Some(parts) = parts else { return Ok(()) };
            let mut sum = args;
            sum[1] = words(&parts);
            let grid = ((rows * n / 4).div_ceil(256) as u32, 1, 1);
            return self.launch(
                "ck_gemm",
                &format!("reduce_{suffix}"),
                grid,
                256,
                0,
                &[&sum[..], &[splits.into()]].concat(),
            );
        }
        ensure!(
            y.dtype == Dtype::F32 && (x.dtype == Dtype::F32) == (w.dtype == Dtype::F32),
            "no GEMM for {:?} · {:?} → {:?}",
            x.dtype,
            w.dtype,
            y.dtype
        );
        let name = format!("any_{}{suffix}", if x.dtype == Dtype::F32 { "f32in_" } else { "" });
        let grid = (n.div_ceil(ANY) as u32, rows.div_ceil(ANY) as u32, 1);
        self.launch(
            "ck_gemm",
            &name,
            grid,
            256,
            2 * 16 * (ANY as u32 + 4) * 4,
            &[&args[..], &[k.into()]].concat(),
        )
    }

    fn plan(&self, rows: usize, n: usize, k: usize, tail: bool) -> (Tile, usize, usize, usize) {
        let tiles = |t: Tile| rows.div_ceil(t.bm) * n.div_ceil(t.bn);
        let t = self.tile(rows, n, k);
        if tail {
            let span = k.div_ceil(TAIL).next_multiple_of(BK);
            return (t, tiles(t), k.div_ceil(span), span);
        }
        match self.split(t, tiles(t), k) {
            (1, _) if t.name == SQUARE.name && 2 * tiles(t) < self.slots(t) => (SMALL, tiles(SMALL), 1, k),
            (splits, span) => (t, tiles(t), splits, span),
        }
    }

    fn head(&self, t: Tile, tiles: usize, splits: usize, n: usize, k: usize) -> Option<usize> {
        let slots = self.slots(t);
        (splits == 1 && k >= TAIL * MIN_SPAN && tiles > slots && 4 * tiles <= 7 * slots)
            .then(|| slots / n.div_ceil(t.bn) * t.bm)
    }

    fn slots(&self, t: Tile) -> usize {
        self.sms * t.resident[usize::from(self.sm >= 80)]
    }

    fn split(&self, t: Tile, tiles: usize, k: usize) -> (usize, usize) {
        let slots = self.slots(t);
        let cost = |s: usize| {
            let span = k.div_ceil(s).next_multiple_of(BK);
            ((tiles * k.div_ceil(span)).div_ceil(slots) * span, span)
        };
        let (base, _) = cost(1);
        if 2 * tiles > slots && k < 8192 {
            return (1, k);
        }
        (2..=8)
            .filter(|&s| k.div_ceil(s) >= MIN_SPAN)
            .map(|s| (cost(s), s))
            .filter(|&((c, _), _)| c * 100 <= base * 85)
            .min_by_key(|&((c, _), _)| c)
            .map_or((1, k), |((_, span), _)| (k.div_ceil(span), span))
    }

    fn tile(&self, rows: usize, n: usize, k: usize) -> Tile {
        let wide = rows.div_ceil(WIDE.bm) * n.div_ceil(WIDE.bn);
        match () {
            () if self.sm < 70 => SIMT,
            () if n <= THIN.bn => THIN,
            () if self.sm >= 80
                || (wide < 2 * self.sms && k < 8192 && wide.next_multiple_of(self.sms) * 9 > wide * 10) =>
            {
                SQUARE
            }
            () => WIDE,
        }
    }
}
