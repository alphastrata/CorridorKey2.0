use eyre::{Result, ensure};

use crate::{Arg, Dtype, Gpu, Tensor};

#[derive(Clone, Copy)]
struct Tile {
    name: &'static str,
    bm: usize,
    bn: usize,
    stages: [u32; 2],
}

const WIDE: Tile = Tile {
    name: "wide",
    bm: 256,
    bn: 128,
    stages: [2, 4],
};
const SQUARE: Tile = Tile {
    name: "square",
    bm: 128,
    bn: 128,
    stages: [2, 3],
};
const THIN: Tile = Tile {
    name: "thin",
    bm: 128,
    bn: 16,
    stages: [2, 4],
};
const ANY: usize = 64;

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
        (ld, off): (usize, usize),
        accumulate: bool,
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
            let t = self.tile(rows, n, k);
            let name = format!("gemm_{}_{suffix}", t.name);
            let tiles = rows.div_ceil(t.bm) * n.div_ceil(t.bn);
            let shared = t.stages[usize::from(self.sm >= 80)] * (t.bm + t.bn) as u32 * 64;
            return self.launch("ck_gemm", &name, (tiles as u32, 1, 1), 256, shared, &args);
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
        self.launch("ck_gemm", &name, grid, 256, 2 * 16 * (ANY as u32 + 4) * 4, &args)
    }

    fn tile(&self, rows: usize, n: usize, k: usize) -> Tile {
        let wide = rows.div_ceil(WIDE.bm) * n.div_ceil(WIDE.bn);
        match () {
            () if n <= THIN.bn => THIN,
            () if self.sm >= 80 || (wide < 2 * self.sms && k < 8192) => SQUARE,
            () => WIDE,
        }
    }
}
