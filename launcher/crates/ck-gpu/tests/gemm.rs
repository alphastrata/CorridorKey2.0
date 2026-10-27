use std::path::Path;

use ck_gpu::{Dtype, Gpu};
use cudarc::driver::sys::CUevent_flags;
use half::{bf16, f16};

fn bin() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../corridorkey/ptx/bin"))
}

fn device() -> usize {
    std::env::var("CK_DEVICE").map_or(0, |d| d.parse().unwrap_or(0))
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    (0..n as u32)
        .map(|i| {
            let h = (i ^ seed)
                .wrapping_mul(0x9E37_79B9)
                .rotate_left(13)
                .wrapping_mul(0x85EB_CA6B);
            (h >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn round(v: &[f32], d: Dtype) -> Vec<f32> {
    v.iter()
        .map(|&x| match d {
            Dtype::Bf16 => f32::from(bf16::from_f32(x)),
            Dtype::F16 => f32::from(f16::from_f32(x)),
            _ => x,
        })
        .collect()
}

#[test]
fn linear_matches_f64() -> eyre::Result<()> {
    let gpu = Gpu::open(device(), bin())?;
    let cases = [
        (37, 24, 40, false),
        (300, 136, 1288, false),
        (2000, 16, 1536, false),
        (777, 8, 512, false),
        (4096, 1152, 64, false),
        (13, 3072, 512, true),
        (3, 512, 1, true),
        (300, 136, 588, false),
        (541, 384, 2056, false),
        (300, 384, 8192, false),
        (2310, 1280, 1024, false),
    ];
    cases.iter().try_for_each(|&(rows, n, k, f32in)| {
        let (xd, wd) = if f32in {
            (Dtype::F32, Dtype::F32)
        } else {
            (Dtype::Bf16, gpu.half())
        };
        let (x, w, bias) = (noise(rows * k, 1), noise(n * k, 2), noise(n, 3));
        let (xr, wr) = (round(&x, xd), round(&w, wd));
        let dots: Vec<f64> = (0..rows * n)
            .map(|i| {
                (0..k)
                    .map(|j| f64::from(xr[i / n * k + j]) * f64::from(wr[i % n * k + j]))
                    .sum()
            })
            .collect();
        let (xt, wt) = (gpu.upload_f32(&x, &[rows, k], xd)?, gpu.upload_f32(&w, &[n, k], wd)?);
        let bt = gpu.upload_f32(&bias, &[n], Dtype::F32)?;
        let (ld, off) = (n + 8, 8);
        let start = noise(rows * ld, 4);
        let outs = if f32in || k % 8 != 0 {
            vec![Dtype::F32]
        } else {
            vec![Dtype::F32, Dtype::Bf16]
        };
        outs.iter().try_for_each(|&yd| {
            [(false, false), (true, false), (false, true), (true, true)]
                .iter()
                .try_for_each(|&(acc, with)| {
                    let y = gpu.upload_f32(&start, &[rows, ld], yd)?;
                    gpu.linear_into(&xt, &wt, with.then_some(&bt), &y, (ld, off), acc)?;
                    let got = gpu.download_f32(&y)?;
                    let base = round(&start, yd);
                    let worst = (0..rows * ld).fold(0.0f64, |m, i| {
                        let (r, c) = (i / ld, i % ld);
                        let want = if c < off {
                            f64::from(base[i])
                        } else {
                            let o = c - off;
                            dots[r * n + o]
                                + if with { f64::from(bias[o]) } else { 0.0 }
                                + if acc { f64::from(base[i]) } else { 0.0 }
                        };
                        m.max((f64::from(got[i]) - want).abs() / (1.0 + want.abs()))
                    });
                    let tol = if yd == Dtype::F32 {
                        1e-4 * (k as f64 / 2048.0).max(1.0)
                    } else {
                        8e-3
                    };
                    assert!(
                        worst <= tol,
                        "{rows}x{n}x{k} {xd:?}→{yd:?} acc {acc} bias {with}: {worst}"
                    );
                    Ok::<_, eyre::Report>(())
                })
        })
    })
}

#[test]
#[ignore = "timing: CK_SHAPES=<count,stage,rows,n,k,ld,x,w,y,epi,acc csv>"]
fn timing() -> eyre::Result<()> {
    let gpu = Gpu::open(device(), bin())?;
    let shapes = std::fs::read_to_string(std::env::var("CK_SHAPES")?)?;
    let event = || gpu.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT));
    let (start, end) = (event()?, event()?);
    println!("stage,count,rows,n,k,ld,y,epi,acc,ms");
    shapes.lines().try_for_each(|line| {
        let f: Vec<&str> = line.split(',').collect();
        let [count, stage, rows, n, k, ld, x, _, y, epi, acc] = f[..] else {
            return Ok(());
        };
        let [rows, n, k, ld]: [usize; 4] = [rows, n, k, ld].map(|v| v.parse().unwrap_or(0));
        let (xd, wd) = if x == "F32" {
            (Dtype::F32, Dtype::F32)
        } else {
            (Dtype::Bf16, gpu.half())
        };
        let yd = if y == "F32" { Dtype::F32 } else { Dtype::Bf16 };
        let xt = gpu.upload_f32(&noise(rows * k, 1), &[rows, k], xd)?;
        let wt = gpu.upload_f32(&noise(n * k, 2), &[n, k], wd)?;
        let bt = gpu.upload_f32(&noise(n, 3), &[n], Dtype::F32)?;
        let yt = gpu.zeros(&[rows, ld], yd)?;
        let b = (epi == "Bias").then_some(&bt);
        let cycles = 2.0 * ((rows * n * k) as f64);
        let reps = (2e11 / cycles).ceil().clamp(3.0, 200.0).max(0.0) as usize;
        let run = || gpu.linear_into(&xt, &wt, b, &yt, (ld, 0), acc == "true");
        let ms = (0..5).try_fold(f32::MAX, |best, _| {
            run()?;
            start.record(&gpu.stream)?;
            (0..reps).try_for_each(|_| run())?;
            end.record(&gpu.stream)?;
            Ok::<_, eyre::Report>(best.min(start.elapsed_ms(&end)? / reps as f32))
        })?;
        println!("{stage},{count},{rows},{n},{k},{ld},{y},{epi},{acc},{ms:.4}");
        Ok(())
    })
}
