fn periods(dim: usize, base: f32) -> Vec<f32> {
    (0..dim / 2).map(|i| base.powf(i as f32 * 2.0 / dim as f32)).collect()
}

fn coords(n: usize) -> Vec<f32> {
    (0..n).map(|i| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0).collect()
}

fn axis(n: usize, dim: usize, base: f32, stride: f32) -> Vec<Vec<f32>> {
    let p = periods(dim, base);
    coords(n)
        .into_iter()
        .map(|c| {
            p.iter()
                .map(|q| 2.0 * std::f32::consts::PI * (c * stride) / q)
                .collect()
        })
        .collect()
}

#[must_use]
pub fn angles(head: usize, (t, h, w): (usize, usize, usize), spatial: bool) -> Vec<f32> {
    let pairs = head / 2;
    let base = 100.0;
    let dims: Vec<(usize, usize)> = if spatial {
        vec![(h, pairs), (w, pairs)]
    } else {
        let third = pairs / 3;
        let rem = pairs - 3 * third;
        vec![
            (t, 2 * third),
            (h, 2 * (third + usize::from(rem >= 2))),
            (w, 2 * (third + usize::from(rem >= 1))),
        ]
    };
    let tables: Vec<Vec<Vec<f32>>> = dims.iter().map(|&(n, d)| axis(n, d, base, 1.0)).collect();
    let tokens = t * h * w;
    (0..tokens)
        .flat_map(|i| {
            let idx = if spatial {
                vec![i / w % h, i % w]
            } else {
                vec![i / (h * w), i / w % h, i % w]
            };
            tables
                .iter()
                .zip(idx)
                .flat_map(|(tab, j)| tab[j].clone())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[must_use]
pub fn tables(head: usize, block: (usize, usize, usize), spatial: bool) -> (Vec<f32>, Vec<f32>) {
    let a = angles(head, block, spatial);
    (a.iter().map(|x| x.cos()).collect(), a.iter().map(|x| x.sin()).collect())
}
