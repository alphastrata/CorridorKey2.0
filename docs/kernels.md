# Kernels

## 1. Build

After editing anything in `kernels/`, rebuild the compiled kernels:

```
just ptx
```

Commit the changed files under `corridorkey/ptx/bin/` together with your source change, so users never need to compile kernels themselves.

## 2. Support every GPU generation

Your change must build for all of them:

| Folder | Cards |
|---|---|
| `sm_61` | GTX 10-series |
| `sm_75` | RTX 20-series, TITAN RTX |
| `sm_80` | A100, and any newer card without its own folder |
| `sm_86` | RTX 30-series, A10, A40 (empty: these run `sm_80`) |
| `sm_89` | RTX 40-series, L4, L40S (empty: these run `sm_80`) |
| `sm_90a` | H100, H200 |
| `sm_100a` | B200 |
| `sm_120` | RTX 50-series |

`just lint` checks every folder. Without a card of that generation, `CK_SM=86` (say) makes a newer card run that folder's PTX.

## 3. Test

From `launcher/`, on each GPU you have:

```
CK_DEVICE=0 cargo nextest run --release -p ck-gpu
```

For the whole pipeline against PyTorch, make the fixtures once, then use `--workspace` in place of `-p ck-gpu`:

```
uv run python launcher/parity/dump.py --tier 720
uv run python launcher/parity/dump.py --tier 1080
```

No GTX 10-series card? Add `CK_SM=61` and a newer card will run the GTX 10-series kernels instead:

```
CK_SM=61 CK_DEVICE=0 cargo nextest run --release -p ck-gpu
```

## 4. Prove it is faster

A speed-up is welcome if it helps 720p, 1080p or 4K by 0.5% or more without slowing anything else. Measure on every GPU you have; see [Benchmarks](benchmarks.md).
