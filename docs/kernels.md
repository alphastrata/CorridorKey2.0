# Kernels

## 1. Build

After editing anything in `kernels/`, rebuild the compiled kernels:

```
just ptx
```

Commit the changed files under `corridorkey/ptx/bin/` together with your source change, so users never need to compile kernels themselves.

## 2. Support every GPU generation

Your change must build for both:

| Folder | Cards |
|---|---|
| `sm_75` | RTX 20-series, TITAN RTX |
| `sm_120` | RTX 50-series |

`just lint` checks both.

## 3. Test

From `launcher/`, on each GPU you have:

```
CK_DEVICE=0 cargo nextest run --release -p ck-gpu
```

## 4. Prove it is faster

A speed-up is welcome if it helps 720p, 1080p or 4K by 0.5% or more without slowing anything else. Measure on every GPU you have; see [Benchmarks](benchmarks.md).
