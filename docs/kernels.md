# Kernels

## 1. Build

After editing anything in `kernels/`, rebuild the compiled kernels:

```
just ptx
```

Commit the changed files under `corridorkey/ptx/bin/` together with your source change, so users never need to compile kernels themselves.

## 2. Target

Kernels build for `sm_120` (RTX 50-series).

## 3. Test

`just suite` on each GPU you have; see [Benchmarks](benchmarks.md).

## 4. Prove it is faster

A speed-up is welcome if it helps 720p, 1080p or 4K by 0.5% or more without slowing anything else. Measure on every GPU you have; see [Benchmarks](benchmarks.md).
