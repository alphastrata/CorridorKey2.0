# Contributing

## 1. Set up

You need an NVIDIA GPU, its driver, `ffmpeg` and `curl`. Then run:

```
scripts/dev-setup.sh
```

This installs everything else (uv, Rust nightly, the `ck` launcher, the model weights and sample clips).

## 2. Find your way around

- `launcher/` — the `ck` program, written in Rust.
- `kernels/` — the GPU kernels, also in Rust.
- `corridorkey/ptx/bin/` — the compiled kernels, one folder per GPU generation.
- `bench/` — benchmarks.

## 3. Change a kernel

After editing anything in `kernels/`, rebuild the compiled kernels:

```
just ptx
```

Commit the changed files under `corridorkey/ptx/bin/` together with your source change, so users never need to compile kernels themselves.

Your change must build for all three GPU generations we support:

| Folder | Cards |
|---|---|
| `sm_61` | GTX 10-series |
| `sm_75` | RTX 20-series, TITAN RTX |
| `sm_120` | RTX 50-series |

## 4. Test

From `launcher/`, run the tests on each GPU you have (`CK_DEVICE=0`, `1`, …):

```
CK_DEVICE=0 cargo nextest run --release -p ck-gpu
```

No GTX 10-series card? Add `CK_SM=61` and a newer card will run the GTX 10-series kernels instead:

```
CK_SM=61 CK_DEVICE=0 cargo nextest run --release -p ck-gpu
```

## 5. Lint

```
just lint
```

It must pass with no warnings. Don't silence warnings with `#[allow]`; fix the code.

## 6. Commit

- One change per commit.
- Message format: `type(scope): summary`, e.g. `perf(ck_gemm): faster split-K`.
- Never commit footage or weights.

## Style

- British spelling.
- Few comments; clear names instead.
- Rust: add crates with `cargo add`. Python: add packages with `uv add`, and use type hints.
- Speed-ups are welcome if they help 720p, 1080p or 4K by 0.5% or more without slowing anything else.
