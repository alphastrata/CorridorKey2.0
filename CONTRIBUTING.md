# Contributing

## 1. Set up

You need an NVIDIA GPU, its driver, `ffmpeg` and `curl`. Then run:

```
scripts/dev-setup.sh
```

This installs everything else (uv, Rust nightly, `just`, `rg`, the model weights and sample clips).

## 2. Find your way around

- `kernels/` — the GPU kernels, in Rust.
- `corridorkey/ptx/` — routes the model through them.
- `corridorkey/ptx/bin/` — the compiled kernels, one folder per GPU generation.
- `bench/` — benchmarks and profiling.

## 3. Test

```
just suite
```

It checks every kernel against an fp32 reference on each GPU.

## 4. Lint

```
just lint
```

It must pass with no warnings. Don't silence warnings with `#[allow]`; fix the code.

## 5. Commit

- One change per commit.
- Message format: `type(scope): summary`, e.g. `perf(ck_attn): fewer stages`.
- Never commit footage or weights.

## Style

- Few comments; clear names instead.
- Rust: add crates with `cargo add`. Python: add packages with `uv add`, and use type hints.

## More

- [Kernels](docs/kernels.md) — changing and testing GPU kernels.
- [Benchmarks](docs/benchmarks.md) — measuring speed.
- [Profiling](docs/profiling.md) — finding where the time goes.
