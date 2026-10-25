# Contributing

## 1. Set up

You need an NVIDIA GPU, its driver, `ffmpeg` and `curl`. Then run:

```
scripts/dev-setup.sh
```

This installs everything else (uv, `just`, the model weights and sample clips).

## 2. Find your way around

- `corridorkey/` — the model and pipeline.
- `bench/` — benchmarks and profiling.

## 3. Lint

```
just lint
```

It must pass with no warnings.

## 4. Commit

- One change per commit.
- Message format: `type(scope): summary`, e.g. `perf(bench): fewer syncs`.
- Never commit footage or weights.

## Style

- Few comments; clear names instead.
- Add packages with `uv add`, and use type hints.

## More

- [Benchmarks](docs/benchmarks.md) — measuring speed.
- [Profiling](docs/profiling.md) — finding where the time goes.
