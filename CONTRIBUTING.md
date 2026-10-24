# Contributing

## Setup

```
scripts/dev-setup.sh
```

This installs uv, nightly Rust, `ck`, the weights and the sample clips. Pipeline overview: [arch.md](arch.md).

## Layout

| Path | What |
|---|---|
| `launcher/` | `ck` (Rust launcher) and `ck-gpu` (GPU runtime, GEMM dispatch) |
| `kernels/` | CUDA kernels in Rust on the [ptx](https://codeberg.org/P-E-T-R-A/ptx) crate |
| `corridorkey/ptx/bin/sm_{61,75,120}/` | built PTX, committed so users need no toolchain |
| `corridorkey/`, `infer.py` | Python reference path |
| `bench/` | benchmarks and profiling |

## Kernels

- Run `just ptx` after any change to `kernels/`, and commit the PTX alongside the source.
- Every change must build for sm_61, sm_75 and sm_120.
  - sm_61 is compiled as sm_70 and then relabelled, because rustc no longer targets Pascal.
  - Validate it with CUDA 12's `ptxas -arch=sm_61`.
- Architecture-specific code goes in `ck_arch/src/sm*.rs`, a crate's `tune/` directory, or a `cfg(feature = "sm_*")` module.
- If a kernel's `ptxas -v` report shows a stack frame, fix that first. It usually means a large closure was not inlined; mark it `#[inline(always)]`.

## Testing

```
CK_DEVICE=0 cargo nextest run --release -p ck-gpu
CK_DEVICE=1 cargo nextest run --release -p ck-gpu
CK_SM=61 CK_DEVICE=0 cargo nextest run --release -p ck-gpu
```

- Run these from `launcher/`.
- `CK_SM` caps the architecture so that a newer card runs older PTX, which makes Pascal testable without a Pascal card.
- Test on every GPU you have.
- For end-to-end checks, compare `--exr float` alpha output against the previous build.

## Performance

- Benchmark at 720p, 1080p and 2160p, on every architecture you can.
- A gain of 0.5% or more at any of those sizes, with no regression elsewhere, is worth taking.
- A simplification is worth taking even if it does not change performance.

## Style

- `just lint` must pass: clippy pedantic for each kernel architecture, ruff, and typos in `en-gb`.
- Never use `#[allow]` or `#[expect]`. Fix the code, or set the policy in `[workspace.lints]`.
- Rust: nightly, edition 2024, iterators over loops. Add crates with `cargo add`.
- Python: typed, run through uv. Add packages with `uv add`.
- Keep comments to a minimum; code should be self-documenting.
- Use British spelling.

## Commits

- Keep commits atomic, in the form `type(scope): summary`, e.g. `perf(ck_gemm): …`.
- Do not commit footage or weights.
