# Profiling

Needs `nsys` and `ncu`, which come with the CUDA toolkit.

## Which kernels take the time

```
just kprof NAME <clip>
```

Runs `ck` once under `nsys` and prints the 25 slowest kernels. Any `ck` option can follow the clip, e.g. `--frames 8`. The full trace is saved as `bench/results/NAME.nsys-rep`; open it in Nsight Systems.

## Which stage takes the time

```
just profile
```

Prints the time and peak memory of stage 1 and stage 2. Options:

- `--hw 1088x1920` — frame size.
- `--frames 8` — window length.
- `--ck` — use our kernels.
- `--nvtx` — mark stages for `nsys`.

## Why one kernel is slow

Use `ncu` on that kernel's test. From `launcher/`:

```
cargo nextest list --release -p ck-gpu --message-format json | jq -r '."rust-suites"[]."binary-path"'
```

This prints the test programs. Pick the one for your kernel, e.g. `.../gemm-<hash>`, then:

```
CK_DEVICE=0 ncu --set full -k regex:gemm -c 5 -o gemm <program> <test name>
```

Open `gemm.ncu-rep` in Nsight Compute.
