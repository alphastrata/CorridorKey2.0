# Benchmarks

Results go to `bench/results/`.

## Quick: frames per second

```
just bench
```

Runs the Python pipeline on a sample clip at 720p, 1080p and 4K, on every GPU, and prints one table: frames per second, memory, power.

Change the frame count or resolutions with:

```
just bench --frames 100 --tiers 1080 2160
```

## Full: every part, every resolution

1. Cut frame sets from the footage (once):

   ```
   just prep
   ```

2. Measure on every GPU:

   ```
   just suite
   ```

   Options: `--tiers 2160 1080 720`, `--clips 4`, `--frames 8`.

3. Build the report:

   ```
   just render
   ```

   Read `bench/results/README.md`, then `bench/results/<gpu>.md` for detail: time per stage, memory, time by kernel, and precision against an fp32 reference.
