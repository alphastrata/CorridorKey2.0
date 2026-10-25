# Benchmarks

Results go to `bench/results/`.

## Frames per second

```
just bench
```

Runs the Python pipeline on a sample clip at 720p, 1080p and 4K, on every GPU, and prints one table: frames per second, memory, power.

Change the frame count or resolutions with:

```
just bench --frames 100 --tiers 1080 2160
```
