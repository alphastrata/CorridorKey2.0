# Profiling

## Which stage takes the time

```
just profile
```

Prints the time and peak memory of stage 1 and stage 2. Options:

- `--hw 1088x1920` — frame size.
- `--frames 8` — window length.
- `--nvtx` — mark stages for `nsys`.
