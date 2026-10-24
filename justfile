out := justfile_directory() / "corridorkey/ptx/bin"

install:
    scripts/dev-setup.sh

lint:
    cd launcher && cargo clippy --release --workspace --all-targets 2>&1 | rg -n "error(\[|:)|warning:" || true
    uv run ruff check corridorkey/ptx bench
    typos launcher/crates bench scripts/bench.sh scripts/dev-setup.sh scripts/download_sample_clips.sh

profile *args:
    uv run python bench/randprof.py weights/corridorkey_v2.safetensors {{args}}

prep:
    uv run python bench/prep.py

suite *args:
    #!/usr/bin/env bash
    set -euo pipefail
    for d in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
      uv run python bench/suite.py --device cuda:$d {{args}}
    done

render:
    uv run python bench/render.py

bench:
    scripts/bench.sh

kprof name *args:
    bench/kprof.sh {{name}} {{args}}

[no-cd, positional-arguments]
run *args:
    @cargo run -q --release --manifest-path {{justfile_directory()}}/launcher/Cargo.toml -p ck -- \
      --ptx {{out}} \
      --weights {{justfile_directory()}}/weights/corridorkey_v2.safetensors \
      --vfms {{justfile_directory()}}/weights/vfms.safetensors "$@"
