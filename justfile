out := justfile_directory() / "corridorkey/ptx/bin"

ptx:
    cd kernels && cargo ptx build -r --archs sm_120 --emit ptx --out-dir {{out}}

install:
    scripts/dev-setup.sh

lint:
    uv run ruff check corridorkey/ptx bench
    cd kernels && for sm in 120; do RUSTFLAGS="-C target-cpu=sm_$sm" cargo clippy --release --workspace --features ck_attn/sm_$sm,ck_dw/sm_$sm,ck_glue/sm_$sm,ck_ops/sm_$sm 2>&1 | rg -n "error(\[|:)|warning:" || true; done

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

bench *args:
    uv run python bench/fps.py {{args}}
