out := justfile_directory() / "corridorkey/ptx/bin"

ptx:
    cd kernels && cargo ptx build -r --archs sm_61,sm_75,sm_80,sm_90a,sm_100a,sm_120 --emit ptx --out-dir {{out}}

install:
    scripts/dev-setup.sh

lint:
    cd launcher && cargo clippy --release --workspace --all-targets 2>&1 | rg -n "error(\[|:)|warning:" || true
    uv run ruff check corridorkey/ptx bench
    cd kernels && for sm in 61 75 80 90a 100a 120; do cpu=$sm; [ "${sm%a}" -lt 70 ] && cpu=70; RUSTFLAGS="-C target-cpu=sm_$cpu" cargo clippy --release --workspace --features ck_attn/sm_$sm,ck_dw/sm_$sm,ck_gemm/sm_$sm,ck_glue/sm_$sm,ck_ops/sm_$sm 2>&1 | rg -n "error(\[|:)|warning:" || true; done

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

kprof name *args:
    uv run python bench/kprof.py {{name}} {{args}}

[no-cd, positional-arguments]
run *args:
    @cargo run -q --release --manifest-path {{justfile_directory()}}/launcher/Cargo.toml -p ck -- "$@"

[no-cd, positional-arguments]
gui *args:
    #!/usr/bin/env bash
    set -euo pipefail
    clips=()
    for a in "$@"; do clips+=("$(realpath "$a")"); done
    run="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    if [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
        sock=$(find "$run" -maxdepth 1 -type s -name 'wayland-[0-9]*' -printf '%f\n' 2>/dev/null | sort | head -1)
        [[ -n "$sock" ]] && export WAYLAND_DISPLAY="$sock" XDG_RUNTIME_DIR="$run"
    fi
    cargo run -q --release --manifest-path {{justfile_directory()}}/launcher/Cargo.toml -p ck-gui -- ${clips[@]+"${clips[@]}"}
