out := justfile_directory() / "corridorkey/ptx/bin"

ptx:
    #!/usr/bin/env bash
    set -euo pipefail
    cd kernels
    meta=$(cargo metadata --no-deps --format-version 1)
    for c in $(jq -r '.packages[] | select(any(.targets[]; .crate_types | index("cdylib"))) | .name' <<< "$meta"); do
      for sm in $(jq -r --arg c "$c" '.packages[] | select(.name == $c) | .metadata.ck.archs // [75, 120] | .[]' <<< "$meta"); do
        mkdir -p target/ptx/sm_$sm {{out}}/sm_$sm
        RUSTFLAGS="-C target-cpu=sm_$sm" cargo rustc -q -p $c --release --features sm_$sm -- \
          -Zunstable-options -Clinker-flavor=llbc --emit=asm -o target/ptx/sm_$sm/$c.s \
          2> >(rg -v 'due to multiple output types|ignoring --out-dir|^$' >&2 || true)
        bounds=$(jq -r --arg c "$c" '.packages[] | select(.name == $c) | .metadata.ck.launch_bounds | if type == "object" then .["'$sm'"] else . end // empty | "\n.maxntid \(.[0]), 1, 1\n.minnctapersm \(.[1])"' <<< "$meta")
        perl -0pe "s/(\.visible \.entry \w+\([^)]*\))/\$1$bounds/g" target/ptx/sm_$sm/$c.ptx > {{out}}/sm_$sm/$c.ptx
      done
    done

lint:
    cd kernels && for sm in 75 120; do cargo clippy --release --workspace --features ck_arch/sm_$sm 2>&1 | rg -n "error(\[|:)|warning:" || true; done
    cd kernels && typos --locale en-gb .

profile *args:
    uv run python kernels/bench/randprof.py weights/corridorkey_v2.safetensors {{args}}

prep:
    uv run python kernels/bench/prep.py

bench *args:
    #!/usr/bin/env bash
    set -euo pipefail
    for d in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
      uv run python kernels/bench/suite.py --device cuda:$d {{args}}
    done

render:
    uv run python kernels/bench/render.py

[no-cd, positional-arguments]
run *args:
    @cargo run -q --release --manifest-path {{justfile_directory()}}/launcher/Cargo.toml -p ck -- \
      --ptx {{out}} \
      --weights {{justfile_directory()}}/weights/corridorkey_v2.safetensors \
      --vfms {{justfile_directory()}}/weights/vfms.safetensors "$@"

wall *args:
    kernels/bench/wall.sh {{args}}

kprof name *args:
    kernels/bench/kprof.sh {{name}} {{args}}
