# stock vs ours on every GPU, then the report in kernels/bench/results/
bench *args:
    just kernels/bench {{args}}

# rebuild the report from measured JSON
render:
    just kernels/render

# stage timings at --hw HxW (--ck for our kernels, --nvtx for nsys)
profile *args:
    just kernels/profile {{args}}

# consecutive bench frames per resolution
prep:
    just kernels/prep

# per-arch PTX into corridorkey/ptx/bin/sm_XX/
ptx:
    just kernels/ptx

# clippy (both arches) and typos
lint:
    just kernels/lint

# key clips with the Rust launcher from wherever you are: just run clip.mp4 [clips…] [ck flags, see `just run --help`]
[no-cd, positional-arguments]
run *args:
    @cargo run -q --release --manifest-path {{justfile_directory()}}/launcher/Cargo.toml -p ck -- \
      --ptx {{justfile_directory()}}/corridorkey/ptx/bin \
      --weights {{justfile_directory()}}/weights/corridorkey_v2.safetensors \
      --vfms {{justfile_directory()}}/weights/vfms.safetensors "$@"
