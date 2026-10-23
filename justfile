bench *args:
    just kernels/bench {{args}}

render:
    just kernels/render

profile *args:
    just kernels/profile {{args}}

prep:
    just kernels/prep

ptx:
    just kernels/ptx

lint:
    just kernels/lint

[no-cd, positional-arguments]
run *args:
    @cargo run -q --release --manifest-path {{justfile_directory()}}/launcher/Cargo.toml -p ck -- \
      --ptx {{justfile_directory()}}/corridorkey/ptx/bin \
      --weights {{justfile_directory()}}/weights/corridorkey_v2.safetensors \
      --vfms {{justfile_directory()}}/weights/vfms.safetensors "$@"
