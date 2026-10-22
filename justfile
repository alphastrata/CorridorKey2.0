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
