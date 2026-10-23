#!/usr/bin/env bash
# usage: kprof.sh NAME [ck args…] : nsys one ck run, print the top kernels by GPU time → results/NAME.nsys-rep
set -eu
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
B=$R/bench/results
NAME=$1; shift
mkdir -p $B
cargo build -q --release --manifest-path $R/launcher/Cargo.toml -p ck
CK=$(cargo metadata -q --format-version 1 --no-deps --manifest-path $R/launcher/Cargo.toml | jq -r .target_directory)/release/ck
cd $R
nsys profile -f true -o $B/$NAME $CK --ptx corridorkey/ptx/bin \
  --weights weights/corridorkey_v2.safetensors --vfms weights/vfms.safetensors -o $B/out/$NAME "$@" > /dev/null 2>&1
nsys stats -q --force-export=true -r cuda_gpu_kern_sum -f csv $B/$NAME.nsys-rep | python3 -c '
import csv, sys
r = list(csv.reader(sys.stdin))
h = next(i for i, x in enumerate(r) if x and x[0] == "Time (%)")
rows = [x for x in r[h + 1:] if len(x) > 1]
print(f"total {sum(int(x[1]) for x in rows) / 1e6:.1f} ms")
for x in rows[:25]:
    print(f"{int(x[1]) / 1e6:9.1f} ms {x[2]:>6} {x[-1][:70]}")'
