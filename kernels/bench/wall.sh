#!/usr/bin/env bash
# usage: wall.sh [clip] [frames] : end-to-end wall time per GPU and tier for master, stock, ours and ck → results/wall.csv
set -u
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
B=$R/kernels/bench/results
CLIP=${1:-pexels_9489598}
FRAMES=${2:-48}
CSV=$B/wall.csv
GPUS=$(( $(nvidia-smi -L | wc -l) - 1 ))
mkdir -p $B/logs

loop() {
  local src=$R/kernels/bench/frames/$CLIP/$1 dst=$B/loop/$1
  rm -rf $dst && mkdir -p $dst
  local pngs=($src/*.png)
  seq 0 $((FRAMES - 1)) | while read -r i; do
    ln -s ${pngs[$((i % ${#pngs[@]}))]} $dst/$(printf %04d.png $((i + 1)))
  done
}

master() {
  local m=$B/master
  rm -rf $m && mkdir -p $m
  git -C $R archive master | tar -x -C $m
  ln -sf $R/weights/*.safetensors $m/weights/
  for d in $R/vfms/*/; do ln -sfn $d* $m/vfms/$(basename $d)/; done
}

run() {
  local gpu=$1 tier=$2 cfg=$3 dir=$4; shift 4
  local log=$B/logs/${gpu/,/+}_${tier}_$cfg.log s=$(date +%s.%N)
  (cd $dir && CUDA_VISIBLE_DEVICES=$gpu /usr/bin/time -f "RSS %M" "$@" > $log 2>&1)
  local rc=$? wall=$(echo "$(date +%s.%N) - $s" | bc)
  local rss=$(rg -o "^RSS \d+" $log | tail -1 | awk '{print int($2/1024)}')
  local set=$(rg -o "hann_chunk=\d+, hann_stride=\d+, low_vram=\w+|window \d+, stride \d+" $log | head -1 | tr ',' ';')
  echo "${gpu/,/+},$tier,$cfg,$(printf %.1f $wall),$rc,$rss,$set" | tee -a $CSV
}

py() {
  run $1 $2 $3 $4 "${@:5}" infer.py --input $B/loop/$2 --output_dir $B/out/$1_$2_$3 --num_frames $FRAMES
}

ck() {
  run $1 $2 ck_$3 $R $CK --ptx corridorkey/ptx/bin --weights weights/corridorkey_v2.safetensors \
    --vfms weights/vfms.safetensors --frames $FRAMES --chunk auto --codec $3 -o $B/out/${1/,/+}_$2_ck_$3 $B/loop/$2
}

cargo build -q --release --manifest-path $R/launcher/Cargo.toml -p ck
CK=$(cargo metadata -q --format-version 1 --no-deps --manifest-path $R/launcher/Cargo.toml | jq -r .target_directory)/release/ck
master
echo "gpu,tier,config,wall_s,exit,max_rss_mb,setting" > $CSV
for tier in 720p 1080p 2160p; do
  loop $tier
  for gpu in $(seq 0 $GPUS); do
    py $gpu $tier master $B/master $R/.venv/bin/python
    py $gpu $tier stock $R env CORRIDORKEY_PTX=0 $R/.venv/bin/python
    py $gpu $tier ours $R $R/.venv/bin/python
    for codec in libx264 hevc_nvenc; do ck $gpu $tier $codec; done
  done
  [ $GPUS -gt 0 ] && for codec in libx264 hevc_nvenc; do ck $(seq -s, 0 $GPUS) $tier $codec; done
done
