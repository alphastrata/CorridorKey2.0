#!/usr/bin/env bash
# usage: [FRAMES=48] [TIERS="720 1080 2160"] [PARTS="e2e stages kernels suite"] bench.sh
# every runner (master, stock, ours, ck) on every GPU and one native clip per tier → kernels/bench/results/bench/REPORT.md
set -uo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
B=$R/kernels/bench/results/bench
F=$R/kernels/bench/footage
FRAMES=${FRAMES:-48}
TIERS=${TIERS:-720 1080 2160}
PARTS=${PARTS:-e2e stages kernels suite}
PY=$R/.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)

frames() {
  local file=$(awk -F'\t' -v t=$1 '$4 == t {print $1}' $F/SOURCES.tsv)
  local d=$B/clips/${file%.*}/$1p
  if [ "$(ls $d 2>/dev/null | wc -l)" -ne $FRAMES ]; then
    rm -rf $d && mkdir -p $d
    local dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $F/$file)
    ffmpeg -v error -ss $(echo "$dur / 4" | bc -l) -i $F/$file -frames:v $FRAMES $d/%03d.png
  fi
  echo $d
}

master() {
  rm -rf $B/master && mkdir -p $B/master
  git -C $R archive master | tar -x -C $B/master
  ln -sf $R/weights/*.safetensors $B/master/weights/
  for d in $R/vfms/*/; do ln -sfn $d* $B/master/vfms/$(basename $d)/; done
}

run() {
  local gpus=$1 tier=$2 cfg=$3 dir=$4; shift 4
  local tag=${gpus//,/+}_${tier}p_$cfg
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i $gpus > $B/samples/$tag.base
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits \
    -i $gpus -lms 100 > $B/samples/$tag.csv &
  local smi=$! s=$(date +%s.%N)
  (cd $dir && CUDA_VISIBLE_DEVICES=$gpus /usr/bin/time -f "RSS %M" "$@" > $B/logs/$tag.log 2>&1)
  local rc=$? wall=$(echo "$(date +%s.%N) - $s" | bc)
  kill $smi
  local rss=$(rg -o "^RSS \d+" $B/logs/$tag.log | tail -1 | cut -d' ' -f2)
  echo "${gpus//,/+},$tier,$cfg,$wall,$rc,${rss:-0}" >> $B/e2e.csv
  printf "%-8s %5sp %-16s %7.1f s  exit %s\n" "${gpus//,/+}" $tier $cfg $wall $rc
}

ck() {
  local gpus=$1 tier=$2 codec=$3; shift 3
  echo $CK --ptx corridorkey/ptx/bin --weights weights/corridorkey_v2.safetensors --vfms weights/vfms.safetensors \
    --frames $FRAMES --chunk auto --codec $codec -o $B/out/${gpus//,/+}_${tier}p_$codec "$@" $(frames $tier)
}

e2e() {
  master
  echo "gpus,tier,config,wall_s,exit,rss_kib" > $B/e2e.csv
  for tier in $TIERS; do
    local d=$(frames $tier)
    for g in ${GPUS//,/ }; do
      for cfg in master stock ours; do
        local dir=$R env=()
        [ $cfg = master ] && dir=$B/master
        [ $cfg = stock ] && env=(env CORRIDORKEY_PTX=0)
        run $g $tier $cfg $dir "${env[@]}" $PY infer.py --input $d --output_dir $B/out/${g}_${tier}p_$cfg --num_frames $FRAMES
      done
      for codec in libx264 hevc_nvenc; do run $g $tier ck_$codec $R $(ck $g $tier $codec); done
    done
    [[ $GPUS == *,* ]] && for codec in libx264 hevc_nvenc; do run $GPUS $tier ck_$codec $R $(ck $GPUS $tier $codec); done
  done
}

stages() {
  for tier in $TIERS; do
    for g in ${GPUS//,/ }; do
      (cd $R && CK_TIME=1 CK_MEM=1 CUDA_VISIBLE_DEVICES=$g $(ck $g $tier libx264) > $B/logs/stages_${g}_${tier}p.log 2>&1)
      echo "stages $g ${tier}p"
    done
  done
}

kernels() {
  for tier in $TIERS; do
    for g in ${GPUS//,/ }; do
      local tag=${g}_${tier}p s=$(date +%s.%N)
      (cd $R && CUDA_VISIBLE_DEVICES=$g nsys profile -f true -o $B/kernels/$tag $(ck $g $tier libx264) > /dev/null 2>&1)
      echo "$(date +%s.%N) - $s" | bc > $B/kernels/$tag.wall
      nsys stats -q --force-export=true -r cuda_gpu_kern_sum -f csv $B/kernels/$tag.nsys-rep > $B/kernels/$tag.csv 2>/dev/null
      echo "kernels $tag"
    done
  done
}

suite() {
  for g in ${GPUS//,/ }; do
    (cd $R && CUDA_VISIBLE_DEVICES=$g $PY kernels/bench/suite.py --device cuda:0 --tiers $TIERS --clips 1 --frames 8 \
      --frames-dir $B/clips > $B/logs/suite_$g.log 2>&1)
    echo "suite $g"
  done
  (cd $R && $PY kernels/bench/render.py > /dev/null)
}

"$R/scripts/download_sample_clips.sh"
cargo build -q --release --manifest-path $R/launcher/Cargo.toml -p ck
CK=$(cargo metadata -q --format-version 1 --no-deps --manifest-path $R/launcher/Cargo.toml | jq -r .target_directory)/release/ck
mkdir -p $B/logs $B/samples $B/kernels $B/out
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits > $B/gpus.csv
for part in $PARTS; do $part; done
$PY $R/kernels/bench/report.py $B $FRAMES > $B/REPORT.md && cat $B/REPORT.md
