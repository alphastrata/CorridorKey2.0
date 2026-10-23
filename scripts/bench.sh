#!/usr/bin/env bash
# usage: [FRAMES=50] [TIERS="720 1080 2160"] bench.sh : python (stock kernels) and ck on one clip per tier and GPU → one csv, one table
set -uo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
B=$R/bench/results/bench
FRAMES=${FRAMES:-50}
TIERS=${TIERS:-720 1080 2160}
CSV=$B/bench.csv
PY=$R/.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
declare -A NAME BASE

frames() {
  echo $R/bench/footage/sample/$1p
}

f() { awk "BEGIN { printf \"$1\", $2 }"; }

stats() {
  awk -F', ' -v base="$1" '
    BEGIN { n = split(base, b, " "); for (i = 1; i <= n; i += 2) idle[b[i]] = b[i + 1] }
    { split($1, t, " "); split(t[2], h, ":"); s = h[1] * 3600 + h[2] * 60 + h[3]; g = $2
      if ($3 - idle[g] > peak[g]) peak[g] = $3 - idle[g]
      util[g] += $4; seen[g]++
      if (g in last) joules += (s - last[g]) * $5
      last[g] = s; watts += $5 }
    END { for (g in peak) { vram += peak[g]; u += util[g] / seen[g]; k++ }
          printf "%.2f,%.0f,%.0f,%.0f", vram / 1024, k ? u / k : 0, NR ? watts / NR * k : 0, joules }' $2
}

run() {
  local gpus=$1 tier=$2 launcher=$3; shift 3
  local tag=${gpus//,/+}_${tier}p_$launcher label=""
  for g in ${gpus//,/ }; do label+=${label:+ + }${NAME[$g]}; done
  local base=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i $gpus | tr -d , | tr '\n' ' ')
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits \
    -i $gpus -lms 100 > $B/samples/$tag.csv &
  local smi=$! s=$(date +%s.%N)
  (cd $R && CUDA_VISIBLE_DEVICES=$gpus /usr/bin/time -f "RSS %M" "$@" > $B/logs/$tag.log 2>&1)
  local rc=$? wall=$(echo "$(date +%s.%N) - $s" | bc)
  kill $smi; wait $smi 2>/dev/null
  local rss=$(rg -o "^RSS \d+" $B/logs/$tag.log | tail -1 | cut -d' ' -f2)
  IFS=, read -r vram util watts joules <<< "$(stats "$base" $B/samples/$tag.csv)"
  [ $launcher = python ] && BASE[$gpus$tier]=$wall
  local ref=${BASE[$gpus$tier]:-} ok=$([ $rc = 0 ] && echo 1)
  printf "%s,%sp,%s,%d,%s,%.1f,%s,%s,%s,%s,%s,%s,%s,%.1f\n" "$label" $tier $launcher $FRAMES $rc $wall \
    "${ok:+$(f "%.2f" $FRAMES/$wall)}" "${ok:+$(f "%.0f" 1000*$wall/$FRAMES)}" "${ok:+${ref:+$(f "%.2f" $ref/$wall)}}" \
    $vram $util $watts "${ok:+$(f "%.0f" $joules/$FRAMES)}" $(f "%.1f" ${rss:-0}/1048576) >> $CSV
  tail -1 $CSV
}

ck() {
  run $1 $2 ck $CK --ptx corridorkey/ptx/bin --weights weights/corridorkey_v2.safetensors --vfms weights/vfms.safetensors \
    --frames $FRAMES --chunk auto -o $B/out/${1//,/+}_$2p_ck $(frames $2)
}

"$R/scripts/download_sample_clips.sh" $FRAMES
command -v cargo > /dev/null || { echo "no Rust toolchain: run scripts/dev-setup.sh" >&2; exit 1; }
cargo build -q --release --manifest-path $R/launcher/Cargo.toml -p ck
CK=$(cargo metadata -q --format-version 1 --no-deps --manifest-path $R/launcher/Cargo.toml | jq -r .target_directory)/release/ck
mkdir -p $B/logs $B/samples $B/out
while IFS=, read -r i n; do NAME[$i]=${n# NVIDIA }; done < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)
GPUS=$(printf "%s\n" "${!NAME[@]}" | sort -n | paste -sd,)
echo "gpu,tier,launcher,frames,exit,wall_s,fps,ms_frame,x_python,vram_gib,util_pct,power_w,j_frame,rss_gib" > $CSV
for tier in $TIERS; do
  d=$(frames $tier)
  for g in ${GPUS//,/ }; do
    run $g $tier python env CORRIDORKEY_PTX=0 $PY infer.py --input $d --output_dir $B/out/${g}_${tier}p_python --num_frames $FRAMES
    ck $g $tier
  done
  [[ $GPUS == *,* ]] && ck $GPUS $tier
done
echo; column -ts, $CSV
