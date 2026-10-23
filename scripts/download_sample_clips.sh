#!/usr/bin/env bash
# usage: download_sample_clips.sh : the 4K green-screen benchmark clip (CC0) into kernels/bench/footage
set -euo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT=$R/kernels/bench/footage
FILE=commons_4k_video_of_a_Chicken_fillet_roll___green_screen.webm
URL=https://upload.wikimedia.org/wikipedia/commons/8/8e/4k_video_of_a_Chicken_fillet_roll_-_green_screen_alan_butler_VP9.webm
mkdir -p "$OUT"
[ -s "$OUT/$FILE" ] && exit 0
curl -fsSL --retry 3 -A "CorridorKeyBench/0.1" -o "$OUT/$FILE.part" "$URL" && mv "$OUT/$FILE.part" "$OUT/$FILE"
