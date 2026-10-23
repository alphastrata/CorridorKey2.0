#!/usr/bin/env bash
# usage: download_sample_clips.sh : one native green-screen clip per tier into kernels/bench/footage, listed in SOURCES.tsv
set -euo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT=$R/kernels/bench/footage
C=https://upload.wikimedia.org/wikipedia/commons
CLIPS="commons_Lydia_Owen___Yuri.webm	$C/0/01/Lydia_Owen_-_Yuri.webm	CC BY 3.0	720
commons_The_Pelones___Green_Screen.webm	$C/a/a1/The_Pelones_-_Green_Screen.webm	CC BY 3.0	1080
commons_4k_video_of_a_Chicken_fillet_roll___green_screen.webm	$C/8/8e/4k_video_of_a_Chicken_fillet_roll_-_green_screen_alan_butler_VP9.webm	CC0	2160"

mkdir -p "$OUT"
echo "$CLIPS" > "$OUT/SOURCES.tsv"
while IFS=$'\t' read -r file url licence tier; do
  [ -s "$OUT/$file" ] && { echo "have $file (${tier}p)"; continue; }
  echo "fetching $file (${tier}p, $licence)"
  curl -fsSL --retry 3 -A "CorridorKeyBench/0.1" -o "$OUT/$file.part" "$url" && mv "$OUT/$file.part" "$OUT/$file"
done <<< "$CLIPS"
