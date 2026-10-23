#!/usr/bin/env bash
# usage: download_sample_clips.sh [frames=50] : the 4K green-screen clip (CC0), plus 720p/1080p/2160p frame sets of it
set -euo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT=$R/bench/footage
FRAMES=${1:-50}
FILE=$OUT/commons_4k_video_of_a_Chicken_fillet_roll___green_screen.webm
URL=https://upload.wikimedia.org/wikipedia/commons/8/8e/4k_video_of_a_Chicken_fillet_roll_-_green_screen_alan_butler_VP9.webm
mkdir -p "$OUT"
[ -s "$FILE" ] || { curl -fsSL --retry 3 -A "CorridorKeyBench/0.1" -o "$FILE.part" "$URL" && mv "$FILE.part" "$FILE"; }
command -v ffmpeg > /dev/null || { echo "ffmpeg not found: only the 4K source is available"; exit 0; }
for tier in 720 1080 2160; do
  d=$OUT/sample/${tier}p
  [ "$(ls "$d" 2>/dev/null | wc -l)" -eq "$FRAMES" ] && continue
  rm -rf "$d" && mkdir -p "$d"
  ffmpeg -v error -ss 5 -i "$FILE" -frames:v "$FRAMES" -vf "scale=-2:$tier:flags=lanczos" "$d/%03d.png"
  echo "$d"
done
