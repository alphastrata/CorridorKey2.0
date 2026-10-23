#!/usr/bin/env bash
# usage: download_sample_clips.sh [name…] : green-screen sample clips into kernels/bench/footage (all by default)
set -euo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT=$R/kernels/bench/footage
C=https://upload.wikimedia.org/wikipedia/commons
P=https://www.pexels.com/download/video
CLIPS="pexels_9489598.mp4	$P/9489598/	Pexels License
pexels_9489384.mp4	$P/9489384/	Pexels License
pexels_9488366.mp4	$P/9488366/	Pexels License
pexels_29985923.mp4	$P/29985923/	Pexels License
pexels_9075217.mp4	$P/9075217/	Pexels License
pexels_7670822.mp4	$P/7670822/	Pexels License
commons_Chroma_key_demonstration.ogv	$C/2/20/Chroma_key_demonstration.ogv	Public domain
commons_Filming_with_the_green_screen.webm	$C/d/d8/Filming_with_the_green_screen.webm	Public domain
commons_Lydia_Owen___Yuri.webm	$C/0/01/Lydia_Owen_-_Yuri.webm	CC BY 3.0
commons_Pelo_Does___The_Green_Screen.webm	$C/0/0a/Pelo_Does_-_The_Green_Screen.webm	CC BY 3.0
commons_The_Pelones___Green_Screen.webm	$C/a/a1/The_Pelones_-_Green_Screen.webm	CC BY 3.0
commons_We_Made_a_Bunch_of_Green_Screen_Footage_.webm	$C/5/51/We_Made_a_Bunch_of_Green_Screen_Footage%21.webm	CC BY 3.0
commons_4k_video_of_a_Chicken_fillet_roll___green_screen.webm	$C/8/8e/4k_video_of_a_Chicken_fillet_roll_-_green_screen_alan_butler_VP9.webm	CC0"

mkdir -p "$OUT"
echo "$CLIPS" > "$OUT/SOURCES.tsv"
while IFS=$'\t' read -r file url licence; do
  [ $# -gt 0 ] && [[ ! " $* " =~ " ${file%.*} " ]] && continue
  [ -s "$OUT/$file" ] && { echo "have $file"; continue; }
  echo "fetching $file [$licence]"
  curl -fL --retry 3 -A "CorridorKeyBench/0.1" -o "$OUT/$file.part" "$url" && mv "$OUT/$file.part" "$OUT/$file"
done <<< "$CLIPS"
