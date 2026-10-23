#!/usr/bin/env bash
# usage: bench.sh [clip] [frames] : fetch the clip, extract frames, then time master, stock, ours and ck on every GPU and tier
set -euo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
CLIP=${1:-pexels_9489598}
FRAMES=${2:-48}
"$R/scripts/download_sample_clips.sh" "$CLIP"
[ -d "$R/kernels/bench/frames/$CLIP" ] || (cd "$R" && uv run --no-sync python kernels/bench/prep.py)
"$R/kernels/bench/wall.sh" "$CLIP" "$FRAMES"
column -ts, "$R/kernels/bench/results/wall.csv"
