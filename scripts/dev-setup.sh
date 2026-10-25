#!/usr/bin/env bash
# usage: dev-setup.sh : uv + Python deps, just, weights and the sample clip
set -euo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
need() { command -v "$1" > /dev/null || { echo "missing $1: $2" >&2; exit 1; }; }

need nvidia-smi "install the NVIDIA driver"
need ffmpeg "install ffmpeg (decode, encode and the sample frame sets)"
need curl "install curl"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

(cd "$R" && uv sync)
command -v just > /dev/null || uv tool install rust-just
(cd "$R" && uv run python scripts/download_weights.py)
(cd "$R" && uv run python bench/clips.py)
echo "ready: just bench"
