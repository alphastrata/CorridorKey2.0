#!/usr/bin/env bash
# usage: dev-setup.sh : uv + Python deps, nightly Rust, just, cargo-ptx, the ck launcher, weights and the sample clip
set -euo pipefail
R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
need() { command -v "$1" > /dev/null || { echo "missing $1: $2" >&2; exit 1; }; }

need nvidia-smi "install the NVIDIA driver"
need ffmpeg "install ffmpeg (decode, encode and the sample frame sets)"
need curl "install curl"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v cargo > /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

(cd "$R" && uv sync)
(cd "$R/launcher" && rustup toolchain install --no-self-update && cargo build --release -p ck)
(cd "$R/kernels" && rustup toolchain install --no-self-update)
command -v just > /dev/null || cargo install --locked just
command -v cargo-ptx > /dev/null || (cd "$R/kernels" && cargo install --locked --git https://codeberg.org/P-E-T-R-A/cargo-ptx.git --branch develop)
(cd "$R" && uv run python scripts/download_weights.py)
"$R/scripts/download_sample_clips.sh"
echo "ready: just run <clip> -o out, or scripts/bench.sh"
