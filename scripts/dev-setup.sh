#!/usr/bin/env bash
# usage: dev-setup.sh : uv + Python deps, nightly Rust, just, ripgrep, typos, cargo-nextest, the ck launcher, weights and the sample clip
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
tool() { command -v "$1" > /dev/null || (cd "$R/launcher" && cargo install --locked "$2"); }
tool just just
tool rg ripgrep
tool typos typos-cli
tool cargo-nextest cargo-nextest
(cd "$R" && uv run python scripts/download_weights.py)
(cd "$R" && uv run python bench/clips.py)
echo "ready: just run <clip> -o out, or just bench"
