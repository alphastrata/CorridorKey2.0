"""Write a reduced-precision copy of the CorridorKey v2 weights next to the original.

fp16: matrices and convolutions in fp16, 1-D tensors (biases, norms) kept fp32.

uv run python scripts/quantise.py [--dtype fp16] [--weights weights/corridorkey_v2.safetensors]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def fp16(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() < 2 else t.half()


def mib(p: Path) -> float:
    return p.stat().st_size / 2**20


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--weights", type=Path, default=ROOT / "weights/corridorkey_v2.safetensors"
    )
    ap.add_argument("--dtype", choices=["fp16"], default="fp16")
    a = ap.parse_args()
    out = a.weights.with_suffix(f".{a.dtype}.safetensors")
    with safe_open(str(a.weights), "pt") as f:
        meta, names = f.metadata(), f.keys()
        tensors = {n: fp16(f.get_tensor(n)).contiguous() for n in names}
    save_file(tensors, str(out), metadata=meta)
    print(f"{out}: {mib(out):.0f} MiB (from {mib(a.weights):.0f} MiB)")


if __name__ == "__main__":
    main()
