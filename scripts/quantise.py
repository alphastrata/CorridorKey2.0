"""Write a reduced-precision copy of the CorridorKey v2 weights next to the original.

Matrices and convolutions are cast; 1-D tensors (biases, norms) stay fp32.
bf16 suits Ampere and newer (ck computes in bf16 there); fp16 suits Turing.

uv run python scripts/quantise.py [--dtype bf16|fp16] [--weights weights/corridorkey_v2.safetensors]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t if t.dim() < 2 else t.to(dtype)


def mib(p: Path) -> float:
    return p.stat().st_size / 2**20


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--weights", type=Path, default=ROOT / "weights/corridorkey_v2.safetensors"
    )
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    a = ap.parse_args()
    out = a.weights.with_suffix(f".{a.dtype}.safetensors")
    with safe_open(str(a.weights), "pt") as f:
        meta, names = f.metadata(), f.keys()
        tensors = {
            n: cast(f.get_tensor(n), DTYPES[a.dtype]).contiguous() for n in names
        }
    save_file(tensors, str(out), metadata=meta)
    print(f"{out}: {mib(out):.0f} MiB (from {mib(a.weights):.0f} MiB)")


if __name__ == "__main__":
    main()
