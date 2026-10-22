"""Fixtures for the launcher's parity tests: real-footage inputs and every stage-2 block's input/output, stock bf16
and fp32 reference, as ``launcher/fixtures/<name>.safetensors``.

uv run python launcher/parity/dump.py [--tier 720] [--frames 4]
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).parents[2]
sys.path[:0] = [str(REPO), str(REPO / "kernels/bench")]

from harness import config, frame2, load, stage1, vfm

from corridorkey.runtime import RuntimePaths, load_model

OUT = REPO / "launcher/fixtures"


def blocks(model: torch.nn.Module, fn) -> dict[str, torch.Tensor]:
    """Inputs and outputs of each stage-2 block over one call of ``fn``."""
    seen: dict[str, torch.Tensor] = {}
    s2 = model.stage2
    named = [(f"pre{i}", b) for i, b in enumerate(s2.pre_blocks)] + [
        (f"final{i}", b) for i, b in enumerate(s2.final_blocks)
    ]

    def hook(name: str):
        def grab(_m: torch.nn.Module, args: tuple, out: torch.Tensor) -> None:
            seen[f"{name}.in"] = args[0].float().cpu().contiguous()
            seen[f"{name}.out"] = out.float().cpu().contiguous()

        return grab

    hooks = [b.register_forward_hook(hook(n)) for n, b in named]
    seen["logits"] = fn().float().cpu().contiguous()
    for h in hooks:
        h.remove()
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=720)
    ap.add_argument("--frames", type=int, default=4)
    a = ap.parse_args()
    model = load_model(RuntimePaths.from_package(REPO, None)).cuda()
    clip = load(min((REPO / "kernels/bench/frames").glob(f"*/{a.tier}p")), a.frames)
    x, display = clip.x.cuda(), clip.display.cuda()
    with config(model, "stock"):
        baton = stage1(model, x, vfm(model, display))[0][:, :1].contiguous()
    rgb = x[:, :1].contiguous()
    out = {"rgb": rgb[0].float().cpu(), "baton": baton[0].cpu()}
    for name in ("stock", "ref"):
        with config(model, name):
            got = blocks(model, functools.partial(frame2, model, rgb, baton))
        out |= {f"{name}.{k}": v for k, v in got.items()}
    OUT.mkdir(exist_ok=True)
    path = OUT / f"stage2_{a.tier}.safetensors"
    save_file({k: v.contiguous() for k, v in out.items()}, path)
    print(
        path,
        {
            k: tuple(v.shape)
            for k, v in out.items()
            if "." not in k or k.endswith("logits")
        },
    )


if __name__ == "__main__":
    main()
