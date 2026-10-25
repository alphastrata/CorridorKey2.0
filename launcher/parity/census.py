"""Aten ops and leaf modules each part runs (stock, bf16 autocast): the port's checklist.

uv run python launcher/parity/census.py [--hw 256 384] [--frames 2]
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

REPO = Path(__file__).parents[2]
sys.path[:0] = [str(REPO), str(REPO / "bench")]

from harness import config, frame2, stage1, vfm

from corridorkey.runtime import RuntimePaths, load_model


class Ops(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.seen: collections.Counter[str] = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen[str(func.overloadpacket)] += 1
        return func(*args, **kwargs or {})


def leaves(m: torch.nn.Module) -> collections.Counter[str]:
    hit: collections.Counter[str] = collections.Counter()
    hooks = [
        s.register_forward_hook(lambda mod, *_: hit.update([type(mod).__name__]))
        for s in m.modules()
    ]
    return hit, hooks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", type=int, nargs=2, default=[256, 384])
    ap.add_argument("--frames", type=int, default=2)
    a = ap.parse_args()
    model = load_model(RuntimePaths.from_package(REPO, None)).cuda()
    h, w = a.hw
    x = torch.rand(1, a.frames, 3, h, w, device="cuda")
    with config(model, "stock"):
        f = vfm(model, x)
        baton = stage1(model, x, f)[0][:, :1]
        parts = {
            "vfm": (model.foundations, lambda: vfm(model, x)),
            "stage1": (model.stage1, lambda: stage1(model, x, f)),
            "stage2": (model.stage2, lambda: frame2(model, x[:, :1], baton)),
        }
        for name, (mod, fn) in parts.items():
            hit, hooks = leaves(mod)
            with Ops() as ops:
                fn()
            for hk in hooks:
                hk.remove()
            print(
                f"## {name}\nmodules: {dict(hit.most_common())}\nops: {dict(ops.seen.most_common())}\n"
            )


if __name__ == "__main__":
    main()
