"""Fixtures for the launcher's parity tests on real footage: each stage's inputs, every block's input and output,
and the stage outputs, stock bf16 and fp32 reference, as ``launcher/fixtures/stage{1,2}_<tier>.safetensors``.

uv run python launcher/parity/dump.py [--tier 720] [--frames 4]
"""

import argparse
import functools
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).parents[2]
sys.path[:0] = [str(REPO), str(REPO / "bench")]

from harness import Feats, config, frame2, load, stage1, vfm  # ty: ignore[unresolved-import]

from corridorkey.runtime import RuntimePaths, load_model

OUT = REPO / "launcher/fixtures"
Tensors = dict[str, torch.Tensor]


def cpu(t: torch.Tensor) -> torch.Tensor:
    return t.float().cpu().contiguous()


def traced(
    named: list[tuple[str, torch.nn.Module]], fn: Callable[[], tuple[torch.Tensor, ...]]
) -> Tensors:
    """Each block's first input and output, plus ``fn``'s outputs ``out0``, ``out1``, ..."""
    seen: Tensors = {}

    def hook(name: str) -> Callable[..., None]:
        def grab(_m: torch.nn.Module, args: tuple, out: torch.Tensor) -> None:
            seen[f"{name}.in"], seen[f"{name}.out"] = cpu(args[0]), cpu(out)

        return grab

    hooks = [b.register_forward_hook(hook(n)) for n, b in named]
    seen |= {f"out{i}": cpu(t) for i, t in enumerate(fn())}
    for h in hooks:
        h.remove()
    return seen


def both(model: torch.nn.Module, named: list, fn: Callable[[], tuple]) -> Tensors:
    out: Tensors = {}
    for name in ("stock", "ref"):
        with config(model, name):
            out |= {f"{name}.{k}": v for k, v in traced(named, fn).items()}
    return out


def save(name: str, tensors: Tensors) -> None:
    OUT.mkdir(exist_ok=True)
    save_file(tensors, OUT / f"{name}.safetensors")
    print(OUT / f"{name}.safetensors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=720)
    ap.add_argument("--frames", type=int, default=4)
    a = ap.parse_args()
    model = load_model(RuntimePaths.from_package(REPO, None), bf16_weights=False).cuda()
    clip = load(min((REPO / "bench/frames").glob(f"*/{a.tier}p")), a.frames)
    x, display = clip.x.cuda().float(), clip.display.cuda()
    fm = model.foundations
    assert fm.cradio is not None and fm.cradio.radio_model is not None
    assert fm.moge is not None and fm.moge.encoder is not None
    assert fm.rvm is not None
    named0 = [

        (f"{tag}{i}", b)
        for tag, blocks in (
            ("cradio", fm.cradio.radio_model.model.blocks),
            ("moge", fm.moge.encoder.backbone.blocks),
            ("rvm", fm.rvm.encoder_layers),
        )
        for i, b in enumerate(blocks)
    ]
    feats = both(model, named0, lambda: tuple(vfm(model, display).values()))
    save(
        f"vfm_{a.tier}",
        {"display": cpu(display[0])}

        | {k: v for k, v in feats.items() if not k.endswith(".in")},
    )
    with config(model, "stock"):
        f: Feats = vfm(model, display)
        baton = stage1(model, x, f)[0][:, :1].contiguous()

    s1 = model.stage1
    named1 = [
        (f"{part}{i}", b)
        for part in ("enc_16", "body_64", "mid_16", "dec_16")
        for i, b in enumerate(getattr(s1, part))
    ]
    hint = torch.cat([x, torch.zeros_like(x[:, :, :1])], 2)
    inputs = {
        "x": cpu(hint.flatten(0, 1)),
        "quality": torch.full((x.shape[1],), -1.0),
    } | {k: cpu(v.flatten(0, 1)) for k, v in f.items()}
    save(
        f"stage1_{a.tier}",
        inputs | both(model, named1, functools.partial(stage1, model, x, f)),
    )

    s2 = model.stage2
    named2 = [(f"pre{i}", b) for i, b in enumerate(s2.pre_blocks)] + [
        (f"final{i}", b) for i, b in enumerate(s2.final_blocks)
    ]
    rgb = x[:, :1].contiguous()
    run2 = functools.partial(frame2, model, rgb, baton)
    save(
        f"stage2_{a.tier}",
        {"rgb": cpu(rgb[0]), "baton": cpu(baton[0])}
        | both(model, named2, lambda: (run2(),)),
    )


if __name__ == "__main__":
    main()
