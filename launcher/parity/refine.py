"""Edge-refiner fixture: a native frame, the frame and CorridorKey matte at reduced resolution, and the reference
refiner's intermediates and outputs (fp32), as ``launcher/fixtures/refine.safetensors``.

uv run python launcher/parity/refine.py NATIVE.png LOW.png ALPHA.exr FG.exr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

REPO = Path(__file__).parents[2]
OUT = REPO / "launcher/fixtures/refine.safetensors"
REFINER = "mantelmoritz/CorridorKey-Realtime-EdgeRefiner"
REVISION = "75bb21da1ea3f22b57c7d16a28191e11eb43c2a6"
CROP = 4
KNEE = 0.008
DENOM = 8.29405


def image(path: str) -> torch.Tensor:
    probe = ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", path]
    w, h = map(int, subprocess.run(probe, capture_output=True, text=True, check=True).stdout.split(","))
    exr = path.endswith(".exr")
    fmt = "gbrpf32le" if exr else "rgb48le"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", fmt, "-"], capture_output=True, check=True).stdout
    if exr:
        g, b, r = np.frombuffer(raw, np.float32).reshape(3, h, w)
        x = np.stack([r, g, b])
    else:
        x = np.frombuffer(raw, np.uint16).reshape(h, w, 3).transpose(2, 0, 1).astype(np.float32) / 65535.0
    return torch.from_numpy(np.ascontiguousarray(x))[None]


def window(alpha: torch.Tensor, size: tuple[int, int]) -> tuple[int, int]:
    unc = ((alpha > 0.004) & (alpha < 0.996)).float()[0, 0]
    ys, xs = torch.nonzero(unc, as_tuple=True)
    h, w = unc.shape
    cy, cx = (int(ys.float().mean()), int(xs.float().mean())) if len(ys) else (h // 2, w // 2)
    y = min(max(cy - size[0] // 2, 0), h - size[0]) // CROP * CROP
    x = min(max(cx - size[1] // 2, 0), w - size[1]) // CROP * CROP
    return y, x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("native")
    ap.add_argument("low")
    ap.add_argument("alpha")
    ap.add_argument("fg")
    ap.add_argument("--low_size", type=int, nargs=2, default=(256, 384))
    a = ap.parse_args()

    root = Path(snapshot_download(REFINER, revision=REVISION, allow_patterns=["*.py", "*v10w.safetensors"]))
    sys.path.insert(0, str(root))
    import corridorkey_refiner as ref

    rgb, rgb_low = image(a.native), image(a.low)
    alpha_low = image(a.alpha)[:, :1]
    fg_low = ref.linear_to_srgb(image(a.fg)).clamp(0.0, 1.0)
    k = rgb.shape[-1] // rgb_low.shape[-1]
    assert rgb.shape[-2:] == (rgb_low.shape[-2] * k, rgb_low.shape[-1] * k), "native must be an integer multiple"
    (lh, lw), (y, x) = a.low_size, window(alpha_low, a.low_size)

    def low(t: torch.Tensor) -> torch.Tensor:
        return t[..., y : y + lh, x : x + lw].contiguous()

    rgb, rgb_low, alpha_low, fg_low = rgb[..., y * k : (y + lh) * k, x * k : (x + lw) * k].contiguous(), *map(low, (rgb_low, alpha_low, fg_low))

    net = ref.load_refiner(str(root / "corridorkey_edge_refiner_v10w.safetensors"), device="cpu", dtype=torch.float32)
    H, W = rgb.shape[-2:]

    def up(z: torch.Tensor) -> torch.Tensor:
        return F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)

    with torch.inference_mode():
        coef = ref.guided_coef(rgb_low, alpha_low)
        bg = ref.background_low(rgb_low, alpha_low)
        band = ref.refiner_band(alpha_low)
        a_gf = ((up(coef)[:, :3] * rgb).sum(1, keepdim=True) + up(coef)[:, 3:4]).clamp(0.0, 1.0)
        fg_in = (rgb + up(fg_low - rgb_low)).clamp(0.0, 1.0)
        alpha, fg = ref.refine(net, rgb, rgb_low, alpha_low, fg_low)
    def gbr(t: torch.Tensor) -> torch.Tensor:
        return t[:, [1, 2, 0]]
    z = ref.srgb_to_linear(fg_low) / KNEE
    tensors = {
        "native": gbr(rgb), "low": gbr(rgb_low), "logits": torch.cat([alpha_low, torch.asinh(z) / DENOM], 1),
        "coef": coef, "bg_low": bg, "band_low": band, "a_gf": a_gf, "fg_in": fg_in,
        "alpha": alpha, "fg": fg, "bilinear": up(alpha_low),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({n: t.float().contiguous() for n, t in tensors.items()}, OUT)
    print(f"{OUT}: native {W}x{H}, low {lw}x{lh} at ({x}, {y}), band {float((band > 0).float().mean()):.1%}")


if __name__ == "__main__":
    main()
