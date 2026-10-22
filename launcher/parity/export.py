"""The VFMs' weights and feature stats, f32, as ``weights/vfms.safetensors`` under ``cradio.``, ``moge.``, ``rvm.``
and ``stats.``, for the Rust launcher.

uv run python launcher/parity/export.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).parents[2]
sys.path[:0] = [str(REPO)]

from corridorkey.runtime import RuntimePaths, load_model


def prefixed(prefix: str, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"{prefix}.{k}": v.float().cpu().contiguous() for k, v in sd.items()}


def main() -> None:
    f = load_model(RuntimePaths.from_package(REPO, None)).foundations
    stats = {k: getattr(f, k) for k in ("cradio_mean", "cradio_std", "moge_mean", "moge_std", "rvm_mean", "rvm_std")}
    out = (
        prefixed("cradio", f.cradio.radio_model.state_dict())
        | prefixed("moge", f.moge.encoder.state_dict())
        | prefixed("rvm", f.rvm.state_dict() | {"base_posenc": f.rvm.base_posenc})
        | prefixed("stats", stats)
    )
    path = REPO / "weights/vfms.safetensors"
    save_file(out, path)
    print(path, len(out))


if __name__ == "__main__":
    main()
