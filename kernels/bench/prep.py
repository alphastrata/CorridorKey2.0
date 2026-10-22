"""Extract ``FRAMES`` consecutive frames from the middle of each clip in ``footage/`` at every tier (short side;
never upscaled): ``frames/<clip>/<tier>p/NNN.png``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
FOOTAGE = ROOT / "footage"
OUT = ROOT / "frames"
FRAMES = 16
TIERS = (2160, 1080, 720)
VIDEO = {".mp4", ".webm", ".ogv", ".mov"}


def probe(clip: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json"]
        + ["-show_entries", "stream=width,height:format=duration", str(clip)],
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(out.stdout)
    s = meta["streams"][0]
    return s["width"], s["height"], float(meta["format"]["duration"])


def extract(clip: Path, tier: int, w: int, h: int, duration: float) -> None:
    dest = OUT / clip.stem / f"{tier}p"
    if len(list(dest.glob("*.png"))) == FRAMES:
        return
    dest.mkdir(parents=True, exist_ok=True)
    scale = f"scale=-2:{tier}" if h <= w else f"scale={tier}:-2"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{duration / 2:.3f}", "-i", str(clip)]
        + [
            "-frames:v",
            str(FRAMES),
            "-vf",
            f"{scale}:flags=lanczos",
            str(dest / "%03d.png"),
        ],
        check=True,
    )


def main() -> None:
    for clip in sorted(p for p in FOOTAGE.iterdir() if p.suffix in VIDEO):
        w, h, duration = probe(clip)
        for tier in (t for t in TIERS if t <= min(w, h)):
            extract(clip, tier, w, h, duration)
            print(clip.stem, f"{tier}p")


if __name__ == "__main__":
    main()
