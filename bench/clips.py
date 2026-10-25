"""The 4K green-screen sample clip (CC0) and its 720p/1080p/2160p frame sets: ``footage/sample/<tier>p/NNN.png``.

uv run python bench/clips.py [--frames 50]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path

FOOTAGE = Path(__file__).parent / "footage"
CLIP = FOOTAGE / "commons_4k_video_of_a_Chicken_fillet_roll___green_screen.webm"
URL = "https://upload.wikimedia.org/wikipedia/commons/8/8e/4k_video_of_a_Chicken_fillet_roll_-_green_screen_alan_butler_VP9.webm"
TIERS = (720, 1080, 2160)


def tier(t: int) -> Path:
    return FOOTAGE / "sample" / f"{t}p"


def fetch(frames: int) -> list[Path]:
    FOOTAGE.mkdir(parents=True, exist_ok=True)
    if not CLIP.exists() or CLIP.stat().st_size == 0:
        part = CLIP.with_suffix(".part")
        request = urllib.request.Request(
            URL, headers={"User-Agent": "CorridorKeyBench/0.1"}
        )
        with urllib.request.urlopen(request) as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        part.replace(CLIP)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found: only the 4K source is available")
        return []
    made = []
    for t in TIERS:
        d = tier(t)
        if d.is_dir() and len(list(d.iterdir())) == frames:
            continue
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                "5",
                "-i",
                str(CLIP),
                "-frames:v",
                str(frames),
            ]
            + ["-vf", f"scale=-2:{t}:flags=lanczos", str(d / "%03d.png")],
            check=True,
        )
        made.append(d)
    return made


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=50)
    for d in fetch(p.parse_args().frames):
        print(d)
