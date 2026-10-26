"""Job settings and the command line they turn into: infer.py, or the Rust ck launcher.

The infer.py command matches what the terminal wizard (run_interactive.py) assembles,
plus the infer.py switches the wizard leaves at their defaults.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from run_interactive import DEFAULT_TEMP_DIR

from .clips import HINT_FIRST_FRAME, HINT_VIDEO, Clip, frame_files

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "infer.py"
ALL_OUTPUTS = ("alpha", "fg", "cutout", "checker")
BACKENDS = ("ck", "python-ptx", "stock")


@dataclass
class JobSettings:
    backend: str = "ck"              # ck | python-ptx | stock
    codec: str = "libx264"           # ck only: libx264 | hevc_nvenc
    outputs: list[str] = field(default_factory=lambda: list(ALL_OUTPUTS))
    output_format: str = "video"     # video | exr
    exr_dtype: str = "half"          # half | float
    cutout_linear: bool = False
    despill_strength: float = 0.5
    hint_quality: float = 0.75
    carry_hint: bool = True
    carry_hint_quality: float = 0.95
    frame_colour: str = "auto"       # auto (EXR linear, PNG/JPG sRGB) | linear | srgb
    frame_dir_fps: float = 24.0
    auto_vram: bool = True
    low_vram: bool = False
    hann_chunk: int = 48
    hann_stride: int = 24
    temp_dir: str = DEFAULT_TEMP_DIR
    num_frames: int = 0              # 0 = every frame
    start_mode: str = "begin"        # begin | middle | random_middle
    output_parent: str = ""          # "" = out_<clip> next to each clip

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "JobSettings":
        names = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in data.items() if k in names})
        except TypeError:
            return cls()


def frames_are_linear(clip: Clip, settings: JobSettings) -> bool:
    if settings.frame_colour != "auto":
        return settings.frame_colour == "linear"
    files = frame_files(clip.source)
    return bool(files) and files[0].suffix.lower() == ".exr"


def infer_args(clip: Clip, s: JobSettings, output_dir: Path) -> list[str]:
    args = [str(INFER), "--input", str(clip.source), "--output_dir", str(output_dir)]
    if clip.hint_path is not None and clip.hint_kind == HINT_FIRST_FRAME:
        args += ["--hint_first_frame", str(clip.hint_path)]
    elif clip.hint_path is not None and clip.hint_kind == HINT_VIDEO:
        args += ["--hint_video", str(clip.hint_path)]
    args += ["--hint_quality", f"{s.hint_quality:g}"]
    if s.carry_hint:
        args += ["--carry_hint_quality", f"{s.carry_hint_quality:g}"]
    else:
        args.append("--no_carry_hint")
    args += ["--outputs", *s.outputs, "--output_format", s.output_format]
    if s.output_format == "exr":
        args += ["--exr_dtype", s.exr_dtype]
    elif s.cutout_linear:
        args.append("--cutout_linear")
    args += ["--despill_strength", f"{s.despill_strength:g}"]
    if clip.is_frames:
        args += ["--frame_dir_fps", f"{s.frame_dir_fps:g}"]
        if frames_are_linear(clip, s):
            args.append("--frame_dir_linear")
    # With no memory flags, infer.py sizes the windows from free GPU memory.
    if not s.auto_vram:
        if s.low_vram:
            args.append("--low_vram")
        args += ["--hann_chunk", str(s.hann_chunk), "--hann_stride", str(s.hann_stride)]
    args += ["--temp_dir", s.temp_dir]
    if s.num_frames > 0:
        args += ["--num_frames", str(s.num_frames), "--start_mode", s.start_mode]
    return args


def ck_args(clip: Clip, s: JobSettings, output_dir: Path) -> list[str]:
    args = ["-o", str(output_dir), "--outputs", ",".join(s.outputs)]
    if clip.hint_path is not None and clip.hint_kind == HINT_FIRST_FRAME:
        args += ["--hint", str(clip.hint_path)]
    elif clip.hint_path is not None and clip.hint_kind == HINT_VIDEO:
        args += ["--hint-video", str(clip.hint_path)]
    args += ["--hint-quality", f"{s.hint_quality:g}"]
    if s.carry_hint:
        args += ["--carry-quality", f"{s.carry_hint_quality:g}"]
    else:
        args.append("--no-carry")
    if s.output_format == "exr":
        args += ["--exr", s.exr_dtype]
    else:
        args += ["--codec", s.codec]
        if s.cutout_linear:
            args.append("--cutout-linear")
    args += ["--despill", f"{s.despill_strength:g}"]
    if clip.is_frames:
        args += ["--fps", f"{s.frame_dir_fps:g}"]
        if frames_are_linear(clip, s):
            args.append("--linear")
    # ck sizes the window from free GPU memory unless told otherwise.
    if not s.auto_vram:
        args += ["--chunk", str(s.hann_chunk), "--stride", str(s.hann_stride)]
    if s.num_frames > 0:
        args += ["--frames", str(s.num_frames)]
    return [*args, str(clip.source)]


def ck_binary() -> Path | None:
    """ck from $CK_BIN, the PATH, launcher/target, or cargo's (possibly shared) target directory."""
    exe = "ck.exe" if sys.platform == "win32" else "ck"
    found = [os.environ.get("CK_BIN"), shutil.which("ck"), str(ROOT / "launcher/target/release" / exe)]
    if shutil.which("cargo"):
        meta = subprocess.run(
            ["cargo", "metadata", "-q", "--format-version", "1", "--no-deps",
             "--manifest-path", str(ROOT / "launcher/Cargo.toml")],
            capture_output=True, text=True, check=False)
        if meta.returncode == 0:
            found.append(str(Path(json.loads(meta.stdout)["target_directory"]) / "release" / exe))
    return next((Path(p) for p in found if p and Path(p).is_file()), None)


def backend_command(clip: Clip, s: JobSettings, output_dir: Path) -> tuple[str, list[str], dict[str, str]]:
    """Program, arguments and extra environment for the chosen backend."""
    if s.backend == "ck":
        ck = ck_binary()
        if ck is None:
            raise FileNotFoundError("ck is not built: run `cargo build --release -p ck` in launcher/")
        return str(ck), ck_args(clip, s, output_dir), {}
    ptx = "1" if s.backend == "python-ptx" else "0"
    return python_exe(), infer_args(clip, s, output_dir), {"CORRIDORKEY_PTX": ptx}


def python_exe() -> str:
    """The venv's console interpreter. infer.py streams its progress on stdout,
    so it runs under python.exe even when the UI was started with pythonw.exe."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").is_file():
        return str(exe.with_name("python.exe"))
    return str(exe)


def command_line(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv)
