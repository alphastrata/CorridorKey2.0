"""Everything worth knowing about a run, for the stats-for-nerds overlay: the job as
the runner reports it, every GPU through nvidia-smi, and the job's process tree."""
from __future__ import annotations

import re
import shutil
import time

import psutil
from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from .clips import RUNNING, Clip
from .runner import QueueRunner

_GPU_FIELDS = ("index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,"
               "temperature.gpu,clocks.sm,clocks.max.sm,pcie.link.gen.current,"
               "pcie.link.width.current,encoder.stats.sessionCount,encoder.stats.averageFps")
_PROBED = re.compile(r"probed (\d+) frames .*@ (\d+)x(\d+) \(native (\d+)x(\d+)")
_CK_CLIP = re.compile(r"(\d+) frames (\d+)x(\d+) → (\d+)x(\d+), window (\d+), stride (\d+)")
_INFER_WINDOW = re.compile(r"Hann stage1: chunk=(\d+), stride=(\d+), windows=(\d+)")
_WINDOW = re.compile(r"\bwindow (\d+)/(\d+)")
_KEYED = re.compile(r"\b(\d+) frames through stage 2")


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes // 60}:{minutes % 60:02d}:{secs:02d}" if minutes >= 60 else f"{minutes}:{secs:02d}"


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


class Telemetry(QObject):
    changed = Signal(str)

    def __init__(self, runner: QueueRunner, parent=None, interval_ms: int = 1000):
        super().__init__(parent)
        self._runner = runner
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._smi: QProcess | None = None
        self._apps: QProcess | None = None
        self._gpus: list[list[str]] = []
        self._vram: dict[int, float] = {}
        self._procs: dict[int, psutil.Process] = {}
        runner.clip_started.connect(self._on_started)
        runner.clip_changed.connect(self._on_changed)
        runner.log.connect(self._on_line)
        self._reset(None)

    def start(self) -> None:
        self._timer.start()
        self._tick()

    def stop(self) -> None:
        self._timer.stop()

    def _reset(self, clip: Clip | None) -> None:
        self._clip = clip
        self._started = time.monotonic()
        self._ended: float | None = None
        self._phases: list[tuple[str, float]] = [("Starting", self._started)]
        self._frames = 0
        self._size = ""
        self._window = ""
        self._at = (0, 0)
        self._keyed = 0
        self._keying_since: float | None = None
        self._last = ""

    def _on_started(self, clip: Clip) -> None:
        self._reset(clip)

    def _on_changed(self, clip: Clip) -> None:
        if clip is not self._clip:
            return
        phase = "Keying" if clip.detail[:1].isdigit() else re.sub(r"\s*\d.*$", "", clip.detail)
        if not self._phases or self._phases[-1][0] != phase:
            self._phases.append((phase, time.monotonic()))
        if phase == "Keying" and self._keying_since is None:
            self._keying_since = time.monotonic()
        if clip.status != RUNNING and self._ended is None:
            self._ended = time.monotonic()
        self._render()

    def _on_line(self, line: str) -> None:
        text = line.strip()
        if text.startswith("==="):
            return
        if (m := _CK_CLIP.search(text)) is not None:
            self._frames = int(m[1])
            self._size = f"{m[2]}x{m[3]} → model {m[4]}x{m[5]}"
            self._window = f"chunk {m[6]} · stride {m[7]}"
        elif (m := _PROBED.search(text)) is not None:
            self._frames, self._size = int(m[1]), f"{m[4]}x{m[5]} → model {m[2]}x{m[3]}"
        elif (m := _INFER_WINDOW.search(text)) is not None:
            self._window = f"chunk {m[1]} · stride {m[2]} · {m[3]} windows"
        if (m := _WINDOW.search(text)) is not None:
            self._at = (int(m[1]), int(m[2]))
        if (m := _KEYED.search(text)) is not None:
            self._keyed = max(self._keyed, int(m[1]))
        if text:
            self._last = text[-110:]

    def _tick(self) -> None:
        self._poll_gpus()
        self._render()

    def _poll_gpus(self) -> None:
        if not shutil.which("nvidia-smi") or self._smi is not None:
            return
        self._smi = self._query(["--query-gpu=" + _GPU_FIELDS], self._on_gpus)
        self._apps = self._query(["--query-compute-apps=pid,gpu_uuid,used_memory"], self._on_apps)

    def _query(self, args: list[str], done) -> QProcess:
        proc = QProcess(self)
        proc.finished.connect(lambda *_: done(proc))
        proc.start("nvidia-smi", [*args, "--format=csv,noheader,nounits"])
        return proc

    def _rows(self, proc: QProcess) -> list[list[str]]:
        text = proc.readAllStandardOutput().data().decode(errors="replace")
        proc.deleteLater()
        return [[c.strip() for c in line.split(",")] for line in text.splitlines() if line.strip()]

    def _on_gpus(self, proc: QProcess) -> None:
        self._gpus, self._smi = self._rows(proc), None

    def _on_apps(self, proc: QProcess) -> None:
        mine = {p.pid for p in self._tree()}
        uuids = {row[1]: int(row[0]) for row in self._gpus if len(row) > 1}
        vram: dict[int, float] = {}
        for pid, uuid, used in (r for r in self._rows(proc) if len(r) == 3):
            if pid.isdigit() and int(pid) in mine and uuid in uuids and (mb := _num(used)) is not None:
                vram[uuids[uuid]] = vram.get(uuids[uuid], 0.0) + mb
        self._vram, self._apps = vram, None

    def _tree(self) -> list[psutil.Process]:
        pid = self._runner.pid()
        if not pid:
            return []
        try:
            root = self._procs.get(pid) or psutil.Process(pid)
            tree = [root, *root.children(recursive=True)]
        except psutil.Error:
            return []
        self._procs = {p.pid: self._procs.get(p.pid, p) for p in tree}
        return [self._procs[p.pid] for p in tree]

    def _process_line(self) -> str:
        tree = self._tree()
        if not tree:
            return "idle"
        rss = cpu = threads = 0.0
        names: dict[str, int] = {}
        for p in tree:
            try:
                with p.oneshot():
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent()
                    threads += p.num_threads()
                    if p is not tree[0]:
                        names[p.name()] = names.get(p.name(), 0) + 1
            except psutil.Error:
                continue
        kids = ", ".join(f"{n}×{c}" if c > 1 else n for n, c in names.items()) or "none"
        return (f"pid {tree[0].pid} · RSS {rss / 2**30:.1f} GB · CPU {cpu:.0f}% · "
                f"{threads:.0f} threads · children {kids}")

    def _gpu_lines(self) -> list[str]:
        out = []
        if not self._runner.pid():
            self._vram = {}
        for row in self._gpus:
            if len(row) < 15:
                continue
            idx, _, name, util, used, total, watts, cap, temp, sm, sm_max, gen, width, enc, enc_fps = row
            mine = self._vram.get(int(idx)) if idx.isdigit() else None
            job = f" (job {mine / 1024:.1f})" if mine else ""
            nvenc = f" · NVENC {enc} @ {enc_fps} fps" if enc not in ("0", "[N/A]") else ""
            out.append(f"GPU{idx} {name.replace('NVIDIA ', '').replace('GeForce ', '')}")
            out.append(f"  util {util}% · mem {float(used) / 1024:.1f}/{float(total) / 1024:.1f} GB{job} · "
                       f"{_num(watts) or 0:.0f}/{_num(cap) or 0:.0f} W · {temp}°C · "
                       f"SM {sm}/{sm_max} MHz · PCIe {gen}×{width}{nvenc}")
        return out

    def _render(self) -> None:
        s = self._runner.settings()
        clip = self._clip
        now = self._ended or time.monotonic()
        lines = ["Stats for nerds", ""]
        lines.append(f"backend    {s.backend}" + (f" · {self._runner.program()}" if self._runner.program() else ""))
        if clip is not None:
            lines.append(f"clip       {clip.name} · {clip.status}"
                         + (f" · {self._frames} frames {self._size}" if self._frames else ""))
            if self._window:
                at = f"window {self._at[0]}/{self._at[1]} · " if self._at[1] else ""
                lines.append(f"window     {at}{self._window}")
            lines.append(f"progress   {clip.progress:.0%} · {clip.detail}")
            phases: dict[str, float] = {}
            for i, (name, t) in enumerate(self._phases):
                end = self._phases[i + 1][1] if i + 1 < len(self._phases) else now
                if name and name not in ("Done", "Failed", "Stopped"):
                    phases[name] = phases.get(name, 0.0) + end - t
            lines.append(f"elapsed    {_clock(now - self._started)} · "
                         + " · ".join(f"{n.lower()} {d:.1f}s" for n, d in phases.items()))
            if self._keying_since is not None and self._frames:
                spent = max(now - self._keying_since, 1e-3)
                keyed = self._keyed or (self._frames * self._at[0] // self._at[1] if self._at[1] else 0)
                keyed = self._frames if clip.status == "Done" else keyed
                overall = (f" · {self._frames / max(now - self._started, 1e-3):.2f} fps overall"
                           if self._ended is not None else "")
                lines.append(f"throughput {keyed} frames keyed · {keyed / spent:.2f} fps{overall}")
            lines.append(f"process    {self._process_line()}")
        else:
            lines.append("clip       none yet · start the queue")
        lines += self._gpu_lines()
        if self._last:
            lines.append(f"last       {self._last}")
        self.changed.emit("\n".join(lines))
