"""Load Rust-compiled PTX (kernels/) through the CUDA driver API and launch on torch's current stream."""

from __future__ import annotations

import ctypes
import functools
import os
import sys
from pathlib import Path

import torch

BIN = Path(__file__).parent / "bin"
Arg = torch.Tensor | int | float
MAX_DYNAMIC_SHARED = 8


@functools.cache
def _driver() -> ctypes.CDLL | None:
    name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        return ctypes.CDLL(name)
    except OSError:
        return None


def available() -> bool:
    return (
        os.environ.get("CORRIDORKEY_PTX", "1") != "0"
        and torch.cuda.is_available()
        and not torch.version.hip
        and torch.cuda.get_device_capability() >= (7, 5)
        and _driver() is not None
    )


def _check(code: int) -> None:
    if code != 0:
        raise RuntimeError(f"CUDA driver error {code}")


class Module:
    def __init__(self, ptx: bytes) -> None:
        drv = _driver()
        assert drv is not None, "CUDA driver not available"
        torch.cuda.init()
        self._mod = ctypes.c_void_p()
        _check(
            drv.cuModuleLoadData(ctypes.byref(self._mod), ctypes.c_char_p(ptx + b"\0"))
        )
        self._fns: dict[str, ctypes.c_void_p] = {}
        self._shared: dict[str, int] = {}

    def _fn(self, name: str) -> ctypes.c_void_p:
        if name not in self._fns:
            drv = _driver()
            assert drv is not None, "CUDA driver not available"
            fn = ctypes.c_void_p()
            _check(
                drv.cuModuleGetFunction(ctypes.byref(fn), self._mod, name.encode())
            )
            self._fns[name] = fn
        return self._fns[name]

    def launch(
        self,
        name: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[Arg],
        shared: int = 0,
    ) -> None:
        drv = _driver()
        assert drv is not None, "CUDA driver not available"

        if shared > self._shared.get(name, 48 * 1024):
            _check(
                drv.cuFuncSetAttribute(self._fn(name), MAX_DYNAMIC_SHARED, shared)
            )
            self._shared[name] = shared

        params = [p for a in args for p in _lower(a)]
        argv = (ctypes.c_void_p * len(params))(
            *(ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in params)
        )

        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(
            drv.cuLaunchKernel(
                self._fn(name), *grid, *block, shared, stream, argv, None
            )
        )


def _lower(a: Arg) -> list[ctypes._SimpleCData]:
    if isinstance(a, torch.Tensor):
        assert a.is_cuda and a.is_contiguous(), (
            "PTX kernels take contiguous CUDA tensors"
        )
        return [ctypes.c_void_p(a.data_ptr()), ctypes.c_size_t(a.numel())]
    if isinstance(a, float):
        return [ctypes.c_float(a)]
    return [ctypes.c_uint32(a)]


def best(crate: str, capability: tuple[int, int]) -> Path:
    """``bin/sm_XX/<crate>.ptx`` for the highest built ``XX`` the device can run."""
    cap = capability[0] * 10 + capability[1]
    built = [
        (int(d.name[3:]), d / f"{crate}.ptx")
        for d in BIN.glob("sm_*")
        if (d / f"{crate}.ptx").exists()
    ]
    fits = [(sm, p) for sm, p in built if sm <= cap]
    if not fits:
        raise RuntimeError(f"no {crate} PTX for sm_{cap} in {BIN}")
    return max(fits)[1]


def arch(crate: str, device: int) -> int:
    """The ``sm_XX`` that :func:`load` picks for ``crate`` on ``device``."""
    return int(best(crate, torch.cuda.get_device_capability(device)).parent.name[3:])


@functools.cache
def load(crate: str, device: int) -> Module:
    path = best(crate, torch.cuda.get_device_capability(device))
    with torch.cuda.device(device):
        return Module(path.read_bytes())


def enable(model: torch.nn.Module, on: bool = True) -> int:
    """Route ``model``'s depthwise branches and ``corridorkey.blocks``' attention through ck, or back to
    stock; returns how many modules were patched."""
    from . import attn, block, dw, glue

    if not on:
        dw.unpatch(model)
        glue.unpatch(model)
        block.unpatch(model)
        attn.unpatch()
        return 0
    return dw.patch(model) + glue.patch(model) + block.patch(model) + int(attn.patch())
