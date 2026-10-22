"""Stage 1 and stage 2 per-stage timings and peak memory; real weights when the checkpoint is complete, else random."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey import ptx
from corridorkey.stage1 import Stage1
from corridorkey.stage2 import Stage2

VFM = {"cradio_dim": 1152, "moge_dim": 768, "rvm_dim": 384}


def config(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(json.loads(f.read(n))["__metadata__"]["model_config"])


def stages(path: Path) -> tuple[Stage1, Stage2]:
    cfg = config(path)
    s1 = Stage1(
        d_16=cfg["d_16"],
        d_64=cfg["d_64"],
        **VFM,
        n_enc_16=cfg["n_enc_16"],
        n_body_64=cfg["n_body_64"],
        n_dec_16=cfg["n_dec_16"],
        num_heads_16=cfg["num_heads_16"],
        num_heads_64=cfg["num_heads_64"],
        window_HW=cfg["window_HW"],
        mlp_mult=cfg["mlp_mult"],
        local_mix_kernel=cfg["stage1_local_mix_kernel"],
        body64_local_mix_kernel=cfg["stage1_body64_local_mix_kernel"],
        recurrent_64_sites=(),
        hint_adaln_blocks=cfg["stage1_hint_adaln_blocks"],
        mid_16_blocks=cfg["stage1_mid_16_blocks"],
        mlp_dw_kernel=cfg["mlp_dw_kernel"],
        qkv_dw_kernel=cfg["qkv_dw_kernel"],
        qkv_dw_fraction=cfg["qkv_dw_fraction"],
        gradient_checkpoint=False,
        overlap_mult=cfg["overlap_mult"],
        bottleneck_head_stride=cfg["bottleneck_head_stride"],
        rope_base=cfg["rope_base"],
    )
    s2 = Stage2(
        d=cfg["d_stage2"],
        d_baton=cfg["d_16"],
        num_heads=cfg["num_heads_stage2"],
        num_blocks=cfg["n_blocks_stage2"],
        num_final_blocks=cfg["n_final_blocks_stage2"],
        window_size=cfg["window_size_stage2"],
        mlp_mult=cfg["mlp_mult"],
        overlap_mult=cfg["overlap_mult"],
        local_mix_kernel=cfg["stage2_local_mix_kernel"],
        mlp_dw_kernel=cfg["mlp_dw_kernel"],
        qkv_dw_kernel=cfg["qkv_dw_kernel"],
        qkv_dw_fraction=cfg["qkv_dw_fraction"],
        rope_base=cfg["rope_base"],
    )
    if path.suffix == ".safetensors":
        sd = load_file(str(path))
        for name, m in (("stage1", s1), ("stage2", s2)):
            m.load_state_dict(
                {
                    k[len(name) + 1 :]: v
                    for k, v in sd.items()
                    if k.startswith(name + ".")
                }
            )
    return s1.cuda().eval(), s2.cuda().eval()


def ms(fn: object, reps: int = 3) -> float:
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--hw", default="1088x1920")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--ck", action="store_true")
    a = p.parse_args()
    h, w = map(int, a.hw.split("x"))
    s1, s2 = stages(a.checkpoint)
    if a.ck:
        print(f"ck patched {ptx.enable(s1) + ptx.enable(s2)} modules")
    t = a.frames
    rgb = torch.randn(1, t, 3, h, w, device="cuda")
    hint = torch.zeros(1, t, 1, h, w, device="cuda")
    feats = {
        k[:-4] + "_feat": torch.randn(1, t, c, h // 64, w // 64, device="cuda")
        for k, c in VFM.items()
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        run1 = lambda: s1(
            rgb,
            hint,
            feats["moge_feat"],
            cradio_feat=feats["cradio_feat"],
            rvm_feat=feats["rvm_feat"],
        )
        baton = run1()["baton"]
        run2 = lambda: s2(rgb[:, :1], baton[:, :1])
        if a.nvtx:
            for name, fn in (("stage1", run1), ("stage2", run2)):
                fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(name)
                fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        else:
            print(
                f"{h}x{w} T={t}: stage1 {ms(run1):.1f} ms ({ms(run1) / t:.1f}/frame), stage2 {ms(run2):.1f} ms/frame"
            )
            print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
