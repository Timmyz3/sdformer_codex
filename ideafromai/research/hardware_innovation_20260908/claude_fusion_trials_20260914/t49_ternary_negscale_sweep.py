#!/usr/bin/env python3
"""T49-3: 三值 swap 的 `negative_threshold_scale` 零样本扫描（在 sd5ai 上跑）。

背景（T49-2 结果）：ep34 的二值网络直接换成三值，零样本 valid825 全崩
（AEE 1.19951 → 20.18936，FR 5.671% → 46.462%，spikes 72.89G → 597.21G）。
崩的机制是**负半轴**：`official_atlif` 输出 ∈ {0,+θ}，把 h ≤ −θ 的整个负尾
（约 40.8% 元素）直接丢掉；换成 `asymmetric_scale` 后 neg_thre = θ·neg_scale，
scale=1 时负尾全部开火 ⇒ 发放率爆炸。

本脚本只动 `negative_threshold_scale`（1/2/4/8），权重、注意力、优化器全部不动，
回答一个纯量化问题：**这个旋钮单独能把爆炸压回去多少，代价是多少精度？**
若大 scale 能把 FR 压回二值水平而 AEE 仍崩 ⇒ 说明三值的代价不是「阈值没调好」，
而是「权重里没学过负极性」⇒ 必须重训（该结论要写进报告）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
SRC_CFG = (REPO / "hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/"
           "dsec_c12_alpha0125_ep29_resume5_20260830.yml")
SRC_RUN = EXP / "results/dsec_c12_alpha0125_ep29_resume5_20260830"
OUT_ROOT = EXP / "results/t49_ternary_negscale"
GEN_CFGS = EXP / "configs/generated"
EPOCH = 34
PY = "/opt/conda/envs/sdformerflow/bin/python"


def tag_of(scale: float) -> str:
    return ("ns%g" % scale).replace(".", "p")


def build_config(scale: float) -> dict:
    cfg = yaml.safe_load(SRC_CFG.read_text(encoding="utf-8"))

    def flip(d: dict) -> None:
        d["output_mode"] = "ternary"
        d["threshold_mode"] = "asymmetric_scale"

    a = cfg["atlif_ternary_psn"]
    flip(a)
    for g in a.get("target_groups", ()) or ():
        flip(g)
    a["negative_threshold_scale"] = float(scale)
    cfg["experiment"] = "t49_ternary_" + tag_of(scale)
    cfg["note"] = ("T49-3 zero-shot negativity-budget sweep: ternary output, "
                   "asymmetric_scale, negative_threshold_scale=%g. No training." % scale)
    return cfg


def main() -> int:
    scales = [float(x) for x in sys.argv[1:]] or [1.0, 2.0, 4.0, 8.0]
    GEN_CFGS.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SDFORMER_USE_MLFLOW": "0", "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
           "SDFORMER_SNN_BACKEND": "cupy", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    for scale in scales:
        tag = tag_of(scale)
        out_cfg = GEN_CFGS / ("t49_ternary_%s.yml" % tag)
        out_cfg.write_text(yaml.safe_dump(build_config(scale), sort_keys=False), encoding="utf-8")

        run_dir = OUT_ROOT / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / ("checkpoint_epoch%d.pth" % EPOCH)
        if not ckpt.exists():
            shutil.copy2(SRC_RUN / ("checkpoint_epoch%d.pth" % EPOCH), ckpt)

        out_dir = run_dir / "standard_valid825" / ("epoch%d" % EPOCH)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [PY, "-u", "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
               "--config", str(out_cfg), "--checkpoint", str(ckpt),
               "--path_results", str(out_dir), "--mode", "valid"]
        print("[SWEEP] launching %s" % tag, flush=True)
        with open(out_dir / "eval.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print("[SWEEP] %s rc=%d" % (tag, rc), flush=True)

    print("[SWEEP] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
