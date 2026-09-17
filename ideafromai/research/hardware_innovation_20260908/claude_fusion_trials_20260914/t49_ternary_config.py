#!/usr/bin/env python3
"""T49-2: 造一份「把 ATLIF 换成真三值」的 config，并在 ep34 checkpoint 上做 zero-shot eval。

动机（见 T49 诊断）：
  - ep34 的 ATLIF 实际是**固定阈值二值 IF**（official_atlif ⇒ 输出 ∈ {0,+θ}，θ≡1），
    自适应机制被 threshold_eta=0 关掉、残存梯度 ~1e-13 数值上死掉。
  - 而 h60 注意力的 `_ternary_alpha_xnor_token_scores` **本就按有符号三值写的**
    （`_ternary_sign_ste`），喂进去的非负二值让 `opposite` 项恒为空集。
  ⇒ 直接把 output_mode 换 ternary，看零样本会怎样。这一步不训练。

改法（两处必须同时改，否则 __init__ 直接 raise）：
  output_mode: binary → ternary
  threshold_mode: official_atlif → asymmetric_scale   （负阈值 = thre × neg_scale）
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
SRC_CFG = Path("/root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07/"
               "system_handoff/incoming/m2041_ep34_quant_binding_inputs/"
               "dsec_c12_alpha0125_ep29_resume5_20260830.yml")
SRC_RUN = EXP / "results/dsec_c12_alpha0125_ep29_resume5_20260830"
OUT_CFG = EXP / "configs/generated/t49_ternary_zeroshot_ep34.yml"
OUT_RUN = EXP / "results/t49_ternary_zeroshot_ep34"
EPOCH = 34


def build_config() -> dict:
    cfg = yaml.safe_load(SRC_CFG.read_text(encoding="utf-8"))

    def flip(d: dict) -> None:
        d["output_mode"] = "ternary"
        d["threshold_mode"] = "asymmetric_scale"

    a = cfg["atlif_ternary_psn"]
    flip(a)
    for g in a.get("target_groups", ()) or ():
        flip(g)
    # 负阈值 = thre × negative_threshold_scale。ep34 是 1.0 ⇒ 对称 ±θ。
    a["negative_threshold_scale"] = float(a.get("negative_threshold_scale", 1.0) or 1.0)
    cfg["experiment"] = "t49_ternary_zeroshot_ep34"
    cfg["note"] = ("T49-2 zero-shot semantics swap: ATLIFTernaryPSN output_mode "
                   "binary->ternary, threshold_mode official_atlif->asymmetric_scale. "
                   "No training; weights are the binary-trained ep34 checkpoint.")
    return cfg


def main() -> int:
    cfg = build_config()
    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CFG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("wrote", OUT_CFG)
    a = cfg["atlif_ternary_psn"]
    print("atlif:", json.dumps({k: a[k] for k in
                                ("enabled", "output_mode", "threshold_mode",
                                 "negative_threshold_scale", "threshold_eta",
                                 "threshold_init", "min_threshold", "max_threshold")},
                               indent=1))
    print("target_groups:", [(g.get("name"), g.get("output_mode"), g.get("threshold_mode"))
                             for g in a.get("target_groups", ())])

    OUT_RUN.mkdir(parents=True, exist_ok=True)
    for ep in (30, 32, 34):
        src = SRC_RUN / f"checkpoint_epoch{ep}.pth"
        dst = OUT_RUN / f"checkpoint_epoch{ep}.pth"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    out_dir = OUT_RUN / "standard_valid825" / f"epoch{EPOCH}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cmd = [
        "/opt/conda/envs/sdformerflow/bin/python", "-u",
        "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
        "--config", str(OUT_CFG),
        "--checkpoint", str(OUT_RUN / f"checkpoint_epoch{EPOCH}.pth"),
        "--path_results", str(out_dir),
        "--mode", "valid",
    ]
    print("launching:", " ".join(env_cmd))
    log = open(out_dir / "eval.log", "w", encoding="utf-8")
    proc = subprocess.Popen(env_cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ,
                                 "SDFORMER_USE_MLFLOW": "0",
                                 "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
                                 "SDFORMER_SNN_BACKEND": "cupy",
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    print("pid", proc.pid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
