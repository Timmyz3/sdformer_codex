#!/usr/bin/env python3
"""T49-4: 真·三值重训（warm start from ep34 binary checkpoint，fresh optimizer，5 epoch）。

为什么这么起（而不是照 config 里写的 "reload from a baseline checkpoint"）：
  T49-6 的零样本实测已经证明，**H9 的装载顺序是「先按 config 装 overlay，再把
  checkpoint 的 state_dict 灌进去（strict=False）」**——
  `entrypoints/train.py:LOAD_MODEL_PATCH` 把 install 插在 `load_model` 之前，
  加载审计是 `missing=0 unexpected=0`。所以 binary 权重可以合法地灌进
  ternary 模块，`_configure_existing_atlif` 的 mode 守卫**在这条路径上不会触发**
  （它守的是「已装好的模块」被重新配置的场景）。
  ⇒ 可以拿 ep34 的权重直接 warm start，比从 NB0 基线重训 30 epoch 便宜 6 倍。

改什么（单变量，只动神经元语义）：
  output_mode:            binary          → ternary
  threshold_mode:         official_atlif  → asymmetric_scale
  negative_threshold_scale: 由零样本扫描（t49_ternary_negscale_sweep.py）选定
  其余（注意力 mismatch_penalty=0.0、motion alpha=0.125、数据、分辨率）**全部不动**，
  这样「AEE 差多少」可以干净地归因到三值本身。

刻意不动的一轴：`threshold_eta` 保持 0.0。T49 已证 θ≡1 是冗余尺度、且
自适应阈值在硬件上要付逐神经元阈值存储 + 变尺度通路。本次只测三值。

优化器：fresh（不传 --resume），flat low LR（milestones 推到 999 之外），
5 epoch，每个 epoch 存一次 checkpoint 以便看曲线。
"""
from __future__ import annotations

import argparse
import json
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
PY = "/opt/conda/envs/sdformerflow/bin/python"
EPOCH_OFFSET = 35          # raw epoch 0..N-1 落盘为 checkpoint_epoch35..N+34
SRC_EPOCH = 34


def build_config(scale: float, epochs: int, backbone_lr: float,
                 mismatch_penalty: float = 0.0, motion_alpha: float = 0.125,
                 max_train_steps: int = 0) -> dict:
    cfg = yaml.safe_load(SRC_CFG.read_text(encoding="utf-8"))

    def flip(d: dict) -> None:
        d["output_mode"] = "ternary"
        d["threshold_mode"] = "asymmetric_scale"

    a = cfg["atlif_ternary_psn"]
    flip(a)
    for g in a.get("target_groups", ()) or ():
        flip(g)
    a["negative_threshold_scale"] = float(scale)

    # 注意力侧：cell A 保持源配置不动（mismatch 0.0 / motion 0.125），
    # cell B 才把负极性惩罚打开、并关掉丢极性的 motion XOR 项。
    cfg["bsa_attention"]["mismatch_penalty"] = float(mismatch_penalty)
    cfg["bsa_attention"]["binary_motion_xor_alpha"] = float(motion_alpha)

    opt = cfg["optimizer"]
    opt["lr"] = float(backbone_lr)
    opt["milestones"] = [999]                       # 5 epoch 内不衰减 → flat LR
    pg = opt["param_groups"]
    # 复刻源 run（ep29→34）的**实际生效** LR：scheduler 在 ep29 已消费掉
    # milestones [20,25]，γ=0.5 ⇒ 生效值 = 配置值 × 0.25。
    # backbone 1e-4→2.5e-5，neuron 5e-5→1.25e-5，threshold 5e-6→1.25e-6。
    pg["backbone_lr"] = float(backbone_lr)
    pg["norm_lr"] = float(backbone_lr)
    pg["neuron_lr"] = float(backbone_lr) * 0.5
    pg["threshold_lr"] = float(backbone_lr) * 0.05

    cfg["loader"]["n_epochs"] = int(epochs)

    rt = cfg["runtime"]
    rt["epoch_offset"] = EPOCH_OFFSET
    rt["force_save_epochs"] = list(range(epochs))    # 每个 epoch 都存，便于看曲线
    rt["save_only_force_epochs"] = True
    rt["state_save_epochs"] = []
    rt["skip_state_save"] = True
    rt["resume_protocol"] = "t49_ternary_warmstart_from_ep34_fresh_optimizer"
    if max_train_steps:
        rt["max_train_steps"] = int(max_train_steps)
    for key in ("resume_source_epoch", "factorial_parent_checkpoint",
                "factorial_parent_checkpoint_sha256", "source_crop_checkpoint"):
        rt.pop(key, None)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--backbone-lr", type=float, default=2.5e-05)
    ap.add_argument("--name", default="")
    ap.add_argument("--mismatch-penalty", type=float, default=0.0)
    ap.add_argument("--motion-alpha", type=float, default=0.125)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="runtime.max_train_steps; >0 用于冒烟测试（先跑几步确认能起）")
    args = ap.parse_args()

    name = args.name or ("t49_ternary_ws%s_ep%d" % (("%g" % args.scale).replace(".", "p"), args.epochs))
    out_cfg = EXP / "configs/generated" / (name + ".yml")
    out_cfg.write_text(yaml.safe_dump(
        build_config(args.scale, args.epochs, args.backbone_lr,
                     args.mismatch_penalty, args.motion_alpha, args.max_steps),
        sort_keys=False), encoding="utf-8")

    run_dir = EXP / "results" / name
    (run_dir / "warmstart").mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "warmstart" / ("checkpoint_epoch%d.pth" % SRC_EPOCH)
    if not ckpt.exists():
        shutil.copy2(SRC_RUN / ("checkpoint_epoch%d.pth" % SRC_EPOCH), ckpt)

    cmd = [PY, "-u", "neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py",
           "--config", str(out_cfg),
           "--prev_runid", str(ckpt),
           "--save_path", str(run_dir / "checkpoint_epoch{}.pth"),
           "--finetune", "1"]
    env = {**os.environ, "SDFORMER_USE_MLFLOW": "0", "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
           "SDFORMER_SNN_BACKEND": "cupy", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    with open(run_dir / "launch.json", "w", encoding="utf-8") as fh:
        json.dump({"name": name, "scale": args.scale, "epochs": args.epochs,
                   "backbone_lr": args.backbone_lr, "cmd": cmd,
                   "config": str(out_cfg), "warm_start": str(ckpt)}, fh, indent=2)
    print("[T49-4] launching:", " ".join(cmd), flush=True)
    if args.dry_run:
        print("[T49-4] dry-run: config written, training NOT launched", flush=True)
        return 0
    with open(run_dir / "train.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    print("[T49-4] %s rc=%d" % (name, rc), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
