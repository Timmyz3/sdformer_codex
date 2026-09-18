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

刻意不动的一轴：`threshold_eta` 默认保持 0.0（=θ 恒定，自适应关）。T49 已证
θ≡1 是冗余尺度、且自适应阈值在硬件上要付逐神经元阈值存储 + 变尺度通路。
**但自适应正是 ATLIF 相对 LIF 的唯一增量**，所以要单独一格消融：
`--threshold-eta >0`（官方质量型）或 `--target-rate/--target-rate-eta`
（发放率反馈型，需 threshold_mode != official_atlif）；两者都会把
`threshold_freeze_after_step` 置 null 以解除永久冻结。`--binary-official`
用于把这格跑成 binary+official_atlif，对照直接是已发布的 ep34 锚点。

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
                 max_train_steps: int = 0, target_rate: float | None = None,
                 target_rate_eta: float | None = None,
                 threshold_eta: float | None = None,
                 binary_official: bool = False) -> dict:
    cfg = yaml.safe_load(SRC_CFG.read_text(encoding="utf-8"))

    def flip(d: dict) -> None:
        if binary_official:      # 保持源配置的 binary + official_atlif（单变量消融）
            return
        d["output_mode"] = "ternary"
        d["threshold_mode"] = "asymmetric_scale"

    a = cfg["atlif_ternary_psn"]
    flip(a)
    for g in a.get("target_groups", ()) or ():
        flip(g)
    a["negative_threshold_scale"] = float(scale)

    # 自适应阈值对照组：源配置 target_rate=null / threshold_eta=0 ⇒ θ 恒定 1.0，
    # ATLIF 实际退化成"固定阈值的非对称 LIF"。不打开这几个键，就无法回答
    # "自适应阈值到底值不值"，而自适应恰恰是 ATLIF 相对 LIF 的**唯一**增量。
    #
    # 三条路径（2026-09-18 按源码重核）：①`threshold_eta`→`module.sp`（质量型手动项）
    # ②`target_rate`+`target_rate_eta`（反馈型手动项，需 threshold_mode != official_atlif）
    # ③ optimizer 的 `atlif_threshold` 参数组走 `thresh.grad`（**默认开启**，
    #   只有 `freeze_threshold_grad_after_step` 才关，源配置没有这个键）。
    # 所以 `threshold_freeze_after_step` **只冻结 ①② 的手动项**，不冻结 ③。
    # 这里只暴露 ①；②③ 需要另立单变量臂。
    if target_rate is not None:
        a["target_rate"] = float(target_rate)
        for g in a.get("target_groups", ()) or ():
            g["target_rate"] = float(target_rate)
    if target_rate_eta is not None:
        a["target_rate_eta"] = float(target_rate_eta)
        for g in a.get("target_groups", ()) or ():
            g["target_rate_eta"] = float(target_rate_eta)
    if threshold_eta is not None:
        a["threshold_eta"] = float(threshold_eta)
        for g in a.get("target_groups", ()) or ():
            g["threshold_eta"] = float(threshold_eta)
    if threshold_eta or target_rate_eta:
        # 必须"解除"冻结才能让 θ 动。installer 的判据是
        # `freeze_after_step is not None and global_step >= freeze_after_step`
        # ⇒ 写 0 等于"从第 0 步起永久冻结"（恰是反的），只有 null 才不冻结。
        a["threshold_freeze_after_step"] = None

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
    ap.add_argument("--target-rate", type=float, default=None,
                    help="ATLIF 目标发放率；设置后打开自适应阈值（源配置为 null=关闭）")
    ap.add_argument("--target-rate-eta", type=float, default=None,
                    help="目标发放率自适应步长（仅 threshold_mode != official_atlif 生效）")
    ap.add_argument("--threshold-eta", type=float, default=None,
                    help="官方 ATLIF 质量型自适应步长（接到 module.sp；源配置 0.0=关闭）")
    ap.add_argument("--binary-official", action="store_true",
                    help="保持源配置的 binary + official_atlif 不变（自适应单变量消融用）")
    args = ap.parse_args()

    name = args.name or ("t49_ternary_ws%s_ep%d" % (("%g" % args.scale).replace(".", "p"), args.epochs))
    out_cfg = EXP / "configs/generated" / (name + ".yml")
    out_cfg.write_text(yaml.safe_dump(
        build_config(args.scale, args.epochs, args.backbone_lr,
                     args.mismatch_penalty, args.motion_alpha, args.max_steps,
                     args.target_rate, args.target_rate_eta, args.threshold_eta,
                     args.binary_official),
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
