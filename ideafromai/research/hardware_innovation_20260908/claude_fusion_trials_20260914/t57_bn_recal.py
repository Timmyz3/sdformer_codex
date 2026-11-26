#!/usr/bin/env python3
"""T57: BN 固定统计（running）的重校准探针。

背景：锚点 ep34 的评测口径是 `test.bn_policy: no_running` ⇒ batch=1 时 BN 用
**本样本自身的统计**做归一化（等价于 InstanceNorm，逐样本自适配）。这在硬件上
要付两次全域归约，**并且让权重变成输入相关 ⇒ θ 折进 W′ 的前提被破坏**（T55）。

要兑现"自适应阈值零硬件成本"，就必须能换成固定统计（`running`）。已知 ep4 的
零恢复 A/B 里 `running` 亏 AEE +12.7%，但那是在 ep4（几乎没训完）测的。
本脚本在 **ep34 锚点**上重测，并补一格"训练集重校准"：

  N0  ep34 + no_running                      = 锚点 1.19951（已发布，无需重跑）
  N1a ep34 + running（直接用训练 EMA 统计）    ← 本脚本 --mode none
  N1b ep34 + running（训练集重校准统计）       ← 本脚本 --mode recal --split train
  N1c ep34 + running（valid 重校准统计）       ← --mode recal --split valid
                                                  （**直推式上界**，用了测试数据，
                                                   只能当诊断，不能当部署数字）

重校准做法：把所有 `_BatchNorm` 的 `momentum` 置 None（累积平均）、
`reset_running_stats()`（mean=0,var=1,nbt=0），只把 BN 模块切 train()（其余 eval()），
在**无增强**的数据上做纯前向若干 batch ⇒ running 统计变成该数据集上的精确均值/方差。
无增强是刻意的：评测也是无增强的，增强会改变激活的绝对尺度。

用法（远端 /root）：
  python -u /root/t57_bn_recal.py --mode recal --split train \
      --out <ckpt.pth> --batches 300
产出 ckpt 结构与训练落盘一致（{"model_state_dict": ...}），可直接喂
  eval_DSEC_flow_SNN.py --bn-policy running
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision  # noqa: F401 - keep SDFormerFlow import order

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP_ROOT = REPO / "neuron_experiments/H9_bipolar_self_attention"
OVERLAY_ROOT = EXP_ROOT / "overlay"
BASELINE_ROOT = REPO / "third_party/SDformerFlow"
ENTRY = EXP_ROOT / "entrypoints"

sys.path.insert(0, str(OVERLAY_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ENTRY))

# config['data']['path'] 是相对路径（../../data/Datasets/DSEC/saved_flow_data），
# 锚定在 baseline 根 —— 与 eval_DSEC_flow_SNN.py 的 cwd 一致。
os.chdir(BASELINE_ROOT)
os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")

from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402
from torch.nn.modules.batchnorm import _BatchNorm  # noqa: E402

import profile_nts11_hardware_p0 as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=REPO / "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
                                    "dsec_c12_alpha0125_ep29_resume5_20260830.yml")
    ap.add_argument("--checkpoint", type=Path,
                    default=EXP_ROOT / "results/dsec_c12_alpha0125_ep29_resume5_20260830/"
                                        "checkpoint_epoch34.pth")
    ap.add_argument("--mode", choices=("none", "recal"), default="recal")
    ap.add_argument("--split", choices=("train", "valid"), default="train")
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    config, device = P.load_config(args.config.resolve())
    print(f"[t57bn] device={device} mode={args.mode} split={args.split}", flush=True)

    model = P.build_model(config, args.checkpoint.resolve(), device)
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    print(f"[t57bn] batch-norm modules = {len(bns)}", flush=True)

    if args.mode == "recal":
        for m in bns:
            m.reset_running_stats()
            m.momentum = None          # 累积平均：统计量 = 所见过 batch 的精确均值/方差
        model.eval()
        for m in bns:
            m.train()                  # 仅 BN 走 train 分支（用 batch 统计 + 更新 running）

        dataset = DSECDatasetLite(config, file_list=args.split, stereo=False,
                                  scale_factor=config.get("test", {}).get("scale_factor", 1))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            drop_last=True, pin_memory=False, num_workers=0)
        print(f"[t57bn] calibrating on {args.split}: len={len(dataset)} "
              f"bs={args.batch_size} batches={args.batches}", flush=True)

        done = 0
        with torch.no_grad():
            for chunk, mask, label in loader:
                if done >= args.batches:
                    break
                functional.reset_net(model)
                x, _lbl, _mask = P.preprocess_chunk(config, chunk, label, mask, None, device)
                model(x)
                done += 1
                if done % 50 == 0:
                    print(f"[t57bn] batch {done}/{args.batches}", flush=True)
        for m in bns:
            m.eval()
        print(f"[t57bn] calibrated {done} batches", flush=True)
    else:
        model.eval()
        print("[t57bn] mode=none: passing checkpoint through unchanged "
              "(use eval --bn-policy running)", flush=True)

    state = {"model_state_dict": model.state_dict()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    print(f"[t57bn] wrote {args.out}", flush=True)
    print("=== T57BN COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
