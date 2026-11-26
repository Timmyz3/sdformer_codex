#!/usr/bin/env python3
"""T57-N2: 冻结 BN 统计的恢复训练（让网络去适配**固定** BN 统计这条部署口径）。

要解决的问题：锚点评测用 `test.bn_policy: no_running` ⇒ batch=1 时 BN 用**本样本**
统计归一化（≈InstanceNorm）。这在硬件上两次全域归约，**并且让权重输入相关 ⇒
θ 折进 W′（T55b 的"零成本"前提）作废**。

但直接换 `running` 会掉精度（ep4 零恢复 A/B：AEE +12.7%），因为网络是在
"batch 统计"下训出来的，从没见过固定统计。本脚本把训练期的 BN **强制走 eval 分支**
（用 running_mean/var，且不更新它们）⇒ 训练与部署口径一致，网络只能去适配固定统计。

实现方式：**不改 overlay 任何文件**，在进程内 monkeypatch
`torch.nn.modules.batchnorm._BatchNorm.forward`，把 `self.training` 临时置 False。
（`_BatchNorm.forward` 判据是 `self.training or not self.track_running_stats`
来决定是否用 batch 统计，所以置 False 即固定统计 + 不更新 running。）

用法（先用 t49_ternary_retrain.py --dry-run 生成配置与 warmstart checkpoint）：
  python /root/t57_frozenbn_train.py --config <cfg.yml> \
      --prev_runid <ckpt> --save_path <fmt> --finetune 1
"""
from __future__ import annotations

import atexit
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.modules.batchnorm as bn_mod

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
OVERLAY = EXP / "overlay"
BASELINE = REPO / "third_party/SDformerFlow"
TRAIN_PY = EXP / "entrypoints/train.py"

_ORIG_FORWARD = bn_mod._BatchNorm.forward
_CALLS = {"train_mode_forced": 0, "passed_through": 0}


def frozen_forward(self, input):
    was_training = self.training
    if was_training:
        _CALLS["train_mode_forced"] += 1
    else:
        _CALLS["passed_through"] += 1
    self.training = False
    try:
        return _ORIG_FORWARD(self, input)
    finally:
        self.training = was_training


def _report() -> None:
    print(f"[t57-frozenbn] BN forward calls: forced_from_train={_CALLS['train_mode_forced']} "
          f"already_eval={_CALLS['passed_through']}", flush=True)


def main() -> int:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(BASELINE))
    sys.path.insert(0, str(OVERLAY))

    bn_mod._BatchNorm.forward = frozen_forward
    atexit.register(_report)
    print(f"[t57-frozenbn] patched {bn_mod.__file__}::_BatchNorm.forward", flush=True)
    print("[t57-frozenbn] BN 在训练期也走 eval 分支 ⇒ 固定统计、不更新 running",
          flush=True)

    spec = importlib.util.spec_from_file_location("h9_train_main", TRAIN_PY)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    sys.argv = ["train.py", *sys.argv[1:]]
    train_mod.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
