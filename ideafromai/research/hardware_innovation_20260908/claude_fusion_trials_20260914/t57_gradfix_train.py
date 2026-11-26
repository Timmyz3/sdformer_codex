#!/usr/bin/env python3
"""T57: 用「数学上正确的 θ 梯度」跑一小段真实训练，验证 θ 是否真的会动。

背景（T56，2026-09-19 实测）：官方 `OfficialATLIFSurrogate` 的 θ 梯度中位 5.31e-10，
落在 AdamW 默认 `eps=1e-8` 之下 ⇒ `Δθ ≈ lr·m̂/ε ≈ 6.3e-8/步` ⇒ 实际冻结。
把数学正确的 θ 梯度补上后中位 9.31e-3，比 ε 大 10⁶ 倍 ⇒ 每步走满 lr。

**本脚本不修改 overlay 任何文件**（对其他 agent 零风险）：在进程内 monkeypatch
`OfficialATLIFSurrogate.backward`，只改 θ 的梯度，前向逐位不变：

    官方  ∂y/∂h = grad·tmp ,  ∂y/∂θ = -mean(grad·tmp²)
    修正  ∂y/∂h = grad·tmp ,  ∂y/∂θ = Σ grad·(g - tmp)      g = H(h-θ)

（官方 forward 只 save 了 (input, thre)，所以修正版从二者重算 g —— 与 forward 的
`out = (input >= thre).float()` 逐位等价。）

用法：先用 t49_ternary_retrain.py --dry-run 生成配置与 warmstart checkpoint，再
  python /root/t57_gradfix_train.py --config <cfg.yml> \
      --prev_runid <ckpt> --save_path <fmt> --finetune 1
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
OVERLAY = EXP / "overlay"
BASELINE = REPO / "third_party/SDformerFlow"
TRAIN_PY = EXP / "entrypoints/train.py"


def corrected_backward(ctx, grad_input, _dummy):
    input, thre = ctx.saved_tensors
    dtype = grad_input.dtype
    input = input.to(dtype)
    thre = thre.to(dtype)
    out = (input >= thre).to(dtype)
    tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
    grad_h = grad_input * tmp
    grad_thre = (grad_input * (out - tmp)).sum()
    return grad_h, grad_thre, None


def main() -> int:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(BASELINE))
    sys.path.insert(0, str(OVERLAY))

    import models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn as M

    M.OfficialATLIFSurrogate.backward = staticmethod(corrected_backward)
    print(f"[t57-gradfix] patched {M.__file__}", flush=True)
    print(f"[t57-gradfix] backward={M.OfficialATLIFSurrogate.backward}", flush=True)

    spec = importlib.util.spec_from_file_location("h9_train_main", TRAIN_PY)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    sys.argv = ["train.py", *sys.argv[1:]]
    train_mod.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
