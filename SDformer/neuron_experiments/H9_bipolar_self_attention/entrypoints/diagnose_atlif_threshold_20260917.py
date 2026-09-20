"""ATLIF 阈值不更新的最小复现诊断。

目的：区分"阈值≈1"是 (a) 配置关闭（sp=0/activity_eta=0）还是 (b) 代码 bug（grad 恒负/恒零）。

用法：
    python diagnose_atlif_threshold_20260917.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
H11 = HERE.parent.parent / "H11_ternary_event_attention"
sys.path.insert(0, str(H11 / "overlay" / "models" / "STSwinNet_SNN"))

from atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN  # noqa: E402


def probe_grad_sign(mode: str, sp: float, steps: int = 40, lr: float = 0.15) -> dict:
    """跑一个小训练循环，记录阈值、梯度和 update_value。"""
    torch.manual_seed(0)
    T = 10
    neuron = ATLIFTernaryPSN(
        T=T,
        base_psn=None,
        thresh=1.0,
        sparsity_eta=sp,
        negative_threshold_scale=5.0,
        activity_eta=0.0,
        output_mode=mode,
    )
    # 一个可学习读出，让损失对阈值有非零梯度通路
    readout = nn.Linear(T, 1)
    opt = torch.optim.Adam(
        [{"params": [neuron.thresh], "lr": lr}, {"params": list(readout.parameters()), "lr": 1e-3}]
    )

    x = torch.randn(T, 4, 8, 8) * 0.8  # 时间优先 [T, B, H, W]
    target = torch.randn(4, 1)

    traj, grads, updates = [], [], []
    for step in range(steps):
        opt.zero_grad()
        out = neuron(x)                          # [T, B, H, W]
        pooled = out.mean(dim=(2, 3)).t()        # [B, T]
        loss = ((readout(pooled) - target) ** 2).mean()
        loss.backward()
        g = float(neuron.thresh.grad) if neuron.thresh.grad is not None else float("nan")
        grads.append(g)
        traj.append(float(neuron.thresh.item()))
        updates.append(neuron.update_value)
        opt.step()
        neuron.update_value = 0.0

    return {
        "mode": mode,
        "sp": sp,
        "thresh_first": traj[0],
        "thresh_last": traj[-1],
        "thresh_min": min(traj),
        "thresh_max": max(traj),
        "grad_first": grads[0],
        "grad_last": grads[-1],
        "grad_max": max(grads),
        "grad_min": min(grads),
        "grad_all_nonpositive": all(g <= 1e-12 for g in grads),
        "update_abs_max": max(abs(u) for u in updates),
    }


def probe_update_value(mode: str) -> dict:
    """验证 sp=0 时 update_value 是否恒为 0。"""
    torch.manual_seed(0)
    T = 10
    res = {}
    for sp in (0.0, 1e-3):
        neuron = ATLIFTernaryPSN(T=T, thresh=1.0, sparsity_eta=sp, output_mode=mode)
        x = torch.randn(T, 4, 8, 8) * 0.8
        neuron.update_value = 0.0
        _ = neuron(x)
        res[f"sp={sp}"] = neuron.update_value
    return {"mode": mode, "update_value": res}


def main() -> None:
    print("=" * 72)
    print("ATLIF 阈值诊断")
    print("=" * 72)

    print("\n[1] update_value 是否被 sp=0 完全关闭")
    for mode in ("binary", "ternary"):
        r = probe_update_value(mode)
        print(f"  {mode:8s}: " + "  ".join(f"{k} -> {v:.6e}" for k, v in r["update_value"].items()))

    print("\n[2] 阈值/梯度轨迹（Adam lr=0.15, 40 step）")
    header = f"{'mode':8s} {'sp':>6s} {'th_first':>9s} {'th_last':>9s} {'g_max':>12s} {'g_min':>12s} {'g<=0':>6s}"
    print("  " + header)
    for mode in ("binary", "ternary"):
        for sp in (0.0, 1e-3):
            r = probe_grad_sign(mode, sp)
            print(
                f"  {r['mode']:8s} {r['sp']:6.0e} {r['thresh_first']:9.4f} {r['thresh_last']:9.4f} "
                f"{r['grad_max']:12.3e} {r['grad_min']:12.3e} {str(r['grad_all_nonpositive']):>6s}"
            )

    print("\n[3] 判读")
    print("  - 若 update_value(sp=1e-3) 非 0 而 sp=0 时为 0：活动剪枝通道确实被配置关闭。")
    print("  - 若 g<=0 全为 True：backward 的 grad_thre 恒负，缺 ternary 直传项（代码 bug）。")
    print("  - 若 th_last ≈ th_first ≈ 1.0：阈值实质冻结，即观察到的'阈值都约等于 1'。")


if __name__ == "__main__":
    main()
