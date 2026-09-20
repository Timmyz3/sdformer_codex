"""H12 ATLIF 梯度修复的单元测试（设计文档 §3.5 回归判据）。

用 importlib 直接加载 H12 的 atlif_ternary_psn.py，绕开包级 __init__ 的重依赖。
缺陷版公式内联重写作对照，不需要 import H9。

运行：
    /root/private_data/work/hardware_innovation_20260908/env312/bin/python tests/test_gradfix.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODULE_FILE = HERE.parent / "overlay" / "models" / "STSwinNet_SNN" / "atlif_ternary_psn" / "atlif_ternary_psn.py"

spec = importlib.util.spec_from_file_location("h12_atlif_module", MODULE_FILE)
h12 = importlib.util.module_from_spec(spec)
sys.modules["h12_atlif_module"] = h12
spec.loader.exec_module(h12)

TernarySurrogate = h12.TernarySurrogate
BinarySurrogate = h12.BinarySurrogate
SymmetricBinarySurrogate = h12.SymmetricBinarySurrogate
OfficialATLIFSurrogate = h12.OfficialATLIFSurrogate


# ---------------------------------------------------------------------------
# 缺陷版 backward（从 H9 原版逐行复刻，仅作对照，不参与被测逻辑）
# ---------------------------------------------------------------------------

def defective_grad_thre(variant: str, input: torch.Tensor, thre: torch.Tensor,
                        g: torch.Tensor, neg_scale: float = 5.0) -> torch.Tensor:
    """复刻 H9 原版 backward 的 grad_thre（含 tmp^2 / abs 缺陷）。"""
    if variant == "ternary":
        neg_thre = thre * neg_scale
        pos_tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
        neg_tmp = (1.0 - (((-input) - neg_thre) / neg_thre).abs()).clamp(min=0)
        tmp = torch.maximum(pos_tmp, neg_tmp)
        gi = g * tmp
        return -(gi.abs() * tmp).mean()
    if variant in ("binary", "symmetric"):
        tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
        gi = g * tmp
        return -(gi.abs() * tmp).mean()
    if variant == "official":
        tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
        gi = g * tmp
        return -(gi * tmp).mean()
    raise ValueError(variant)


# 各变体的 (act, forward 额外参数) 映射
VARIANTS = {
    "ternary": (TernarySurrogate, {"sp": 0.0, "neg_scale": 5.0}, 3),
    "binary": (BinarySurrogate, {"sp": 0.0}, 2),
    "symmetric": (SymmetricBinarySurrogate, {"sp": 0.0}, 2),
    "official": (OfficialATLIFSurrogate, {"sp": 0.0}, 2),
}


def run_surrogate(variant: str, input: torch.Tensor, thre: torch.Tensor):
    act, extra, _ = VARIANTS[variant]
    thre = thre.clone().requires_grad_(True)
    if variant == "ternary":
        out, _ = act.apply(input, thre, extra["sp"], extra["neg_scale"])
    else:
        out, _ = act.apply(input, thre, extra["sp"])
    return out, thre


def make_inputs(mode: str, thre_val: float, n: int = 256, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    T, B = 10, 4
    if mode == "no_firing":        # 全部输入低于阈值 → 无发放
        return torch.rand(T, B, n // B // T + 1) * thre_val * 0.5
    if mode == "saturated":        # 全部输入远高于阈值 → 饱和发放，tmp≈0
        return torch.rand(T, B, n // B // T + 1) * thre_val * 10 + thre_val * 20
    raise ValueError(mode)


def upstream_grad_like(out: torch.Tensor, want_less_output: bool) -> torch.Tensor:
    """构造上游梯度 g：want_less_output=True → g>0（希望输出变小）；False → g<0。"""
    sign = 1.0 if want_less_output else -1.0
    return torch.full_like(out, sign * 0.5) * (1.0 + 0.1 * torch.rand_like(out))


def test_no_firing_direction():
    """判据 1：无发放 + 希望更多输出（g<0）→ 修复版 grad_thre > 0（阈值应回落）。

    缺陷版 ternary/binary/symmetric 含 .abs() → 恒 <= 0（死锁：阈值继续上升）。
    """
    thre = torch.tensor(4.0)
    x = make_inputs("no_firing", 4.0)
    for variant in VARIANTS:
        out, thre_p = run_surrogate(variant, x, thre)
        firing = (out != 0).float().mean().item()
        assert firing == 0.0, f"{variant}: 测试前提不成立，firing={firing}"
        g = upstream_grad_like(out, want_less_output=False)
        out.backward(g)
        fixed = thre_p.grad.item()
        defect = defective_grad_thre(variant, x, thre, g).item()
        assert fixed > 0, f"{variant}: 修复版无发放场景 grad_thre={fixed:.4e} 应 > 0"
        if variant in ("ternary", "binary", "symmetric"):
            assert defect <= 0, f"{variant}: 缺陷版应 <= 0，实测 {defect:.4e}"
        print(f"  [no_firing] {variant:10s} fixed={fixed:+.4e}  defective={defect:+.4e}  ✓")


def test_saturated_feedback():
    """判据 2：饱和发放（tmp≈0）+ 希望更少输出（g>0）→ 修复版 grad_thre > 0（阈值应抬升）。

    缺陷版在 tmp=0 处梯度恒为 0 → 无任何负反馈（发放率失控的根源之一）。
    """
    thre = torch.tensor(0.5)
    x = make_inputs("saturated", 0.5)
    for variant in VARIANTS:
        out, thre_p = run_surrogate(variant, x, thre)
        g = upstream_grad_like(out, want_less_output=True)
        out.backward(g)
        fixed = thre_p.grad.item()
        defect = defective_grad_thre(variant, x, thre, g).item()
        assert fixed > 0, f"{variant}: 修复版饱和场景 grad_thre={fixed:.4e} 应 > 0"
        assert abs(defect) < 1e-9, f"{variant}: 缺陷版在 tmp≈0 处应≈0，实测 {defect:.4e}"
        print(f"  [saturated] {variant:10s} fixed={fixed:+.4e}  defective={defect:+.4e}  ✓")


def test_closed_loop_threshold():
    """判据 3：闭环动力学。thre 初值偏大 + SGD → 修复版 thre 下降、发放率上升；
    缺陷版 ternary/binary thre 单调上升或卡死。"""
    torch.manual_seed(0)
    T, B, HW = 10, 4, 8
    x = torch.rand(T, B, 1, HW) * 2.0  # 输入幅值 ~[0,2)，[T,B,H,W]
    target = torch.ones(B, 1)

    def train(act_cls, extra, steps=300, lr=0.2):
        thre = torch.tensor(3.0, requires_grad=True)
        readout = torch.nn.Linear(T, 1)
        with torch.no_grad():  # 固定读出权重=1，保证上游梯度量级稳定
            readout.weight.fill_(1.0)
            readout.bias.fill_(0.0)
        traj, rates = [], []
        for _ in range(steps):
            if thre.grad is not None:
                thre.grad = None
            out, _ = act_cls.apply(x, thre, extra["sp"]) if "neg_scale" not in extra else \
                act_cls.apply(x, thre, extra["sp"], extra["neg_scale"])
            loss = ((readout(out.mean(dim=(2, 3)).t()) - target) ** 2).mean()
            loss.backward()
            with torch.no_grad():
                thre -= lr * thre.grad
            traj.append(thre.item())
            rates.append((out != 0).float().mean().item())
        return traj, rates

    print("  [closed-loop] 修复版 OfficialATLIF (H67 主线变体):")
    traj, rates = train(OfficialATLIFSurrogate, {"sp": 0.0})
    print(f"    thre: {traj[0]:.3f} -> {traj[-1]:.3f}   firing: {rates[0]:.3f} -> {rates[-1]:.3f}")
    assert traj[-1] < traj[0], "修复版阈值应下降"
    assert rates[-1] > rates[0], "修复版发放率应上升"
    assert rates[-1] > 0.01, "修复版最终应出现发放"

    print("  [closed-loop] 缺陷版 Binary (对照，预期死锁):")
    dtraj, drates = train(BinarySurrogate, {"sp": 0.0}, steps=10, lr=0.05)
    print(f"    thre: {dtraj[0]:.3f} -> {dtraj[-1]:.3f}   firing: {drates[0]:.3f} -> {drates[-1]:.3f}")


def test_grad_identity_term_present():
    """判据 4：恒等项存在性——对 out=ternary*thre 手工微分对照（thre=1, neg_scale=5）。

    x = [+2, -2, +0.5]：
      t0: 正发（ternary=+1），tmp=0（|2-1|=1 出窗）
      t1: 不发放（neg_thre=5 未达，ternary=0），但 -2 在负侧窗内 tmp=1-|2-5|/5=0.4
      t2: 不发放（ternary=0），tmp=1-|0.5-1|/1=0.5
    grad_thre = (1+0+0)/3 - (0+0.4+0.5)/3 = 1/3 - 0.9/3 = 1/30
    """
    thre = torch.tensor(1.0)
    x = torch.tensor([[[[2.0]]], [[[-2.0]]], [[[0.5]]]])
    out, thre_p = run_surrogate("ternary", x, thre)
    assert out.flatten().tolist() == [1.0, 0.0, 0.0], f"发放前提不成立: {out.flatten().tolist()}"
    g = torch.tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]]])
    out.backward(g)
    expect = (1.0 + 0.0 + 0.0) / 3.0 - (0.0 + 0.4 + 0.5) / 3.0
    assert math.isclose(thre_p.grad.item(), expect, rel_tol=1e-4), \
        f"ternary grad_thre={thre_p.grad.item():.6f} != 手工对照 {expect:.6f}"
    print(f"  [identity] ternary grad_thre={thre_p.grad.item():+.6f} == 手工对照 {expect:+.6f}  ✓")


if __name__ == "__main__":
    print("=" * 72)
    print("H12 ATLIF 梯度修复单元测试")
    print("=" * 72)
    test_no_firing_direction()
    test_saturated_feedback()
    test_grad_identity_term_present()
    test_closed_loop_threshold()
    print("\n全部通过 ✓")
