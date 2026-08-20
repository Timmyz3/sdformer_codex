#!/usr/bin/env python3
"""D3 (h88/local5_a3s) forward 级 CPU 单测。

在 _qk_shiftmax_gate_forward 的注入式 forward 上验证 local5_a3s 分支的
端到端行为（forward 尾部对所有模式统一返回 (x, attn) 两元组）：
  F1  forward 可运行：x/attn 形状正确（(2,15,15) 窗、B*=20 行）
  F2  Δ=0 forward 级 K1：local5_a3s(Δ=0) 与现网 h66_lr/local5 forward 逐位一致
      （x_out 与 attn 均 torch.equal；可注入式训练锚点）
  F3  A3S 账本挂载：_h9_a3s_direction_field（2bit/px 位图）/ delta_bins /
      axis_frac_ew / winner_hit_rate 形状与语义正确
  F4  Δ 注入式渐增：warmup 下 step=0 -> Δ=0（恒等档），中途 -> 部分档，
      满档后 -> a3s_delta_bins（forward 级调度生效）
  F5  回归：h66_lr（Local5）/h87（D1）/h82/compat 分支仍运行（0 删除行纪律）
  F6  STE 梯度：local5_a3s forward 的 x 可反向，梯度非空（CPU）

CPU-only：不训练、不评测、不碰 GPU。
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

# 64 核机器线程池争用（同 check_d3 纪律）
torch.set_num_threads(2)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))

from models.STSwinNet_SNN.bsa_attention import (  # noqa: E402
    ShiftmaxAttentionConfig,
    _qk_shiftmax_gate_forward,
)

DELTA_BINS = 8


def _cfg(mode: str = "local5_a3s", **kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        mode=mode,
        alpha0=0.015625,
        consensus_score_norm="head_dim",
        preserve_mean=False,
        center_scores=True,
        a3s_delta_bins=DELTA_BINS,
        a3s_delta_warmup_steps=0,
        binary_motion_xor_alpha=0.0,
    )
    values.update(kwargs)
    return ShiftmaxAttentionConfig(**values)


class FakeWindowAttention3D:
    """注入式 forward 的最小假模块（Identity 子模块 + 二值 x 输入）。

    与真实流水一致：attention 收到 (2,15,15) 两切片窗（w15 族窗口，空间
    225 token），B* = B×n_pairs×n_sw = 20 行（= 2×5×2）。两个实例同 seed
    构造出逐位相同的输入（可做 Δ=0 vs Local5 的 forward 级恒等对比）。
    """

    def __init__(self, cfg: ShiftmaxAttentionConfig, *, T: int = 2, batch: int = 20,
                 height: int = 15, width: int = 15, channels: int = 32,
                 num_heads: int = 4, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self._h9_shiftmax_cfg = cfg
        self._h9_global_step = 0
        self.training = True
        self.num_heads = num_heads
        self.norm_layer = "None"  # 跳过 BN 路径
        self.proj_sn = torch.nn.Identity()
        self.linear_q = torch.nn.Identity()
        self.sn_q = torch.nn.Identity()
        self.linear_k = torch.nn.Identity()
        self.sn_k = torch.nn.Identity()
        self.positional_encoding = torch.zeros(T, 1, height, width, channels)
        self.attn_drop = torch.nn.Identity()
        self.attn_sn = torch.nn.Identity()
        self.proj = torch.nn.Identity()
        self.proj_bn = torch.nn.Identity()
        self.sn2_q = torch.nn.Identity()
        self._x_shape = (T, batch, height, width, channels)

    def build_input(self) -> torch.Tensor:
        return torch.randint(0, 2, self._x_shape).to(torch.float32)


def run_forward(cfg: ShiftmaxAttentionConfig, *, step: int = 0, **fake_kw) -> tuple[object, tuple]:
    fake = FakeWindowAttention3D(cfg, **fake_kw)
    fake._h9_global_step = step
    fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
    return fake, fake.forward(fake.build_input())


class H88ForwardTests(unittest.TestCase):
    """F1/F2/F3：local5_a3s forward 端到端 + Δ=0 恒等 + 账本挂载。"""

    def setUp(self) -> None:
        self.cfg = _cfg("local5_a3s")
        self.fake, self.out = run_forward(self.cfg)

    def test_forward_runs_and_shapes(self) -> None:
        x_out, attn_out = self.out
        T, B, H, W, C = self.fake._x_shape
        b_star = B
        n_tokens = T * H * W
        self.assertEqual(tuple(x_out.shape), (b_star, n_tokens, C))
        self.assertEqual(tuple(attn_out.shape), (T, b_star, H, W, C))
        self.assertFalse(torch.isnan(x_out).any())
        self.assertFalse(torch.isinf(x_out).any())

    def test_gate_normalization(self) -> None:
        # shiftmax 行和 ∈ (0.5, 1]（h60/Motion 锚点同一约定）；Local5 门为
        # [B, H, N, 5]（5-lane），全元素均值 = 行和均值 / 5（preserve_mean
        # false，Local5 模板同款）。
        row_mean = float(self.fake.h9_shiftmax_row_sum_mean)
        self.assertGreater(row_mean, 0.5)
        self.assertLessEqual(row_mean, 1.0)
        gate_mean = float(self.fake.h9_shiftmax_gate_mean)
        self.assertAlmostEqual(gate_mean * 5, row_mean, places=6)

    def test_delta0_forward_bit_exact_local5(self) -> None:
        # F2：forward 级 K1 —— local5_a3s(Δ=0) 与 h66_lr 现网 forward 逐位一致
        cfg_base = _cfg("binary_axnor_local5_shiftmax", a3s_delta_bins=0)
        cfg_a3s = _cfg("local5_a3s", a3s_delta_bins=0)
        _, out_base = run_forward(cfg_base, seed=3)
        _, out_a3s = run_forward(cfg_a3s, seed=3)
        self.assertTrue(torch.equal(out_base[0], out_a3s[0]), "x_out 逐位不一致")
        self.assertTrue(torch.equal(out_base[1], out_a3s[1]), "attn 逐位不一致")

    def test_delta0_forward_bit_exact_local5_at_warmup_step0(self) -> None:
        # 注入式起调档：warmup 内 step=0 -> Δ=0 -> 与现网逐位一致
        cfg_base = _cfg("binary_axnor_local5_shiftmax")
        cfg_a3s = _cfg("local5_a3s", a3s_delta_warmup_steps=1224)
        _, out_base = run_forward(cfg_base, seed=5)
        _, out_a3s = run_forward(cfg_a3s, seed=5, step=0)
        self.assertTrue(torch.equal(out_base[0], out_a3s[0]), "x_out 逐位不一致")
        self.assertTrue(torch.equal(out_base[1], out_a3s[1]), "attn 逐位不一致")

    def test_a3s_stats_mounted(self) -> None:
        # F3：方向场位图 + 账本挂载（forward 验证用）
        dirs = self.fake._h9_a3s_direction_field
        self.assertEqual(tuple(dirs.shape), (20, 4, 15, 15))  # [B*, H, H, W]
        self.assertEqual(set(torch.unique(dirs).tolist()), {0, 1, 2, 3})
        self.assertEqual(self.fake._h9_a3s_delta_bins, DELTA_BINS)
        self.assertGreaterEqual(self.fake._h9_a3s_axis_frac_ew, 0.0)
        self.assertLessEqual(self.fake._h9_a3s_axis_frac_ew, 1.0)
        self.assertGreaterEqual(self.fake._h9_a3s_winner_hit_rate, 0.0)
        self.assertLessEqual(self.fake._h9_a3s_winner_hit_rate, 1.0)

    def test_delta_full_changes_attn_vs_zero(self) -> None:
        cfg0 = _cfg("local5_a3s", a3s_delta_bins=0)
        cfg8 = _cfg("local5_a3s", a3s_delta_bins=DELTA_BINS)
        _, out0 = run_forward(cfg0, seed=7)
        _, out8 = run_forward(cfg8, seed=7)
        self.assertFalse(torch.equal(out0[0], out8[0]))


class H88WarmupForwardTests(unittest.TestCase):
    """F4：Δ 注入式渐增（forward 级调度）。"""

    def test_warmup_schedule_at_forward_level(self) -> None:
        cfg = _cfg("local5_a3s", a3s_delta_bins=DELTA_BINS, a3s_delta_warmup_steps=100)
        _, out0 = run_forward(cfg, seed=1, step=0)
        fake1, out1 = run_forward(cfg, seed=1, step=50)
        fake2, out2 = run_forward(cfg, seed=1, step=100)
        self.assertEqual(fake1._h9_a3s_delta_bins, 4)  # 半程 -> 4 档
        self.assertEqual(fake2._h9_a3s_delta_bins, DELTA_BINS)  # 满档
        # 起调档 = 现网恒等档；Δ 生效后 attn 变化
        _, out_local5 = run_forward(_cfg("binary_axnor_local5_shiftmax"), seed=1)
        self.assertTrue(torch.equal(out0[0], out_local5[0]))
        self.assertFalse(torch.equal(out2[0], out_local5[0]))

    def test_zero_target_never_touches_scores(self) -> None:
        cfg = _cfg("local5_a3s", a3s_delta_bins=0)
        fake, _ = run_forward(cfg, seed=2)
        self.assertEqual(fake._h9_a3s_delta_bins, 0)


class H88RegressionTests(unittest.TestCase):
    """F5：既有 mode 分支回归（0 删除行纪律）。"""

    def test_local5_regression(self) -> None:
        _, out = run_forward(_cfg("binary_axnor_local5_shiftmax"), seed=9)
        self.assertEqual(tuple(out[0].shape), (20, 450, 32))

    def test_h87_regression(self) -> None:
        cfg = _cfg("h87", temporal_quotient_steps=10, temporal_quotient_len=5,
                   temporal_quotient_batch=2)
        _, out = run_forward(cfg, seed=9)
        self.assertEqual(tuple(out[0].shape), (20, 450, 32))

    def test_h82_regression(self) -> None:
        _, out = run_forward(_cfg("h82"), seed=9)
        self.assertEqual(tuple(out[0].shape), (20, 450, 32))

    def test_compat_regression(self) -> None:
        _, out = run_forward(_cfg("compat_qk_product"), seed=9)
        self.assertEqual(tuple(out[0].shape), (20, 450, 32))


class H88GradientTests(unittest.TestCase):
    """F6：STE 梯度（CPU 反向）。"""

    def test_backward_gradients_non_empty(self) -> None:
        fake = FakeWindowAttention3D(_cfg("local5_a3s"))
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x = fake.build_input().requires_grad_(True)
        out = fake.forward(x)
        out[0].sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
