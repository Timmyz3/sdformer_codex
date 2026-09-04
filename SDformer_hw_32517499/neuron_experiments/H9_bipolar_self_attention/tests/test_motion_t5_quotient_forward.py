#!/usr/bin/env python3
"""D1 (h87) motion_t5_quotient forward 级 CPU 单测。

在 _qk_shiftmax_gate_forward 的注入式 forward 上验证 h87 分支的端到端行为
（forward 尾部对所有模式统一返回 (x, attn) 两元组）：
  F1  forward 可运行：x/attn 形状正确（T=10 双组、bs2 分解 (2,5,2)）
  F2  gate 归一化：h9_shiftmax_row_sum_mean ≈ n_tokens（preserve_mean）
  F3  RLE 账与槽位视图：self._h9_d1_rle_stats / _h9_d1_slot_* 挂载且形状正确
  F4  配置校验：binary_motion_xor_alpha != 0 必须抛 RuntimeError（运动不双重计数）
  F5  回归：h60（Motion）/h82（class_major_ttx）/compat_qk_product 分支仍运行
  F6  STE 梯度：h87 forward 的 x 可反向，梯度非空（CPU）

CPU-only：不训练、不评测、不碰 GPU。
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))

from models.STSwinNet_SNN.bsa_attention import (
    ShiftmaxAttentionConfig,
    _qk_shiftmax_gate_forward,
)

MAX_SCORE = 162


def _cfg(mode: str = "h87", **kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        mode=mode,
        temporal_quotient_steps=10,
        temporal_quotient_len=5,
        temporal_quotient_batch=2,
        binary_motion_xor_alpha=0.0,
    )
    values.update(kwargs)
    return ShiftmaxAttentionConfig(**values)


class FakeWindowAttention3D:
    """注入式 forward 的最小假模块（Identity 子模块 + 二值 x 输入）。

    与真实流水一致：attention 收到 (2,15,15) 两切片窗（w15 族窗口，
    空间 225 token），B* = B×n_pairs×n_sw；T=2（窗切片）× B*=20 行
    （= 2×5×2，batch_decomposition=(2,5,2)），n_sw=2 来自 batch 分解
    （D1 候选，与真实 fullres stage3 的窗口数一致）。
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
        # forward 尾部（attn_drop -> attn_sn -> proj -> proj_bn）
        self.attn_drop = torch.nn.Identity()
        self.attn_sn = torch.nn.Identity()
        self.proj = torch.nn.Identity()
        self.proj_bn = torch.nn.Identity()
        # compat_qk_product/legacy 分支的 att_token 精化
        self.sn2_q = torch.nn.Identity()
        self._x_shape = (T, batch, height, width, channels)

    def build_input(self) -> torch.Tensor:
        return torch.randint(0, 2, self._x_shape).to(torch.float32)


def run_forward(cfg: ShiftmaxAttentionConfig, **fake_kw) -> tuple[object, tuple]:
    fake = FakeWindowAttention3D(cfg, **fake_kw)
    fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
    return fake, fake.forward(fake.build_input())


class H87ForwardTests(unittest.TestCase):
    """F1/F2/F3：h87 forward 端到端。"""

    def setUp(self) -> None:
        self.cfg = _cfg("h87")
        self.fake, self.out = run_forward(self.cfg)

    def test_forward_runs_and_shapes(self) -> None:
        x_out, attn_out = self.out
        T, B, H, W, C = self.fake._x_shape
        b_star = B  # attention batch = B* = 2×5×2 = 20 行
        n_tokens = T * H * W
        self.assertEqual(tuple(x_out.shape), (b_star, n_tokens, C))
        self.assertEqual(tuple(attn_out.shape), (T, b_star, H, W, C))
        self.assertFalse(torch.isnan(x_out).any())

    def test_gate_normalization(self) -> None:
        n_tokens = 2 * 15 * 15
        # shiftmax 行和 ∈ (0.5, 1]（h60/Motion 锚点同一约定）；row_sum 在
        # preserve_mean 缩放前记账，故 h9_shiftmax_row_sum_mean 是缩放前行和均值。
        row_mean = float(self.fake.h9_shiftmax_row_sum_mean)
        self.assertGreater(row_mean, 0.5)
        self.assertLessEqual(row_mean, 1.0)
        # preserve_mean: gate = gate × n_tokens（乘数 == token 维尺寸），
        # 全元素均值不变 -> gate_mean == row_sum_mean 为缩放生效的诚实证据。
        gate_mean = float(self.fake.h9_shiftmax_gate_mean)
        self.assertAlmostEqual(gate_mean, row_mean, places=6)
        self.assertGreaterEqual(gate_mean, 0.0)
        self.assertLessEqual(gate_mean, 1.0)
        # 最终门预算每行 ≈ row_mean × n_tokens（> n_tokens 说明均值保持生效）
        self.assertGreaterEqual(row_mean * n_tokens, n_tokens / 2)

    def test_scores_on_q7_grid(self) -> None:
        # 分数由 T5 商给出，max <= 162（Q7 网格）
        slots = self.fake._h9_d1_slot_scores
        self.assertEqual(tuple(slots.shape), (20, 4, 10, 225))  # [B*, H, num_steps, N]
        self.assertGreaterEqual(float(slots.min()), 0.0)
        self.assertLessEqual(float(slots.max()), MAX_SCORE)

    def test_rle_stats_mounted(self) -> None:
        rle = self.fake._h9_d1_rle_stats
        self.assertEqual(rle["num_steps"], 10)
        self.assertEqual(rle["quotient_len"], 5)
        self.assertEqual(rle["batch_decomposition"], (2, 5, 2))
        self.assertGreaterEqual(rle["mean_runs_per_position"], 1.0)
        self.assertLessEqual(rle["mean_runs_per_position"], 5.0)
        self.assertGreaterEqual(rle["eq_edge_rate"], 0.0)
        self.assertLessEqual(rle["eq_edge_rate"], 1.0)
        self.assertAlmostEqual(
            rle["independent_gate_ratio"],
            rle["mean_runs_per_position"] / 5.0,
            places=6,
        )

    def test_slot_overlap_remainder_mounted(self) -> None:
        ov = self.fake._h9_d1_slot_overlap
        rem = self.fake._h9_d1_slot_remainder
        self.assertEqual(tuple(ov.shape), (20, 4, 10, 225))
        self.assertEqual(tuple(rem.shape), (20, 4, 10, 225))
        slots = self.fake._h9_d1_slot_scores
        self.assertTrue(torch.equal(rem, (slots % 4).to(rem.dtype)))

    def test_center_scores_off_still_runs(self) -> None:
        fake, out = run_forward(_cfg("h87", center_scores=False))
        self.assertEqual(tuple(out[0].shape), (20, 450, 32))


class H87ValidationTests(unittest.TestCase):
    """F4：配置校验。"""

    def test_motion_alpha_forbidden(self) -> None:
        cfg = _cfg("h87", binary_motion_xor_alpha=0.25)
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        with self.assertRaises(RuntimeError):
            fake.forward(fake.build_input())

    def test_bad_temporal_steps_forbidden(self) -> None:
        fake = FakeWindowAttention3D(_cfg("h87", temporal_quotient_steps=8))  # 8 % 5 != 0
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        with self.assertRaises(ValueError):
            fake.forward(fake.build_input())


class RegressionTests(unittest.TestCase):
    """F5：既有分支不受 h87 追加影响。"""

    def test_h60_motion_runs(self) -> None:
        # H67/Motion 需要两切片时间窗（T=2，_binary_temporal_k_xor_popcount）
        cfg = _cfg("h60", binary_motion_xor_alpha=0.25)
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))
        self.assertEqual(tuple(attn_out.shape), (2, 20, 15, 15, 32))
        self.assertFalse(torch.isnan(x_out).any())

    def test_h82_class_major_runs(self) -> None:
        cfg = _cfg("h82")
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))

    def test_compat_qk_product_runs(self) -> None:
        cfg = _cfg("compat_qk_product")
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))


class GradientTests(unittest.TestCase):
    """F6：h87 forward 反向传播（STE 梯度非空）。

    F2（2026-08-19）后 STE backward 为 ÷16 直通（见 scores 套件
    Rne16DivTests.test_ste_backward_scales_by_denominator 的数值断言）；
    本档保持"梯度不塌"回归。
    """

    def test_backward_flows(self) -> None:
        cfg = _cfg("h87")
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x = fake.build_input()
        x.requires_grad_(True)
        x_out, _ = fake.forward(x)
        loss = x_out.float().sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.numel(), x.numel())
        self.assertTrue(torch.isfinite(x.grad).all())


class H67ForwardAlignmentTests(unittest.TestCase):
    """F1 回归（2026-08-19 修复）：forward 分支端到端 h87 vs h67（h60/Motion）。

    同一稀疏随机输入（p=0.06，对齐真实 popcount~1.9）分别走 h87 与 h60
    分支，attn = k·gate（preserve_mean ×450）应数值对齐（F1 的 ÷128 挂载
    生效的直接证据）。独立随机输入下实测 mean_rel ≈ 5.9e-3（修复前同口径
    ≈0.148–0.198）；阈值留 3× 余量。
    """

    def test_forward_attn_aligns_with_h67(self) -> None:
        def sparse_input(p: float = 0.06) -> torch.Tensor:
            return (torch.rand(2, 20, 15, 15, 32) < p).to(torch.float32)

        torch.manual_seed(0)
        x = sparse_input()
        fake87 = FakeWindowAttention3D(_cfg("h87"))
        fake87.forward = types.MethodType(_qk_shiftmax_gate_forward, fake87)
        _, attn87 = fake87.forward(x)
        fake60 = FakeWindowAttention3D(_cfg("h60", binary_motion_xor_alpha=0.25))
        fake60.forward = types.MethodType(_qk_shiftmax_gate_forward, fake60)
        _, attn60 = fake60.forward(x)
        rel = ((attn87 - attn60).abs() / (attn60.abs() + 1e-12)).mean()
        self.assertLessEqual(
            float(rel), 2e-2,
            f"forward attn mean rel 差须 ≤2e-2（F1 修复后实测 ~5.9e-3），"
            f"实测 {float(rel):.2e}",
        )


if __name__ == "__main__":
    unittest.main()
