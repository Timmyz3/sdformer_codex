#!/usr/bin/env python3
"""B2 (h87b) motion_t4_pad_quotient forward 级 CPU 单测。

在 _qk_shiftmax_gate_forward 的注入式 forward 上验证 h87b 分支的端到端行为：
  F1  forward 可运行：x/attn 形状正确（T=10 三组、bs2 分解 (2,5,2)）
  F2  gate 归一化：h9_shiftmax_row_sum_mean ≈ n_tokens（preserve_mean）
  F3  RLE 账与槽位视图挂载：_h9_b2_rle_stats / _h9_b2_slot_* / _h9_b2_pad_mask
      （group_lengths (4,4,2)、pad_slots 2、coverage_edges 7）
  F4  配置校验：binary_motion_xor_alpha != 0 抛 RuntimeError；steps%4==0 /
      len!=4 抛 ValueError
  F5  回归：h87（D1）/h88（local5_a3s）/h89（motion_sw12_overlap）/h60/
      h82/compat 分支仍运行
  F6  STE 梯度：h87b forward 的 x 可反向，梯度非空（CPU）

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


def _cfg(mode: str = "h87b", **kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        mode=mode,
        temporal_quotient_steps=10,
        temporal_quotient_len=4,
        temporal_quotient_batch=2,
        binary_motion_xor_alpha=0.0,
    )
    values.update(kwargs)
    return ShiftmaxAttentionConfig(**values)


class FakeWindowAttention3D:
    """注入式 forward 的最小假模块（Identity 子模块 + 二值 x 输入）。

    与真实流水一致：attention 收到 (2,15,15) 两切片窗（空间 225 token），
    B* = 2×5×2 = 20 行（batch_decomposition=(2,5,2)）。
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


def run_forward(cfg: ShiftmaxAttentionConfig, **fake_kw) -> tuple[object, tuple]:
    fake = FakeWindowAttention3D(cfg, **fake_kw)
    fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
    return fake, fake.forward(fake.build_input())


class H87BForwardTests(unittest.TestCase):
    """F1/F2/F3：h87b forward 端到端。"""

    def setUp(self) -> None:
        self.cfg = _cfg("h87b")
        self.fake, self.out = run_forward(self.cfg)

    def test_forward_runs_and_shapes(self) -> None:
        x_out, attn_out = self.out
        T, B, H, W, C = self.fake._x_shape
        b_star = B
        n_tokens = T * H * W
        self.assertEqual(tuple(x_out.shape), (b_star, n_tokens, C))
        self.assertEqual(tuple(attn_out.shape), (T, b_star, H, W, C))
        self.assertFalse(torch.isnan(x_out).any())

    def test_gate_normalization(self) -> None:
        n_tokens = 2 * 15 * 15
        row_mean = float(self.fake.h9_shiftmax_row_sum_mean)
        self.assertGreater(row_mean, 0.5)
        self.assertLessEqual(row_mean, 1.0)
        gate_mean = float(self.fake.h9_shiftmax_gate_mean)
        self.assertAlmostEqual(gate_mean, row_mean, places=6)
        self.assertGreaterEqual(gate_mean, 0.0)
        self.assertLessEqual(gate_mean, 1.0)
        self.assertGreaterEqual(row_mean * n_tokens, n_tokens / 2)

    def test_scores_on_q7_grid_real_slots(self) -> None:
        slots = self.fake._h9_b2_slot_scores
        # [B*, H, num_steps=10, N]——仅真实槽（pad 槽不进 slot 融合式）
        self.assertEqual(tuple(slots.shape), (20, 4, 10, 225))
        self.assertGreaterEqual(float(slots.min()), 0.0)
        self.assertLessEqual(float(slots.max()), MAX_SCORE)
        rem = self.fake._h9_b2_slot_remainder
        self.assertTrue(torch.equal(rem, (slots % 4).to(rem.dtype)))

    def test_pad_mask_mounted(self) -> None:
        pad_mask = self.fake._h9_b2_pad_mask
        self.assertEqual(tuple(pad_mask.shape), (20, 4, 3, 4, 225))
        self.assertTrue((pad_mask[:, :, 0] == False).all())
        self.assertTrue((pad_mask[:, :, 1] == False).all())
        self.assertTrue((pad_mask[:, :, 2, 0] == False).all())
        self.assertTrue((pad_mask[:, :, 2, 1] == False).all())
        self.assertTrue((pad_mask[:, :, 2, 2] == True).all())
        self.assertTrue((pad_mask[:, :, 2, 3] == True).all())

    def test_rle_stats_mounted(self) -> None:
        rle = self.fake._h9_b2_rle_stats
        self.assertEqual(rle["num_steps"], 10)
        self.assertEqual(rle["quotient_len"], 4)
        self.assertEqual(rle["n_groups"], 3)
        self.assertEqual(rle["group_lengths"], (4, 4, 2))
        self.assertEqual(rle["pad_slots"], 2)
        self.assertEqual(rle["coverage_edges"], 7)
        self.assertEqual(rle["batch_decomposition"], (2, 5, 2))
        self.assertGreaterEqual(rle["mean_runs_per_position"], 3.0)
        self.assertLessEqual(rle["mean_runs_per_position"], 10.0)
        self.assertGreaterEqual(rle["eq_edge_rate"], 0.0)
        self.assertLessEqual(rle["eq_edge_rate"], 1.0)
        self.assertAlmostEqual(
            rle["independent_gate_ratio"],
            rle["mean_runs_per_position"] / 10.0,
            places=6,
        )
        self.assertAlmostEqual(
            rle["broadcast_saving"],
            1.0 - rle["mean_runs_per_position"] / 10.0,
            places=6,
        )
        # 恒等式：总门 = 3 + 7·(1 − eq 边率)
        self.assertAlmostEqual(
            rle["mean_runs_per_position"],
            3.0 + 7.0 * (1.0 - rle["eq_edge_rate"]),
            places=5,
        )

    def test_center_scores_off_still_runs(self) -> None:
        fake, out = run_forward(_cfg("h87b", center_scores=False))
        self.assertEqual(tuple(out[0].shape), (20, 450, 32))


class H87BValidationTests(unittest.TestCase):
    """F4：配置校验。"""

    def test_motion_alpha_forbidden(self) -> None:
        cfg = _cfg("h87b", binary_motion_xor_alpha=0.25)
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        with self.assertRaises(RuntimeError):
            fake.forward(fake.build_input())

    def test_exact_quadruple_division_forbidden(self) -> None:
        fake = FakeWindowAttention3D(_cfg("h87b", temporal_quotient_steps=8))  # 8 % 4 == 0
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        with self.assertRaises(ValueError):
            fake.forward(fake.build_input())

    def test_len_not_four_forbidden(self) -> None:
        fake = FakeWindowAttention3D(_cfg("h87b", temporal_quotient_len=5))
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        with self.assertRaises(ValueError):
            fake.forward(fake.build_input())


class RegressionTests(unittest.TestCase):
    """F5：既有分支不受 h87b 追加影响（h87/h88/h89 及更早路径）。"""

    def test_h87_motion_t5_runs(self) -> None:
        cfg = _cfg("h87", temporal_quotient_len=5)
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))
        self.assertEqual(tuple(attn_out.shape), (2, 20, 15, 15, 32))
        # D1 挂载账仍在
        self.assertEqual(self._fake_rle(fake)["num_steps"], 10)
        self.assertEqual(self._fake_rle(fake)["quotient_len"], 5)

    def test_h88_local5_a3s_runs(self) -> None:
        cfg = _cfg("h88", a3s_delta_bins=0)  # Δ=0 锚点档
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))
        self.assertEqual(tuple(attn_out.shape), (2, 20, 15, 15, 32))
        self.assertEqual(tuple(fake._h9_a3s_direction_field.shape), (20, 4, 15, 15))

    def test_h89_motion_sw12_runs(self) -> None:
        cfg = _cfg(
            "h89",
            sw12_window_size=15,
            sw12_stride=12,
            sw12_num_steps=10,
            sw12_batch=2,
            sw12_window_grid=(1, 2),
        )
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))
        self.assertEqual(tuple(attn_out.shape), (2, 20, 15, 15, 32))
        self.assertEqual(fake._h9_d2_batch_decomposition, (2, 5, 2))

    def test_h60_motion_runs(self) -> None:
        cfg = _cfg("h60", binary_motion_xor_alpha=0.25)
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))
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

    @staticmethod
    def _fake_rle(fake: FakeWindowAttention3D) -> dict:
        return fake._h9_d1_rle_stats


class GradientTests(unittest.TestCase):
    """F6：h87b forward 反向传播（STE 梯度非空）。

    F2（2026-08-19）后 STE backward 为 ÷16 直通（与 D1 同享
    _rne16_div_pow2_ste；数值断言见 scores 套件 F2 梯度测试）。
    """

    def test_backward_flows(self) -> None:
        cfg = _cfg("h87b")
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


if __name__ == "__main__":
    unittest.main()
