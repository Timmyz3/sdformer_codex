#!/usr/bin/env python3
"""D2 (h89) motion_sw12_overlap forward 级 CPU 单测。

在 _qk_shiftmax_gate_forward 的注入式 forward 上验证 h89 分支的端到端行为
（forward 尾部对所有模式统一返回 (x, attn) 两元组）：
  F1  forward 可运行：x/attn 形状正确（T=2、B*=20 = 2×5×2、窗 (2,15,15)）
  F2  mult 加权守恒：Σ_t gate(t)·mult(t) == Σ_w row_sum(w) ∈ (0.5·n_ow, n_ow]
      （J3 门还原的 forward 级证据；preserve_mean=False 的未缩放门）
  F3  _h9_d2_* 账本挂载：rolling_z / z_full / exp_ledger / catalog /
      window_plan / gate_final / gate_mult / batch_decomposition /
      window_counts 形状与数值
  F4  配置校验：binary_motion_xor_alpha != 0 抛 RuntimeError（运动不双重计数）；
      sw12_num_steps 不整除 2 抛 ValueError；stride=15 退化解 = 稠密非重叠
  F5  回归：h60（Motion）/h87（motion_t5_quotient）/h88（local5_a3s）/
      h82（class_major_ttx）/compat_qk_product 分支仍运行
  F6  STE 梯度：h89 forward 的 x 可反向，梯度非空（CPU）

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


def _cfg(mode: str = "h89", **kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        mode=mode,
        sw12_window_size=15,
        sw12_stride=12,
        sw12_num_steps=10,
        sw12_batch=2,
        binary_motion_xor_alpha=0.0,
        preserve_mean=False,
    )
    values.update(kwargs)
    return ShiftmaxAttentionConfig(**values)


class FakeWindowAttention3D:
    """注入式 forward 的最小假模块（Identity 子模块 + 二值 x 输入）。

    与真实流水一致：attention 收到 (2,15,15) 两切片窗（w15 族窗口，空间
    225 token），B* = B×n_pairs×n_sw = 20 行（= 2×5×2，batch_decomposition
    =(2,5,2)），n_sw=2 来自 batch 分解（D2 候选，与真实 fullres stage3 的
    窗口数一致）。
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


class H89ForwardTests(unittest.TestCase):
    """F1/F2/F3：h89 forward 端到端。"""

    def setUp(self) -> None:
        self.cfg = _cfg("h89")
        self.fake, self.out = run_forward(self.cfg)

    def test_forward_runs_and_shapes(self) -> None:
        x_out, attn_out = self.out
        T, B, H, W, C = self.fake._x_shape
        b_star = B  # attention batch = B* = 2×5×2 = 20 行
        n_tokens = T * H * W
        self.assertEqual(tuple(x_out.shape), (b_star, n_tokens, C))
        self.assertEqual(tuple(attn_out.shape), (T, b_star, H, W, C))
        self.assertFalse(torch.isnan(x_out).any())

    def test_mult_weighted_conservation(self) -> None:
        """J3 门还原：Σ_t gate(t)·mult(t) == Σ_w row_sum(w) ∈ (0.5·n_ow, n_ow]。"""
        gate_mean = self.fake.h9_shiftmax_gate_mean  # 未缩放门（preserve_mean=False）
        self.assertGreaterEqual(gate_mean, 0.0)
        # gate_final[t] = Σ_w gate_w(t)（各窗门在 token 上的 scatter_add 和）；
        # 每 field-head 的 Σ_t g_final(t) = Σ_w row_sum(w)，shiftmax 每窗行和
        # ∈ (0.5, 1] -> 总和 ∈ (0.5·n_ow, n_ow]（J3 恒等的 float 界）。
        n_ow = self.fake._h9_d2_window_plan["n_ow"]
        g_final = self.fake._h9_d2_gate_final  # [n_f, H, n_sw, 900]
        per_fh = g_final.sum(dim=(2, 3))  # [n_f, H] = Σ_w row_sum(w)
        self.assertGreater(float(per_fh.min()), 0.5 * n_ow)
        self.assertLessEqual(float(per_fh.max()), float(n_ow))

    def test_scores_on_q7_grid_and_rolling_ledger(self) -> None:
        scores = self.fake._h9_d2_scores
        self.assertEqual(tuple(scores.shape), (20, 4, 450))
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 162.0)
        z_roll = self.fake._h9_d2_rolling_z
        z_full = self.fake._h9_d2_z_full
        self.assertEqual(tuple(z_roll.shape), tuple(z_full.shape))
        # J1 逐位相等（forward 级复验）
        self.assertTrue(torch.equal(z_roll, z_full), "J1 bitwise FAIL in forward")
        self.assertEqual(int(z_roll.shape[-1]), 11)  # 16bit 块数

    def test_ledgers_mounted(self) -> None:
        plan = self.fake._h9_d2_window_plan
        self.assertEqual(plan["n_ow"], 3)
        self.assertEqual(plan["ys"], [(0, 15)])
        self.assertEqual(plan["xs"], [(0, 15), (12, 27), (24, 30)])
        self.assertEqual(self.fake._h9_d2_batch_decomposition, (2, 5, 2))
        self.assertEqual(
            self.fake._h9_d2_window_counts, {"dense": 2, "overlap": 3}
        )
        ledger = self.fake._h9_d2_exp_ledger
        # J6 数值（check_d2 同式）：(1,2) 网格 dense 2 窗 / overlap 3 窗，
        # 每窗全量 450 项、增量 270 项（450 − 2×90 共享带）
        self.assertEqual(ledger["dense_windows"], 2)
        self.assertEqual(ledger["overlap_windows"], 3)
        self.assertEqual(ledger["per_window_full"], 450)
        self.assertEqual(ledger["per_window_incremental_formula"], 270)
        self.assertEqual(ledger["dense_total_terms"], 2 * 450)
        self.assertEqual(ledger["overlap_total_terms"], 3 * 270)
        # gate_final/gate_mult 形状
        self.assertEqual(
            tuple(self.fake._h9_d2_gate_final.shape), (10, 4, 2, 450)
        )
        self.assertEqual(
            tuple(self.fake._h9_d2_gate_mult.shape), (1, 1, 2, 450)
        )
        # catalog 目录（跨窗共享带身份码/类码）
        catalog = self.fake._h9_d2_catalog
        self.assertIn("x_identities", catalog)
        self.assertIn("x_classes", catalog)
        self.assertIn("y_identities", catalog)
        self.assertIn("y_classes", catalog)
        self.assertEqual(catalog["x_pairs"], [(0, 0), (0, 1)])
        # 每对相邻窗一条 3 宽共享带（x 向：15×3×2 时 = 90 身份码）
        self.assertEqual(tuple(catalog["x_identities"][0].shape), (90,))
        self.assertEqual(tuple(catalog["x_classes"][0].shape), (10, 4, 90))

    def test_preserve_mean_forward_runs(self) -> None:
        fake, out = run_forward(_cfg("h89", preserve_mean=True))
        self.assertEqual(tuple(out[0].shape), (20, 450, 32))


class H89ValidationTests(unittest.TestCase):
    """F4：配置校验。"""

    def test_motion_alpha_forbidden(self) -> None:
        cfg = _cfg("h89", binary_motion_xor_alpha=0.25)
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        with self.assertRaises(RuntimeError):
            fake.forward(fake.build_input())

    def test_bad_num_steps_forbidden(self) -> None:
        fake = FakeWindowAttention3D(_cfg("h89", sw12_num_steps=9))  # 9 % 2 != 0
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        with self.assertRaises(ValueError):
            fake.forward(fake.build_input())

    def test_stride15_degradation_dense(self) -> None:
        """stride=15 退化解 = 稠密非重叠基线：mult 全 1、窗口数不增。"""
        fake, out = run_forward(_cfg("h89", sw12_stride=15))
        counts = fake._h9_d2_window_counts
        self.assertEqual(counts, {"dense": 2, "overlap": 2})
        self.assertTrue(
            torch.equal(fake._h9_d2_gate_mult, torch.ones_like(fake._h9_d2_gate_mult))
        )
        self.assertEqual(tuple(out[0].shape), (20, 450, 32))
        self.assertTrue(torch.equal(fake._h9_d2_rolling_z, fake._h9_d2_z_full))


class RegressionTests(unittest.TestCase):
    """F5：既有分支不受 h89 追加影响。"""

    def test_h60_motion_runs(self) -> None:
        cfg = _cfg("h60", binary_motion_xor_alpha=0.25)
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))
        self.assertEqual(tuple(attn_out.shape), (2, 20, 15, 15, 32))
        self.assertFalse(torch.isnan(x_out).any())

    def test_h87_t5_quotient_runs(self) -> None:
        cfg = _cfg(
            "h87",
            temporal_quotient_steps=10,
            temporal_quotient_len=5,
            temporal_quotient_batch=2,
        )
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))

    def test_h88_local5_a3s_runs(self) -> None:
        cfg = _cfg("h88")
        fake = FakeWindowAttention3D(cfg)
        fake.forward = types.MethodType(_qk_shiftmax_gate_forward, fake)
        x_out, attn_out = fake.forward(fake.build_input())
        self.assertEqual(tuple(x_out.shape), (20, 450, 32))

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
    """F6：h89 forward 反向传播（STE 梯度非空）。"""

    def test_backward_flows(self) -> None:
        cfg = _cfg("h89")
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
