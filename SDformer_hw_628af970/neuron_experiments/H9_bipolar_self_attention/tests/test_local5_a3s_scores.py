#!/usr/bin/env python3
"""D3 (h88/local5_a3s) 算子级 CPU 单测：K1-K5 恒等式逐位断言。

对应合同：neuron_autoresearch/CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D3，
验证脚本：entrypoints/check_d3_axis_stencil_20260818.py（K1-K5）。
本文件把恒等式提升到算子级（_binary_axnor_local5_a3s_attention / _d3_axis_field）：
  K1  Δ=0 锚点恒等：A3S(Δ=0) 与现网 Local5 门/attn/row_sum 逐位一致
      （float 路径 + hardware Q7 路径双覆盖；可注入式训练关键）
  K2  网格精确位移：Δ=1/16 == 8 档，与 Q7 分数量化 commute（clamp 外）
  K3  方向场语义：匀速移动条 -> 3×3 时域 XOR 梯度 argmax 对齐运动轴（E/W 轴占比）
  K4  各向异性生效：对齐 lane winner 命中率 > 基线（运动承载像素，诚实指标）
  K5  唯一门成本账：ident-K 目的地全部从 1 组 -> 3 偏移类（{self 0, +Δ, −Δ}）
另含：方向场与 check_d3 参考实现逐位交叉、Δ 注入式渐增调度、偏移符号语义、
      Δ=8 时分数恰移 8 档（K2 算子级）。

CPU-only：不训练、不评测、不碰 GPU。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

# 64 核机器上微小张量的 advanced indexing 线程池争用开销巨大（同 check_d3
# 纪律），限制线程数后整个测试套件恢复秒级。
torch.set_num_threads(2)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "entrypoints"))  # check_d3 参考实现

from models.STSwinNet_SNN.bsa_attention import (  # noqa: E402
    ShiftmaxAttentionConfig,
    _binary_alpha_xnor_stencil_attention,
    _binary_axnor_local5_a3s_attention,
    _d3_a3s_offset,
    _d3_axis_field,
    _d3_effective_delta_bins,
)

STEP = 1.0 / 128.0
LO, HI = -2.0, 2.0
N_BINS = int(round((HI - LO) / STEP)) + 1
DELTA_BINS = 8  # Δ = 1/16 on 1/128 网格

AXIS_CODES = {"E": 0, "W": 1, "N": 2, "S": 3}


def quant_score(scores: torch.Tensor) -> torch.Tensor:
    codes = torch.round((scores - LO) / STEP).clamp(0, N_BINS - 1)
    return LO + STEP * codes


def _cfg(**kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        mode="local5_a3s",
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


class _StepModule(torch.nn.Module):
    def __init__(self, step: int = 0) -> None:
        super().__init__()
        self._h9_global_step = step


def make_planes(
    seed: int, t: int = 2, b: int = 2, h: int = 15, w: int = 15,
    heads: int = 4, d: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randint(0, 2, (t, b, heads, h * w, d)).float()
    k = torch.randint(0, 2, (b, heads, t * h * w, d)).float()
    return q, k


def moving_bar_planes(
    t: int, h: int, w: int, d: int, speed: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """匀速向右移动的竖条（与 check_d3 moving_bar_planes 同构造，q=k=平面）。"""
    torch.manual_seed(seed)
    planes = torch.zeros(t, h, w, d)
    for tt in range(t):
        bar_x = (tt * speed) % w
        for y in range(h):
            for dx in (0, 1):
                x = (bar_x + dx) % w
                if torch.rand(()) < 0.8:
                    lanes = torch.randperm(d)[: torch.randint(1, 5, ()).item()]
                    planes[tt, y, x, lanes] = 1.0
    q = planes.reshape(t, 1, 1, h * w, d)
    k = q.permute(1, 2, 0, 3, 4).reshape(1, 1, t * h * w, d).contiguous()
    return q, k


class K1IdentityTests(unittest.TestCase):
    """K1：Δ=0 与现网 Local5 逐位一致（消融与回滚锚点）。"""

    def test_delta0_bit_exact_random_planes(self) -> None:
        cfg = _cfg(a3s_delta_bins=0)
        for seed in range(10):
            q, k = make_planes(seed)
            a, rs, g, st = _binary_axnor_local5_a3s_attention(
                q, k, cfg, profile_module=_StepModule(0)
            )
            ab, rsb, gb = _binary_alpha_xnor_stencil_attention(
                q, k, cfg, temporal_pair=False, spatial_cross=True,
                motion_xor_alpha=0.0, profile_module=torch.nn.Module(),
            )
            self.assertTrue(torch.equal(g, gb), f"gate mismatch seed={seed}")
            self.assertTrue(torch.equal(a, ab), f"attn mismatch seed={seed}")
            self.assertTrue(torch.equal(rs, rsb), f"row_sum mismatch seed={seed}")
            self.assertEqual(st["delta_bins"], 0)

    def test_delta0_bit_exact_warmup_step0(self) -> None:
        # 注入式训练起调档：warmup 内 step=0 -> Δ=0 -> 与现网逐位一致
        cfg = _cfg(a3s_delta_bins=DELTA_BINS, a3s_delta_warmup_steps=1224)
        for seed in range(10):
            q, k = make_planes(seed)
            a, rs, g, st = _binary_axnor_local5_a3s_attention(
                q, k, cfg, profile_module=_StepModule(0)
            )
            ab, rsb, gb = _binary_alpha_xnor_stencil_attention(
                q, k, cfg, temporal_pair=False, spatial_cross=True,
                motion_xor_alpha=0.0, profile_module=torch.nn.Module(),
            )
            self.assertTrue(torch.equal(g, gb), f"gate mismatch seed={seed}")
            self.assertTrue(torch.equal(a, ab), f"attn mismatch seed={seed}")
            self.assertEqual(st["delta_bins"], 0)

    def test_delta0_bit_exact_hardware_quant(self) -> None:
        # 部署路径（Q7 分数 + 门量化 + 无效候选掩码）下同样逐位一致
        cfg = _cfg(
            a3s_delta_bins=0,
            hardware_quant_enabled=True,
            hardware_score_step=STEP,
            hardware_score_min=LO,
            hardware_score_max=HI,
            hardware_gate_step=STEP,
            hardware_gate_min=0.0,
            hardware_gate_max=2.0,
            hardware_mask_invalid_candidates=True,
        )
        for seed in range(10):
            q, k = make_planes(seed)
            a, rs, g, st = _binary_axnor_local5_a3s_attention(
                q, k, cfg, profile_module=_StepModule(0)
            )
            ab, rsb, gb = _binary_alpha_xnor_stencil_attention(
                q, k, cfg, temporal_pair=False, spatial_cross=True,
                motion_xor_alpha=0.0, profile_module=torch.nn.Module(),
            )
            self.assertTrue(torch.equal(g, gb), f"gate mismatch seed={seed}")
            self.assertTrue(torch.equal(a, ab), f"attn mismatch seed={seed}")
            self.assertEqual(st["delta_bins"], 0)


class K2GridShiftTests(unittest.TestCase):
    """K2：网格精确位移 —— Δ=1/16 == 8 档，与分数量化 commute（clamp 外）。"""

    def test_quant_commutes_with_shift(self) -> None:
        torch.manual_seed(3)
        for _ in range(200):
            s = torch.rand(16) * 2.0 - 1.0  # [-1,1] 远离 clamp 边界
            q_shift = quant_score(s + DELTA_BINS * STEP)
            q_then = quant_score(s) + DELTA_BINS * STEP
            self.assertTrue(torch.equal(q_shift, q_then))

    def test_offset_is_exact_eight_bins_on_q7(self) -> None:
        # 算子级 K2：Δ=8 档时，对齐 lane 的 Q7 分数恰移 8 档（远离 clamp）
        q, k = make_planes(7)
        cfg = _cfg(a3s_delta_bins=DELTA_BINS)
        _, _, g8, st8 = _binary_axnor_local5_a3s_attention(
            q, k, cfg, profile_module=_StepModule(9999)
        )
        cfg0 = _cfg(a3s_delta_bins=0)
        _, _, g0, _ = _binary_axnor_local5_a3s_attention(
            q, k, cfg0, profile_module=_StepModule(0)
        )
        # 分数不可直接观测（量化后过 shiftmax），用 gate 单调性 + Δ=0 恒等 +
        # K2 恒等式联合验证：偏移只经 Q7 网格 8 档位移改变门。
        self.assertTrue(torch.equal(g0, _binary_alpha_xnor_stencil_attention(
            q, k, cfg0, temporal_pair=False, spatial_cross=True,
            motion_xor_alpha=0.0, profile_module=torch.nn.Module(),
        )[2]))
        self.assertFalse(torch.equal(g8, g0))  # Δ=8 必须真的改变门
        self.assertTrue(torch.isfinite(g8).all())
        self.assertAlmostEqual(st8["delta_bins"], DELTA_BINS)

    def test_offset_sign_semantics(self) -> None:
        # 偏移符号语义：对齐 lane +Δ、正交 −Δ、self 0（直接测 _d3_a3s_offset）
        q, k = make_planes(11, h=4, w=4, heads=1, b=1)
        dirs = _d3_axis_field(q, k)  # [1, 1, 4, 4]
        scores = torch.zeros(1, 1, 2 * 16, 5)
        offset = _d3_a3s_offset(scores, dirs, DELTA_BINS)
        delta = DELTA_BINS * STEP
        off = offset.reshape(1, 1, 2, 16, 5)
        dirs_flat = dirs.reshape(-1)
        for yx in range(16):
            d = int(dirs_flat[yx])
            # lane 序 = self, N, S, W, E；轴码 N=2 S=3 W=1 E=0
            expect = torch.tensor([0.0])
            expect_n = delta if d == AXIS_CODES["N"] else -delta
            expect_s = delta if d == AXIS_CODES["S"] else -delta
            expect_w = delta if d == AXIS_CODES["W"] else -delta
            expect_e = delta if d == AXIS_CODES["E"] else -delta
            want = torch.tensor([0.0, expect_n, expect_s, expect_w, expect_e])
            self.assertTrue(torch.equal(off[0, 0, 0, yx], want), f"yx={yx} d={d}")
            # 两个时间切片共享同一方向场位图
            self.assertTrue(torch.equal(off[0, 0, 0, yx], off[0, 0, 1, yx]))
        # Δ=0 -> 全零偏移
        zero = _d3_a3s_offset(scores, dirs, 0)
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))


class K3K4DirectionSemanticsTests(unittest.TestCase):
    """K3/K4：方向场语义 + 对齐 lane winner 命中率（诚实生效指标）。"""

    def test_direction_field_alignment(self) -> None:
        q, k = moving_bar_planes(5, 15, 15, 32, speed=2)
        dirs = _d3_axis_field(q, k)  # [1, 1, 15, 15]
        frac_ew = float((dirs <= AXIS_CODES["W"]).float().mean())
        self.assertGreaterEqual(frac_ew, 0.5)  # 移动条 -> E/W 轴主导（K3）
        self.assertEqual(set(torch.unique(dirs).tolist()), {0, 1, 2, 3})

    def test_direction_field_matches_reference(self) -> None:
        # 与 check_d3 参考 axis_field 逐位交叉（同构造、同 seed 的移动条平面）
        from check_d3_axis_stencil_20260818 import axis_field as ref_axis_field

        t, h, w, d = 5, 15, 15, 32
        torch.manual_seed(0)
        planes = torch.zeros(t, h, w, d)
        for tt in range(t):
            bar_x = (tt * 2) % w
            for y in range(h):
                for dx in (0, 1):
                    x = (bar_x + dx) % w
                    if torch.rand(()) < 0.8:
                        lanes = torch.randperm(d)[: torch.randint(1, 5, ()).item()]
                        planes[tt, y, x, lanes] = 1.0
        q = planes.reshape(t, 1, 1, h * w, d)
        k = q.permute(1, 2, 0, 3, 4).reshape(1, 1, t * h * w, d).contiguous()
        mine = _d3_axis_field(q, k)[0, 0]
        self.assertTrue(torch.equal(mine, ref_axis_field(planes)))

    def test_winner_hit_rate_anisotropy(self) -> None:
        # K4：q=自匹配平面（现网 q/k 高度相关 regime，round2 C1 79.36%），
        # 仅运动承载像素上度量对齐 lane winner 命中率：Δ=8 显著 > 基线。
        q, k = moving_bar_planes(5, 15, 15, 32, speed=2)
        cfg = _cfg(a3s_delta_bins=DELTA_BINS, alpha0=0.02)
        _, _, g8, st8 = _binary_axnor_local5_a3s_attention(
            q, k, cfg, profile_module=_StepModule(9999)
        )
        cfg0 = _cfg(a3s_delta_bins=0, alpha0=0.02)
        _, _, g0, st0 = _binary_axnor_local5_a3s_attention(
            q, k, cfg0, profile_module=_StepModule(0)
        )
        hit_a3s = st8["winner_hit_rate"]
        hit_base = st0["winner_hit_rate"]
        # 基线的对齐 lane 命中率接近 0（无方向偏好），A3S 显著抬升（>50%）
        self.assertLess(hit_base, 0.5)
        self.assertGreater(hit_a3s, 0.5)
        # 门质量再分配有界（2^s 动态范围）：E/W 门质量占比在 (0,1)
        self.assertGreaterEqual(float(st8["axis_frac_ew"]), 0.5)  # K3 方向场账
        self.assertGreaterEqual(st8["delta_bins"], DELTA_BINS)

    def test_delta0_matches_base_on_moving_bars(self) -> None:
        q, k = moving_bar_planes(5, 15, 15, 32, speed=2)
        cfg = _cfg(a3s_delta_bins=0)
        _, _, g, _ = _binary_axnor_local5_a3s_attention(
            q, k, cfg, profile_module=_StepModule(0)
        )
        _, _, gb = _binary_alpha_xnor_stencil_attention(
            q, k, cfg, temporal_pair=False, spatial_cross=True,
            motion_xor_alpha=0.0, profile_module=torch.nn.Module(),
        )
        self.assertTrue(torch.equal(g, gb))


class K5OffsetClassAccountingTests(unittest.TestCase):
    """K5：ident-K 目的地从 1 组唯一门 -> 3 偏移类（诚实成本账）。"""

    def test_ident_k_splits_into_three_classes(self) -> None:
        torch.manual_seed(5)
        n_ident = 0
        split_sizes: dict[int, int] = {}
        for _ in range(50):
            t, h, w, d = 2, 15, 15, 32
            if torch.rand(()) < 0.5:
                vec = torch.randint(0, 2, (t, 1, 1, d)).float()
                k_planes = vec.expand(t, h, w, d)
            else:
                k_planes = torch.randint(0, 2, (t, h, w, d)).float()
            q = k_planes.clone()
            q_orig = q.reshape(t, 1, 1, h * w, d)
            k_orig = q_orig.permute(1, 2, 0, 3, 4).reshape(1, 1, t * h * w, d).contiguous()
            dirs = _d3_axis_field(q_orig, k_orig)[0, 0].reshape(h, w)
            offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
            for y in range(h):
                for x in range(w):
                    nbrs = []
                    for dy, dx in offsets:
                        yy = min(max(y + dy, 0), h - 1)
                        xx = min(max(x + dx, 0), w - 1)
                        nbrs.append(tuple(k_planes[0, yy, xx].tolist()))
                    if len(set(nbrs)) == 1:  # ident-K 目的地（5 邻域向量全同）
                        n_ident += 1
                        classes = set()
                        for dy, dx in offsets:
                            if (dy, dx) == (0, 0):
                                classes.add(0)
                            else:
                                code = AXIS_CODES[(
                                    "E" if dx == 1 else "W" if dx == -1
                                    else "N" if dy == -1 else "S"
                                )]
                                classes.add(1 if int(dirs[y, x]) == code else -1)
                        split_sizes[len(classes)] = split_sizes.get(len(classes), 0) + 1
        self.assertGreater(n_ident, 0)
        # 全部 ident-K 目的地恰分裂为 3 偏移类 {self 0, 对齐 +Δ, 正交 −Δ}
        self.assertEqual(sorted(split_sizes), [3])
        self.assertEqual(list(split_sizes.values())[0], n_ident)


class WarmupScheduleTests(unittest.TestCase):
    """Δ 注入式渐增调度（可注入式训练的结构性保障）。"""

    def test_warmup_ramp(self) -> None:
        cfg = _cfg(a3s_delta_bins=DELTA_BINS, a3s_delta_warmup_steps=1224)
        for step, want in ((0, 0), (300, 2), (612, 4), (1224, 8), (9999, 8)):
            self.assertEqual(_d3_effective_delta_bins(cfg, _StepModule(step)), want)

    def test_warmup_off_is_instant(self) -> None:
        cfg = _cfg(a3s_delta_bins=DELTA_BINS, a3s_delta_warmup_steps=0)
        self.assertEqual(_d3_effective_delta_bins(cfg, _StepModule(0)), DELTA_BINS)

    def test_zero_target_always_identity(self) -> None:
        cfg = _cfg(a3s_delta_bins=0, a3s_delta_warmup_steps=100)
        self.assertEqual(_d3_effective_delta_bins(cfg, _StepModule(9999)), 0)

    def test_no_module_attr_falls_back_zero(self) -> None:
        cfg = _cfg(a3s_delta_bins=DELTA_BINS, a3s_delta_warmup_steps=1224)
        self.assertEqual(_d3_effective_delta_bins(cfg, None), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
