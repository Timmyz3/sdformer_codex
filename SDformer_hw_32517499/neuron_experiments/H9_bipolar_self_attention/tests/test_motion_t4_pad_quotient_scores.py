#!/usr/bin/env python3
"""B2 (h87b) T=4+pad wildcard 时间商算子级 CPU 单测。

对应合同：D1_VARIANT_SEARCH_20260819.md §4.1（T=4+pad12，plan B 预案），
实现说明：neuron_autoresearch/B2_MOTION_T4_PAD_IMPLEMENTATION_20260819.md。
num_steps=10 -> 3 组 T=4：(0,1,2,3)/(4,5,6,7)/(8,9,pad,pad)，末组 2 个 pad。

  P1  pad 掩码恒等式：pad 槽不参与商组——不贡献 run-length 统计（无 eq 边、
      无 run 断点，wildcard 合并）、不进 slot 融合式、广播按掩码跳过；跨组
      边 (3,4)/(7,8) 不得影响任何真实槽分数（I7 同 D1 语义）
  P2  真实槽与 D1 同参数逐位一致：槽 0..7 与 9 与 h87（len=5）逐位相同；
      槽 8 按 B2 组首槽约定采用组内第 1 条边 (8,9)（D1 槽 8 用边 (7,8)）
  P3  run-length 账含 pad 跳过：每位置总独立门 = Σ_g (1 + Σ_{组内真实边}(1−eq))
      = 3 + 7·(1−p̄)；末组 E[runs] = 1 + (1−eq_8,9)；位账 −61.4% 口径
  P4  I1/I2/I5 在真实槽上成立（规范融合式 / r∈{0,1,2} / 反解唯一）
  P5  布局写回、batch 分解、校验（len≠4 / steps%4==0 / 运动 alpha）、STE

CPU-only：不训练、不评测、不碰 GPU。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))

from models.STSwinNet_SNN.bsa_attention import (
    ShiftmaxAttentionConfig,
    _binary_event_ste,
    _binary_t4_pad_quotient_token_scores,
    _binary_t5_quotient_token_scores,
    _d1_decompose_temporal_batch,
    _rne16_div_pow2_ste,
    _tx_sc_fusion_score_pair,
    shiftmax,
)

LANES = 32
MAX_SCORE = 162
NUM_STEPS = 10
QUOTIENT_LEN = 4
N_GROUPS = 3
GROUP_LENGTHS = (4, 4, 2)


def rne16(n: int) -> int:
    """RNE 除以 16（与 check_d1 / RTL 同式）。"""
    q, r = divmod(n, 16)
    return q + (1 if (r > 8 or (r == 8 and (q & 1))) else 0)


def h67_slot_score(o: int, sz: int, motion: int) -> int:
    """规范融合式（部署同式）：min(RNE16(64o + sz + 16m), 162)。"""
    return min(rne16(64 * o + sz + 16 * motion), MAX_SCORE)


def _cfg(**kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        temporal_quotient_steps=NUM_STEPS,
        temporal_quotient_len=QUOTIENT_LEN,
        temporal_quotient_batch=2,
    )
    values.update(kwargs)
    return ShiftmaxAttentionConfig(**values)


def assemble_bins(
    bins: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 10 个 1D (Q, K) 图案装配为 B*=10 的 q_orig/k_orig（H=1, N=1, n_sw=2）。

    同一图案复制到 2 个空间窗：cfg batch=1 -> 分解 (1, 2)，行序
    row = wd·n_sw + s（window_partition_v2 一致），两个窗分数逐位相同。
    """
    assert len(bins) == NUM_STEPS
    q_rows, k_rows = [], []
    for wd in range(5):  # wd-major：row = wd·n_sw + s
        for s in range(2):
            q_rows.append(
                torch.stack(
                    [
                        bins[2 * wd][0].unsqueeze(0).unsqueeze(0),
                        bins[2 * wd + 1][0].unsqueeze(0).unsqueeze(0),
                    ],
                    dim=0,
                )
            )
            k_rows.append(
                torch.stack(
                    [
                        bins[2 * wd][1].unsqueeze(0).unsqueeze(0),
                        bins[2 * wd + 1][1].unsqueeze(0).unsqueeze(0),
                    ],
                    dim=0,
                )
            )
    q_orig = torch.stack(q_rows, dim=1)  # [2, 10, 1, 1, D]
    k_orig = torch.stack(k_rows, dim=1).permute(1, 2, 0, 3, 4).reshape(10, 1, 2, LANES)
    return q_orig, k_orig


def make_pair_window(
    *,
    n_rows: int = 30,
    n_pairs: int = 5,
    heads: int = 1,
    spatial: int = 4,
    cfg: ShiftmaxAttentionConfig | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """构造随机二值事件的 q_orig/k_orig（行序与 window_partition_v2 一致）。"""
    torch.manual_seed(seed)
    if cfg is None:
        cfg = _cfg()
    batch_actual, n_sw = _d1_decompose_temporal_batch(n_rows, n_pairs, cfg)
    n_bins = n_pairs * 2
    bins = torch.randint(
        0, 2, (n_bins, batch_actual, n_sw, heads, spatial, LANES)
    ).to(torch.float32)
    q_parts, k_parts = [], []
    for b in range(batch_actual):
        for wd in range(n_pairs):
            for s in range(n_sw):
                q_parts.append(torch.stack([bins[2 * wd, b, s], bins[2 * wd + 1, b, s]], dim=0))
                k_parts.append(torch.stack([bins[2 * wd, b, s], bins[2 * wd + 1, b, s]], dim=0))
    q_orig = torch.stack(q_parts, dim=1)  # [2, B*, H, N, D]
    k_orig = torch.stack(k_parts, dim=1).permute(1, 2, 0, 3, 4).reshape(
        n_rows, heads, 2 * spatial, LANES
    )
    return q_orig, k_orig, (batch_actual, n_sw)


def reference_slot_scores(q_orig: torch.Tensor, k_orig: torch.Tensor, cfg) -> torch.Tensor:
    """纯 Python 参考实现：逐 (b, s) 真实槽 (o, sz, m) -> 规范融合分数 [B*, H, 10, N]。"""

    t_steps, batch, heads, spatial, lanes = q_orig.shape
    num_steps = int(cfg.temporal_quotient_steps)
    qlen = int(cfg.temporal_quotient_len)
    n_pairs = num_steps // t_steps
    _, n_sw = _d1_decompose_temporal_batch(batch, n_pairs, cfg)
    q_bin = (q_orig > 0).to(torch.int64)
    k_bin = (k_orig > 0).to(torch.int64)
    slots = torch.zeros(batch, heads, num_steps, spatial, dtype=torch.int64)
    for idx in range(batch):
        b = idx // (n_pairs * n_sw)
        s = idx % n_sw
        base_row = b * (n_pairs * n_sw) + s
        for tb in range(num_steps):
            group = tb // qlen
            first = group * qlen
            last = min((group + 1) * qlen, num_steps)
            row = base_row + (tb // 2) * n_sw
            t_local = tb % 2
            q = q_bin[t_local, row]
            k = k_bin[row, :, t_local * spatial : (t_local + 1) * spatial, :]
            for h in range(heads):
                for n in range(spatial):
                    q_t = int(q[h, n].sum().item())
                    k_t = int(k[h, n].sum().item())
                    o_t = int((q[h, n] & k[h, n]).sum().item())
                    sz_t = lanes - q_t - k_t + o_t
                    e = first if tb == first else tb - 1  # 组内首槽复用组内第 1 条边
                    bin_a, bin_b = e, e + 1
                    wd_a, tl_a = divmod(bin_a, 2)
                    wd_b, tl_b = divmod(bin_b, 2)
                    row_a = (b * n_pairs + wd_a) * n_sw + s
                    row_b = (b * n_pairs + wd_b) * n_sw + s
                    ka = k_bin[row_a, h, tl_a * spatial + n, :]
                    kb = k_bin[row_b, h, tl_b * spatial + n, :]
                    m_t = int((ka ^ kb).sum().item())
                    slots[idx, h, tb, n] = h67_slot_score(o_t, sz_t, m_t)
    return slots


class PadMaskIdentityTests(unittest.TestCase):
    """P1：pad 槽 wildcard 掩码恒等式。"""

    def test_pad_mask_shape_and_content(self) -> None:
        q_orig, k_orig, _ = make_pair_window(n_rows=10, heads=2, spatial=4,
                                             cfg=_cfg(temporal_quotient_batch=1), seed=5)
        _, _, views = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, _cfg(temporal_quotient_batch=1))
        pad_mask = views["pad_mask"]
        self.assertEqual(tuple(pad_mask.shape), (10, 2, N_GROUPS, QUOTIENT_LEN, 4))
        # 末组 2 个 pad 槽在组布局的 2、3 位；其余组无 pad
        self.assertTrue((pad_mask[:, :, 0, :, :] == False).all())
        self.assertTrue((pad_mask[:, :, 1, :, :] == False).all())
        self.assertTrue((pad_mask[:, :, 2, 0, :] == False).all())
        self.assertTrue((pad_mask[:, :, 2, 1, :] == False).all())
        self.assertTrue((pad_mask[:, :, 2, 2, :] == True).all())
        self.assertTrue((pad_mask[:, :, 2, 3, :] == True).all())

    def test_slot_views_contain_only_real_slots(self) -> None:
        q_orig, k_orig, _ = make_pair_window(n_rows=20, heads=1, spatial=6, seed=11)
        _, _, views = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, _cfg())
        self.assertEqual(tuple(views["scores"].shape), (20, 1, NUM_STEPS, 6))
        self.assertEqual(tuple(views["overlap"].shape), (20, 1, NUM_STEPS, 6))
        self.assertEqual(tuple(views["motion"].shape), (20, 1, NUM_STEPS, 6))
        self.assertEqual(tuple(views["grouped_runs"].shape), (20, 1, N_GROUPS, 6))

    def test_cross_group_edges_never_affect_real_slots(self) -> None:
        # 跨组边 (3,4) 与 (7,8) 的内容差异不得改变任何真实槽分数（I7 wildcard）
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k_a = k.clone()
        k_a[0] = 1.0
        k_b = k.clone()
        k_b[1] = 1.0
        base_bins = [(q, k)] * NUM_STEPS
        # A：边 (7,8) 有差异（XOR=2）；B：边 (7,8) 无差异
        bins_a = list(base_bins)
        bins_a[7] = (q, k_a)
        bins_a[8] = (q, k_b)
        bins_b = list(base_bins)
        # 变体 B：边 (7,8) 两侧相同（都用 k_a 图案的 popcount），(8,9) 保持无差异
        bins_b[7] = (q, k_a)
        bins_b[8] = (q, k_a)
        cfg = _cfg(temporal_quotient_batch=1)
        _, _, va = _binary_t4_pad_quotient_token_scores(*assemble_bins(bins_a), cfg)
        _, _, vb = _binary_t4_pad_quotient_token_scores(*assemble_bins(bins_b), cfg)
        self.assertTrue(
            torch.equal(va["scores"], vb["scores"]),
            "跨组边 (7,8) 内容不得影响任何真实槽分数（wildcard 掩码）",
        )
        # 同理跨组边 (3,4)
        bins_c = list(base_bins)
        bins_c[3] = (q, k_a)
        bins_c[4] = (q, k_b)
        bins_d = list(base_bins)
        bins_d[3] = (q, k_a)
        bins_d[4] = (q, k_a)
        _, _, vc = _binary_t4_pad_quotient_token_scores(*assemble_bins(bins_c), cfg)
        _, _, vd = _binary_t4_pad_quotient_token_scores(*assemble_bins(bins_d), cfg)
        self.assertTrue(torch.equal(vc["scores"], vd["scores"]))

    def test_uniform_scores_full_broadcast_with_pad_skip(self) -> None:
        # 全部真实槽同分（同图案窗口）：每组 1 个独立门，末组 pad 槽
        # wildcard 跳过不增加门 -> 每位置总门 3、saving 70%（非 80%：
        # 第三组 pad 不参与商组，按 len-2 记账 1+1(1−eq)=1）
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k[:5] = 1.0
        q[:3] = 1.0
        q_orig, k_orig = assemble_bins([(q, k)] * NUM_STEPS)
        _, rle, views = _binary_t4_pad_quotient_token_scores(
            q_orig, k_orig, _cfg(temporal_quotient_batch=1)
        )
        self.assertAlmostEqual(
            rle["mean_runs_per_position"], 3.0, places=5,
            msg="全同分 -> 每组 1 门，pad 不增加门，总门 3",
        )
        self.assertAlmostEqual(
            rle["broadcast_saving"], 1.0 - 3.0 / NUM_STEPS, places=5
        )
        self.assertAlmostEqual(
            float(views["grouped_runs"][:, :, 2].mean().item()), 1.0, places=5,
            msg="末组 2 真实槽同分 -> 1 个独立门（pad 不产生 run 断点）",
        )
        # 对照：同样的全同分在 D1（T=5，无 pad）下 saving = 80%（2 组 × 5 槽）
        d1_cfg = ShiftmaxAttentionConfig(
            temporal_quotient_steps=NUM_STEPS, temporal_quotient_len=5,
            temporal_quotient_batch=1,
        )
        _, rle_d1, _ = _binary_t5_quotient_token_scores(q_orig, k_orig, d1_cfg)
        self.assertAlmostEqual(rle_d1["broadcast_saving"], 0.8, places=5)

    def test_pad_group_runs_formula(self) -> None:
        # 末组 E[runs] = 1 + (1 - eq(8,9))：eq(8,9)=1 -> 1；eq(8,9)=0 -> 2
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        # 边 (8,9) 的 m = popcount(K8 ⊕ K9)；K8/K9 popcount 不同 -> 分数必不同
        k_a = k.clone()
        k_a[0] = 1.0  # k_a：1 个 1
        k_b = k.clone()
        k_b[:17] = 1.0  # k_b：17 个 1（o=0 -> sz 15 vs 31，RNE 后必差 1 档）
        cfg = _cfg(temporal_quotient_batch=1)
        # eq(8,9)=1：两槽同图案
        bins = [(q, k)] * NUM_STEPS
        _, _, views = _binary_t4_pad_quotient_token_scores(*assemble_bins(bins), cfg)
        runs_g2 = views["grouped_runs"][:, :, 2]
        self.assertAlmostEqual(float(runs_g2.mean().item()), 1.0, places=5)
        # eq(8,9)=0：K8 与 K9 图案不同（分数必不同 -> run 断点）
        bins2 = list(bins)
        bins2[8] = (q, k_a)
        bins2[9] = (q, k_b)
        _, rle2, views2 = _binary_t4_pad_quotient_token_scores(*assemble_bins(bins2), cfg)
        runs_g2 = views2["grouped_runs"][:, :, 2]
        self.assertAlmostEqual(float(runs_g2.mean().item()), 2.0, places=5,
                               msg="末组 E[runs] = 1 + (1-eq_8,9)（len-2 记账）")
        self.assertEqual(rle2["group_lengths"], GROUP_LENGTHS)
        self.assertEqual(rle2["pad_slots"], 2)
        self.assertEqual(rle2["coverage_edges"], 7)

    def test_rle_identity_with_manual_account(self) -> None:
        # 每位置总独立门 = Σ_g (1 + Σ_{组内真实边}(1-eq)) = 3 + Σ_7(1-eq)
        q_orig, k_orig, _ = make_pair_window(n_rows=60, heads=2, spatial=6, seed=23)
        _, rle, views = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, _cfg())
        scores = views["scores"]  # [B*, H, 10, N]
        total = torch.zeros_like(scores[:, :, 0, :], dtype=torch.float)
        n_eq = 0
        for g in range(N_GROUPS):
            first = g * QUOTIENT_LEN
            length = GROUP_LENGTHS[g]
            grp = scores[:, :, first : first + length, :]
            eq = grp[:, :, :-1].eq(grp[:, :, 1:]).to(torch.float)
            n_eq += int(eq.sum().item())
            total = total + (1.0 + (length - 1) - eq.sum(dim=2))
        manual_mean = float(total.float().mean().item())
        self.assertAlmostEqual(manual_mean, rle["mean_runs_per_position"], places=5)
        self.assertAlmostEqual(
            rle["broadcast_saving"], 1.0 - manual_mean / NUM_STEPS, places=6
        )
        self.assertAlmostEqual(rle["independent_gate_ratio"], manual_mean / NUM_STEPS, places=6)
        # eq 边率只数 7 条真实边
        n_pos = total.numel()
        manual_eq_rate = n_eq / (7 * n_pos)
        self.assertAlmostEqual(rle["eq_edge_rate"], manual_eq_rate, places=6)
        self.assertGreaterEqual(rle["eq_edge_rate"], 0.0)
        self.assertLessEqual(rle["eq_edge_rate"], 1.0)
        # 每位置总门数 ∈ [3, 10]（3 组下限 / 无广播上限）；位账恒等式
        # 3 + 7·(1−p̄)（p̄ 为真实 eq 边率）在 contract 口径 −61.4%（p̄≈0.879）
        # 处成立；随机二值数据下 p̄ 偏低、门数偏高，仅验恒等式与界。
        self.assertAlmostEqual(
            rle["mean_runs_per_position"],
            3.0 + 7.0 * (1.0 - rle["eq_edge_rate"]),
            places=5,
        )
        self.assertGreaterEqual(rle["mean_runs_per_position"], 3.0)
        self.assertLessEqual(rle["mean_runs_per_position"], 10.0)


class D1BitwiseConsistencyTests(unittest.TestCase):
    """P2：真实槽与 D1（h87, len=5）同参数逐位一致。"""

    def _run_both(self, cfg_kwargs: dict | None = None):
        cfg = _cfg(temporal_quotient_batch=1)
        q_orig, k_orig, _ = make_pair_window(
            n_rows=10, heads=1, spatial=8, cfg=cfg, seed=9
        )
        d1_cfg = ShiftmaxAttentionConfig(
            temporal_quotient_steps=NUM_STEPS,
            temporal_quotient_len=5,
            temporal_quotient_batch=1,
        )
        _, rle_d1, v_d1 = _binary_t5_quotient_token_scores(q_orig, k_orig, d1_cfg)
        _, rle_b2, v_b2 = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, cfg)
        return q_orig, k_orig, rle_d1, v_d1, rle_b2, v_b2

    def test_real_slots_bitwise_equal_to_h87(self) -> None:
        q_orig, k_orig, rle_d1, v_d1, rle_b2, v_b2 = self._run_both()
        s_d1 = v_d1["scores"]
        s_b2 = v_b2["scores"]
        # 槽 0..3：两组划分共享组 0（边 0,1,2）-> 逐位一致
        self.assertTrue(torch.equal(s_d1[:, :, :4], s_b2[:, :, :4]),
                        "槽 0..3 必须与 D1 逐位一致（同 (o,sz) 同边）")
        # 槽 9：两组划分都指定边 (8,9) -> 逐位一致
        self.assertTrue(torch.equal(s_d1[:, :, 9:10], s_b2[:, :, 9:10]),
                        "槽 9（边 (8,9)）必须与 D1 逐位一致")
        # 槽 4..8 是分组边界的预期差异（I7）：D1 的槽 4..8 用边 (3,4)..(7,8)，
        # B2 的槽 4..8 用边 (4,5)..(8,9)。分数由同一规范融合式给出——
        # 用 Python 参考实现（T=4 分组）逐位验证 B2 全部真实槽。
        ref = reference_slot_scores(q_orig, k_orig, _cfg(temporal_quotient_batch=1)).to(torch.float32)
        self.assertTrue(torch.equal(v_b2["scores"], ref),
                        "B2 全部真实槽分数必须与 Python 参考实现逐位一致")
        # 同 (o,sz,m) 输入下融合式逐位一致（I1 共享）：用运动全 0 构造，
        # 槽 4..8 的 (o,sz) 与 D1 相同且 m=0 -> 全部 10 槽逐位一致（退化路径）
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k[:5] = 1.0
        q[:3] = 1.0
        q_orig0, k_orig0 = assemble_bins([(q, k)] * NUM_STEPS)
        _, _, v0_d1 = _binary_t5_quotient_token_scores(
            q_orig0, k_orig0,
            ShiftmaxAttentionConfig(temporal_quotient_steps=NUM_STEPS,
                                    temporal_quotient_len=5,
                                    temporal_quotient_batch=1),
        )
        _, _, v0_b2 = _binary_t4_pad_quotient_token_scores(
            q_orig0, k_orig0, _cfg(temporal_quotient_batch=1)
        )
        self.assertTrue(torch.equal(v0_d1["scores"], v0_b2["scores"]),
                        "m=0 退化路径下 10 个真实槽必须与 D1 逐位一致")

    def test_slot8_adopts_group_first_edge(self) -> None:
        # 构造：边 (7,8) XOR=2（D1 槽 8 会看到），(8,9) XOR=1（k vs k_b）。
        # B2 槽 8 的 m 必须 = 1（组内首槽采用组内第 1 条边 (8,9)），
        # D1 槽 8 的 m = 2（边 (7,8)）——跨组边 (7,8) 对 B2 不可见。
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k_a = k.clone()
        k_a[0] = 1.0
        k_b = k.clone()
        k_b[1] = 1.0
        bins = [(q, k)] * NUM_STEPS
        bins[7] = (q, k_a)
        bins[8] = (q, k_b)  # 边 (7,8) = k_a⊕k_b = 2；边 (8,9) = k_b⊕k = 1
        cfg = _cfg(temporal_quotient_batch=1)
        _, _, v_b2 = _binary_t4_pad_quotient_token_scores(*assemble_bins(bins), cfg)
        motion = v_b2["motion"].reshape(10, 10)
        self.assertEqual(int(motion[0, 8].item()), 1,
                         "B2 槽 8 采用组内第 1 条边 (8,9)=1，而非 D1 的 (7,8)=2")
        self.assertEqual(int(motion[0, 9].item()), 1, "槽位 9 采用边 (8,9)=1")
        d1_cfg = ShiftmaxAttentionConfig(
            temporal_quotient_steps=NUM_STEPS, temporal_quotient_len=5,
            temporal_quotient_batch=1,
        )
        _, _, v_d1 = _binary_t5_quotient_token_scores(*assemble_bins(bins), d1_cfg)
        motion_d1 = v_d1["motion"].reshape(10, 10)
        self.assertEqual(int(motion_d1[0, 8].item()), 2, "D1 槽 8 用边 (7,8)（对照）")

    def test_h87_path_unchanged(self) -> None:
        # h87 的校验原样保留：len=4 必须抛 ValueError（h87 合同钉死 5）
        q_orig, k_orig, _ = make_pair_window(
            n_rows=10, heads=1, spatial=4,
            cfg=_cfg(temporal_quotient_batch=1), seed=3,
        )
        with self.assertRaises(ValueError):
            _binary_t5_quotient_token_scores(q_orig, k_orig, _cfg(temporal_quotient_batch=1))


class FusedFormAndDecompositionTests(unittest.TestCase):
    """P4：I1 规范融合式 / I2 槽位分解 / I5 反解（真实槽）。"""

    def test_fused_form_bitwise(self) -> None:
        for q_count, k_count, o in (
            (8, 8, 4),
            (16, 16, 12),
            (3, 5, 2),
            (20, 20, 18),
            (32, 32, 32),
            (0, 0, 0),
        ):
            sz = LANES - q_count - k_count + o
            for m in (0, 1, 7, 16, 31, 32):
                q_pat = torch.zeros(LANES)
                q_pat[:q_count] = 1.0
                k_pat = torch.zeros(LANES)
                k_pat[:o] = 1.0
                k_pat[q_count : q_count + (k_count - o)] = 1.0
                diff = torch.zeros(LANES)
                diff[:m] = 1.0
                k_slot_alt = ((k_pat + diff) % 2).to(torch.float32)
                bins = [(q_pat, k_pat)] * NUM_STEPS
                bins[1] = (q_pat, k_slot_alt)
                o1 = int(((q_pat > 0) & (k_slot_alt > 0)).sum().item())
                k1 = int(k_slot_alt.sum().item())
                sz1 = LANES - q_count - k1 + o1
                # T=4 组 0：(0,1,2,3) 首槽 0 用边 (0,1)=m；槽 1 用 k_slot_alt
                # 与边 (0,1)=m；槽 2 用边 (1,2)=m；槽 3 用边 (2,3)=0。
                # 组 1/组 2（(4..7)/(8,9)）全部边 = 0。
                expected = {
                    0: h67_slot_score(o, sz, m),
                    1: h67_slot_score(o1, sz1, m),
                    2: h67_slot_score(o, sz, m),
                    3: h67_slot_score(o, sz, 0),
                    4: h67_slot_score(o, sz, 0),
                    5: h67_slot_score(o, sz, 0),
                    6: h67_slot_score(o, sz, 0),
                    7: h67_slot_score(o, sz, 0),
                    8: h67_slot_score(o, sz, 0),  # 组 2 首槽：边 (8,9)=0
                    9: h67_slot_score(o, sz, 0),
                }
                q_orig, k_orig = assemble_bins(bins)
                cfg = _cfg(temporal_quotient_batch=1)
                _, rle, views = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, cfg)
                slot_scores = views["scores"].reshape(10, 10)
                for wd in range(5):
                    for tl in range(2):
                        tb = 2 * wd + tl
                        self.assertEqual(
                            int(slot_scores[wd, tb].item()), expected[tb],
                            f"wd={wd} tb={tb} (o,sz,m)=({o},{sz},{m})",
                        )
                self.assertGreaterEqual(rle["mean_runs_per_position"], 1.0)
                self.assertLessEqual(rle["mean_runs_per_position"], 10.0)

    def test_slot_decomposition_and_inverse_unique(self) -> None:
        q_orig, k_orig, _ = make_pair_window(n_rows=60, heads=2, spatial=6, seed=31)
        _, _, views = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, _cfg())
        slot_scores = views["scores"].reshape(-1).to(torch.int64)
        overlap = views["overlap"].reshape(-1).to(torch.int64)
        motion = views["motion"].reshape(-1).to(torch.int64)
        sm = slot_scores - motion
        o_recovered = torch.div(sm, 4, rounding_mode="floor")
        r_recovered = torch.remainder(sm, 4)
        self.assertTrue(((r_recovered >= 0) & (r_recovered <= 2)).all(),
                        "物理域内反解 r = (s-m)%4 ∈ {0,1,2}（I5）")
        self.assertTrue(torch.equal(o_recovered, overlap), "反解 (s-m)//4 必须恢复 o_t")
        # m=0 构造：无 s%4==3（I2）
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k[:5] = 1.0
        q[:3] = 1.0
        _, _, views0 = _binary_t4_pad_quotient_token_scores(
            *assemble_bins([(q, k)] * NUM_STEPS), _cfg(temporal_quotient_batch=1)
        )
        s0 = views0["scores"].reshape(-1).to(torch.int64)
        self.assertTrue(torch.equal(views0["motion"].reshape(-1),
                                    torch.zeros_like(views0["motion"].reshape(-1))))
        self.assertFalse((s0 % 4 == 3).any(), "m=0 槽位分数无 s%4==3（I2）")


class EdgeCoverageTests(unittest.TestCase):
    """P1/I7：时间边覆盖 7/9——(3,4) 与 (7,8) 不可见。"""

    def test_only_intra_group_edges_visible(self) -> None:
        # 边 e 的 XOR=2 若进入某个槽位，该槽运动项 == 2；跨组边 (3,4)/(7,8)
        # 的 XOR=2 不得出现（相邻组内边的 XOR=1 可正常出现，max == 1）。
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k_a = k.clone()
        k_a[0] = 1.0
        k_b = k.clone()
        k_b[1] = 1.0
        visible = set()
        invisible = set()
        for e in range(9):
            bins = [(q, k)] * NUM_STEPS
            bins[e] = (q, k_a)
            bins[e + 1] = (q, k_b)
            _, _, views = _binary_t4_pad_quotient_token_scores(
                *assemble_bins(bins), _cfg(temporal_quotient_batch=1)
            )
            motion = views["motion"].reshape(10, 10)
            m_max = float(motion.max().item())
            if m_max >= 2.0:
                visible.add(e)
            else:
                invisible.add(e)
                self.assertLessEqual(m_max, 1.0, f"边 {e} 的 XOR=2 不得入槽")
        self.assertEqual(visible, {0, 1, 2, 4, 5, 6, 8},
                         "可见边必须恰为 7/9（(3,4)/(7,8) 跨组不可见）")
        self.assertEqual(invisible, {3, 7})

    def test_cross_group_edges_never_in_motion(self) -> None:
        # 边 (3,4) 与 (7,8) 的差异（XOR=2）不得进入任何槽位运动项；
        # 相邻组内边（(2,3),(4,5),(6,7),(8,9)）的 XOR=1 正常入槽（max ≤ 1）。
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k_a = k.clone()
        k_a[0] = 1.0
        k_b = k.clone()
        k_b[1] = 1.0
        bins = [(q, k)] * NUM_STEPS
        bins[3] = (q, k_a)
        bins[4] = (q, k_b)
        bins[7] = (q, k_a)
        bins[8] = (q, k_b)
        _, _, views = _binary_t4_pad_quotient_token_scores(
            *assemble_bins(bins), _cfg(temporal_quotient_batch=1)
        )
        motion = views["motion"].reshape(10, 10)
        self.assertLessEqual(
            float(motion.max().item()), 1.0,
            "跨组边 (3,4)/(7,8)（XOR=2）不得进入任何槽位的运动项",
        )
        # 对照：D1（T=5）中边 (7,8) 是组 1 组内边，槽 8 能看到 XOR=2
        d1_cfg = ShiftmaxAttentionConfig(
            temporal_quotient_steps=NUM_STEPS, temporal_quotient_len=5,
            temporal_quotient_batch=1,
        )
        _, _, v_d1 = _binary_t5_quotient_token_scores(*assemble_bins(bins), d1_cfg)
        self.assertEqual(float(v_d1["motion"].max().item()), 2.0,
                         "对照：D1 中 (7,8) 为组内边，XOR=2 必须可见")


class LayoutAndBoundaryTests(unittest.TestCase):
    """P5：分组、布局写回、batch 分解、校验、STE。"""

    def test_grouping_matches_python_reference(self) -> None:
        cfg = _cfg()
        q_orig, k_orig, _ = make_pair_window(n_rows=30, heads=2, spatial=6, cfg=cfg, seed=41)
        _, _, views = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, cfg)
        ref = reference_slot_scores(q_orig, k_orig, cfg).to(torch.float32)
        self.assertTrue(torch.equal(views["scores"], ref),
                        "算子槽位分数必须与 Python 参考实现逐位一致")
        # 布局写回：token (t_local, n) 于行 idx 取槽 2*wd(idx)+t_local
        scores, _, views = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, cfg)
        batch, heads, n2, lanes = k_orig.shape
        n_pairs = cfg.temporal_quotient_steps // 2
        _, n_sw = _d1_decompose_temporal_batch(batch, n_pairs, cfg)
        expected_out = torch.zeros_like(scores)
        for idx in range(batch):
            wd = (idx // n_sw) % n_pairs
            for tl in range(2):
                expected_out[idx, :, tl * 6 : (tl + 1) * 6, 0] = ref[idx, :, 2 * wd + tl, :]
        self.assertTrue(torch.equal(scores, expected_out), "分数必须写回原 token 布局")

    def test_first_slot_adopts_group_first_edge(self) -> None:
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k_a = k.clone()
        k_a[0] = 1.0
        k_b = k.clone()
        k_b[1] = 1.0
        bins = [(q, k)] * NUM_STEPS
        bins[8] = (q, k_a)
        bins[9] = (q, k_b)
        _, _, views = _binary_t4_pad_quotient_token_scores(
            *assemble_bins(bins), _cfg(temporal_quotient_batch=1)
        )
        motion = views["motion"].reshape(10, 10)
        self.assertEqual(int(motion[0, 8].item()), 2,
                         "组 2 首槽 t=8 必须采用组内第 1 条边 (8,9)")
        self.assertEqual(int(motion[0, 9].item()), 2, "槽位 t=9 采用边 (8,9)")

    def test_batch_decomposition(self) -> None:
        cfg2 = _cfg(temporal_quotient_batch=2)
        cfg1 = _cfg(temporal_quotient_batch=1)
        self.assertEqual(_d1_decompose_temporal_batch(880, 5, cfg2), (2, 88))
        self.assertEqual(_d1_decompose_temporal_batch(440, 5, cfg2), (1, 88))
        self.assertEqual(_d1_decompose_temporal_batch(440, 5, cfg1), (1, 88))
        self.assertEqual(_d1_decompose_temporal_batch(240, 5, cfg2), (2, 24))

    def test_validation_errors(self) -> None:
        q_orig, k_orig, _ = make_pair_window(
            n_rows=10, heads=1, spatial=4,
            cfg=_cfg(temporal_quotient_batch=1), seed=47,
        )
        # 组长度钉死 4
        with self.assertRaises(ValueError):
            _binary_t4_pad_quotient_token_scores(
                q_orig, k_orig, _cfg(temporal_quotient_len=5, temporal_quotient_batch=1)
            )
        # steps % 4 == 0 必须拒绝（无 pad 的整除归 h87）
        with self.assertRaises(ValueError):
            _binary_t4_pad_quotient_token_scores(
                q_orig, k_orig, _cfg(temporal_quotient_steps=8, temporal_quotient_batch=1)
            )
        with self.assertRaises(ValueError):
            _binary_t4_pad_quotient_token_scores(
                q_orig, k_orig, _cfg(temporal_quotient_steps=12, temporal_quotient_batch=1)
            )
        # steps 必须为偶（T=2 窗）
        with self.assertRaises(ValueError):
            _binary_t4_pad_quotient_token_scores(
                q_orig, k_orig, _cfg(temporal_quotient_steps=9, temporal_quotient_batch=1)
            )
        # T != 2 与 batch 分解失败
        with self.assertRaises(ValueError):
            _binary_t4_pad_quotient_token_scores(
                q_orig[:1], k_orig, _cfg(temporal_quotient_batch=1)
            )
        with self.assertRaises(ValueError):
            _d1_decompose_temporal_batch(7, 5, _cfg())
        # STE 梯度路径存在（不塌）
        q_orig.requires_grad_(True)
        k_orig.requires_grad_(True)
        scores, _, _ = _binary_t4_pad_quotient_token_scores(
            q_orig, k_orig, _cfg(temporal_quotient_batch=1)
        )
        loss = scores.float().sum()
        loss.backward()
        self.assertIsNotNone(q_orig.grad)
        self.assertIsNotNone(k_orig.grad)
        self.assertEqual(q_orig.grad.numel(), q_orig.numel())

    def test_rne16_helper_matches_python(self) -> None:
        torch.manual_seed(53)
        nums = torch.randint(0, 2600, (3000,)).to(torch.float32)
        got = _rne16_div_pow2_ste(nums.clone())
        for i in range(3000):
            self.assertEqual(int(got[i].item()), rne16(int(nums[i].item())))


class H67FixedPointAlignmentTests(unittest.TestCase):
    """F1 回归（2026-08-19 修复）：h87b（T=4+pad）真实槽 Q7 分数 ÷128 后
    与 h67 锚点 gate/attn 数值对齐（B2 与 D1 同受 F1 修复；验证标准同 D1：
    gate mean abs ≤3e-5、attn mean rel ≤5e-4。实测 5.8e-6 / 1.8e-4）。"""

    @staticmethod
    def synthetic_batch(
        *, B: int = 2, n_sw: int = 2, heads: int = 4, N: int = 225,
        dim: int = LANES, p: float = 0.06, seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(seed)
        t_steps = 10
        events = (rng.random((t_steps, B, n_sw, N, dim)) < p).astype(np.float32)
        rows = [(b, wd, s) for b in range(B) for wd in range(5) for s in range(n_sw)]
        q_orig = torch.zeros(2, len(rows), heads, N, dim)
        k_orig = torch.zeros(len(rows), heads, 2 * N, dim)
        for r, (b, wd, s) in enumerate(rows):
            for j in (0, 1):
                t = 2 * wd + j
                q_orig[j, r] = torch.from_numpy(events[t, b, s]).unsqueeze(0).expand(heads, N, dim)
                k_orig[r, :, j * N:(j + 1) * N] = torch.from_numpy(
                    events[t, b, s]
                ).unsqueeze(0).expand(heads, N, dim)
        return q_orig, k_orig

    def test_h87b_fixed_point_gate_attn_aligns_with_h67(self) -> None:
        q_orig, k_orig = self.synthetic_batch(seed=0)
        cfg_b2 = _cfg()  # temporal_quotient_len=4, batch=2, steps=10
        s_b2, _, _ = _binary_t4_pad_quotient_token_scores(q_orig, k_orig, cfg_b2)
        s_b2f = s_b2 / 128.0  # F1 定点解释（与 h87b forward 分支同式）
        g_b2 = shiftmax(s_b2f - s_b2f.mean(dim=2, keepdim=True), dim=2, eps=1e-6)
        # h67 锚点（h60 分支 float 路径）
        cfg_m = ShiftmaxAttentionConfig(
            mode="h60", binary_motion_xor_alpha=0.25, alpha0=0.02,
            mismatch_penalty=0.0, single_active_penalty=0.0,
            consensus_score_norm="head_dim", score_scale=1.0,
            center_scores=True, preserve_mean=True, eps=1e-6,
        )
        tx, _sc = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg_m)
        s67 = tx - tx.mean(dim=2, keepdim=True)
        g67 = shiftmax(s67, dim=2, eps=1e-6)
        k_ev = _binary_event_ste(k_orig)
        n_tok = 2 * 225
        a_b2 = k_ev * g_b2 * n_tok
        a67 = k_ev * g67 * n_tok
        gate_abs = float((g_b2 - g67).abs().mean())
        attn_rel = float(((a_b2 - a67).abs() / (a67.abs() + 1e-12)).mean())
        self.assertLessEqual(
            gate_abs, 3e-5,
            f"h87b÷128 后 gate mean abs 差须 ≤3e-5（实测 {gate_abs:.2e}）",
        )
        self.assertLessEqual(
            attn_rel, 5e-4,
            f"h87b÷128 后 attn mean rel 差须 ≤5e-4（实测 {attn_rel:.2e}）",
        )


if __name__ == "__main__":
    unittest.main()
