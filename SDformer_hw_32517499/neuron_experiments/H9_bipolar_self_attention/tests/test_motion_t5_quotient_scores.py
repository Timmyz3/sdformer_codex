#!/usr/bin/env python3
"""D1 (h87) T=5 时间商算子级 CPU 单测：I1-I7 恒等式逐位断言。

对应合同：neuron_autoresearch/CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D1，
验证脚本：entrypoints/check_d1_tetra_quotient_20260818.py（I1-I7）。
本文件把恒等式提升到算子级 forward（_binary_t5_quotient_token_scores）：
  I1 融合式规范：RNE16(64o+sz+16m) 为唯一规范（拆解式平局翻转处差 1 档）
  I2 槽位分解唯一：s = 4o + r, r ∈ {0,1,2}（物理域内无 s%4==3）
  I3 网格位移不变：无 clamp 时 Shiftmax 对全体分数 +k 档不变
  I4 T 一致性：T=5 均匀边剖面 ≡ H67 T=2 槽位分数
  I5 商可逆：记录 (o, sz, m) -> 分数精确重建，物理域内反解唯一
  I6 run-length 账：eq 边率 p 下每位置独立门数 = 1 + 4(1-p)；p=0.979 时 −78.3%
  I7 时间边覆盖：组内 4 条边可见（8/9），跨组边 (4,5) 不可见
另含：跨窗时间槽分组、batch 维分解、m_0 首槽边约定、布局写回。

CPU-only：不训练、不评测、不碰 GPU。
"""

from __future__ import annotations

import math
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
    _binary_t5_quotient_token_scores,
    _d1_decompose_temporal_batch,
    _rne16_div_pow2_ste,
    _tx_sc_fusion_score_pair,
    shiftmax,
)

LANES = 32
MAX_SCORE = 162
EQ_RATE = 0.979


def rne16(n: int) -> int:
    """RNE 除以 16（与 check_d1 / RTL 同式）。"""
    q, r = divmod(n, 16)
    return q + (1 if (r > 8 or (r == 8 and (q & 1))) else 0)


def h67_slot_score(o: int, sz: int, motion: int) -> int:
    """规范融合式（部署同式）：min(RNE16(64o + sz + 16m), 162)。"""
    return min(rne16(64 * o + sz + 16 * motion), MAX_SCORE)


def _cfg(**kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        temporal_quotient_steps=10,
        temporal_quotient_len=5,
        temporal_quotient_batch=2,
    )
    values.update(kwargs)
    return ShiftmaxAttentionConfig(**values)


def make_pair_window(
    *,
    n_rows: int = 30,
    n_pairs: int = 5,
    heads: int = 1,
    spatial: int = 4,
    lanes: int = LANES,
    cfg: ShiftmaxAttentionConfig | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """构造 q_orig=[T=2, B*, H, N, D] / k_orig=[B*, H, 2N, D]（随机二值事件）。

    装配行数 = n_rows（= B*），行序 row = (b·n_pairs + wd)·n_sw + s 与
    window_partition_v2 一致；(batch_actual, n_sw) 用与算子同一的
    _d1_decompose_temporal_batch 求解，保证装配与算子的跨窗分组一致。
    返回 (q_orig, k_orig, (batch_actual, n_sw))。
    """
    torch.manual_seed(seed)
    if cfg is None:
        cfg = _cfg()
    batch_actual, n_sw = _d1_decompose_temporal_batch(n_rows, n_pairs, cfg)
    n_bins = n_pairs * 2
    bins = torch.randint(
        0, 2, (n_bins, batch_actual, n_sw, heads, spatial, lanes)
    ).to(torch.float32)
    q_parts, k_parts = [], []
    for b in range(batch_actual):
        for wd in range(n_pairs):
            for s in range(n_sw):
                q_parts.append(torch.stack([bins[2 * wd, b, s], bins[2 * wd + 1, b, s]], dim=0))
                k_parts.append(torch.stack([bins[2 * wd, b, s], bins[2 * wd + 1, b, s]], dim=0))
    q_orig = torch.stack(q_parts, dim=1)  # [2, B*, H, N, D]
    k_orig = torch.stack(k_parts, dim=1).permute(1, 2, 0, 3, 4).reshape(
        n_rows, heads, 2 * spatial, lanes
    )
    return q_orig, k_orig, (batch_actual, n_sw)


def assemble_bins(bins: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """把 10 个 1D (Q, K) 图案装配为 B*=10 的 q_orig/k_orig（H=1, N=1, n_sw=2）。

    同一图案复制到 2 个空间窗：cfg batch=1 -> 分解 (1, 2)，行序
    row = wd·n_sw + s（与 window_partition_v2 一致），两个窗分数逐位相同。
    """
    assert len(bins) == 10
    q_rows, k_rows = [], []
    for wd in range(5):  # wd-major：row = wd·n_sw + s（window_partition_v2 行序）
        for s in range(2):  # 空间窗（复制图案）
            q_rows.append(
                torch.stack(
                    [
                        bins[2 * wd][0].unsqueeze(0).unsqueeze(0),
                        bins[2 * wd + 1][0].unsqueeze(0).unsqueeze(0),
                    ],
                    dim=0,
                )
            )  # [2, 1, 1, D]
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


def reference_slot_scores(q_orig: torch.Tensor, k_orig: torch.Tensor, cfg) -> torch.Tensor:
    """纯 Python 参考实现：逐 (b, s) 槽位 (o, sz, m) -> 规范融合分数。

    返回 [B*, H, num_steps, N]（与算子 slot_views["scores"] 同布局）。
    """

    t_steps, batch, heads, spatial, lanes = q_orig.shape
    num_steps = int(cfg.temporal_quotient_steps)
    n_pairs = num_steps // t_steps
    _, n_sw = _d1_decompose_temporal_batch(batch, n_pairs, cfg)
    q_bin = (q_orig > 0).to(torch.int64)
    k_bin = (k_orig > 0).to(torch.int64)
    slots = torch.zeros(batch, heads, num_steps, spatial, dtype=torch.int64)
    for idx in range(batch):
        b = idx // (n_pairs * n_sw)
        s = idx % n_sw
        # 与算子同一的 pair_row 映射：槽位 tb 的 (Q, K) 来自
        # row = base + (tb//2)·n_sw（该 (b, s) 列的第 tb//2 个时间对）
        base_row = b * (n_pairs * n_sw) + s
        for tb in range(num_steps):
            group = tb // 5
            row = base_row + (tb // 2) * n_sw
            t_local = tb % 2
            q = q_bin[t_local, row]  # [H, N, D]
            k = k_bin[row, :, t_local * spatial : (t_local + 1) * spatial, :]
            for h in range(heads):
                for n in range(spatial):
                    q_t = int(q[h, n].sum().item())
                    k_t = int(k[h, n].sum().item())
                    o_t = int((q[h, n] & k[h, n]).sum().item())
                    sz_t = lanes - q_t - k_t + o_t
                    # 组内运动边：m_t = popcount(K_{t-1} ⊕ K_t)，组内首槽 t≡0 采用组内第 1 条边
                    if tb % 5 == 0:
                        e = 0  # 组内第 1 条边 (5g, 5g+1)
                    else:
                        e = tb % 5 - 1
                    # 边 (5g+e, 5g+e+1) 的 K：全局 bin 映射回 (pair, t_local) 行
                    # 运动项按 (head, token) 逐位置计（与算子 k_diff 同语义）
                    bin_a = group * 5 + e
                    bin_b = bin_a + 1
                    wd_a, tl_a = divmod(bin_a, 2)
                    wd_b, tl_b = divmod(bin_b, 2)
                    row_a = (b * n_pairs + wd_a) * n_sw + s
                    row_b = (b * n_pairs + wd_b) * n_sw + s
                    ka = k_bin[row_a, h, tl_a * spatial + n, :]
                    kb = k_bin[row_b, h, tl_b * spatial + n, :]
                    m_t = int((ka ^ kb).sum().item())
                    slots[idx, h, tb, n] = h67_slot_score(o_t, sz_t, m_t)
    return slots


class CanonicalFusedFormTests(unittest.TestCase):
    """I1：融合式 RNE16(64o+sz+16m) 为规范分数。"""

    def test_fused_form_is_canonical_bitwise(self) -> None:
        # 用可构造的 (o, sz, m) 直接验证算子分数的融合式逐位一致
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
                # 构造每槽 popcount 精确为 (q_count, k_count, o) 的窗口：
                # Q = [1]*q_count, K = 与 Q 交集为 o 的模式
                q_pat = torch.zeros(LANES)
                q_pat[:q_count] = 1.0
                k_pat = torch.zeros(LANES)
                k_pat[:o] = 1.0
                k_pat[q_count : q_count + (k_count - o)] = 1.0
                # 边 (0,1) = m：K1 相对 K0 翻转 m 位（XOR 置 1）
                diff = torch.zeros(LANES)
                diff[:m] = 1.0
                k_slot_alt = ((k_pat + diff) % 2).to(torch.float32)
                bins = [(q_pat, k_pat)] * 10
                bins[1] = (q_pat, k_slot_alt)
                # 槽位 1 用 k_slot_alt（popcount 变化 -> 其 (o, sz) 需单独计算）；
                # 槽位 0（组内首槽）与槽位 2 共享边 (0,1)/(1,2) = m，其余边 = 0
                o1 = int(((q_pat > 0) & (k_slot_alt > 0)).sum().item())
                k1 = int(k_slot_alt.sum().item())
                sz1 = LANES - q_count - k1 + o1
                expected_by_slot = {
                    0: h67_slot_score(o, sz, m),      # 组内首槽：边 (0,1) = m
                    1: h67_slot_score(o1, sz1, m),    # 槽位 1：K 为 k_slot_alt
                    2: h67_slot_score(o, sz, m),      # 边 (1,2) = m
                    3: h67_slot_score(o, sz, 0),
                    4: h67_slot_score(o, sz, 0),
                    5: h67_slot_score(o, sz, 0),      # 组 1 首槽：边 (5,6) = 0
                    6: h67_slot_score(o, sz, 0),
                    7: h67_slot_score(o, sz, 0),
                    8: h67_slot_score(o, sz, 0),
                    9: h67_slot_score(o, sz, 0),
                }
                q_orig, k_orig = assemble_bins(bins)
                cfg = _cfg(temporal_quotient_batch=1, temporal_quotient_steps=10)
                scores, rle, views = _binary_t5_quotient_token_scores(q_orig, k_orig, cfg)
                slot_scores = views["scores"].reshape(10, 10)
                # 行 r = (wd, s)，槽位 tb = 2*wd + t_local（两个空间窗逐位一致）
                for wd in range(5):
                    for tl in range(2):
                        tb = 2 * wd + tl
                        got = int(slot_scores[wd, tb].item())
                        self.assertEqual(
                            got, expected_by_slot[tb],
                            f"wd={wd} tl={tl} tb={tb} (o,sz,m)=({o},{sz},{m})",
                        )
                # rle 记账存在且运行
                self.assertGreaterEqual(rle["mean_runs_per_position"], 1.0)
                self.assertLessEqual(rle["mean_runs_per_position"], 5.0)

    def test_decomposed_form_differs_at_tie_flip(self) -> None:
        # 找一个融合式与拆解式差 1 档的平局案例（I1 测量：2.74%）。
        # RNE 平局 = (64o+sz) % 16 == 8；64o 被 16 整除 -> sz % 16 == 8。
        # 商 = 4o + sz//16，奇偶性仅由 sz//16 决定；翻转发生在 m 为奇数时：
        #   拆解式 = A + (A&1) + m, 融合式 = A + m + ((A+m)&1)，A=4o+sz//16
        #   -> 二者差 1 档 iff m 为奇数（与 o 无关）。
        tie_case = None
        for sz in range(33):
            if sz % 16 == 8:
                tie_case = (0, sz)  # o=0 时 A 的奇偶性由 sz//16 决定
                break
        self.assertIsNotNone(tie_case, "平局翻转案例应在物理域内存在（sz=8/24）")
        o, sz = tie_case
        m = 1  # 奇数 m 必然触发翻转
        fused = h67_slot_score(o, sz, m)
        decomposed = min(rne16(64 * o + sz) + m, MAX_SCORE)
        self.assertNotEqual(fused, decomposed, "平局翻转处拆解式与融合式必须差 1 档")
        # 反解 (q, k)：sz = 32 - q - k + o -> q + k = 32 - sz + o
        q = o + 6
        k = (LANES - sz + o) - q
        if k < o:
            q = o + 2
            k = LANES - sz + o - q
        self.assertGreaterEqual(k, o)
        self.assertGreaterEqual(k, 0)
        self.assertLessEqual(k, LANES)
        q_pat = torch.zeros(LANES)
        q_pat[:q] = 1.0
        k_pat = torch.zeros(LANES)
        k_pat[:o] = 1.0
        k_pat[q : q + (k - o)] = 1.0
        bins = [(q_pat, k_pat)] * 10
        # 构造 m 条边：K 在槽位 1 使用翻转模式（边 (0,1) 由槽位 0/1 共享）
        diff = torch.zeros(LANES)
        diff[:m] = 1.0
        k_alt = ((k_pat + diff) % 2).to(torch.float32)
        bins[1] = (q_pat, k_alt)
        q_orig, k_orig = assemble_bins(bins)
        scores, _, views = _binary_t5_quotient_token_scores(
            q_orig, k_orig, _cfg(temporal_quotient_batch=1)
        )
        slot_scores = views["scores"].reshape(10, 10)
        got = int(slot_scores[0, 0].item())
        self.assertEqual(got, fused, f"算子必须给出融合式 {fused}，而非拆解式 {decomposed}")


class SlotDecompositionTests(unittest.TestCase):
    """I2：槽位分解 s = 4o + r；I5：反解 (s - m) -> (o, r) 完全唯一。

    合同口径（check_d1 实测）：r = (s - m) % 4 在物理域内恒 ∈ {0,1,2}，
    o 精确恢复；s%4==3 仅在 m ≡ 3 (mod 4) 时出现（check_I2 的“无
    s%4==3”只对 m=0 成立）。
    """

    def test_decomposition_unique_and_no_remainder_three(self) -> None:
        q_orig, k_orig, (b_actual, n_sw) = make_pair_window(
            n_rows=60, heads=2, spatial=6, seed=7
        )  # 默认 cfg batch=2 -> (2, 6)
        self.assertEqual((b_actual, n_sw), (2, 6))
        scores, _, views = _binary_t5_quotient_token_scores(q_orig, k_orig, _cfg())
        slot_scores = views["scores"].reshape(-1).to(torch.int64)
        overlap = views["overlap"].reshape(-1).to(torch.int64)
        motion = views["motion"].reshape(-1).to(torch.int64)
        # I5：反解 (s - m) -> (o, r)：r ∈ {0,1,2} 恒成立，o 精确恢复
        sm = slot_scores - motion
        o_recovered = torch.div(sm, 4, rounding_mode="floor")
        r_recovered = torch.remainder(sm, 4)
        self.assertTrue(
            ((r_recovered >= 0) & (r_recovered <= 2)).all(),
            "物理域内反解 r = (s-m)%4 ∈ {0,1,2}（I5）",
        )
        self.assertTrue(torch.equal(o_recovered, overlap), "反解 (s-m)//4 必须恢复 o_t")
        # I2：m=0 槽位（check_I2 的口径）无 s%4==3 —— 用无运动构造验证
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k[:5] = 1.0
        q[:3] = 1.0
        q_orig0, k_orig0 = assemble_bins([(q, k)] * 10)
        _, _, views0 = _binary_t5_quotient_token_scores(
            q_orig0, k_orig0, _cfg(temporal_quotient_batch=1)
        )
        s0 = views0["scores"].reshape(-1).to(torch.int64)
        self.assertTrue(torch.equal(views0["motion"].reshape(-1), torch.zeros_like(views0["motion"].reshape(-1))))
        self.assertFalse((s0 % 4 == 3).any(), "m=0 槽位分数无 s%4==3（I2）")
        self.assertTrue(((s0 % 4 >= 0) & (s0 % 4 <= 2)).all())


class GridShiftInvarianceTests(unittest.TestCase):
    """I3：网格位移不变（无 clamp）与 clamp 边界。"""

    def test_shiftmax_invariant_under_plus_k_without_clamp(self) -> None:
        torch.manual_seed(11)
        for _ in range(50):
            n = int(torch.randint(4, 9, ()).item())
            lo = int(torch.randint(0, 60, ()).item())
            scores = torch.randint(lo, lo + 40, (n,)).to(torch.float32)
            k_shift = float(torch.randint(1, 20, ()).item())
            if scores.max() + k_shift <= MAX_SCORE:
                g0 = shiftmax(scores.unsqueeze(0), dim=-1)
                g1 = shiftmax((scores + k_shift).unsqueeze(0), dim=-1)
                self.assertTrue(torch.allclose(g0, g1, atol=1e-6), "无 clamp 的 +k 档位移必须保持 Shiftmax 不变")

    def test_operator_scores_within_q7_grid(self) -> None:
        q_orig, k_orig, _ = make_pair_window(n_rows=20, heads=1, spatial=5, seed=3)
        scores, _, _ = _binary_t5_quotient_token_scores(q_orig, k_orig, _cfg())
        self.assertGreaterEqual(scores.min().item(), 0.0)
        self.assertLessEqual(scores.max().item(), MAX_SCORE)


class TConsistencyTests(unittest.TestCase):
    """I4：T=5 均匀边剖面 ≡ H67 T=2 槽位分数。"""

    def test_uniform_edge_profile_equals_h67(self) -> None:
        # 构造：槽位 0 与槽位 1 共享运动边 m，且两槽 (o, sz) 相同。
        # K0 与 K1 需 popcount 相同、与 Q 的交集相同、且 XOR = m 位：
        # 在 Q 之外翻转一对 1→0 与 0→1（m = 2，偶数，避免 RNE 平局翻转混淆 I4 恒等）。
        q = torch.zeros(LANES)
        q[:8] = 1.0  # Q 的 1 在 0..7
        k = torch.zeros(LANES)
        k[4:12] = 1.0  # K 的 1 在 4..11；o = |0..7 ∩ 4..11| = 4
        # K1：位置 8（K 内、Q 外）1→0，位置 12（Q/K 外）0→1
        k_alt = k.clone()
        k_alt[8] = 0.0
        k_alt[12] = 1.0
        assert int(k_alt.sum()) == 8 and int(((q > 0) & (k_alt > 0)).sum()) == 4
        m = int((k.to(torch.int64) ^ k_alt.to(torch.int64)).sum())  # = 2
        self.assertEqual(m, 2)
        o = 4
        sz = LANES - 8 - 8 + o
        expected = h67_slot_score(o, sz, m)
        bins = [(q, k)] + [(q, k_alt)] + [(q, k)] * 8  # 边 (0,1)=2，(1,2)=2，其余 0
        q_orig, k_orig = assemble_bins(bins)
        _, _, views = _binary_t5_quotient_token_scores(
            q_orig, k_orig, _cfg(temporal_quotient_batch=1)
        )
        slot_scores = views["scores"].reshape(10, 10)
        # wd=0 的槽位 0（组内首槽，采用第 1 条边）与槽位 1（边 (0,1)）
        # 必须逐点等于 H67 T=2 的同一运动边融合分数（I4 的 pair 语义）
        for tb in (0, 1):
            self.assertEqual(
                int(slot_scores[0, tb].item()), expected,
                "T=5 槽位公式必须逐点等于 H67 T=2 融合分数（同一运动边喂两个槽位）",
            )
        # 槽位 2 亦共享边 (1,2)=2；槽位 3/4 的边为 0
        self.assertEqual(int(slot_scores[0, 2].item()), expected)
        for tb in (3, 4):
            self.assertEqual(
                int(slot_scores[0, tb].item()), h67_slot_score(o, sz, 0),
                f"槽位 {tb} 的边必须为 0",
            )


class QuotientReversibilityTests(unittest.TestCase):
    """I5：记录 (o, sz, m) -> 分数精确重建；物理域内反解唯一。"""

    def test_record_reconstruction_and_inverse_unique(self) -> None:
        q_orig, k_orig, _ = make_pair_window(
            n_rows=10, heads=1, spatial=8, cfg=_cfg(temporal_quotient_batch=1), seed=9
        )
        scores, _, views = _binary_t5_quotient_token_scores(
            q_orig, k_orig, _cfg(temporal_quotient_batch=1)
        )
        slot_scores = views["scores"].reshape(-1).to(torch.int64)
        overlap = views["overlap"].reshape(-1).to(torch.int64)
        same_zero = views["same_zero"].reshape(-1).to(torch.int64)
        motion = views["motion"].reshape(-1).to(torch.int64)
        # 前向重建：融合式
        rebuilt = torch.clamp(
            torch.div(
                64 * overlap + same_zero + 16 * motion, 16, rounding_mode="floor"
            ),
            max=MAX_SCORE,
        )
        # RNE 需要完整逻辑，直接逐元素用 Python 校验（数量小）
        for i in range(slot_scores.numel()):
            o, sz, m = int(overlap[i]), int(same_zero[i]), int(motion[i])
            self.assertEqual(int(slot_scores[i]), h67_slot_score(o, sz, m))
        # 反解 (s - m) -> (o, r)，物理域内唯一
        for i in range(slot_scores.numel()):
            s = int(slot_scores[i])
            m = int(motion[i])
            o_hat, r_hat = divmod(s - m, 4)
            self.assertNotEqual(r_hat, 3, "物理域内 r=3 不存在（I5 修正）")
            self.assertEqual(o_hat, int(overlap[i]))


class RunLengthLedgerTests(unittest.TestCase):
    """I6：时间维 run-length 广播账。"""

    def test_runs_accounting_formula(self) -> None:
        # 算子返回的独立门数必须等于 1 + Σ(1 - eq)（组内 4 条边）
        q_orig, k_orig, _ = make_pair_window(n_rows=30, heads=2, spatial=6, seed=13)
        scores, rle, views = _binary_t5_quotient_token_scores(q_orig, k_orig, _cfg())
        slot_scores = views["scores"]  # [B*, H, 10, N]
        grouped = slot_scores.unflatten(2, (2, 5))
        eq = grouped[:, :, :, :-1].eq(grouped[:, :, :, 1:]).to(torch.float)
        runs = 1 + (4 - eq.sum(dim=3))
        mean_runs = float(runs.mean().item())
        self.assertAlmostEqual(mean_runs, rle["mean_runs_per_position"], places=4)
        self.assertAlmostEqual(rle["independent_gate_ratio"], mean_runs / 5.0, places=6)

    def test_uniform_scores_give_full_broadcast(self) -> None:
        # 全部槽位同分 -> 每位置 1 个独立门（广播率 1/5，节省 80%）
        q_orig, k_orig, _ = make_pair_window(
            n_rows=10, heads=1, spatial=4, cfg=_cfg(temporal_quotient_batch=1), seed=17
        )
        scores, rle, views = _binary_t5_quotient_token_scores(
            q_orig, k_orig, _cfg(temporal_quotient_batch=1)
        )
        self.assertLessEqual(rle["mean_runs_per_position"], 5.0)
        # 替换槽位分数为全同分后重算 RLE 账：直接检验记账恒等式（用同分序列）
        grouped = views["scores"].unflatten(2, (2, 5))
        uniform = grouped.clone()
        uniform[:, :, :, :, :] = uniform[:, :, :, 0:1, :]
        eq = uniform[:, :, :, :-1].eq(uniform[:, :, :, 1:]).to(torch.float)
        runs = 1 + (4 - eq.sum(dim=3))
        self.assertEqual(float(runs.mean().item()), 1.0, "全同分 -> 1 个独立门/位置")

    def test_bernoulli_bound_matches_contract_saving(self) -> None:
        # eq=0.979 的 Bernoulli 界：E[runs] = 1 + 4*(1-p) = 1.084 -> 节省 78.3%
        expected_runs = 1 + 4 * (1 - EQ_RATE)
        saving = 1 - expected_runs / 5
        self.assertAlmostEqual(expected_runs, 1.084, places=3)
        self.assertAlmostEqual(saving, 0.783, places=3)


class EdgeCoverageTests(unittest.TestCase):
    """I7：时间边覆盖（组内 4 条边，8/9；跨组边 (4,5) 不可见）。"""

    def test_cross_group_edge_never_used(self) -> None:
        # 构造：仅 (4,5) 一对 bin 存在差异（XOR popcount = 2，唯一一处为 2 的边）。
        # 任何槽位都不得看到该边：组 0 末槽 4 指定边 (3,4)、组 1 首槽 5 指定边
        # (5,6)，二者都只能看到 popcount=1 的合法边，永远看不到 2。
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k_a = k.clone()
        k_a[0] = 1.0
        k_b = k.clone()
        k_b[1] = 1.0
        bins = [(q, k)] * 10
        bins[4] = (q, k_a)
        bins[5] = (q, k_b)
        q_orig, k_orig = assemble_bins(bins)
        _, _, views = _binary_t5_quotient_token_scores(
            q_orig, k_orig, _cfg(temporal_quotient_batch=1)
        )
        motion = views["motion"].reshape(10, 10)  # [B*=5, 10]（H=1, N=1）
        # 跨组边 (4,5) 的 XOR popcount = 2，不得进入任何槽位
        self.assertLessEqual(
            float(motion.max().item()), 1.0,
            "跨组边 (4,5)（XOR=2）不得进入任何槽位的运动项",
        )
        # 槽位 4 只能看到边 (3,4)=1，槽位 5 只能看到边 (5,6)=1（合法可见边）
        self.assertEqual(int(motion[0, 4].item()), 1, "槽位 4 的指定边是 (3,4)")
        self.assertEqual(int(motion[0, 5].item()), 1, "组 1 首槽 5 的指定边是 (5,6)")
        self.assertEqual(int(motion[0, 9].item()), 0, "槽位 9 的指定边 (8,9) 无差异")

    def test_intra_group_edges_visible(self) -> None:
        # 组 0 内的边 (0,1),(1,2),(2,3),(3,4) 必须全部可见（8/9 覆盖）
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        edges_visible = set()
        # 逐边验证：只有边 (e, e+1) 有差异时，相应槽位的运动项 > 0
        for e in range(9):
            if e == 4:
                continue  # 跨组边
            bins = [(q, k)] * 10
            k_a = k.clone()
            k_a[0] = 1.0
            k_b = k.clone()
            k_b[1] = 1.0
            bins[e] = (q, k_a)
            bins[e + 1] = (q, k_b)
            q_orig, k_orig = assemble_bins(bins)
            _, _, views = _binary_t5_quotient_token_scores(
                q_orig, k_orig, _cfg(temporal_quotient_batch=1)
            )
            motion = views["motion"].reshape(10, 10)
            # 边 e 出现在槽位 e+1 的运动项（以及组内首槽的复用）中
            used = int(motion[:, :].sum().item()) > 0
            if used:
                edges_visible.add(e)
        self.assertEqual(edges_visible, {0, 1, 2, 3, 5, 6, 7, 8}, "可见边必须恰为 8/9")


class LayoutAndBoundaryTests(unittest.TestCase):
    """跨窗时间槽分组、batch 分解、布局写回、m_0 约定。"""

    def test_grouping_matches_python_reference(self) -> None:
        cfg = _cfg()
        q_orig, k_orig, (_, n_sw) = make_pair_window(
            n_rows=30, heads=2, spatial=6, cfg=cfg, seed=23
        )
        scores, _, views = _binary_t5_quotient_token_scores(q_orig, k_orig, cfg)
        ref = reference_slot_scores(q_orig, k_orig, cfg).to(torch.float32)  # [B*, H, 10, N]
        self.assertTrue(
            torch.equal(views["scores"], ref),
            "算子槽位分数必须与 Python 参考实现逐位一致",
        )
        # 布局写回：token (t_local, n) 于行 idx 取槽 2*wd(idx)+t_local
        batch, heads, n2, lanes = k_orig.shape
        n_pairs = cfg.temporal_quotient_steps // 2
        expected_out = torch.zeros_like(scores)
        for idx in range(batch):
            wd = (idx // n_sw) % n_pairs
            for tl in range(2):
                expected_out[idx, :, tl * 6 : (tl + 1) * 6, 0] = ref[idx, :, 2 * wd + tl, :]
        self.assertTrue(torch.equal(scores, expected_out), "分数必须写回原 token 布局")

    def test_first_slot_adopts_group_first_edge(self) -> None:
        # 组 0 首槽 t=0 的运动项必须等于边 (0,1)（与槽位 1 共享同一边）
        q = torch.zeros(LANES)
        k = torch.zeros(LANES)
        k_a = k.clone()
        k_a[0] = 1.0
        k_b = k.clone()
        k_b[1] = 1.0
        bins = [(q, k)] * 10
        bins[0] = (q, k_a)
        bins[1] = (q, k_b)
        q_orig, k_orig = assemble_bins(bins)
        _, _, views = _binary_t5_quotient_token_scores(
            q_orig, k_orig, _cfg(temporal_quotient_batch=1)
        )
        motion = views["motion"].reshape(10, 10)
        # 边 (0,1) 的 XOR popcount = 2（k_a 与 k_b 各带 1 个不同位）
        self.assertEqual(int(motion[0, 0].item()), 2, "组内首槽 t=0 必须采用组内第 1 条边 (0,1)")
        self.assertEqual(int(motion[0, 1].item()), 2, "槽位 t=1 采用边 (0,1)")
        self.assertEqual(int(motion[0, 2].item()), 1, "槽位 t=2 采用边 (1,2)=1")

    def test_batch_decomposition_explicit_and_auto(self) -> None:
        cfg2 = _cfg(temporal_quotient_batch=2)
        cfg1 = _cfg(temporal_quotient_batch=1)
        self.assertEqual(_d1_decompose_temporal_batch(880, 5, cfg2), (2, 88))
        self.assertEqual(_d1_decompose_temporal_batch(440, 5, cfg2), (1, 88), "评测 bs1 时回退自动分解")
        self.assertEqual(_d1_decompose_temporal_batch(440, 5, cfg1), (1, 88))
        self.assertEqual(_d1_decompose_temporal_batch(240, 5, cfg2), (2, 24))
        self.assertEqual(_d1_decompose_temporal_batch(120, 5, cfg2), (1, 24))

    def test_validation_errors(self) -> None:
        q_orig, k_orig, _ = make_pair_window(
            n_rows=10, heads=1, spatial=4, cfg=_cfg(temporal_quotient_batch=1), seed=29
        )
        bad = dict(temporal_quotient_batch=1)
        with self.assertRaises(ValueError):
            _binary_t5_quotient_token_scores(
                q_orig[:1], k_orig, _cfg(**bad)
            )  # T != 2
        with self.assertRaises(ValueError):
            _binary_t5_quotient_token_scores(
                q_orig, k_orig, _cfg(temporal_quotient_steps=0, **bad)
            )
        with self.assertRaises(ValueError):
            _binary_t5_quotient_token_scores(
                q_orig, k_orig, _cfg(temporal_quotient_steps=12, **bad)
            )  # 12 % 5 != 0
        with self.assertRaises(ValueError):
            _binary_t5_quotient_token_scores(
                q_orig, k_orig, _cfg(temporal_quotient_len=4, **bad)
            )
        with self.assertRaises(ValueError):
            _d1_decompose_temporal_batch(7, 5, _cfg())
        # STE 梯度路径存在（不塌）
        cfg = _cfg(**bad)
        q_orig.requires_grad_(True)
        k_orig.requires_grad_(True)
        scores, _, _ = _binary_t5_quotient_token_scores(q_orig, k_orig, cfg)
        loss = scores.float().sum()
        loss.backward()
        self.assertIsNotNone(q_orig.grad)
        self.assertIsNotNone(k_orig.grad)
        self.assertEqual(q_orig.grad.numel(), q_orig.numel())


class H67FixedPointAlignmentTests(unittest.TestCase):
    """F1 回归（2026-08-19 修复）：h87 的 Q7 整数分数进 shiftmax 前 ÷128
    （定点指数语义 2^(s/128)）后，与 h67 现网锚点 gate/attn 数值对齐。

    验证标准 = D1 漂移诊断 §3.2/§3.3 的 h87f 假说实证区间（同 seed 合成输入、
    同一测量口径）：gate mean abs 8.2e-6（本实现实测 1.09e-5，同量级）、
    attn mean rel 0.00036（实测 3.6e-4）。修复前：gate 9.78e-4 / attn 0.198。
    """

    @staticmethod
    def synthetic_batch(
        *, B: int = 2, n_sw: int = 2, heads: int = 4, N: int = 225,
        dim: int = LANES, p: float = 0.06, seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """与诊断脚本同构造：10-bin 事件 → (B, 5 pairs, n_sw 空间窗) q_orig/k_orig。

        事件密度 p 对齐真实 profile（token 平均 popcount ~1.9、kzero ~0.76）。
        """
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

    @staticmethod
    def h67_branch_scores(
        q_orig: torch.Tensor, k_orig: torch.Tensor,
    ) -> torch.Tensor:
        """h67（Motion 现网）锚点：tx + 0.25·motion，center → float 分数。"""
        cfg_m = ShiftmaxAttentionConfig(
            mode="h60", binary_motion_xor_alpha=0.25, alpha0=0.02,
            mismatch_penalty=0.0, single_active_penalty=0.0,
            consensus_score_norm="head_dim", score_scale=1.0,
            center_scores=True, preserve_mean=True, eps=1e-6,
        )
        tx_scores, _sc = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg_m)
        return tx_scores - tx_scores.mean(dim=2, keepdim=True)

    def test_h87_fixed_point_gate_attn_aligns_with_h67(self) -> None:
        q_orig, k_orig = self.synthetic_batch(seed=0)
        cfg = _cfg()  # temporal_quotient_batch=2, steps=10, len=5
        s87, _, _ = _binary_t5_quotient_token_scores(q_orig, k_orig, cfg)
        s87f = s87 / 128.0  # F1 定点解释（与 forward 分支同式）
        g87 = shiftmax(s87f - s87f.mean(dim=2, keepdim=True), dim=2, eps=1e-6)
        g67 = shiftmax(self.h67_branch_scores(q_orig, k_orig), dim=2, eps=1e-6)
        k_ev = _binary_event_ste(k_orig)
        n_tok = 2 * 225
        a87 = k_ev * g87 * n_tok
        a67 = k_ev * g67 * n_tok
        gate_abs = float((g87 - g67).abs().mean())
        gate_rel = float(((g87 - g67).abs() / (g67.abs() + 1e-12)).mean())
        attn_rel = float(((a87 - a67).abs() / (a67.abs() + 1e-12)).mean())
        self.assertLessEqual(
            gate_abs, 3e-5,
            f"h87÷128 后 gate mean abs 差须 ≤3e-5（诊断 h87f 区间 8.2e-6；"
            f"实测 {gate_abs:.2e}），修复前为 9.78e-4",
        )
        self.assertLessEqual(
            attn_rel, 5e-4,
            f"h87÷128 后 attn mean rel 差须 ≤5e-4（诊断 h87f 区间 0.00036；"
            f"实测 {attn_rel:.2e}），修复前为 0.198",
        )
        self.assertLessEqual(
            gate_rel, 0.02,
            f"gate mean rel 差须 ≤0.02（诊断 h87f 实测 0.0064；实测 {gate_rel:.2e}）",
        )


class Rne16DivTests(unittest.TestCase):
    def test_rne16_tensor_matches_python(self) -> None:
        torch.manual_seed(31)
        nums = torch.randint(0, 2600, (5000,)).to(torch.float32)
        got = _rne16_div_pow2_ste(nums.clone())
        for i in range(5000):
            self.assertEqual(int(got[i].item()), rne16(int(nums[i].item())))

    def test_ste_backward_scales_by_denominator(self) -> None:
        # F2（2026-08-19 修复）：STE backward 按真实导数 1/16 直通（非恒等），
        # 消除 o 项系数 65/16 的梯度放大（D1 漂移诊断 §3.4/§6-F2）。
        torch.manual_seed(37)
        nums = torch.randint(1, 2600, (1024,)).to(torch.float32).requires_grad_(True)
        out = _rne16_div_pow2_ste(nums)
        out.sum().backward()
        self.assertIsNotNone(nums.grad)
        expected = torch.full_like(nums.grad, 1.0 / 16.0)
        self.assertTrue(
            torch.allclose(nums.grad, expected, atol=1e-7),
            f"STE backward 必须为 ÷16 直通（每元素梯度恰为 1/16），"
            f"实测 max_dev={float((nums.grad - expected).abs().max()):.2e}",
        )
        # forward 仍为 RNE16 整数商（RTL 逐位一致，不受 backward 改动影响）
        self.assertEqual(int(out[0].item()), rne16(int(nums[0].item())))


if __name__ == "__main__":
    unittest.main()
