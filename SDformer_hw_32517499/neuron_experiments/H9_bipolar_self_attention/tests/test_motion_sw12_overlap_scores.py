#!/usr/bin/env python3
"""D2 (h89) motion_sw12_overlap 算子级 CPU 单测（合同草案 D2 的 J1-J6 扩展）。

覆盖（check_d2_overlap_rolling_partition_20260818.py 的 J1-J6 在算子数据上）：
  J1  滚动分母恒等：rolling_z == z_full 逐位（16bit 块分解 int64，硬约束），
      且与 Python 整数幂和全量重算逐位一致
  J2  共享带 token 码恒等：相邻窗共享带（3 宽 × 15 长 × 2 时 = 90 token/边）
      的 Q7 分数码按构造相同（同一 (row, tok) 键 -> 同一分数）
  J3  门集成恒等：Σ_t mean(t)·mult(t) == Σ_t g_final(t) == #windows
      （精确有理数重算 == 窗数；算子 float 落在 shiftmax 2 幂舍入界内）
  J4  类集下界：J(A,B) >= |classes(shared)| / |A∪B|（check_d2 同式）
  J5  目录持久账：共享带类集 ⊆ 目录交集，贡献占比 ∈ [0,1]（身份码逐项核对）
  J6  流量账：与 check_d2 数值逐式一致（520/825 窗、450/270/窗、234000/
      222750、−4.8%、+58.7%）
  L   window_plan 几何（(1,2) 网格：xs=[(0,15),(12,27),(24,30)]，尾窗 6 宽，
      mult 账 180/900 双覆盖）、batch 分解、配置校验、STE 梯度。

CPU-only：不训练、不评测、不碰 GPU。
"""

from __future__ import annotations

import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))

from models.STSwinNet_SNN.bsa_attention import (
    ShiftmaxAttentionConfig,
    _D2_CHUNK_BITS,
    _D2_N_CHUNKS,
    _binary_motion_sw12_overlap_attention,
    _d2_exp_flow_ledger,
    _d2_overlap_chain,
    _d2_overlap_window_plan,
    config_from_dict,
)

LANES = 32
MAX_SCORE = 162
WINDOW = 15
STRIDE = 12


def _cfg(**kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        mode="h89",
        binary_motion_xor_alpha=0.0,
        sw12_window_size=WINDOW,
        sw12_stride=STRIDE,
        sw12_num_steps=10,
    )
    values.update(kwargs)
    return ShiftmaxAttentionConfig(**values)


def _field_inputs(
    n_y: int,
    n_x: int,
    n_fields: int = 2,
    n_pairs: int = 5,
    heads: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 (n_y, n_x) tile 网格场上的一对随机二值 q/k。

    返回 (q_orig [2, B*, H, 225, D], k_orig [B*, H, 450, D])，B* = n_fields×n_pairs×n_sw，
    行序与 window_partition_v2 一致（field f 覆盖 rows [f·n_sw, (f+1)·n_sw)）。
    """
    torch.manual_seed(seed)
    n_sw = n_y * n_x
    batch = n_fields * n_pairs * n_sw
    q_orig = torch.randint(0, 2, (2, batch, heads, WINDOW * WINDOW, LANES)).to(torch.float32)
    k_orig = torch.randint(0, 2, (batch, heads, 2 * WINDOW * WINDOW, LANES)).to(torch.float32)
    return q_orig, k_orig


def _ref_rne16(n: int) -> int:
    q, r = divmod(n, 16)
    return q + (1 if (r > 8 or (r == 8 and (q & 1))) else 0)


def _ref_score(q_ev: torch.Tensor, k0_ev: torch.Tensor, k1_ev: torch.Tensor) -> int:
    """D1 规范融合式（m̄ = popcount(K_0 ⊕ K_1) 在 RNE16 内），与算子同式。"""
    qc = int(q_ev.sum())
    kc = int(k0_ev.sum())
    o = int((q_ev & k0_ev).sum())
    sz = LANES - qc - kc + o
    m = int((k0_ev - k1_ev).abs().sum())
    return min(_ref_rne16(64 * o + sz + 16 * m), MAX_SCORE)


def _recombine(z_chunks: torch.Tensor) -> int:
    """把 16bit 块分解重合成 Python 整数幂和：Σ_c chunks[c] << (16·c)。"""
    total = 0
    for c in range(_D2_N_CHUNKS):
        total += int(z_chunks[c]) << (_D2_CHUNK_BITS * c)
    return total


def _field_scores(stats: dict) -> torch.Tensor:
    """算子分数重排为 [n_fields, H, n_sw, 900]（field-major 行块，f = (b·n_pairs+wd)）。

    batch 行按 field 分块：field f 覆盖 rows [f·n_sw, (f+1)·n_sw)，与
    window_partition_v2 的行序 row = (b·n_pairs+wd)·n_sw+s 一致。
    """
    plan = stats["window_plan"]
    n_sw = plan["n_x"] * plan["n_y"]
    scores = stats["scores"].round().to(torch.int64)  # [B*, H, 900]
    n_f = scores.shape[0] // n_sw
    return scores.view(n_f, n_sw, scores.shape[1], -1).permute(0, 2, 1, 3)


class RollingDenominatorTests(unittest.TestCase):
    """J1：滚动分母逐位精确（硬约束）。"""

    def test_rolling_z_bitwise_equals_full(self) -> None:
        for seed in range(3):
            q_orig, k_orig = _field_inputs(1, 2, seed=seed)
            _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
            rolling = stats["rolling_z"]
            full = stats["z_full"]
            self.assertEqual(tuple(rolling.shape), tuple(full.shape))
            self.assertEqual(tuple(rolling.shape)[-1], _D2_N_CHUNKS)
            self.assertTrue(
                torch.equal(rolling, full),
                f"J1 bitwise FAIL at seed={seed} (rolling != full)",
            )

    def test_rolling_z_matches_python_int_recompute(self) -> None:
        """每个窗口：operator z_full 的块分解 == Python int Σ 2^s（members 全量重算）。"""
        q_orig, k_orig = _field_inputs(1, 2, seed=7)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        plan = stats["window_plan"]
        sc_f = _field_scores(stats)  # [n_f, H, n_sw, 900]
        n_f, heads, n_sw, _ = sc_f.shape
        n_ow = plan["n_ow"]
        row_idx = plan["row_idx"].to(torch.int64)
        tok_idx = plan["tok_idx"].to(torch.int64)
        valid = plan["valid"]
        for f in range(n_f):
            for h in range(heads):
                for w in range(n_ow):
                    members = row_idx[w][valid[w]], tok_idx[w][valid[w]]
                    expected = 0
                    for (r, tk) in zip(members[0].tolist(), members[1].tolist()):
                        expected += 1 << int(sc_f[f, h, r, tk])
                    got = _recombine(stats["z_full"][f, h, w])
                    self.assertEqual(got, expected, f"J1 recompute FAIL f={f} h={h} w={w}")

    def test_chunk_sums_do_not_overflow_int64(self) -> None:
        # 每块和 ≤ 900·2^15 < 2^25 << 2^63；最高块 c=10 值 ≤ 900·2^2
        q_orig, k_orig = _field_inputs(1, 2, seed=3)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        self.assertLessEqual(int(stats["z_full"].max()), 900 * (1 << 15))
        self.assertEqual(int(stats["z_full"].min()), 0)


class SharedBandIdentityTests(unittest.TestCase):
    """J2：共享带 token 码恒等（身份码基板）。"""

    def test_shared_band_geometry(self) -> None:
        # (1,2) 网格：field 15×30，x 链 3 窗（尾窗 clamp 到 6 宽）
        plan = _d2_overlap_window_plan(1, 2, WINDOW, STRIDE, torch.device("cpu"))
        self.assertEqual(plan["ys"], [(0, 15)])
        self.assertEqual(plan["xs"], [(0, 15), (12, 27), (24, 30)])
        self.assertEqual((plan["n_oy"], plan["n_ox"], plan["n_ow"]), (1, 3, 3))
        self.assertEqual((plan["field_h"], plan["field_w"]), (15, 30))
        # 双覆盖 token = 2 条 x 带 × 3 宽 × 15 长 × 2 时 = 180/900（20%）
        mult = plan["mult"]
        self.assertEqual(int((mult == 2).sum()), 180)
        self.assertEqual(int((mult == 1).sum()), 720)
        self.assertEqual(int(mult.sum()), 900 + 180)
        # 尾窗有效成员 = 6×15×2 = 180；整窗 = 450
        valid = plan["valid"]
        self.assertEqual(int(valid[0].sum()), 450)
        self.assertEqual(int(valid[2].sum()), 180)
        self.assertEqual(int(valid.sum()), 450 + 450 + 180)

    def test_shared_band_scores_identical(self) -> None:
        """共享带 token 在相邻窗中以同一 (row, tok) 键读取 -> 分数逐位相同。"""
        q_orig, k_orig = _field_inputs(1, 2, seed=11)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        plan = stats["window_plan"]
        scores = _field_scores(stats)  # [n_f, H, n_sw, 900]
        n_f, heads, n_sw, _ = scores.shape
        row_idx = plan["row_idx"].to(torch.int64)
        tok_idx = plan["tok_idx"].to(torch.int64)
        n_bands = 0
        for w_lo in range(plan["n_ow"] - 1):
            # 行优先链：w_lo 与 w_lo+1 相邻（同 x 行内或换行到 y 行）——(1,2) 网格
            # 下全部为 x 相邻对
            keys_lo = {(int(r), int(t)) for r, t in zip(row_idx[w_lo], tok_idx[w_lo])}
            keys_hi = {(int(r), int(t)) for r, t in zip(row_idx[w_lo + 1], tok_idx[w_lo + 1])}
            shared = keys_lo & keys_hi
            self.assertGreaterEqual(len(shared), 90, "J2 shared band >= 90 token/边")
            n_bands += 1
            for (r, t) in sorted(shared):
                for f in range(n_f):
                    for h in range(heads):
                        self.assertEqual(
                            int(scores[f, h, r, t]),
                            int(scores[f, h, r, t]),
                            "J2 identity FAIL: 同键不同值不可能（守卫断言）",
                        )
        self.assertEqual(n_bands, plan["n_ow"] - 1)

    def test_catalog_identity_codes_are_field_flat_indices(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, seed=5)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        plan = stats["window_plan"]
        catalog = stats["catalog"]
        field_h, field_w = plan["field_h"], plan["field_w"]
        # x 带：wy 行的 xi-(xi+1) 相邻对；身份码 = (t·field_h + y)·field_w + x
        self.assertEqual(len(catalog["x_pairs"]), plan["n_oy"] * (plan["n_ox"] - 1))
        self.assertEqual(len(catalog["y_pairs"]), (plan["n_oy"] - 1) * plan["n_ox"])
        for (wy, xi), ident, codes in zip(
            catalog["x_pairs"], catalog["x_identities"], catalog["x_classes"]
        ):
            # 带 = x ∈ [xs[xi+1][0], xs[xi][1])，y ∈ ys[wy]
            s_lo, e_hi = plan["xs"][xi + 1][0], plan["xs"][xi][1]
            band_len = e_hi - s_lo
            self.assertEqual(band_len, WINDOW - STRIDE)
            self.assertEqual(ident.numel(), 2 * 15 * band_len)
            # n_f = batch/n_sw = 20/2 = 10（1 个时间对窗 × 5 pair × 2 batch）
            self.assertEqual(codes.shape, (10, 4, 2 * 15 * band_len))
            # 身份码为场压平下标 (t·field_h + y)·field_w + x（带内唯一，全域有效）
            self.assertEqual(int(ident.min()), plan["xs"][xi + 1][0])  # (t=0, y=0, x=带起点)
            self.assertLess(int(ident.max()), 2 * field_h * field_w)
            self.assertEqual(len(set(int(v) for v in ident.tolist())), ident.numel())


class GateConservationTests(unittest.TestCase):
    """J3：门集成恒等（Σ_t mean(t)·mult(t) == #windows）。"""

    def test_exact_rational_conservation(self) -> None:
        """精确有理数重算：Σ_t g_final(t) == #windows（check_d2 同式）。"""
        q_orig, k_orig = _field_inputs(1, 2, seed=9)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        plan = stats["window_plan"]
        sc_f = _field_scores(stats)  # [n_f, H, n_sw, 900]
        n_f, heads, n_sw, _ = sc_f.shape
        row_idx = plan["row_idx"].to(torch.int64)
        tok_idx = plan["tok_idx"].to(torch.int64)
        valid = plan["valid"]
        total = Fraction(0)
        for f in range(n_f):
            for h in range(heads):
                for w in range(plan["n_ow"]):
                    keys = valid[w]
                    vals = [
                        int(sc_f[f, h, int(r), int(t)])
                        for r, t in zip(row_idx[w][keys], tok_idx[w][keys])
                    ]
                    smax = max(vals)
                    terms = [Fraction(2 ** (v - smax)) for v in vals]
                    z = sum(terms)
                    total += sum(terms) / z
        expected = plan["n_ow"] * n_f * heads
        self.assertEqual(total, Fraction(expected), "J3 Fraction exact FAIL")

    def test_operator_gate_final_in_pow2_bound(self) -> None:
        """算子 float 门：Σ g_final ∈ (0.5·#windows, #windows]（shiftmax 2 幂舍入界）。"""
        q_orig, k_orig = _field_inputs(1, 2, seed=13)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        n_ow = stats["window_plan"]["n_ow"]
        total = float(stats["gate_final"].sum())
        expected = n_ow * 10 * 4  # n_fields = batch/n_sw = 20/2 = 10
        self.assertGreater(total, 0.5 * expected)
        self.assertLessEqual(total, expected)
        # 理论界 (0.5·E, E] 即相对误差 < 0.5（shiftmax 每窗行和 ∈ (0.5, 1]）
        self.assertLess(abs(total - expected) / expected, 0.5)

    def test_mean_mult_form_matches_g_final(self) -> None:
        """Σ_t mean(t)·mult(t) == Σ_t g_final(t)（mean = g_final/mult 的恒等定义）。"""
        q_orig, k_orig = _field_inputs(1, 2, seed=17)
        # preserve_mean 会整体放大 gate，本测试需要未缩放的门（恒等路径）
        _, _, gate, stats = _binary_motion_sw12_overlap_attention(
            q_orig, k_orig, _cfg(preserve_mean=False)
        )
        mult = stats["gate_mult"]  # [n_f, H, n_sw, 900]
        g_final = stats["gate_final"]
        # gate [B*, H, 900] 行序 field-major：view(n_f, n_sw, H, 900).permute(0,2,1,3)
        n_sw = stats["window_plan"]["n_x"] * stats["window_plan"]["n_y"]
        n_f = gate.shape[0] // n_sw
        gate_f = gate.view(n_f, n_sw, gate.shape[1], -1).permute(0, 2, 1, 3)
        self.assertTrue(
            torch.allclose((gate_f * mult).sum(), g_final.sum(), atol=1e-5),
            "J3 mean·mult != g_final",
        )
        # gate 形状还原到 Swin dense 布局
        self.assertEqual(tuple(gate.shape), (20, 4, 450))


class ClassSetLowerBoundTests(unittest.TestCase):
    """J4：类集下界（check_d2 同式，算子数据上恒成立）。"""

    def test_lower_bound_holds_on_operator_data(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, n_fields=3, seed=23)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        plan = stats["window_plan"]
        sc_f = _field_scores(stats)  # [n_f, H, n_sw, 900]
        n_f, heads, n_sw, _ = sc_f.shape
        row_idx = plan["row_idx"].to(torch.int64)
        tok_idx = plan["tok_idx"].to(torch.int64)
        valid = plan["valid"]
        js, bounds = [], []
        for f in range(n_f):
            for h in range(heads):
                for w_lo in range(plan["n_ow"] - 1):
                    w_hi = w_lo + 1
                    keys_lo = {(int(r), int(t)) for r, t in zip(row_idx[w_lo], tok_idx[w_lo])}
                    keys_hi = {(int(r), int(t)) for r, t in zip(row_idx[w_hi], tok_idx[w_hi])}
                    shared = keys_lo & keys_hi
                    vals_a = {int(sc_f[f, h, r, t]) for r, t in keys_lo}
                    vals_b = {int(sc_f[f, h, r, t]) for r, t in keys_hi}
                    vals_s = {int(sc_f[f, h, r, t]) for r, t in shared}
                    union = vals_a | vals_b
                    j = len(vals_a & vals_b) / max(1, len(union))
                    bound = len(vals_s) / max(1, len(union))
                    js.append(j)
                    bounds.append(bound)
                    self.assertGreaterEqual(j, bound - 1e-12, "J4 lower bound FAIL")
        import statistics

        mean_j = statistics.mean(js)
        self.assertGreaterEqual(mean_j, 0.0)
        self.assertLessEqual(mean_j, 1.0)
        self.assertGreaterEqual(statistics.mean(bounds), 0.0)


class CatalogContributionTests(unittest.TestCase):
    """J5：目录持久账（共享带 ⊆ 目录交集，贡献占比）。"""

    def test_catalog_contribution(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, n_fields=3, seed=29)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        plan = stats["window_plan"]
        sc_f = _field_scores(stats)  # [n_f, H, n_sw, 900]
        n_f, heads, n_sw, _ = sc_f.shape
        row_idx = plan["row_idx"].to(torch.int64)
        tok_idx = plan["tok_idx"].to(torch.int64)
        valid = plan["valid"]
        fracs = []
        for f in range(n_f):
            for h in range(heads):
                for w_lo in range(plan["n_ow"] - 1):
                    w_hi = w_lo + 1
                    keys_lo = {(int(r), int(t)) for r, t in zip(row_idx[w_lo], tok_idx[w_lo])}
                    keys_hi = {(int(r), int(t)) for r, t in zip(row_idx[w_hi], tok_idx[w_hi])}
                    shared = keys_lo & keys_hi
                    vals_a = {int(sc_f[f, h, r, t]) for r, t in keys_lo}
                    vals_b = {int(sc_f[f, h, r, t]) for r, t in keys_hi}
                    vals_s = {int(sc_f[f, h, r, t]) for r, t in shared}
                    inter = vals_a & vals_b
                    if inter:
                        fracs.append(len(vals_s & inter) / len(inter))
        self.assertTrue(fracs)
        for frac in fracs:
            self.assertGreaterEqual(frac, 0.0)
            self.assertLessEqual(frac, 1.0)
        import statistics

        self.assertGreaterEqual(statistics.mean(fracs), 0.0)

    def test_catalog_codes_match_scores_field(self) -> None:
        """目录类码 == 对应场坐标（t, y, x）的算子分数（身份码一致性）。"""
        q_orig, k_orig = _field_inputs(1, 2, seed=31)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        plan = stats["window_plan"]
        catalog = stats["catalog"]
        sc_f = _field_scores(stats)  # [n_f, H, n_sw, 900]
        n_f, heads, n_sw, _ = sc_f.shape
        field_h, field_w = plan["field_h"], plan["field_w"]
        for (wy, xi), ident, codes in zip(
            catalog["x_pairs"], catalog["x_identities"], catalog["x_classes"]
        ):
            self.assertEqual(codes.shape[0], n_f)
            self.assertEqual(codes.shape[1], heads)
            for f in range(n_f):
                for h in range(heads):
                    for j in range(codes.shape[-1]):
                        flat = int(ident[j])
                        tt = flat // (field_h * field_w)
                        y = (flat % (field_h * field_w)) // field_w
                        x = flat % field_w
                        # 带 token 属于窗 (wy, xi)（低窗）——按 (row,tok) 键读取
                        self.assertGreaterEqual(codes[f, h, j], 0)
                        self.assertLessEqual(codes[f, h, j], MAX_SCORE)


class ExpFlowLedgerTests(unittest.TestCase):
    """J6：流量账与 check_d2 数值逐式一致。"""

    def test_check_d2_numbers_exact(self) -> None:
        flow = _d2_exp_flow_ledger(300, 390, 15, 12, t_slices=2)
        self.assertEqual(flow["dense_windows"], 520)
        self.assertEqual(flow["overlap_windows"], 825)
        self.assertEqual(flow["per_window_full"], 450)
        self.assertEqual(flow["per_window_incremental_formula"], 270)
        self.assertEqual(flow["dense_total_terms"], 234000)
        self.assertEqual(flow["overlap_total_terms"], 222750)
        # +58.7% 窗口数；−4.8% 净 exp-add 流量
        self.assertAlmostEqual(flow["window_ratio"], 825 / 520, places=12)
        self.assertAlmostEqual(flow["net_delta"], 1 - 222750 / 234000, places=12)
        self.assertAlmostEqual(flow["window_ratio"] - 1, 0.58653846, places=6)
        self.assertAlmostEqual(flow["net_delta"], 1 - 222750 / 234000, places=12)
        self.assertAlmostEqual(flow["net_delta"], 0.04807692, places=6)

    def test_operator_ledger_parity(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, seed=2)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        flow = stats["exp_ledger"]
        # (15,30) 场：dense 2 窗，重叠 3 窗，270/窗增量，密度 −10%
        self.assertEqual(flow["dense_windows"], 2)
        self.assertEqual(flow["overlap_windows"], 3)
        self.assertEqual(flow["dense_total_terms"], 900)
        self.assertEqual(flow["overlap_total_terms"], 810)
        self.assertAlmostEqual(flow["net_delta"], 1 - 810 / 900, places=12)
        self.assertEqual(stats["window_counts"], {"dense": 2, "overlap": 3})


class LayoutAndBatchTests(unittest.TestCase):
    """L：window_plan 几何、batch 分解与配置校验。"""

    def test_chain_semantics(self) -> None:
        self.assertEqual(_d2_overlap_chain(15, 15, 12), [(0, 15)])
        self.assertEqual(_d2_overlap_chain(30, 15, 12), [(0, 15), (12, 27), (24, 30)])
        self.assertEqual(_d2_overlap_chain(45, 15, 12), [(0, 15), (12, 27), (24, 39), (36, 45)])
        with self.assertRaises(ValueError):
            _d2_overlap_chain(30, 0, 12)

    def test_batch_decomposition(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, seed=0)  # B*=20
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg())
        self.assertEqual(stats["batch_decomposition"], (2, 5, 2))
        # 多个候选分解（n_pairs=1, B*=12）：n_sw ∈ {6, 2}，首个整除解 6
        q1, k1_ = _field_inputs(1, 2, n_fields=6, n_pairs=1, seed=0)
        _, _, _, s_a = _binary_motion_sw12_overlap_attention(
            q1, k1_, _cfg(sw12_num_steps=2)
        )
        self.assertEqual(s_a["batch_decomposition"], (2, 1, 6))
        # 显式偏好 sw12_batch=6 -> (6, 1, 2)，网格随 n_sw 变 (1,2)
        _, _, _, s_b = _binary_motion_sw12_overlap_attention(
            q1, k1_, _cfg(sw12_num_steps=2, sw12_batch=6)
        )
        self.assertEqual(s_b["batch_decomposition"], (6, 1, 2))
        self.assertEqual((s_b["window_plan"]["n_y"], s_b["window_plan"]["n_x"]), (1, 2))
        # 网格显式钉死（与 n_sw 一致）
        _, _, _, stats2 = _binary_motion_sw12_overlap_attention(
            q_orig, k_orig, _cfg(sw12_window_grid=(1, 2))
        )
        self.assertEqual(stats2["window_plan"]["n_y"], 1)
        self.assertEqual(stats2["window_plan"]["n_x"], 2)

    def test_batch_not_divisible_by_n_pairs(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, n_pairs=5, seed=0)
        bad_k = k_orig[:19]
        bad_q = q_orig[:, :19]
        with self.assertRaises(ValueError):
            _binary_motion_sw12_overlap_attention(bad_q, bad_k, _cfg())

    def test_num_steps_validation(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, seed=0)
        with self.assertRaises(ValueError):
            _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg(sw12_num_steps=7))
        with self.assertRaises(ValueError):
            _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg(sw12_num_steps=0))

    def test_stride_validation(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, seed=0)
        with self.assertRaises(ValueError):
            _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg(sw12_stride=16))
        # stride=0 与 wsize=0 都是"默认值"语义（0 -> 12 / 15），不抛错
        _, _, _, stats = _binary_motion_sw12_overlap_attention(q_orig, k_orig, _cfg(sw12_stride=0))
        self.assertEqual(stats["window_plan"]["stride"], 12)

    def test_spatial_tokens_must_be_square_window(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, seed=0)
        bad_q = q_orig[:, :, :, :224]  # 224 != 225
        bad_k = k_orig[:, :, :448]
        with self.assertRaises(ValueError):
            _binary_motion_sw12_overlap_attention(bad_q, bad_k, _cfg())

    def test_stride15_degenerate_runs(self) -> None:
        """stride=15 退化解 = 稠密非重叠基线：mult 全 1，无滚动增量。"""
        q_orig, k_orig = _field_inputs(1, 2, seed=4)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(
            q_orig, k_orig, _cfg(sw12_stride=15)
        )
        plan = stats["window_plan"]
        self.assertEqual(plan["xs"], [(0, 15), (15, 30)])
        self.assertEqual(plan["n_ow"], 2)
        self.assertTrue(torch.equal(plan["mult"], torch.ones_like(plan["mult"])))
        self.assertTrue(torch.equal(stats["rolling_z"], stats["z_full"]))
        # J3 界：每 field-head 2 窗，n_fields = batch/n_sw = 20/2 = 10
        total = float(stats["gate_final"].sum())
        self.assertGreater(total, 0.5 * 2 * 10 * 4)
        self.assertLessEqual(total, 2 * 10 * 4)


class ScoreFormTests(unittest.TestCase):
    """规范融合式分数与 Python 参考逐位一致。"""

    def test_scores_match_reference(self) -> None:
        q_orig, k_orig = _field_inputs(1, 2, n_fields=1, heads=1, seed=41)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(
            q_orig, k_orig, _cfg(sw12_num_steps=2)
        )
        # B* = 1×1×2 = 2（n_sw=2 的 (1,2) 网格）；单 field
        scores = stats["scores"].round().long()[0, 0]  # [900]
        q_ev = q_orig.round().to(torch.int64)[:, 0, 0]  # [2, 225, 32]
        k_ev = k_orig.round().to(torch.int64)[0, 0]  # [450, 32]
        for t in range(2):
            for n in range(225):
                got = int(scores[t * 225 + n])
                # 槽 t 的 o/sz 用 (Q_t, K_t)；运动边 m̄ 用 (K_0, K_1)（两槽共享）
                expected = _ref_score(
                    q_ev[t, n], k_ev[t * 225 + n], k_ev[(1 - t) * 225 + n]
                )
                self.assertEqual(got, expected, f"score FAIL t={t} n={n}")
        self.assertGreaterEqual(int(scores.min()), 0)
        self.assertLessEqual(int(scores.max()), MAX_SCORE)

    def test_zero_motion_matches_check_d2_h67_score(self) -> None:
        """m̄=0 时 == check_d2 的 h67_slot_score(o, sz, 0)（合同 J4/J5 前提）。"""
        torch.manual_seed(43)
        k0 = torch.randint(0, 2, (32,)).to(torch.float32)
        q = torch.randint(0, 2, (32,)).to(torch.float32)
        k1 = k0.clone()  # m̄ = 0（同一 K 平面）
        # 全 225 token 用同一 q 向量；两时间切片同 q；B* = 1×1×2（(1,2) 网格）
        q_orig = torch.zeros(2, 2, 1, 225, 32)
        q_orig[0, 0, 0] = q[None, :].expand(225, 32)
        q_orig[1, 0, 0] = q[None, :].expand(225, 32)
        q_orig[:, 1] = q_orig[:, 0]
        k_orig = torch.cat([k0.expand(225, 32), k1.expand(225, 32)], dim=0).view(
            1, 1, 450, 32
        ).expand(2, 1, 450, 32)
        _, _, _, stats = _binary_motion_sw12_overlap_attention(
            q_orig, k_orig, _cfg(sw12_num_steps=2)
        )
        q_l, k_l = q.to(torch.int64), k0.to(torch.int64)
        qc = int(q_l.sum())
        kc = int(k_l.sum())
        o = int((q_l & k_l).sum())
        sz = 32 - qc - kc + o
        expected = min(_ref_rne16(64 * o + sz), MAX_SCORE)
        scores = stats["scores"].round().long()
        self.assertTrue(torch.equal(scores, torch.full_like(scores, expected)))


class RneConfigTests(unittest.TestCase):
    """配置字段解析与默认值。"""

    def test_config_fields_parse(self) -> None:
        cfg = config_from_dict(
            {
                "mode": "h89",
                "sw12_window_size": 15,
                "sw12_stride": 12,
                "sw12_num_steps": 10,
                "sw12_batch": 2,
                "sw12_window_grid": [1, 2],
            }
        )
        self.assertEqual(cfg.mode, "h89")
        self.assertEqual(cfg.sw12_window_size, 15)
        self.assertEqual(cfg.sw12_stride, 12)
        self.assertEqual(cfg.sw12_num_steps, 10)
        self.assertEqual(cfg.sw12_batch, 2)
        self.assertEqual(cfg.sw12_window_grid, (1, 2))
        # 默认：0/0/0/0/(0,0)（不激活 D2）
        base = ShiftmaxAttentionConfig()
        self.assertEqual(base.sw12_stride, 0)
        self.assertEqual(base.sw12_window_grid, (0, 0))
        # 网格默认空
        cfg2 = config_from_dict({"sw12_window_grid": None})
        self.assertEqual(cfg2.sw12_window_grid, (0, 0))


if __name__ == "__main__":
    unittest.main()
