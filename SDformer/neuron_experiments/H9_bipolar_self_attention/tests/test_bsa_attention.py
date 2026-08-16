from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()


class DummyStage(nn.Module):
    def __init__(self, blocks: int):
        super().__init__()
        self.swin_blocks = nn.ModuleList(DummyBlock() for _ in range(blocks))


class DummySwin3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyStage(2), DummyStage(1)])


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin3d = DummySwin3D()


class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = DummyEncoder()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sttmultires_unet = DummyUNet()


class ShiftmaxAttentionTest(unittest.TestCase):
    def test_hardware_score_clip_stats_are_strict_and_read_only(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _hardware_score_clip_stats,
            config_from_dict,
        )

        scores = torch.tensor([-3.0, -2.0, 0.0, 2.0, 3.0])
        original = scores.clone()
        cfg = config_from_dict({
            "hardware_quant_enabled": True,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
        })
        stats = _hardware_score_clip_stats(scores, cfg)
        self.assertEqual(stats["score_quant_total"], 5)
        self.assertEqual(stats["score_clip_low"], 1)
        self.assertEqual(stats["score_clip_high"], 1)
        self.assertAlmostEqual(stats["score_clip_ratio"], 0.4)
        self.assertTrue(torch.equal(scores, original))

    def test_true_token_time_bundle_counts(self):
        from models.STSwinNet_SNN.bsa_attention import _token_time_bundle_stats

        q = torch.zeros(2, 1, 1, 4, 4, dtype=torch.bool)
        k = torch.zeros_like(q)
        q[0, 0, 0, 0, 0] = True
        k[0, 0, 0, 2, 1] = True
        stats = _token_time_bundle_stats(q, k)

        self.assertEqual(stats["ttb_tok1_total"], 4)
        self.assertEqual(stats["ttb_tok1_empty"], 2)
        self.assertEqual(stats["ttb_tok1_kzero"], 3)
        self.assertEqual(stats["ttb_tok1_motion_zero"], 3)
        self.assertEqual(stats["ttb_tok1_active_le2"], 2)
        self.assertEqual(stats["ttb_tok1_active_le12"], 2)
        self.assertEqual(stats["ttb_tok1_active_lane_sum_le12"], 2)
        self.assertEqual(stats["ttb_tok1_active_histogram"], [2, 2] + [0] * 7)
        self.assertEqual(stats["ttb_tok2_total"], 2)
        self.assertEqual(stats["ttb_tok2_empty"], 0)
        self.assertEqual(stats["ttb_tok4_total"], 1)
        self.assertEqual(stats["ttb_tok4_active_lanes"], 2)

    def test_binary_temporal_pair_stats_reconstruct_ttx_and_h67_scores(self):
        import base64
        import zlib

        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_pair_stats

        q = torch.zeros(2, 1, 1, 2, 4, dtype=torch.bool)
        k = torch.zeros_like(q)
        q[:, 0, 0, 0, :2] = True
        k[0, 0, 0, 0, 1:3] = True
        k[1, 0, 0, 0, 2] = True
        stats = _binary_temporal_pair_stats(q, k, include_ordered_trace=True)

        self.assertEqual(stats["pair_total"], 2)
        self.assertEqual(stats["pair_empty"], 1)
        self.assertEqual(stats["pair_motion_zero"], 1)
        self.assertEqual(stats["pair_update_zero"], 1)
        self.assertEqual(stats["pair_kzero_both"], 1)
        self.assertEqual(stats["pair_kzero_one"], 0)
        self.assertEqual(stats["token_total"], 4)
        self.assertEqual(stats["token_kzero"], 2)
        self.assertEqual(stats["four_vector_event_histogram"][7], 1)
        self.assertEqual(stats["four_vector_union_histogram"][3], 1)
        self.assertEqual(stats["k_temporal_baseline_reads"], 3)
        self.assertEqual(stats["k_temporal_union_reads"], 2)
        self.assertEqual(stats["k_temporal_intersection_reuse"], 1)
        self.assertEqual(stats["projection_baseline_active_lanes"], 3)
        self.assertEqual(stats["projection_class_channel_terms_ttx"], 3)
        self.assertEqual(stats["projection_class_channel_terms_h67"], 3)
        self.assertEqual(stats["k_temporal_intersection_histogram"][1], 1)
        self.assertEqual(stats["k_temporal_union_histogram"][2], 1)
        self.assertEqual(stats["ttx_score_q7_histogram"][4], 1)
        self.assertEqual(stats["h67_score_q7_histogram"][5], 1)
        self.assertEqual(stats["pair_score_equal_h67_qf5"], 1)
        self.assertEqual(stats["pair_score_equal_h67_qf6"], 1)
        self.assertEqual(stats["pair_score_equal_h67_qf7"], 1)
        self.assertEqual(stats["pair_score_equal_h67_qf8"], 1)
        self.assertEqual(stats["row_all_occupied_classes_sum_h67"], 3)
        self.assertEqual(stats["row_all_occupied_classes_sum_ttx"], 2)
        self.assertEqual(stats["row_kzero_fold_classes_sum_h67"], 1)
        self.assertEqual(stats["row_kzero_fold_classes_sum_ttx"], 1)
        self.assertEqual(stats["pair_kzero_same_class_h67"], 1)
        self.assertEqual(stats["pair_kzero_dual_class_h67"], 0)
        self.assertEqual(stats["row_score_span_h67_histogram"][5], 1)
        self.assertEqual(stats["row_score_span_ttx_histogram"][4], 1)

        encoded = stats["pair_overlap_ordered_trace"]
        raw = zlib.decompress(base64.b64decode(encoded["data"]))
        decoded = torch.frombuffer(bytearray(raw), dtype=torch.int16).reshape(encoded["shape"])
        expected = torch.tensor([[[[1, 0]]], [[[0, 0]]]], dtype=torch.int16)
        self.assertTrue(torch.equal(decoded, expected))

        encoded = stats["pair_k_temporal_intersection_ordered_trace"]
        raw = zlib.decompress(base64.b64decode(encoded["data"]))
        decoded = torch.frombuffer(bytearray(raw), dtype=torch.int16).reshape(encoded["shape"])
        self.assertTrue(torch.equal(decoded, torch.tensor([[[1, 0]]], dtype=torch.int16)))

    def test_projection_class_channel_terms_capture_exact_multicast_reuse(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_pair_stats

        q = torch.zeros(2, 1, 1, 2, 4, dtype=torch.bool)
        k = torch.zeros_like(q)
        k[0, 0, 0, 0, 1] = True
        k[0, 0, 0, 1, 1] = True
        stats = _binary_temporal_pair_stats(q, k)

        self.assertEqual(stats["projection_baseline_active_lanes"], 2)
        self.assertEqual(stats["projection_class_channel_terms_ttx"], 1)
        self.assertEqual(stats["projection_class_channel_terms_h67"], 1)

    def test_factorized_class_lane_segments_preserve_physical_fragments(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_pair_stats

        q = torch.zeros(2, 1, 1, 40, 2, dtype=torch.bool)
        k = torch.zeros_like(q)
        # row token 0 与 75 属于相同H67 score class/lane，但跨越64-token段。
        k[0, 0, 0, 0, 0] = True
        k[1, 0, 0, 35, 0] = True
        stats = _binary_temporal_pair_stats(
            q,
            k,
            include_ordered_trace=True,
        )

        self.assertEqual(stats["projection_class_channel_terms_h67"], 1)
        self.assertEqual(stats["projection_h67_factor_class_segments"], 2)
        self.assertEqual(
            stats["projection_h67_factor_class_lane_segments"],
            2,
        )
        self.assertIn(
            "projection_h67_factor_class_lane_segments_ordered_trace",
            stats,
        )

    def test_gate_class_terms_capture_cross_window_reuse(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_pair_stats

        q = torch.zeros(2, 2, 1, 2, 4, dtype=torch.bool)
        k = torch.zeros_like(q)
        k[0, 0, 0, 0, 1] = True
        k[0, 1, 0, 0, 1] = True
        gate_code = torch.full((2, 1, 4), 64, dtype=torch.long)
        stats = _binary_temporal_pair_stats(
            q, k, gate_q17_code=gate_code, include_ordered_trace=True
        )

        self.assertEqual(stats["projection_baseline_active_lanes"], 2)
        self.assertEqual(stats["projection_gate_class_channel_terms_deploy"], 2)
        self.assertEqual(stats["projection_gate_group_terms_g1"], 2)
        self.assertEqual(stats["projection_gate_group_terms_g2"], 1)
        self.assertEqual(stats["projection_gate_group_terms_g16"], 1)
        self.assertEqual(stats["projection_gate_group_active_lanes_g2"], 2)
        self.assertEqual(stats["projection_gate_group_active_classes_g2"], 1)
        self.assertEqual(stats["projection_gate_group_max_fanout_g2"], 2)
        self.assertEqual(stats["projection_gate_group_delivery_g2_m1"], 2)
        self.assertEqual(stats["projection_gate_group_delivery_g2_m2"], 1)
        self.assertEqual(stats["projection_gate_group_window_count_g2"], 2)
        self.assertEqual(stats["row_active_projection_gate_classes_sum_deploy"], 2)
        self.assertEqual(stats["projection_gate_multicast_delivery_m1"], 2)
        self.assertEqual(stats["projection_gate_multicast_delivery_m16"], 2)
        self.assertEqual(stats["projection_gate_class_channel_term_histogram"][64], 2)
        self.assertEqual(stats["projection_active_lane_gate_q17_histogram"][64], 2)
        self.assertIn("projection_gate_group_terms_g2_ordered_trace", stats)
        self.assertIn("projection_gate_group_window_count_g2_ordered_trace", stats)

    def test_gate_ppdi_preserves_destination_parity_constraint(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_pair_stats

        q = torch.zeros(2, 1, 1, 4, 1, dtype=torch.bool)
        k = torch.zeros_like(q)
        # 同一 gate/lane 的两个 destination 都是偶数。无约束 M2 可一拍，
        # PPDI 受偶/奇双端口约束必须两拍。
        k[0, 0, 0, 0, 0] = True
        k[0, 0, 0, 2, 0] = True
        gate_code = torch.full((1, 1, 8), 64, dtype=torch.long)
        stats = _binary_temporal_pair_stats(
            q,
            k,
            gate_q17_code=gate_code,
            include_ordered_trace=True,
        )

        self.assertEqual(stats["projection_gate_multicast_delivery_m1"], 2)
        self.assertEqual(stats["projection_gate_multicast_delivery_m2"], 1)
        self.assertEqual(stats["projection_gate_ppdi_delivery_exact"], 2)
        self.assertEqual(stats["projection_gate_group_ppdi_delivery_g1"], 2)
        self.assertIn(
            "projection_gate_ppdi_delivery_exact_ordered_trace",
            stats,
        )

    def test_gate_ppdi_uses_flattened_temporal_token_parity(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_pair_stats

        q = torch.zeros(2, 1, 1, 81, 1, dtype=torch.bool)
        k = torch.zeros_like(q)
        # 展平 token 0 与 81 分属偶/奇端口；按空间 ID 取模会把二者都误判为偶数。
        k[0, 0, 0, 0, 0] = True
        k[1, 0, 0, 0, 0] = True
        gate_code = torch.full((1, 1, 162), 64, dtype=torch.long)
        stats = _binary_temporal_pair_stats(q, k, gate_q17_code=gate_code)

        self.assertEqual(stats["projection_gate_multicast_delivery_m1"], 2)
        self.assertEqual(stats["projection_gate_ppdi_delivery_exact"], 1)

    def test_zero_gate_does_not_create_projection_work(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_pair_stats

        q = torch.zeros(2, 1, 1, 2, 1, dtype=torch.bool)
        k = torch.zeros_like(q)
        k[0, 0, 0, 0, 0] = True
        gate_code = torch.zeros((1, 1, 4), dtype=torch.long)
        stats = _binary_temporal_pair_stats(q, k, gate_q17_code=gate_code)

        self.assertEqual(stats["projection_baseline_active_lanes"], 1)
        self.assertEqual(stats["projection_gate_class_channel_terms_deploy"], 0)
        self.assertEqual(stats["projection_gate_multicast_delivery_m1"], 0)
        self.assertEqual(stats["projection_gate_ppdi_delivery_exact"], 0)

    def test_ordered_count_trace_promotes_to_int32_without_overflow(self):
        import base64
        import zlib

        from models.STSwinNet_SNN.bsa_attention import _encode_ordered_count_trace

        encoded = _encode_ordered_count_trace(torch.tensor([0, 40000], dtype=torch.long))
        self.assertEqual(encoded["dtype"], "int32_le")
        raw = zlib.decompress(base64.b64decode(encoded["data"]))
        decoded = torch.frombuffer(bytearray(raw), dtype=torch.int32)
        self.assertTrue(torch.equal(decoded, torch.tensor([0, 40000], dtype=torch.int32)))

    def test_gate_window_group_never_crosses_sample_boundary(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_pair_stats

        q = torch.zeros(2, 6, 1, 1, 4, dtype=torch.bool)
        k = torch.zeros_like(q)
        # 两个活动窗口分处相邻样本的边界两侧。若把batch_windows直接连续分组，
        # G=2会错误地把它们合并成一个乘积项。
        k[0, 2, 0, 0, 1] = True
        k[0, 3, 0, 0, 1] = True
        gate_code = torch.full((6, 1, 2), 64, dtype=torch.long)
        stats = _binary_temporal_pair_stats(
            q,
            k,
            gate_q17_code=gate_code,
            windows_per_sample=3,
            include_ordered_trace=True,
        )

        self.assertEqual(stats["projection_baseline_active_lanes"], 2)
        self.assertEqual(stats["projection_gate_group_terms_g2"], 2)
        self.assertEqual(stats["projection_gate_group_terms_g4"], 2)
        self.assertEqual(stats["projection_gate_group_terms_g16"], 2)
        self.assertEqual(stats["projection_gate_group_window_count_g16"], 6)

    def test_spatial_pair_locality_and_bank_mappings(self):
        from models.STSwinNet_SNN.bsa_attention import _spatial_pair_locality_stats

        q = torch.zeros(2, 1, 1, 9, 2, dtype=torch.bool)
        k = torch.zeros_like(q)
        q[0, 0, 0, 0, 0] = True
        q[1, 0, 0, 0, 1] = True
        k[0, 0, 0, 4, 0] = True
        k[1, 0, 0, 8, 1] = True
        stats = _spatial_pair_locality_stats(q, k)

        self.assertEqual(stats["spatial_row_total"], 1)
        self.assertEqual(stats["spatial_union_tokens"], 3)
        self.assertEqual(stats["spatial_persistent_tokens"], 1)
        self.assertEqual(stats["spatial_changed_tokens"], 2)
        self.assertEqual(stats["spatial_diag_down_adjacent_active"], 2)
        self.assertEqual(stats["spatial_horizontal_adjacent_active"], 0)
        self.assertEqual(stats["spatial_bank4_diagonal_cycles_sum"], 2)
        self.assertEqual(stats["spatial_bank4_rowmajor_cycles_sum"], 3)

    def test_match_code_offsets_match_registered_cardinality(self):
        from models.STSwinNet_SNN.bsa_attention import AX17_OFFSETS, DE9_OFFSETS, MC49_OFFSETS

        self.assertEqual(len(DE9_OFFSETS), 9)
        self.assertEqual(len(set(DE9_OFFSETS)), 9)
        self.assertEqual(len(MC49_OFFSETS), 49)
        self.assertEqual(len(set(MC49_OFFSETS)), 49)
        self.assertEqual(len(AX17_OFFSETS), 17)
        self.assertEqual(len(set(AX17_OFFSETS)), 17)
        self.assertEqual(set(AX17_OFFSETS), {(0, x) for x in range(-4, 5)} | {(y, 0) for y in range(-4, 5)})
        ring_counts = {
            radius: sum(abs(dy) + abs(dx) == radius for dy, dx in MC49_OFFSETS)
            for radius in range(9)
        }
        self.assertEqual(ring_counts, {0: 1, 1: 4, 2: 8, 3: 12, 4: 0, 5: 16, 6: 0, 7: 8, 8: 0})

    def test_de9_and_mc49_match_code_shapes_boundaries_and_gradients(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _ensure_match_code,
            _match_code_attention,
            config_from_dict,
        )

        class MatchModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 1
                self.linear_q = nn.Linear(4, 4, bias=False)

        for mode, descriptor_dim, dual in (
            ("binary_de9_match_code", 18, True),
            ("binary_mc49_match_code", 49, False),
            ("binary_ax17_match_code", 17, False),
        ):
            module = MatchModule()
            cfg = config_from_dict({"mode": mode, "alpha0": 1.0 / 64.0})
            _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
            q_orig = torch.rand(2, 1, 1, 81, 4, requires_grad=True)
            k_orig = torch.rand(1, 1, 162, 4, requires_grad=True)
            offsets = None
            if mode == "binary_ax17_match_code":
                from models.STSwinNet_SNN.bsa_attention import AX17_OFFSETS
                offsets = AX17_OFFSETS
            out, rows, descriptor, scores = _match_code_attention(
                module, q_orig, k_orig, cfg, dual_evidence=dual, offsets=offsets
            )

            self.assertEqual(tuple(out.shape), (1, 1, 162, 4))
            self.assertEqual(tuple(rows.shape), (1, 1, 162))
            self.assertEqual(tuple(descriptor.shape), (1, 1, 162, descriptor_dim))
            self.assertEqual(tuple(scores.shape), tuple(descriptor.shape))
            self.assertFalse(torch.isnan(out).any())
            self.assertEqual(tuple(module._h9_match_code_weight.shape), (1, descriptor_dim, 4))
            out.square().mean().backward()
            self.assertIsNotNone(module._h9_match_code_weight.grad)
            self.assertIsNotNone(q_orig.grad)
            self.assertIsNotNone(k_orig.grad)

    def test_match_code_int8_weight_quantization_is_on_grid(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _ensure_match_code,
            _quantized_match_code_weight,
            config_from_dict,
        )

        class MatchModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 1
                self.linear_q = nn.Linear(4, 4, bias=False)

        module = MatchModule()
        cfg = config_from_dict({
            "mode": "binary_de9_match_code",
            "match_code_weight_quant_enabled": True,
            "match_code_weight_step": 1.0 / 128.0,
        })
        _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
        quantized = _quantized_match_code_weight(module, cfg)
        self.assertTrue(torch.allclose(quantized * 128.0, torch.round(quantized * 128.0)))
        self.assertGreaterEqual(float(quantized.min()), -1.0)
        self.assertLessEqual(float(quantized.max()), 127.0 / 128.0)

    def test_pc9_patch_consistency_has_exact_boundary_normalization(self):
        from models.STSwinNet_SNN.bsa_attention import (
            DE9_OFFSETS,
            PC9_PATCH_WEIGHTS,
            _cross_time_match_counts,
            _ensure_match_code,
            _pc9_patch_match_code_attention,
            config_from_dict,
        )

        class MatchModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 1
                self.linear_q = nn.Linear(4, 4, bias=False)

        self.assertEqual(PC9_PATCH_WEIGHTS, (1, 2, 1, 2, 4, 2, 1, 2, 1))
        module = MatchModule()
        cfg = config_from_dict({"mode": "binary_pc9_patch_match_code", "alpha0": 1.0 / 64.0})
        _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
        q_orig = torch.zeros(2, 1, 1, 81, 4, requires_grad=True)
        k_orig = torch.zeros(1, 1, 162, 4, requires_grad=True)
        out, rows, descriptor, scores = _pc9_patch_match_code_attention(
            module, q_orig, k_orig, cfg
        )
        _, _, valid = _cross_time_match_counts(q_orig, k_orig, DE9_OFFSETS)

        self.assertEqual(tuple(out.shape), (1, 1, 162, 4))
        self.assertEqual(tuple(descriptor.shape), (1, 1, 162, 9))
        self.assertEqual(tuple(scores.shape), (1, 1, 162, 9))
        self.assertTrue(torch.allclose(scores[valid], torch.full_like(scores[valid], 1.0 / 64.0)))
        self.assertTrue(torch.all(rows > 0.5 - 1.0e-6))
        self.assertTrue(torch.all(rows <= 1.0 + 1.0e-6))
        out.square().mean().backward()
        self.assertIsNotNone(q_orig.grad)
        self.assertIsNotNone(k_orig.grad)

    def test_lc4_initialization_is_axnor_equivalent_and_dyadic(self):
        from models.STSwinNet_SNN.bsa_attention import (
            DE9_OFFSETS,
            _ensure_match_code,
            _lc4_match_code_attention,
            _match_code_attention,
            _quantized_lc4_coefficients,
            config_from_dict,
        )

        class MatchModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 1
                self.linear_q = nn.Linear(4, 4, bias=False)

        module = MatchModule()
        cfg = config_from_dict({
            "mode": "binary_lc4_match_code",
            "alpha0": 1.0 / 64.0,
            "lc4_coefficient_quant_enabled": True,
            "lc4_coefficient_step": 1.0 / 64.0,
        })
        _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
        expected_init = torch.tensor([[1.0, 0.0, 0.0, 1.0 / 64.0]])
        self.assertTrue(torch.equal(module._h9_lc4_coefficients.detach(), expected_init))
        quantized = _quantized_lc4_coefficients(module, cfg)
        self.assertTrue(torch.allclose(quantized * 64.0, torch.round(quantized * 64.0)))

        q_orig = torch.rand(2, 1, 1, 81, 4)
        k_orig = torch.rand(1, 1, 162, 4)
        lc_out, lc_rows, lc_descriptor, lc_scores = _lc4_match_code_attention(
            module, q_orig, k_orig, cfg
        )
        ax_out, ax_rows, ax_descriptor, ax_scores = _match_code_attention(
            module, q_orig, k_orig, cfg, dual_evidence=False, offsets=DE9_OFFSETS
        )
        self.assertTrue(torch.equal(lc_scores, ax_scores))
        self.assertTrue(torch.equal(lc_descriptor, ax_descriptor))
        self.assertTrue(torch.equal(lc_rows, ax_rows))
        self.assertTrue(torch.equal(lc_out, ax_out))

    def test_g4_uses_four_fixed_eight_lane_shiftmax_groups(self):
        from models.STSwinNet_SNN.bsa_attention import (
            G4_MATCH_GROUP_DIM,
            G4_MATCH_GROUPS,
            _ensure_match_code,
            _g4_match_code_attention,
            config_from_dict,
        )

        class MatchModule(nn.Module):
            def __init__(self, head_dim: int):
                super().__init__()
                self.num_heads = 1
                self.linear_q = nn.Linear(head_dim, head_dim, bias=False)

        self.assertEqual((G4_MATCH_GROUPS, G4_MATCH_GROUP_DIM), (4, 8))
        module = MatchModule(32)
        cfg = config_from_dict({"mode": "binary_g4_match_code", "alpha0": 1.0 / 64.0})
        _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
        q_orig = torch.rand(2, 1, 1, 81, 32, requires_grad=True)
        k_orig = torch.rand(1, 1, 162, 32, requires_grad=True)
        out, rows, descriptor, scores = _g4_match_code_attention(module, q_orig, k_orig, cfg)

        self.assertEqual(tuple(module._h9_match_code_weight.shape), (1, 36, 32))
        self.assertEqual(tuple(out.shape), (1, 1, 162, 32))
        self.assertEqual(tuple(descriptor.shape), (1, 1, 162, 36))
        self.assertEqual(tuple(scores.shape), (1, 1, 162, 36))
        group_rows = descriptor.reshape(1, 1, 162, 4, 9).sum(dim=-1)
        self.assertTrue(torch.all(group_rows > 0.5 - 1.0e-6))
        self.assertTrue(torch.all(group_rows <= 1.0 + 1.0e-6))
        self.assertTrue(torch.allclose(rows, group_rows.sum(dim=-1)))
        out.square().mean().backward()
        self.assertIsNotNone(q_orig.grad)
        self.assertIsNotNone(k_orig.grad)

        bad_module = MatchModule(16)
        _ensure_match_code(bad_module, cfg, "layers.0.swin_blocks.1.attn")
        with self.assertRaisesRegex(ValueError, "head_dim=32"):
            _g4_match_code_attention(
                bad_module,
                torch.rand(2, 1, 1, 81, 16),
                torch.rand(1, 1, 162, 16),
                cfg,
            )

    def test_round3_match_code_has_no_native_k_carrier(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _ensure_match_code,
            _g4_match_code_attention,
            _lc4_match_code_attention,
            _pc9_patch_match_code_attention,
            config_from_dict,
        )

        cases = (
            ("binary_pc9_patch_match_code", _pc9_patch_match_code_attention, 4),
            ("binary_lc4_match_code", _lc4_match_code_attention, 4),
            ("binary_g4_match_code", _g4_match_code_attention, 32),
        )
        for mode, attention_fn, head_dim in cases:
            module = nn.Module()
            module.num_heads = 1
            module.linear_q = nn.Linear(head_dim, head_dim, bias=False)
            cfg = config_from_dict({"mode": mode, "alpha0": 1.0 / 64.0})
            _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
            with torch.no_grad():
                module._h9_match_code_weight.zero_()
            q_orig = torch.rand(2, 1, 1, 81, head_dim)
            k_orig = torch.rand(1, 1, 162, head_dim) + 1.0
            out, _, _, _ = attention_fn(module, q_orig, k_orig, cfg)
            self.assertTrue(torch.equal(out, torch.zeros_like(out)), mode)

    def test_cf10_formula_is_dyadic_and_null_codeword_is_fixed_zero(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _cf10_null_score,
            _effective_cf10_match_code_weight,
            _ensure_match_code,
            _quantized_cf10_beta,
            config_from_dict,
        )

        module = nn.Module()
        module.num_heads = 2
        module.linear_q = nn.Linear(8, 8, bias=False)
        cfg = config_from_dict({
            "mode": "binary_cf10_match_code",
            "cf10_beta_step": 1.0 / 64.0,
            "cf10_beta_min": -1.0,
            "cf10_beta_max": 1.0,
        })
        _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
        with torch.no_grad():
            module._h9_cf10_beta.copy_(torch.tensor([[0.10, -0.10], [2.0, -2.0]]))

        beta = _quantized_cf10_beta(module, cfg)
        self.assertTrue(torch.equal(beta[0] * 64.0, torch.tensor([6.0, -6.0])))
        self.assertTrue(torch.equal(beta[1], torch.tensor([1.0, -1.0])))
        scores = torch.tensor([
            [
                [[0.75, 0.50, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0, -1.25]],
                [[0.60, 0.40, 0.20, 0.0, -0.20, -0.4, -0.60, -0.8, -1.0]],
            ]
        ])
        activity = torch.tensor([[[0.75], [0.25]]])
        null = _cf10_null_score(scores, activity, module, cfg)
        expected = torch.tensor([[
            [0.75 - 1.0 + (6.0 / 64.0) * 0.25 - (6.0 / 64.0) * 0.25],
            [0.60 - 1.0 + 1.0 * 0.20 - 1.0 * (0.25 - 0.5)],
        ]])
        self.assertTrue(torch.allclose(null, expected))

        effective_weight = _effective_cf10_match_code_weight(module, cfg)
        self.assertEqual(tuple(module._h9_match_code_weight.shape), (2, 9, 4))
        self.assertEqual(tuple(effective_weight.shape), (2, 10, 4))
        self.assertTrue(torch.equal(effective_weight[:, -1], torch.zeros_like(effective_weight[:, -1])))
        self.assertNotIn("_h9_cf10_null_weight", dict(module.named_parameters()))

    def test_cf10_shapes_boundaries_gradients_and_no_native_carrier(self):
        from models.STSwinNet_SNN.bsa_attention import (
            DE9_OFFSETS,
            _cf10_match_code_attention,
            _cross_time_match_counts,
            _ensure_match_code,
            config_from_dict,
        )

        module = nn.Module()
        module.num_heads = 1
        module.linear_q = nn.Linear(4, 4, bias=False)
        cfg = config_from_dict({"mode": "binary_cf10_match_code", "alpha0": 1.0 / 64.0})
        _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
        q_orig = torch.rand(2, 1, 1, 81, 4, requires_grad=True)
        k_orig = torch.rand(1, 1, 162, 4, requires_grad=True)
        out, rows, descriptor, scores = _cf10_match_code_attention(module, q_orig, k_orig, cfg)
        _, _, valid = _cross_time_match_counts(q_orig, k_orig, DE9_OFFSETS)

        self.assertEqual(tuple(out.shape), (1, 1, 162, 4))
        self.assertEqual(tuple(rows.shape), (1, 1, 162))
        self.assertEqual(tuple(descriptor.shape), (1, 1, 162, 10))
        self.assertEqual(tuple(scores.shape), (1, 1, 162, 10))
        self.assertTrue(torch.equal(descriptor[..., :9][~valid], torch.zeros_like(descriptor[..., :9][~valid])))
        out.square().mean().backward()
        self.assertIsNotNone(module._h9_match_code_weight.grad)
        self.assertIsNotNone(module._h9_cf10_beta.grad)
        self.assertIsNotNone(q_orig.grad)
        self.assertIsNotNone(k_orig.grad)

        with torch.no_grad():
            module._h9_match_code_weight.zero_()
        zero_out, _, _, _ = _cf10_match_code_attention(
            module, q_orig.detach(), k_orig.detach() + 1.0, cfg
        )
        self.assertTrue(torch.equal(zero_out, torch.zeros_like(zero_out)))

    def test_dn9_incoming_edges_match_dense_reference_at_boundaries(self):
        from models.STSwinNet_SNN.bsa_attention import (
            DE9_OFFSETS,
            _dn9_destination_gate,
            _dn9_edge_indices,
            config_from_dict,
            shiftmax,
        )

        cfg = config_from_dict({"mode": "binary_dn9_match_code"})
        incoming_index, incoming_valid, _, source_valid = _dn9_edge_indices(torch.device("cpu"))
        counts = incoming_valid.sum(dim=-1).reshape(2, 9, 9)
        self.assertTrue(torch.equal(counts[:, 0, 0], torch.tensor([4, 4])))
        self.assertTrue(torch.equal(counts[:, 0, 4], torch.tensor([6, 6])))
        self.assertTrue(torch.equal(counts[:, 4, 4], torch.tensor([9, 9])))
        valid_edge_ids = torch.arange(162 * 9).reshape(162, 9)[source_valid]
        self.assertTrue(torch.equal(torch.sort(incoming_index[incoming_valid]).values, valid_edge_ids))

        torch.manual_seed(80)
        raw_scores = torch.randn(1, 1, 162, 9)
        scores = raw_scores.masked_fill(
            ~source_valid.view(1, 1, 162, 9), torch.finfo(raw_scores.dtype).min
        )
        actual, actual_valid = _dn9_destination_gate(scores, cfg)
        reference = torch.zeros_like(actual)
        incoming = {(t, y, x): [] for t in range(2) for y in range(9) for x in range(9)}
        for source_t in range(2):
            for source_y in range(9):
                for source_x in range(9):
                    source_index = source_t * 81 + source_y * 9 + source_x
                    for offset_index, (dy, dx) in enumerate(DE9_OFFSETS):
                        target_y, target_x = source_y + dy, source_x + dx
                        if 0 <= target_y < 9 and 0 <= target_x < 9:
                            incoming[(1 - source_t, target_y, target_x)].append(
                                (source_index, offset_index)
                            )
        for edges in incoming.values():
            edge_scores = torch.stack([scores[0, 0, source, offset] for source, offset in edges])
            gates = shiftmax(edge_scores, dim=0, eps=cfg.eps)
            for gate, (source, offset) in zip(gates, edges):
                reference[0, 0, source, offset] = gate
        self.assertTrue(torch.equal(actual_valid, source_valid.view(1, 1, 162, 9)))
        self.assertTrue(torch.allclose(actual, reference, atol=1.0e-7, rtol=0.0))

    def test_dn9_shapes_gradients_q17_product_and_no_native_carrier(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _dn9_match_code_attention,
            _ensure_match_code,
            config_from_dict,
        )

        module = nn.Module()
        module.num_heads = 1
        module.linear_q = nn.Linear(4, 4, bias=False)
        cfg = config_from_dict({"mode": "binary_dn9_match_code", "alpha0": 1.0 / 64.0})
        _ensure_match_code(module, cfg, "layers.0.swin_blocks.0.attn")
        q_orig = torch.rand(2, 1, 1, 81, 4, requires_grad=True)
        k_orig = torch.rand(1, 1, 162, 4, requires_grad=True)
        out, rows, descriptor, scores = _dn9_match_code_attention(module, q_orig, k_orig, cfg)

        self.assertEqual(tuple(module._h9_match_code_weight.shape), (1, 9, 4))
        self.assertEqual(tuple(out.shape), (1, 1, 162, 4))
        self.assertEqual(tuple(rows.shape), (1, 1, 162))
        self.assertEqual(tuple(descriptor.shape), (1, 1, 162, 9))
        self.assertEqual(tuple(scores.shape), (1, 1, 162, 9))
        self.assertTrue(torch.allclose(descriptor * 128.0, torch.round(descriptor * 128.0)))
        out.square().mean().backward()
        self.assertIsNotNone(module._h9_match_code_weight.grad)
        self.assertIsNotNone(q_orig.grad)
        self.assertIsNotNone(k_orig.grad)

        with torch.no_grad():
            module._h9_match_code_weight.zero_()
        zero_out, _, _, _ = _dn9_match_code_attention(
            module, q_orig.detach(), k_orig.detach() + 1.0, cfg
        )
        self.assertTrue(torch.equal(zero_out, torch.zeros_like(zero_out)))

    def test_round4_assignment_modes_are_default_off(self):
        from models.STSwinNet_SNN.bsa_attention import config_from_dict

        cfg = config_from_dict(None)
        self.assertFalse(cfg.enabled)
        self.assertNotIn(cfg.mode, {"binary_cf10_match_code", "binary_dn9_match_code"})

    def test_delta_locality_stats_are_raw_and_bundle_weightable(self):
        import base64
        import zlib

        from models.STSwinNet_SNN.bsa_attention import _delta_locality_stats

        q = torch.zeros(1, 1, 5, 8, dtype=torch.bool)
        k = torch.zeros_like(q)
        q[0, 0, 1, 0] = True
        k[0, 0, 2, :2] = True
        q[0, 0, 4, :5] = True
        stats = _delta_locality_stats(q, k)

        self.assertEqual(stats["delta_token_heads"], 5)
        self.assertEqual(stats["delta_zero_update_token_heads"], 2)
        self.assertEqual(stats["delta_changed_token_heads"], 3)
        self.assertEqual(stats["delta_changed_token_runs"], 2)
        self.assertEqual(stats["delta_update_count_1"], 1)
        self.assertEqual(stats["delta_update_count_2"], 1)
        self.assertEqual(stats["delta_update_count_5_8"], 1)
        self.assertEqual(stats["delta_active_le12"], 3)
        self.assertEqual(stats["delta_active_lane_sum_le12"], 8)
        self.assertEqual(stats["delta_update_histogram"], [2, 1, 1, 0, 0, 1, 0, 0, 0])
        self.assertEqual(stats["delta_bundle4_total"], 2)
        self.assertEqual(stats["delta_bundle4_empty"], 0)
        self.assertEqual(stats["delta_bundle8_total"], 1)
        self.assertEqual(stats["delta_bundle8_empty"], 0)

        traced = _delta_locality_stats(q, k, include_ordered_trace=True)
        encoded = traced["delta_update_ordered_trace"]
        raw = zlib.decompress(base64.b64decode(encoded["data"]))
        decoded = torch.frombuffer(bytearray(raw), dtype=torch.int16).reshape(encoded["shape"])
        self.assertTrue(torch.equal(decoded, torch.tensor([[[0, 1, 2, 0, 5]]], dtype=torch.int16)))

    def test_window_context_broadcast_matches_parameter_free_cb(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _window_context_broadcast,
            config_from_dict,
        )

        tokens = torch.tensor([[[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]]])
        enabled = config_from_dict({"context_broadcast_enabled": True})
        actual = _window_context_broadcast(tokens, enabled)
        expected = 0.5 * (tokens + tokens.mean(dim=2, keepdim=True))
        self.assertTrue(torch.allclose(actual, expected))

        disabled = config_from_dict({"context_broadcast_enabled": False})
        self.assertTrue(torch.equal(_window_context_broadcast(tokens, disabled), tokens))

    def test_event_selective_temperature_uses_dyadic_activity_bins(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _event_selective_temperature,
            config_from_dict,
        )

        q_orig = torch.zeros(2, 1, 1, 2, 8)
        k_orig = torch.zeros(1, 1, 4, 8)
        k_orig[0, 0, 1, :1] = 1.0
        k_orig[0, 0, 2, :2] = 1.0
        k_orig[0, 0, 3, :4] = 1.0
        scores = torch.ones(1, 1, 4, 1)

        enabled = config_from_dict({
            "event_temperature_enabled": True,
            "event_temperature_max_shift": 3,
        })
        scaled = _event_selective_temperature(scores, q_orig, k_orig, enabled)
        self.assertEqual(scaled.flatten().tolist(), [1.0, 2.0, 4.0, 8.0])

        disabled = config_from_dict({"event_temperature_enabled": False})
        identity = _event_selective_temperature(scores, q_orig, k_orig, disabled)
        self.assertTrue(torch.equal(identity, scores))

    def test_castling_auxiliary_is_training_only_and_anneals_to_zero(self):
        from models.STSwinNet_SNN.bsa_attention import _castling_aux_weight, config_from_dict

        module = nn.Identity()
        cfg = config_from_dict({
            "castling_matrix_aux_weight": 0.5,
            "castling_matrix_aux_end_step": 360,
        })
        module.train()
        module._h9_global_step = 0
        self.assertAlmostEqual(_castling_aux_weight(module, cfg), 0.5)
        module._h9_global_step = 180
        self.assertAlmostEqual(_castling_aux_weight(module, cfg), 0.25)
        module._h9_global_step = 360
        self.assertEqual(_castling_aux_weight(module, cfg), 0.0)
        module._h9_global_step = 0
        module.eval()
        self.assertEqual(_castling_aux_weight(module, cfg), 0.0)

    def test_castling_matrix_output_can_align_with_fp32_h60_under_amp(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _castling_binary_matrix_output,
            config_from_dict,
        )

        cfg = config_from_dict({"enabled": True, "center_scores": True})
        q_orig = torch.randint(0, 2, (2, 1, 1, 2, 4)).half()
        k_orig = torch.randint(0, 2, (1, 1, 4, 4)).half()
        h60_output = torch.randn(1, 1, 4, 4, dtype=torch.float32)

        matrix_aux = _castling_binary_matrix_output(q_orig, k_orig, cfg).to(h60_output.dtype)
        mixed = torch.lerp(h60_output, matrix_aux, 0.5)

        self.assertEqual(mixed.dtype, torch.float32)
        self.assertEqual(tuple(mixed.shape), tuple(h60_output.shape))

    def test_binary_temporal_k_xor_popcount_pairs_same_position(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_temporal_k_xor_popcount

        q_orig = torch.zeros(2, 1, 1, 2, 4)
        k_orig = torch.tensor(
            [[[[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]]]
        )

        motion = _binary_temporal_k_xor_popcount(q_orig, k_orig)

        self.assertEqual(tuple(motion.shape), (1, 1, 4, 1))
        self.assertEqual(motion.flatten().tolist(), [1.0, 2.0, 1.0, 2.0])

    def test_binary_axnor_stencil_attention_shapes_and_border_mask(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _binary_alpha_xnor_stencil_attention,
            config_from_dict,
        )

        cfg = config_from_dict({
            "enabled": True,
            "alpha0": 1.0 / 64.0,
            "consensus_score_norm": "head_dim",
            "preserve_mean": False,
        })
        q_orig = torch.randint(0, 2, (2, 1, 1, 81, 4)).float()
        k_orig = torch.randint(0, 2, (1, 1, 162, 4)).float()

        tp_out, tp_rows, tp_gate = _binary_alpha_xnor_stencil_attention(
            q_orig, k_orig, cfg, temporal_pair=True, spatial_cross=False
        )
        lr_out, lr_rows, lr_gate = _binary_alpha_xnor_stencil_attention(
            q_orig, k_orig, cfg, temporal_pair=False, spatial_cross=True
        )

        self.assertEqual(tuple(tp_out.shape), (1, 1, 162, 4))
        self.assertEqual(tuple(tp_gate.shape), (1, 1, 162, 2))
        self.assertEqual(tuple(lr_out.shape), (1, 1, 162, 4))
        self.assertEqual(tuple(lr_gate.shape), (1, 1, 162, 5))
        self.assertTrue(torch.all(tp_rows > 0.5 - 1e-6))
        self.assertTrue(torch.all(lr_rows > 0.5 - 1e-6))
        # Top-left token has no up/left neighbor, so those two lanes are zero.
        self.assertEqual(float(lr_gate[0, 0, 0, 1]), 0.0)
        self.assertEqual(float(lr_gate[0, 0, 0, 3]), 0.0)

        # Paper full-resolution protocol changes the spatial window to 15x15
        # while retaining the exact same five stencil lanes.
        q_w15 = torch.randint(0, 2, (2, 1, 1, 225, 4)).float()
        k_w15 = torch.randint(0, 2, (1, 1, 450, 4)).float()
        w15_out, w15_rows, w15_gate = _binary_alpha_xnor_stencil_attention(
            q_w15, k_w15, cfg, temporal_pair=False, spatial_cross=True
        )
        self.assertEqual(tuple(w15_out.shape), (1, 1, 450, 4))
        self.assertEqual(tuple(w15_gate.shape), (1, 1, 450, 5))
        self.assertTrue(torch.all(w15_rows > 0.5 - 1e-6))

        # Scheme A: Local-5 + temporal peer -> 6 candidates.
        l5tp_out, l5tp_rows, l5tp_gate = _binary_alpha_xnor_stencil_attention(
            q_orig, k_orig, cfg, temporal_pair=True, spatial_cross=True
        )
        self.assertEqual(tuple(l5tp_out.shape), (1, 1, 162, 4))
        self.assertEqual(tuple(l5tp_gate.shape), (1, 1, 162, 6))
        self.assertTrue(torch.all(l5tp_rows > 0.5 - 1e-6))

        # Local-5 + motion bias on self lane only (must change gate vs pure Local-5).
        torch.manual_seed(0)
        q_m = torch.randint(0, 2, (2, 1, 1, 81, 4)).float()
        k_m = torch.randint(0, 2, (1, 1, 162, 4)).float()
        _, _, gate0 = _binary_alpha_xnor_stencil_attention(
            q_m, k_m, cfg, temporal_pair=False, spatial_cross=True, motion_xor_alpha=0.0
        )
        _, _, gate_m = _binary_alpha_xnor_stencil_attention(
            q_m, k_m, cfg, temporal_pair=False, spatial_cross=True, motion_xor_alpha=0.25
        )
        self.assertFalse(torch.allclose(gate0, gate_m))

    def test_rtl_shiftmax_true_mask_excludes_invalid_candidates(self):
        from models.STSwinNet_SNN.bsa_attention import _rtl_shiftmax_gate_q17

        scores = torch.tensor([[[[0.25, 0.125, -2.0, -2.0, -2.0]]]])
        valid = torch.tensor([[[[True, True, False, False, False]]]])
        masked = _rtl_shiftmax_gate_q17(
            scores,
            dim=-1,
            preserve_mean=False,
            valid_mask=valid,
        )
        two_candidate = _rtl_shiftmax_gate_q17(
            scores[..., :2],
            dim=-1,
            preserve_mean=False,
        )

        self.assertTrue(torch.equal(masked[..., :2], two_candidate))
        self.assertTrue(torch.equal(masked[..., 2:], torch.zeros_like(masked[..., 2:])))

    def test_shiftmax_row_sum_is_power_two_bounded(self):
        from models.STSwinNet_SNN.bsa_attention import shiftmax

        scores = torch.randn(4, 3, 17, 1)
        probs = shiftmax(scores, dim=2)
        row_sum = probs.sum(dim=2)

        self.assertTrue(torch.all(row_sum > 0.5 - 1e-6))
        self.assertTrue(torch.all(row_sum <= 1.0 + 1e-6))
        self.assertFalse(torch.isnan(probs).any())

    def test_rtl_shiftmax_q17_matches_lut_normalization_and_saturation(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _rtl_shiftmax_gate_q17,
            config_from_dict,
        )

        cfg = config_from_dict({"hardware_rtl_shiftmax_enabled": True})
        self.assertTrue(cfg.hardware_rtl_shiftmax_enabled)

        scores = torch.tensor([[[[0.0], [-1.0]]]])
        gate = _rtl_shiftmax_gate_q17(scores, dim=2, preserve_mean=True)
        self.assertTrue(torch.equal(gate, torch.tensor([[[[1.0], [0.5]]]])))

        translated = _rtl_shiftmax_gate_q17(scores + 0.375, dim=2, preserve_mean=True)
        self.assertTrue(torch.equal(gate, translated))

        sparse_scores = torch.full((1, 1, 162, 1), -100.0)
        sparse_scores[:, :, 0, :] = 0.0
        sparse_gate = _rtl_shiftmax_gate_q17(sparse_scores, dim=2, preserve_mean=True)
        self.assertEqual(float(sparse_gate[0, 0, 0, 0]), 2.0)
        self.assertTrue(torch.all(sparse_gate >= 0.0))
        self.assertTrue(torch.all(sparse_gate <= 2.0))

    def test_shiftnorm_row_sum_is_power_two_bounded(self):
        from models.STSwinNet_SNN.bsa_attention import l1norm, shiftnorm

        scores = torch.randint(0, 9, (4, 3, 17, 1)).float()
        probs = shiftnorm(scores, dim=2)
        row_sum = probs.sum(dim=2)

        self.assertTrue(torch.all(row_sum > 0.5 - 1e-6))
        self.assertTrue(torch.all(row_sum <= 1.0 + 1e-6))
        self.assertFalse(torch.isnan(probs).any())

        l1_probs = l1norm(scores, dim=2)
        self.assertTrue(torch.allclose(l1_probs.sum(dim=2), torch.ones_like(row_sum), atol=1e-6))
        self.assertFalse(torch.isnan(l1_probs).any())

    def test_target_block_selection_reports_missing_blocks(self):
        from models.STSwinNet_SNN.bsa_attention import _iter_attention_modules, config_from_dict

        model = DummyModel()
        cfg = config_from_dict({"enabled": True, "target_blocks": ["0:1"]})
        pairs = list(_iter_attention_modules(model, cfg))
        self.assertEqual([name for name, _ in pairs], ["layers.0.swin_blocks.1.attn"])

        bad_cfg = config_from_dict({"enabled": True, "target_blocks": ["9:9"]})
        with self.assertRaises(KeyError):
            list(_iter_attention_modules(model, bad_cfg))

    def test_qk_bsa_mode_runs_on_tiny_attention(self):
        from models.STSwinNet_SNN.bsa_attention import _qk_shiftmax_gate_forward, config_from_dict

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": "qk_bsa",
                        "center_scores": False,
                        "preserve_mean": False,
                    }
                )

        module = TinyAttention()
        x = torch.randn(1, 2, 1, 2, 4)
        out, spikes = _qk_shiftmax_gate_forward(module, x)

        self.assertEqual(tuple(out.shape), (2, 2, 4))
        self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
        self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.0)
        self.assertLessEqual(module.h9_shiftmax_row_sum_mean, 1.0)

    def test_h13_consensus_modes_run_on_tiny_attention(self):
        from models.STSwinNet_SNN.bsa_attention import _qk_shiftmax_gate_forward, config_from_dict

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, mode: str):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": mode,
                        "center_scores": False,
                        "preserve_mean": False,
                        "consensus_bias": 1.0,
                        "single_active_penalty": 0.2,
                    }
                )

        for mode in ("signed_consensus_shiftmax", "signed_consensus_shiftnorm", "signed_consensus_popcount_l1"):
            module = TinyAttention(mode)
            x = torch.randn(1, 2, 1, 2, 4)
            out, spikes = _qk_shiftmax_gate_forward(module, x)

            self.assertEqual(tuple(out.shape), (2, 2, 4))
            self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
            self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.0)
            self.assertLessEqual(module.h9_shiftmax_row_sum_mean, 1.0 + 1e-6)
            self.assertFalse(torch.isnan(out).any())

    def test_single_active_penalty_covers_zero_nonzero_mismatch(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _signed_consensus_token_scores,
            _ternary_alpha_xnor_matrix_scores,
            _ternary_alpha_xnor_matrix_scores_ste,
            _ternary_alpha_xnor_token_scores,
            config_from_dict,
        )

        base_cfg = config_from_dict(
            {
                "enabled": True,
                "consensus_score_norm": "none",
                "alpha0": 0.2,
                "mismatch_penalty": 0.5,
            }
        )
        cfg = config_from_dict(
            {
                "enabled": True,
                "consensus_score_norm": "none",
                "alpha0": 0.2,
                "mismatch_penalty": 0.5,
                "single_active_penalty": 0.3,
            }
        )
        q_orig = torch.tensor([[[[[1.0], [0.0], [1.0], [0.0]]]]])
        k_orig = torch.tensor([[[[0.0], [1.0], [-1.0], [0.0]]]])

        base_alpha_token = _ternary_alpha_xnor_token_scores(q_orig, k_orig, base_cfg)
        self.assertTrue(
            torch.allclose(
                base_alpha_token.reshape(-1),
                torch.tensor([0.0, 0.0, -0.5, 0.2]),
                atol=1e-6,
            )
        )

        alpha_token = _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg)
        self.assertTrue(
            torch.allclose(
                alpha_token.reshape(-1),
                torch.tensor([-0.3, -0.3, -0.5, 0.2]),
                atol=1e-6,
            )
        )

        consensus_token = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        self.assertTrue(
            torch.allclose(
                consensus_token.reshape(-1),
                torch.tensor([-0.3, -0.3, -1.0, 0.0]),
                atol=1e-6,
            )
        )

        alpha_matrix = _ternary_alpha_xnor_matrix_scores_ste(q_orig, k_orig, cfg)
        self.assertAlmostEqual(float(alpha_matrix[0, 0, 0, 0]), -0.3, places=6)
        self.assertAlmostEqual(float(alpha_matrix[0, 0, 1, 1]), -0.3, places=6)
        self.assertAlmostEqual(float(alpha_matrix[0, 0, 2, 2]), -0.5, places=6)
        self.assertAlmostEqual(float(alpha_matrix[0, 0, 3, 3]), 0.2, places=6)

        alpha_matrix_hard = _ternary_alpha_xnor_matrix_scores(q_orig, k_orig, cfg)
        self.assertTrue(torch.allclose(alpha_matrix_hard, alpha_matrix, atol=1e-6))

    def test_dualrail_binary_tx_penalizes_opposite_rails(self):
        from models.STSwinNet_SNN.bsa_attention import _dualrail_binary_tx_token_scores, config_from_dict

        cfg = config_from_dict(
            {
                "enabled": True,
                "consensus_score_norm": "none",
                "alpha0": 0.02,
                "mismatch_penalty": 0.5,
                "single_active_penalty": 0.25,
            }
        )
        q_orig = torch.tensor([[[[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]]]])
        k_orig = torch.tensor([[[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]])

        scores = _dualrail_binary_tx_token_scores(q_orig, k_orig, cfg)

        self.assertTrue(
            torch.allclose(
                scores.reshape(-1),
                torch.tensor([2.0, -1.0, -0.5]),
                atol=1e-6,
            )
        )

    def test_direct_group_shiftmax_outputs_group_gates_without_k_carrier(self):
        from models.STSwinNet_SNN.bsa_attention import _direct_group_shiftmax_output, config_from_dict

        cfg = config_from_dict(
            {
                "enabled": True,
                "alpha0": 1.0 / 64.0,
                "center_scores": True,
                "preserve_mean": True,
                "direct_shiftmax_groups": 2,
            }
        )
        q_orig = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]]]])
        k_a = torch.tensor([[[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]]])
        k_b = k_a * 7.0

        out_a, row_sum, gate, _ = _direct_group_shiftmax_output(q_orig, k_a, cfg)
        out_b, _, _, _ = _direct_group_shiftmax_output(q_orig, k_b, cfg)

        self.assertEqual(tuple(out_a.shape), tuple(k_a.shape))
        self.assertEqual(tuple(gate.shape), (1, 1, 2, 2))
        self.assertEqual(tuple(row_sum.shape), (1, 1, 2))
        self.assertTrue(torch.allclose(out_a, out_b))
        self.assertTrue(torch.allclose(out_a[..., 0], out_a[..., 1]))
        self.assertTrue(torch.allclose(out_a[..., 2], out_a[..., 3]))

    def test_direct_group_shiftmax_requires_divisible_head_dim(self):
        from models.STSwinNet_SNN.bsa_attention import _binary_tx_group_scores, config_from_dict

        cfg = config_from_dict({"direct_shiftmax_groups": 3})
        q_orig = torch.zeros(1, 1, 1, 2, 4)
        k_orig = torch.zeros(1, 1, 2, 4)
        with self.assertRaises(ValueError):
            _binary_tx_group_scores(q_orig, k_orig, cfg)

    def test_direct_token_channel_shiftmax_has_full_channel_shape_without_k_carrier(self):
        from models.STSwinNet_SNN.bsa_attention import _direct_token_channel_shiftmax_output, config_from_dict

        cfg = config_from_dict(
            {
                "alpha0": 1.0 / 64.0,
                "center_scores": True,
                "preserve_mean": True,
                "direct_shiftmax_center_output": True,
            }
        )
        q_orig = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]]])
        k_a = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]])
        k_b = k_a * 9.0

        out_a, row_sum, _, _ = _direct_token_channel_shiftmax_output(q_orig, k_a, cfg)
        out_b, _, _, _ = _direct_token_channel_shiftmax_output(q_orig, k_b, cfg)

        self.assertEqual(tuple(out_a.shape), tuple(k_a.shape))
        self.assertEqual(tuple(row_sum.shape), (1, 1, 1))
        self.assertTrue(torch.allclose(out_a, out_b))
        self.assertFalse(torch.allclose(out_a[..., 0], out_a[..., 1]))

    def test_direct_signed_tx_counts_positive_and_negative_matches_equally(self):
        from models.STSwinNet_SNN.bsa_attention import _direct_tx_channel_evidence, config_from_dict

        cfg = config_from_dict(
            {"alpha0": 1.0 / 64.0, "direct_shiftmax_signed_events": True}
        )
        q_orig = torch.tensor([[[[[1.0, -1.0, 1.0, 0.0]]]]])
        k_orig = torch.tensor([[[[1.0, -1.0, -1.0, 0.0]]]])

        evidence = _direct_tx_channel_evidence(q_orig, k_orig, cfg).reshape(-1)

        self.assertTrue(
            torch.allclose(evidence, torch.tensor([1.0, 1.0, 0.0, 1.0 / 64.0]))
        )

    def test_motion_alpha_zero_matches_disabled_saliency(self):
        from models.STSwinNet_SNN.bsa_attention import _signed_consensus_token_scores, config_from_dict

        torch.manual_seed(0)
        q_orig = torch.randn(2, 1, 2, 3, 4)
        k_orig = torch.randn(1, 2, 6, 4)
        base_cfg = config_from_dict(
            {"enabled": True, "consensus_score_norm": "none", "motion_weight_alpha": 0.0}
        )
        off_cfg = config_from_dict({"enabled": True, "consensus_score_norm": "none"})
        base_score = _signed_consensus_token_scores(q_orig, k_orig, base_cfg)
        off_score = _signed_consensus_token_scores(q_orig, k_orig, off_cfg)
        self.assertTrue(torch.allclose(base_score, off_score, atol=1e-6))

    def test_temporal_motion_token_alignment_and_first_frame_zero(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _signed_consensus_token_scores,
            _temporal_motion_from_q_orig,
            config_from_dict,
        )

        q_orig = torch.zeros(2, 1, 2, 3, 4)
        q_orig[1, 0, 0, 1, :] = 2.0
        q_orig[1, 0, 1, :, :] = -1.5
        k_orig = torch.zeros(1, 2, 6, 4)
        cfg = config_from_dict(
            {"enabled": True, "consensus_score_norm": "none", "motion_weight_alpha": 0.1}
        )

        motion = _temporal_motion_from_q_orig(q_orig, cfg)
        self.assertEqual(tuple(motion.shape), (1, 2, 6, 1))
        self.assertTrue(torch.all(motion[:, :, :3, :] == 0))
        self.assertTrue(motion.abs().sum() > 0)

        score = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        self.assertEqual(tuple(score.shape), (1, 2, 6, 1))

    def test_temporal_motion_normalizes_per_head(self):
        from models.STSwinNet_SNN.bsa_attention import _temporal_motion_from_q_orig, config_from_dict

        q_orig = torch.zeros(2, 1, 2, 2, 2)
        q_orig[1, 0, 0, 0, :] = 10.0
        q_orig[1, 0, 1, 1, :] = 0.01
        cfg = config_from_dict({"enabled": True, "motion_weight_alpha": 0.1})

        motion = _temporal_motion_from_q_orig(q_orig, cfg)
        # Flatten order is t0*n0, t0*n1, t1*n0, t1*n1 per head.
        self.assertAlmostEqual(float(motion[0, 0, 2, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(motion[0, 1, 3, 0]), 1.0, places=5)
        self.assertLess(float(motion[0, 1, 2, 0]), 0.1)

    def test_signed_consensus_ste_single_active_keeps_forward_and_adds_gradient(self):
        from models.STSwinNet_SNN.bsa_attention import _signed_consensus_token_scores, config_from_dict

        hard_cfg = config_from_dict(
            {
                "enabled": True,
                "consensus_score_norm": "none",
                "single_active_penalty": 0.3,
                "single_active_penalty_grad": "hard",
            }
        )
        ste_cfg = config_from_dict(
            {
                "enabled": True,
                "consensus_score_norm": "none",
                "single_active_penalty": 0.3,
                "single_active_penalty_grad": "ste",
                "single_active_ste_slope": 4.0,
                "single_active_ste_margin": 0.25,
            }
        )
        q_hard = torch.tensor([[[[[1.0, 0.0]]]]], requires_grad=True)
        k_hard = torch.tensor([[[[0.0, 1.0]]]], requires_grad=True)
        q_ste = q_hard.detach().clone().requires_grad_(True)
        k_ste = k_hard.detach().clone().requires_grad_(True)

        hard_score = _signed_consensus_token_scores(q_hard, k_hard, hard_cfg)
        ste_score = _signed_consensus_token_scores(q_ste, k_ste, ste_cfg)
        self.assertTrue(torch.allclose(hard_score, ste_score, atol=1e-6))
        self.assertAlmostEqual(float(ste_score.reshape(-1)[0]), -0.6, places=6)

        hard_score.sum().backward()
        ste_score.sum().backward()
        self.assertAlmostEqual(float(q_hard.grad.reshape(-1)[0]), 0.0, places=6)
        self.assertGreater(abs(float(q_ste.grad.reshape(-1)[0])), 0.0)
        self.assertAlmostEqual(float(k_hard.grad.reshape(-1)[1]), 0.0, places=6)
        self.assertGreater(abs(float(k_ste.grad.reshape(-1)[1])), 0.0)

    def test_h49_token_single_active_ste_keeps_forward_and_adds_gradient(self):
        from models.STSwinNet_SNN.bsa_attention import _ternary_alpha_xnor_token_scores, config_from_dict

        hard_cfg = config_from_dict(
            {
                "enabled": True,
                "consensus_score_norm": "none",
                "alpha0": 0.2,
                "mismatch_penalty": 0.5,
                "single_active_penalty": 0.3,
                "single_active_penalty_grad": "hard",
            }
        )
        ste_cfg = config_from_dict(
            {
                "enabled": True,
                "consensus_score_norm": "none",
                "alpha0": 0.2,
                "mismatch_penalty": 0.5,
                "single_active_penalty": 0.3,
                "single_active_penalty_grad": "ste",
                "single_active_ste_slope": 4.0,
                "single_active_ste_margin": 0.25,
            }
        )
        q_hard = torch.tensor([[[[[1.0, 0.0]]]]], requires_grad=True)
        k_hard = torch.tensor([[[[0.0, 1.0]]]], requires_grad=True)
        q_ste = q_hard.detach().clone().requires_grad_(True)
        k_ste = k_hard.detach().clone().requires_grad_(True)

        hard_score = _ternary_alpha_xnor_token_scores(q_hard, k_hard, hard_cfg)
        ste_score = _ternary_alpha_xnor_token_scores(q_ste, k_ste, ste_cfg)
        self.assertTrue(torch.allclose(hard_score, ste_score, atol=1e-6))
        self.assertAlmostEqual(float(ste_score.reshape(-1)[0]), -0.6, places=6)
        self.assertFalse(hard_score.requires_grad)
        self.assertTrue(ste_score.requires_grad)

        ste_score.sum().backward()
        self.assertIsNone(q_hard.grad)
        self.assertGreater(abs(float(q_ste.grad.reshape(-1)[0])), 0.0)
        self.assertIsNone(k_hard.grad)
        self.assertGreater(abs(float(k_ste.grad.reshape(-1)[1])), 0.0)

    def test_h54_bipolar_score_components_split_tx_evidence(self):
        from models.STSwinNet_SNN.bsa_attention import _bipolar_token_score_components, config_from_dict

        cfg = config_from_dict(
            {
                "enabled": True,
                "consensus_score_norm": "none",
                "alpha0": 0.2,
                "mismatch_penalty": 0.5,
                "single_active_penalty": 0.3,
            }
        )
        q_orig = torch.tensor([[[[[1.0, 1.0, 0.0, -1.0]]]]])
        k_orig = torch.tensor([[[[1.0, -1.0, 0.0, -1.0]]]])

        tx_score, same_score, opp_score = _bipolar_token_score_components(q_orig, k_orig, cfg)
        self.assertAlmostEqual(float(same_score.reshape(-1)[0]), 2.2, places=6)
        self.assertAlmostEqual(float(opp_score.reshape(-1)[0]), 1.0, places=6)
        self.assertAlmostEqual(float(tx_score.reshape(-1)[0]), 1.7, places=6)

    def test_h54_bipolar_modes_run_on_tiny_attention_and_can_make_signed_gate(self):
        from models.STSwinNet_SNN.bsa_attention import _qk_shiftmax_gate_forward, config_from_dict

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, mode: str):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": mode,
                        "center_scores": False,
                        "preserve_mean": False,
                        "consensus_score_norm": "none",
                        "alpha0": 0.02,
                        "mismatch_penalty": 0.25,
                        "single_active_penalty": 0.2,
                        "bipolar_mu": 0.5,
                        "bipolar_lambda": 1.0,
                    }
                )

        for mode in ("bipolar_qkselector_shiftmax", "tx_bipolar_qkselector_shiftmax"):
            module = TinyAttention(mode)
            x = torch.randn(1, 2, 1, 2, 4)
            out, spikes = _qk_shiftmax_gate_forward(module, x)

            self.assertEqual(tuple(out.shape), (2, 2, 4))
            self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
            self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.0)
            self.assertFalse(torch.isnan(out).any())

    def test_tx_sc_fusion_applies_k_mag_on_tx_only(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _signed_consensus_token_scores,
            _ternary_alpha_xnor_token_scores,
            _tx_sc_fusion_score_pair,
            config_from_dict,
        )

        cfg = config_from_dict(
            {
                "enabled": True,
                "mode": "tx_sc_score_residual_shiftmax",
                "center_scores": False,
                "preserve_mean": False,
                "consensus_score_norm": "none",
                "alpha0": 0.02,
                "mismatch_penalty": 0.25,
                "single_active_penalty": 0.0,
                "k_magnitude_alpha": 0.2,
            }
        )
        q_orig = torch.tensor(
            [[[[[1.5, 0.0], [0.0, -1.0], [0.5, 0.0], [0.0, 1.2]]]]],
            dtype=torch.float32,
        )
        k_orig = torch.tensor(
            [[[[2.0, 0.0], [0.0, -2.0], [1.0, 0.0], [0.0, 2.5]]]],
            dtype=torch.float32,
        )

        tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        tx_direct = _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg)
        sc_direct = _signed_consensus_token_scores(q_orig, k_orig, cfg)

        self.assertTrue(torch.allclose(tx_scores, tx_direct))
        self.assertFalse(torch.allclose(sc_scores, sc_direct))
        from dataclasses import asdict

        cfg_no_kmag = config_from_dict({**asdict(cfg), "k_magnitude_alpha": 0.0})
        sc_without_kmag = _signed_consensus_token_scores(q_orig, k_orig, cfg_no_kmag)
        self.assertTrue(torch.allclose(sc_scores, sc_without_kmag))

    def test_h57_tx_sc_residual_selector_runs_on_tiny_attention(self):
        from models.STSwinNet_SNN.bsa_attention import _qk_shiftmax_gate_forward, config_from_dict

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, mu: float):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": "tx_sc_residual_selector_shiftmax",
                        "center_scores": False,
                        "preserve_mean": False,
                        "consensus_score_norm": "none",
                        "alpha0": 0.02,
                        "mismatch_penalty": 0.25,
                        "single_active_penalty": 0.05,
                        "single_active_penalty_grad": "ste",
                        "bipolar_mu": mu,
                        "bipolar_lambda": 0.4,
                    }
                )

        for mu in (0.0, 0.15):
            module = TinyAttention(mu)
            x = torch.randn(1, 2, 1, 2, 4)
            out, spikes = _qk_shiftmax_gate_forward(module, x)

            self.assertEqual(tuple(out.shape), (2, 2, 4))
            self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
            self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.0)
            self.assertFalse(torch.isnan(out).any())

        module = TinyAttention(0.05)
        module._h9_shiftmax_cfg = config_from_dict(
            {
                "enabled": True,
                "mode": "tx_sc_score_residual_shiftmax",
                "center_scores": False,
                "preserve_mean": False,
                "consensus_score_norm": "none",
                "alpha0": 0.02,
                "mismatch_penalty": 0.25,
                "single_active_penalty": 0.05,
                "single_active_penalty_grad": "ste",
                "bipolar_mu": 0.05,
                "bipolar_lambda": 0.4,
            }
        )
        x = torch.randn(1, 2, 1, 2, 4)
        out, spikes = _qk_shiftmax_gate_forward(module, x)
        self.assertEqual(tuple(out.shape), (2, 2, 4))
        self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
        self.assertFalse(torch.isnan(out).any())

    def test_h58_late_residual_schedule_matches_endpoint_mu(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _qk_shiftmax_gate_forward,
            config_from_dict,
            set_shiftmax_attention_step,
        )

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, mode: str, mu: float, schedule: bool = False):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": mode,
                        "center_scores": False,
                        "preserve_mean": False,
                        "consensus_score_norm": "none",
                        "alpha0": 0.02,
                        "mismatch_penalty": 0.25,
                        "single_active_penalty": 0.05,
                        "single_active_penalty_grad": "ste",
                        "bipolar_mu": mu,
                        "bipolar_lambda": 0.4,
                        "sc_mu_schedule_enabled": schedule,
                        "sc_mu_start_step": 10,
                        "sc_mu_warmup_steps": 10,
                        "sc_mu_start": 0.0,
                    }
                )

        torch.manual_seed(7)
        control0 = TinyAttention("tx_sc_residual_selector_shiftmax", 0.0)
        scheduled = TinyAttention("tx_sc_late_residual_selector_shiftmax", 0.10, schedule=True)
        fixed = TinyAttention("tx_sc_residual_selector_shiftmax", 0.10)
        scheduled.load_state_dict(control0.state_dict())
        fixed.load_state_dict(control0.state_dict())
        x = torch.randn(1, 2, 1, 2, 4)

        set_shiftmax_attention_step(scheduled, 0)
        out_start, _ = _qk_shiftmax_gate_forward(scheduled, x)
        out_control, _ = _qk_shiftmax_gate_forward(control0, x)
        self.assertTrue(torch.allclose(out_start, out_control, atol=1e-6))

        set_shiftmax_attention_step(scheduled, 20)
        out_final, _ = _qk_shiftmax_gate_forward(scheduled, x)
        out_fixed, _ = _qk_shiftmax_gate_forward(fixed, x)
        self.assertTrue(torch.allclose(out_final, out_fixed, atol=1e-6))

    def test_h60_no_carrier_schedule_matches_endpoint_mu(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _qk_shiftmax_gate_forward,
            config_from_dict,
            set_shiftmax_attention_step,
        )

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, mu: float, schedule: bool = False):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": "h60",
                        "center_scores": False,
                        "preserve_mean": False,
                        "consensus_score_norm": "none",
                        "alpha0": 0.02,
                        "mismatch_penalty": 0.25,
                        "single_active_penalty": 0.05,
                        "single_active_penalty_grad": "ste",
                        "bipolar_mu": mu,
                        "bipolar_lambda": 0.4,
                        "k_magnitude_alpha": 0.0,
                        "sc_mu_schedule_enabled": schedule,
                        "sc_mu_start_step": 10,
                        "sc_mu_warmup_steps": 10,
                        "sc_mu_start": 0.0,
                    }
                )

        torch.manual_seed(11)
        control0 = TinyAttention(0.0)
        scheduled = TinyAttention(0.10, schedule=True)
        fixed = TinyAttention(0.10)
        scheduled.load_state_dict(control0.state_dict())
        fixed.load_state_dict(control0.state_dict())
        x = torch.randn(1, 2, 1, 2, 4)

        set_shiftmax_attention_step(scheduled, 0)
        out_start, _ = _qk_shiftmax_gate_forward(scheduled, x)
        out_control, _ = _qk_shiftmax_gate_forward(control0, x)
        self.assertTrue(torch.allclose(out_start, out_control, atol=1e-6))

        set_shiftmax_attention_step(scheduled, 20)
        out_final, _ = _qk_shiftmax_gate_forward(scheduled, x)
        out_fixed, _ = _qk_shiftmax_gate_forward(fixed, x)
        self.assertTrue(torch.allclose(out_final, out_fixed, atol=1e-6))

    def test_strict_bsa_matrix_modes_use_bounded_shiftmax(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _ensure_independent_value_branch,
            _qk_shiftmax_gate_forward,
            config_from_dict,
        )

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, value_mode: str):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": "strict_bsa_shiftmax",
                        "center_scores": True,
                        "preserve_mean": False,
                        "consensus_score_norm": "head_dim",
                        "value_mode": value_mode,
                    }
                )

        for value_mode in ("sign", "threshold"):
            module = TinyAttention(value_mode)
            x = torch.randn(1, 2, 1, 2, 4)
            out, spikes = _qk_shiftmax_gate_forward(module, x)

            self.assertEqual(tuple(out.shape), (2, 2, 4))
            self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
            self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.5 - 1e-6)
            self.assertLessEqual(module.h9_shiftmax_row_sum_mean, 1.0 + 1e-6)
            self.assertFalse(torch.isnan(out).any())

        module = TinyAttention("sign")
        cfg = config_from_dict(
            {
                "enabled": True,
                "mode": "strict_bsa_qkv_shiftmax",
                "center_scores": True,
                "preserve_mean": False,
                "consensus_score_norm": "sqrt_head_dim",
                "value_mode": "sign",
            }
        )
        module._h9_shiftmax_cfg = cfg
        _ensure_independent_value_branch(module, cfg)
        self.assertTrue(hasattr(module, "linear_v"))
        self.assertTrue(hasattr(module, "sn_v"))
        self.assertIsNot(module.linear_v, module.linear_k)
        x = torch.randn(1, 2, 1, 2, 4)
        out, spikes = _qk_shiftmax_gate_forward(module, x)
        self.assertEqual(tuple(out.shape), (2, 2, 4))
        self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
        self.assertFalse(torch.isnan(out).any())

    def test_direct_tx_matrix_diag_bias_changes_attention_output(self):
        from models.STSwinNet_SNN.bsa_attention import _add_matrix_diag_bias, config_from_dict

        scores = torch.zeros(1, 2, 3, 3)
        cfg = config_from_dict({"matrix_diag_bias": 1.25})
        biased = _add_matrix_diag_bias(scores, cfg)

        self.assertTrue(torch.allclose(torch.diagonal(biased[0, 0]), torch.full((3,), 1.25)))
        self.assertEqual(float(biased[0, 0, 0, 1]), 0.0)
        self.assertEqual(float(biased[0, 0, 1, 0]), 0.0)

    def test_independent_value_branch_can_sync_from_loaded_k(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _ensure_independent_value_branch,
            config_from_dict,
            sync_independent_value_branch_from_k,
        )

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class Spiking_QK_WindowAttention3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_k = IdentitySN()

        model = DummyModel()
        attn = Spiking_QK_WindowAttention3D()
        model.sttmultires_unet.encoders.swin3d.layers[0].swin_blocks[0].attn = attn
        cfg = config_from_dict(
            {
                "enabled": True,
                "mode": "ternary_alpha_xnor_ssa_qkv_shiftmax",
                "target_blocks": ["0:0"],
                "value_init": "copy_k",
            }
        )
        with torch.no_grad():
            attn.linear_k.weight.fill_(0.25)
        _ensure_independent_value_branch(attn, cfg)
        with torch.no_grad():
            attn.linear_k.weight.fill_(2.0)

        synced = sync_independent_value_branch_from_k(
            model,
            {
                "enabled": True,
                "mode": "ternary_alpha_xnor_ssa_qkv_shiftmax",
                "target_blocks": ["0:0"],
                "value_init": "copy_k",
            },
        )

        self.assertEqual(synced, 1)
        self.assertTrue(torch.allclose(attn.linear_v.weight, attn.linear_k.weight))
        self.assertTrue(getattr(attn, "_h9_v_initialized_from_loaded_k"))

    def test_h18_paper_backed_modes_run_on_tiny_attention(self):
        from models.STSwinNet_SNN.bsa_attention import _qk_shiftmax_gate_forward, config_from_dict

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, mode: str):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": mode,
                        "center_scores": False,
                        "preserve_mean": False,
                        "consensus_bias": 0.02,
                        "alpha0": 0.02,
                        "mismatch_penalty": 0.5,
                    }
                )

        for mode in (
            "ternary_alpha_xnor_shiftmax",
            "ternary_alpha_xnor_l1",
            "a2os2a_gate",
            "alpha_xnor_matrix_shiftmax",
            "alpha_xnor_matrix_l1",
            "binary_alpha_xnor_matrix_shiftmax",
            "binary_alpha_xnor_matrix_l1",
            "a2os2a_direct",
            "a2os2a_qkv_l1",
            "hamming_binary_direct",
            "hamming_ternary_active_direct",
        ):
            module = TinyAttention(mode)
            x = torch.randn(1, 2, 1, 2, 4)
            out, spikes = _qk_shiftmax_gate_forward(module, x)

            self.assertEqual(tuple(out.shape), (2, 2, 4))
            self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
            self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.0)
            self.assertFalse(torch.isnan(out).any())


    def test_faps_sparse_k_mag_respects_confidence_min_active(self):
        from models.STSwinNet_SNN.bsa_attention import (
            _faps_flow_aligned_token_scores,
            config_from_dict,
        )

        cfg = config_from_dict(
            {
                "enabled": True,
                "mode": "faps",
                "center_scores": False,
                "preserve_mean": False,
                "consensus_score_norm": "none",
                "directional_channels_enabled": False,
                "k_magnitude_alpha": 0.2,
                "confidence_min_active": 4,
                "kmag_quantize_bits": 2,
            }
        )
        q_orig = torch.tensor(
            [[[[[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]],
            dtype=torch.float32,
        )
        k_orig = torch.tensor(
            [[[[[3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]],
            dtype=torch.float32,
        )
        low_active = _faps_flow_aligned_token_scores(q_orig, k_orig, cfg)
        cfg_open = config_from_dict(
            {
                "enabled": True,
                "mode": "faps",
                "center_scores": False,
                "preserve_mean": False,
                "consensus_score_norm": "none",
                "directional_channels_enabled": False,
                "k_magnitude_alpha": 0.2,
                "confidence_min_active": 0,
                "kmag_quantize_bits": 2,
            }
        )
        all_active = _faps_flow_aligned_token_scores(q_orig, k_orig, cfg_open)
        self.assertNotEqual(float(low_active[0, 0, 0, 0]), float(all_active[0, 0, 0, 0]))

    def test_faps_mode_runs_on_tiny_attention(self):
        from models.STSwinNet_SNN.bsa_attention import _qk_shiftmax_gate_forward, config_from_dict

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, *, directional: bool, kmag: float):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": "faps",
                        "center_scores": True,
                        "preserve_mean": True,
                        "consensus_score_norm": "head_dim",
                        "alpha0": 0.02,
                        "mismatch_penalty": 0.25,
                        "single_active_penalty": 0.05,
                        "single_active_penalty_grad": "ste",
                        "directional_channels_enabled": directional,
                        "directional_merge_mode": "mean",
                        "k_magnitude_alpha": kmag,
                        "confidence_min_active": 2 if kmag > 0 else 0,
                        "kmag_quantize_bits": 2,
                    }
                )

        for directional, kmag in ((True, 0.0), (True, 0.03125), (False, 0.0)):
            module = TinyAttention(directional=directional, kmag=kmag)
            x = torch.randn(1, 2, 1, 2, 4)
            out, spikes = _qk_shiftmax_gate_forward(module, x)
            self.assertEqual(tuple(out.shape), (2, 2, 4))
            self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
            self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.0)
            self.assertFalse(torch.isnan(out).any())

    def test_h62_confidence_is_high_for_active_agreement(self):
        from models.STSwinNet_SNN.bsa_attention import _event_agree_confidence, config_from_dict

        cfg = config_from_dict({"enabled": True, "mode": "h62"})
        q_event = torch.tensor([[[[1.0, 1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]]]])
        k_event = torch.tensor([[[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]]])
        conf = _event_agree_confidence(q_event, k_event, cfg)
        self.assertGreater(float(conf[0, 0, 0, 0]), 0.70)
        self.assertEqual(float(conf[0, 0, 1, 0]), 0.0)

    def test_h62_mode_runs_on_tiny_attention(self):
        from models.STSwinNet_SNN.bsa_attention import _qk_shiftmax_gate_forward, config_from_dict

        class IdentitySN(nn.Module):
            def forward(self, x):
                return x

        class TinyAttention(nn.Module):
            def __init__(self, *, gamma: float, schedule: bool):
                super().__init__()
                self.num_heads = 2
                self.norm_layer = None
                self.proj_sn = IdentitySN()
                self.linear_q = nn.Linear(4, 4, bias=False)
                self.linear_k = nn.Linear(4, 4, bias=False)
                self.sn_q = IdentitySN()
                self.sn_k = IdentitySN()
                self.sn2_q = IdentitySN()
                self.attn_drop = nn.Identity()
                self.attn_sn = IdentitySN()
                self.proj = nn.Linear(4, 4, bias=False)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 2, 2, 2))
                self._h9_shiftmax_cfg = config_from_dict(
                    {
                        "enabled": True,
                        "mode": "h62",
                        "center_scores": True,
                        "preserve_mean": True,
                        "consensus_score_norm": "head_dim",
                        "alpha0": 0.02,
                        "mismatch_penalty": 0.25,
                        "single_active_penalty": 0.05,
                        "single_active_penalty_grad": "ste",
                        "bipolar_mu": 0.05,
                        "k_magnitude_alpha": 0.02,
                        "directional_residual_gamma": gamma,
                        "sc_mu_schedule_enabled": schedule,
                        "sc_mu_start": 0.0,
                        "sc_mu_warmup_steps": 10,
                    }
                )
                self._h9_global_step = 5

        for gamma, schedule in ((0.0, False), (0.02, False), (0.02, True)):
            module = TinyAttention(gamma=gamma, schedule=schedule)
            x = torch.randn(1, 2, 1, 2, 4)
            out, spikes = _qk_shiftmax_gate_forward(module, x)
            self.assertEqual(tuple(out.shape), (2, 2, 4))
            self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
            self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.0)
            self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
