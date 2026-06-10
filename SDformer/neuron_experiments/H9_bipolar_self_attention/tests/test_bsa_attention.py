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
    def test_shiftmax_row_sum_is_power_two_bounded(self):
        from models.STSwinNet_SNN.bsa_attention import shiftmax

        scores = torch.randn(4, 3, 17, 1)
        probs = shiftmax(scores, dim=2)
        row_sum = probs.sum(dim=2)

        self.assertTrue(torch.all(row_sum > 0.5 - 1e-6))
        self.assertTrue(torch.all(row_sum <= 1.0 + 1e-6))
        self.assertFalse(torch.isnan(probs).any())

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
