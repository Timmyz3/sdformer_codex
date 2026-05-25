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


if __name__ == "__main__":
    unittest.main()
