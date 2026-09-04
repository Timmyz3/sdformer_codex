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
    def test_ternary_event_score_weakens_negative_agreement(self):
        from models.STSwinNet_SNN.bsa_attention import ternary_event_score

        q = torch.tensor([[[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
        k = torch.tensor([[[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]])

        score = ternary_event_score(q, k, alpha=0.25, beta=1.0)

        self.assertAlmostEqual(float(score[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(score[0, 0, 1, 0]), 0.25)
        self.assertAlmostEqual(float(score[0, 0, 2, 0]), -1.0)

    def test_shiftmax_row_sum_is_power_two_bounded(self):
        from models.STSwinNet_SNN.bsa_attention import shiftmax

        scores = torch.randn(4, 3, 17, 1)
        probs = shiftmax(scores, dim=2)
        row_sum = probs.sum(dim=2)

        self.assertTrue(torch.all(row_sum > 0.5 - 1e-6))
        self.assertTrue(torch.all(row_sum <= 1.0 + 1e-6))
        self.assertFalse(torch.isnan(probs).any())

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

    def test_ternary_event_compat_mode_runs_on_tiny_attention(self):
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
                        "mode": "ternary_event_compat",
                        "event_alpha": 0.25,
                        "event_beta": 1.0,
                        "center_scores": True,
                        "preserve_mean": True,
                    }
                )

        module = TinyAttention()
        x = torch.randn(1, 2, 1, 2, 4)
        out, spikes = _qk_shiftmax_gate_forward(module, x)

        self.assertEqual(tuple(out.shape), (2, 2, 4))
        self.assertEqual(tuple(spikes.shape), (1, 2, 1, 2, 4))
        self.assertGreater(module.h9_shiftmax_row_sum_mean, 0.0)
        self.assertGreater(module.h9_shiftmax_gate_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
