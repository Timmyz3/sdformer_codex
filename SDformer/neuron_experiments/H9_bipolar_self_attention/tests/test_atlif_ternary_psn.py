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


class DummyPSN(nn.Module):
    def __init__(self, T: int = 3):
        super().__init__()
        self.T = T
        self.weight = nn.Parameter(torch.eye(T))
        self.bias = nn.Parameter(torch.zeros(T, 1))


class DummyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.spiking_neuron = DummyPSN(T=3)


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sn_q = DummyWrapper()
        self.sn_k = DummyWrapper()


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
        self.custom_high_sops = DummyWrapper()


class ATLIFTernaryPSNTest(unittest.TestCase):
    def test_forward_outputs_signed_threshold_values(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=1e-3,
            negative_threshold_scale=2.0,
        )
        x = torch.tensor([[1.0, -1.1, 0.1, -0.1], [0.6, -1.2, 0.0, 0.2], [0.0, 0.7, -1.3, 0.1]])

        out = node(x)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(set(torch.unique(out).tolist()).issubset({-0.5, 0.0, 0.5}))
        self.assertGreater(float(node.update_value), 0.0)
        self.assertGreater(float(node.r), 0.0)
        self.assertGreater(float(node.pos_r), 0.0)
        self.assertGreater(float(node.neg_r), 0.0)

    def test_threshold_update_respects_max_threshold(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(T=3, base_psn=DummyPSN(T=3), thresh=0.5, sparsity_eta=1e-3)
        _ = node(torch.full((3, 4), 0.5))

        stats = threshold_update(
            node,
            lr=10.0,
            raw_config={"enabled": True, "threshold_lr_scale": 1000.0, "max_threshold": 0.51},
        )

        self.assertLessEqual(float(node.thresh.detach()), 0.51)
        self.assertEqual(float(node.update_value), 0.0)
        self.assertGreater(float(stats["raw_update_mean"]), 0.0)

    def test_installer_and_trainable_mode(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, apply_trainable_mode, install_atlif_ternary_psn

        model = DummyModel()
        installed = install_atlif_ternary_psn(
            model,
            {
                "enabled": True,
                "stage_selection": "layer0_only",
                "target": "qk",
                "threshold_eta": 1e-3,
            },
        )

        self.assertEqual(len(installed), 4)
        first = model.sttmultires_unet.encoders.swin3d.layers[0].swin_blocks[0].attn
        self.assertIsInstance(first.sn_q.spiking_neuron, ATLIFTernaryPSN)
        stats = apply_trainable_mode(model, {"trainable": "threshold_only"})
        self.assertEqual(stats["trainable_parameters"], 4)

    def test_installer_applies_stage_overrides(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn

        model = DummyModel()
        installed = install_atlif_ternary_psn(
            model,
            {
                "enabled": True,
                "stage_selection": "all",
                "target": "qk",
                "activity_eta": 0.5,
                "negative_threshold_scale": 30.0,
                "max_threshold": 0.13,
                "stage_activity_eta": {"0": 0.2, "1": 1.0},
                "stage_negative_threshold_scale": {"0": 20.0, "1": 40.0},
                "stage_max_threshold": {"0": 0.11, "1": 0.12},
                "stage_target_rate": {"0": 0.04, "1": 0.02},
                "target_rate_eta": 5e-4,
            },
        )

        self.assertEqual(len(installed), 6)
        stage0 = model.sttmultires_unet.encoders.swin3d.layers[0].swin_blocks[0].attn.sn_q.spiking_neuron
        stage1 = model.sttmultires_unet.encoders.swin3d.layers[1].swin_blocks[0].attn.sn_q.spiking_neuron
        self.assertEqual(stage0.activity_eta, 0.2)
        self.assertEqual(stage1.activity_eta, 1.0)
        self.assertEqual(stage0.negative_threshold_scale, 20.0)
        self.assertEqual(stage1.negative_threshold_scale, 40.0)
        self.assertEqual(stage0.max_threshold, 0.11)
        self.assertEqual(stage1.max_threshold, 0.12)
        self.assertEqual(stage0.target_rate, 0.04)
        self.assertEqual(stage1.target_rate, 0.02)

    def test_target_rate_feedback_can_lower_threshold(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=0.0,
            target_rate=0.5,
            target_rate_eta=1e-3,
        )
        _ = node(torch.zeros(3, 4))
        before = float(node.thresh.detach())

        stats = threshold_update(
            node,
            lr=1.0,
            raw_config={"enabled": True, "threshold_lr_scale": 100.0, "min_threshold": 0.1},
        )

        self.assertLess(float(node.thresh.detach()), before)
        self.assertLess(float(stats["target_feedback_mean"]), 0.0)

    def test_negative_rate_feedback_can_keep_negative_trigger_active(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=0.0,
            negative_threshold_scale=8.0,
            negative_target_rate=0.01,
            negative_target_eta=10.0,
            negative_scale_min=2.0,
            negative_scale_max=12.0,
        )
        node.neg_r = 0.0
        before = float(node.negative_threshold_scale)

        stats = threshold_update(node, lr=1.0, raw_config={"enabled": True})

        self.assertLess(float(node.negative_threshold_scale), before)
        self.assertGreaterEqual(float(node.negative_threshold_scale), 2.0)
        self.assertLess(float(stats["negative_scale_feedback_mean"]), 0.0)

    def test_negative_rate_feedback_can_prevent_dense_negative_spikes(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=0.0,
            negative_threshold_scale=3.0,
            negative_target_rate=0.01,
            negative_target_eta=10.0,
            negative_scale_min=2.0,
            negative_scale_max=12.0,
        )
        node.neg_r = 0.05
        before = float(node.negative_threshold_scale)

        stats = threshold_update(node, lr=1.0, raw_config={"enabled": True})

        self.assertGreater(float(node.negative_threshold_scale), before)
        self.assertLessEqual(float(node.negative_threshold_scale), 12.0)
        self.assertGreater(float(stats["negative_scale_feedback_mean"]), 0.0)

    def test_target_groups_apply_path_specific_strength(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, install_atlif_ternary_psn

        model = DummyModel()
        installed = install_atlif_ternary_psn(
            model,
            {
                "enabled": True,
                "stage_selection": "layer0_only",
                "target": "q",
                "activity_eta": 2.0,
                "max_threshold": 0.13,
                "target_groups": [
                    {
                        "paths": ["custom_high_sops"],
                        "activity_eta": 0.1,
                        "max_threshold": 0.11,
                        "negative_threshold_scale": 10.0,
                        "output_mode": "binary",
                    }
                ],
            },
        )

        self.assertEqual(len(installed), 3)
        self.assertIsInstance(model.custom_high_sops.spiking_neuron, ATLIFTernaryPSN)
        self.assertEqual(model.custom_high_sops.spiking_neuron.activity_eta, 0.1)
        self.assertEqual(model.custom_high_sops.spiking_neuron.max_threshold, 0.11)
        self.assertEqual(model.custom_high_sops.spiking_neuron.negative_threshold_scale, 10.0)
        self.assertEqual(model.custom_high_sops.spiking_neuron.output_mode, "binary")

    def test_binary_mode_has_no_negative_spikes(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=1e-3,
            output_mode="binary",
        )
        x = torch.tensor([[1.0, -1.1, 0.1, -0.1], [0.6, -1.2, 0.0, 0.2], [0.0, 0.7, -1.3, 0.1]])

        out = node(x)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(set(torch.unique(out).tolist()).issubset({0.0, 0.5}))
        self.assertGreater(float(node.pos_r), 0.0)
        self.assertEqual(float(node.neg_r), 0.0)


if __name__ == "__main__":
    unittest.main()
