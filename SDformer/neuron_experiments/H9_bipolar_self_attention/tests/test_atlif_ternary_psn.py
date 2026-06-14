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

    def test_installer_all_non_qk_binary_group(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, install_atlif_ternary_psn

        model = DummyModel()
        extra_sn = type("Spiking_neuron", (DummyWrapper,), {})()
        model.sttmultires_unet.extra_sn = extra_sn

        installed = install_atlif_ternary_psn(
            model,
            {
                "enabled": True,
                "stage_selection": "all",
                "target": "qk",
                "output_mode": "ternary",
                "threshold_mode": "symmetric_bsa_tsn",
                "target_groups": [
                    {
                        "name": "all_non_qk_binary",
                        "path_selection": "all_non_qk",
                        "output_mode": "binary",
                        "threshold_mode": "official_atlif",
                        "center_mode": "zero",
                    }
                ],
            },
        )

        self.assertTrue(any("extra_sn" in item for item in installed))
        qk = model.sttmultires_unet.encoders.swin3d.layers[0].swin_blocks[0].attn.sn_q.spiking_neuron
        non_qk = model.sttmultires_unet.extra_sn.spiking_neuron
        self.assertIsInstance(qk, ATLIFTernaryPSN)
        self.assertEqual(qk.output_mode, "ternary")
        self.assertIsInstance(non_qk, ATLIFTernaryPSN)
        self.assertEqual(non_qk.output_mode, "binary")
        self.assertEqual(non_qk.threshold_mode, "official_atlif")

    def test_installer_all_non_qk_respects_exclude_prefixes(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, install_atlif_ternary_psn

        SpikingNeuron = type("Spiking_neuron", (DummyWrapper,), {})

        class DummyDecoderHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.sn = SpikingNeuron()

        model = DummyModel()
        model.sttmultires_unet.decoders = nn.ModuleList([DummyDecoderHead()])
        model.sttmultires_unet.extra_sn = SpikingNeuron()

        installed = install_atlif_ternary_psn(
            model,
            {
                "enabled": True,
                "stage_selection": "all",
                "target": "qk",
                "output_mode": "ternary",
                "threshold_mode": "symmetric_bsa_tsn",
                "target_groups": [
                    {
                        "name": "encoder_only_binary",
                        "path_selection": "all_non_qk",
                        "exclude_path_prefixes": ["sttmultires_unet.decoders."],
                        "output_mode": "binary",
                        "threshold_mode": "official_atlif",
                        "center_mode": "zero",
                    }
                ],
            },
        )

        self.assertTrue(any("extra_sn" in item for item in installed))
        self.assertFalse(any("decoders.0.sn" in item for item in installed))
        self.assertIsInstance(model.sttmultires_unet.decoders[0].sn.spiking_neuron, DummyPSN)  # still vanilla
        self.assertIsInstance(model.sttmultires_unet.extra_sn.spiking_neuron, ATLIFTernaryPSN)

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

    def test_target_rate_upper_bound_does_not_lower_threshold_by_default(self):
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

        self.assertEqual(float(node.thresh.detach()), before)
        self.assertEqual(float(stats["target_feedback_mean"]), 0.0)

    def test_target_rate_bidirectional_feedback_can_lower_threshold(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=0.0,
            target_rate=0.5,
            target_rate_eta=1e-3,
            target_rate_mode="bidirectional",
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

    def test_summary_counts_only_active_target_rate_control(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, atlif_ternary_summary

        active = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            target_rate=0.5,
            target_rate_eta=1e-3,
        )
        eta_zero = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            target_rate=0.5,
            target_rate_eta=0.0,
        )
        official = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            target_rate=0.5,
            target_rate_eta=1e-3,
            output_mode="binary",
            threshold_mode="official_atlif",
        )
        model = nn.ModuleList([active, eta_zero, official])

        stats = atlif_ternary_summary(model)

        self.assertEqual(stats["target_rate_control_modules"], 1)
        self.assertEqual(stats["target_rate_upper_bound_modules"], 1)
        self.assertEqual(stats["target_rate_bidirectional_modules"], 0)

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

    def test_asymmetric_scale_keeps_legacy_negative_threshold(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=0.0,
            negative_threshold_scale=30.0,
            threshold_mode="asymmetric_scale",
        )
        x = torch.tensor([[0.6, -1.0], [0.0, 0.0], [0.0, 0.0]])

        out = node(x)

        unique = set(torch.unique(out).tolist())
        self.assertIn(0.5, unique)
        self.assertNotIn(-0.5, unique)
        self.assertGreater(node.pos_r, 0.0)
        self.assertEqual(node.neg_r, 0.0)

    def test_symmetric_bsa_tsn_ignores_negative_scale_for_trigger(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=0.0,
            negative_threshold_scale=30.0,
            threshold_mode="symmetric_bsa_tsn",
        )
        x = torch.tensor([[0.6, -0.6, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        out = node(x)

        self.assertTrue(set(torch.unique(out).tolist()).issubset({-0.5, 0.0, 0.5}))
        self.assertGreater(node.pos_r, 0.0)
        self.assertGreater(node.neg_r, 0.0)

    def test_bias_center_mode_treats_psn_bias_as_silent_center(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

        class BiasOnlyPSN(nn.Module):
            T = 2

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2, 2))
                self.bias = nn.Parameter(torch.full((2, 1), -1.0))

        x = torch.zeros(2, 1, 1)
        zero_center = ATLIFTernaryPSN(
            T=2,
            base_psn=BiasOnlyPSN(),
            thresh=0.1,
            output_mode="ternary",
            threshold_mode="symmetric_target_rate",
            center_mode="zero",
        )
        bias_center = ATLIFTernaryPSN(
            T=2,
            base_psn=BiasOnlyPSN(),
            thresh=0.1,
            output_mode="ternary",
            threshold_mode="symmetric_target_rate",
            center_mode="bias",
        )

        self.assertLess(zero_center(x).sum().item(), 0.0)
        self.assertEqual(bias_center(x).abs().sum().item(), 0.0)

    def test_symmetric_target_rate_uses_total_rate_not_negative_scale_feedback(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=0.0,
            negative_threshold_scale=30.0,
            negative_target_rate=0.1,
            negative_target_eta=10.0,
            target_rate=0.8,
            target_rate_eta=1e-3,
            target_rate_mode="bidirectional",
            threshold_mode="symmetric_target_rate",
        )
        _ = node(torch.zeros(3, 4))
        before_scale = float(node.negative_threshold_scale)
        before_thresh = float(node.thresh.detach())

        stats = threshold_update(
            node,
            lr=1.0,
            raw_config={"enabled": True, "threshold_lr_scale": 100.0, "min_threshold": 0.1},
        )

        self.assertEqual(float(node.negative_threshold_scale), before_scale)
        self.assertLess(float(node.thresh.detach()), before_thresh)
        self.assertLess(float(stats["target_feedback_mean"]), 0.0)
        self.assertEqual(float(stats["negative_scale_feedback_mean"]), 0.0)

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
                        "threshold_mode": "symmetric_bsa_tsn",
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
        self.assertEqual(model.custom_high_sops.spiking_neuron.threshold_mode, "symmetric_bsa_tsn")
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

    def test_official_atlif_binary_matches_source_update_scale(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=1e-3,
            output_mode="binary",
            threshold_mode="official_atlif",
        )
        _ = node(torch.full((3, 4), 0.5))

        self.assertAlmostEqual(float(node.update_value), 1e-3, places=7)
        before = float(node.thresh.detach())
        stats = threshold_update(
            node,
            lr=10.0,
            raw_config={"enabled": True, "threshold_lr_scale": 1.0, "min_threshold": None, "max_threshold": None},
        )

        self.assertAlmostEqual(float(node.thresh.detach()), before + 1e-2, places=6)
        self.assertEqual(float(node.update_value), 0.0)
        self.assertEqual(stats["official_atlif_modules"], 1)

    def test_threshold_update_reports_actual_mean_delta_with_module_scales(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node_a = ATLIFTernaryPSN(T=3, base_psn=DummyPSN(T=3), thresh=1.0, sparsity_eta=0.0)
        node_b = ATLIFTernaryPSN(T=3, base_psn=DummyPSN(T=3), thresh=1.0, sparsity_eta=0.0)
        node_a.update_value = torch.tensor(1.0)
        node_b.update_value = torch.tensor(1.0)
        node_a.threshold_lr_scale = 2.0
        node_b.threshold_lr_scale = 4.0
        model = nn.ModuleList([node_a, node_b])

        stats = threshold_update(
            model,
            lr=0.5,
            raw_config={"enabled": True, "threshold_lr_scale": 100.0, "min_threshold": None, "max_threshold": None},
        )

        self.assertAlmostEqual(float(node_a.thresh.detach()), 2.0, places=6)
        self.assertAlmostEqual(float(node_b.thresh.detach()), 3.0, places=6)
        self.assertAlmostEqual(float(stats["effective_update_mean"]), 1.5, places=6)

    def test_threshold_update_freezes_after_configured_global_step(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(T=3, base_psn=DummyPSN(T=3), thresh=1.0, sparsity_eta=0.0)
        node.update_value = torch.tensor(1.0)

        stats = threshold_update(
            node,
            lr=0.5,
            raw_config={
                "enabled": True,
                "threshold_lr_scale": 4.0,
                "threshold_freeze_after_step": 100,
                "_global_step": 100,
                "min_threshold": None,
                "max_threshold": None,
            },
        )

        self.assertAlmostEqual(float(node.thresh.detach()), 1.0, places=6)
        self.assertAlmostEqual(float(stats["effective_update_mean"]), 0.0, places=6)
        self.assertEqual(int(stats["threshold_updates_frozen"]), 1)

    def test_official_atlif_rejects_ternary_output(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

        with self.assertRaises(ValueError):
            ATLIFTernaryPSN(
                T=3,
                base_psn=DummyPSN(T=3),
                output_mode="ternary",
                threshold_mode="official_atlif",
            )

    def test_quantile_guard_slows_update_after_threshold_reaches_distribution_budget(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=1.0,
            sparsity_eta=1e-3,
            output_mode="ternary",
            threshold_mode="symmetric_bsa_tsn",
            quantile_q=0.5,
            quantile_momentum=0.0,
            quantile_guard_margin=0.25,
            quantile_min_guard=0.0,
        )
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        _ = node(x)
        before = float(node.thresh.detach())

        stats = threshold_update(
            node,
            lr=10.0,
            raw_config={"enabled": True, "threshold_lr_scale": 1000.0, "min_threshold": None},
        )

        self.assertEqual(float(node.thresh.detach()), before)
        self.assertLess(float(stats["quantile_guard_mean"]), 0.01)
        self.assertGreater(float(stats["raw_update_mean"]), 0.0)

    def test_importance_guard_uses_backward_saliency_without_blocking_gradients(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, threshold_update

        node = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(T=3),
            thresh=0.5,
            sparsity_eta=1e-3,
            output_mode="binary",
            threshold_mode="official_atlif",
            importance_enabled=True,
            importance_momentum=0.0,
            importance_scale=10.0,
            importance_min_guard=0.1,
        )
        x = torch.full((3, 4), 0.5, requires_grad=True)
        out = node(x)
        loss = (out * 3.0).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)
        self.assertGreater(float(node.importance_ema), 0.0)
        before = float(node.thresh.detach())
        stats = threshold_update(
            node,
            lr=10.0,
            raw_config={"enabled": True, "threshold_lr_scale": 1.0, "min_threshold": None, "max_threshold": None},
        )

        self.assertGreater(float(node.thresh.detach()), before)
        self.assertLess(float(node.thresh.detach()), before + 1e-2)
        self.assertLess(float(stats["importance_guard_mean"]), 1.0)

    def test_importance_saliency_is_stable_under_global_loss_scaling(self):
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

        def run(scale: float) -> float:
            node = ATLIFTernaryPSN(
                T=3,
                base_psn=DummyPSN(T=3),
                thresh=0.5,
                sparsity_eta=1e-3,
                output_mode="binary",
                threshold_mode="official_atlif",
                importance_enabled=True,
                importance_momentum=0.0,
                importance_scale=10.0,
            )
            x = torch.full((3, 4), 0.5, requires_grad=True)
            (node(x).sum() * scale).backward()
            return float(node.importance_ema)

        self.assertAlmostEqual(run(1.0), run(100.0), places=5)


if __name__ == "__main__":
    unittest.main()
