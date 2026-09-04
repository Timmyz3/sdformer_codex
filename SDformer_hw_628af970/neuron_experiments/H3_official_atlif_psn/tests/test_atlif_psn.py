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
        self.bias = nn.Parameter(torch.full((T, 1), -0.1))


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
        self.custom = DummyWrapper()


class ATLIFPSNTest(unittest.TestCase):
    def test_forward_accumulates_official_threshold_update(self):
        from models.STSwinNet_SNN.atlif_psn.atlif_psn import ATLIFPSN

        node = ATLIFPSN(T=3, base_psn=DummyPSN(T=3), thresh=0.5, sparsity_eta=1e-3)
        x = torch.ones(3, 4)

        out = node(x)

        self.assertEqual(out.shape, x.shape)
        self.assertGreater(float(node.update_value), 0.0)
        self.assertGreaterEqual(float(node.r), 0.0)
        self.assertTrue(torch.is_tensor(node.act_value))

    def test_threshold_update_changes_threshold_and_resets_update_value(self):
        from models.STSwinNet_SNN.atlif_psn.atlif_psn import ATLIFPSN
        from models.STSwinNet_SNN.atlif_psn.training import threshold_update

        node = ATLIFPSN(T=3, base_psn=DummyPSN(T=3), thresh=0.5, sparsity_eta=1e-3)
        _ = node(torch.ones(3, 4))
        before = float(node.thresh.detach())

        stats = threshold_update(node, lr=0.1, raw_config={"enabled": True, "threshold_lr_scale": 1.0})

        self.assertGreater(float(node.thresh.detach()), before)
        self.assertEqual(float(node.update_value), 0.0)
        self.assertGreater(float(stats["raw_update_mean"]), 0.0)
        self.assertGreater(float(stats["effective_update_mean"]), 0.0)

    def test_installer_replaces_selected_attention_qk(self):
        from models.STSwinNet_SNN.atlif_psn.atlif_psn import ATLIFPSN
        from models.STSwinNet_SNN.atlif_psn.installer import apply_trainable_mode, install_atlif_psn_qk

        model = DummyModel()
        installed = install_atlif_psn_qk(
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
        self.assertIsInstance(first.sn_q.spiking_neuron, ATLIFPSN)
        untouched = model.sttmultires_unet.encoders.swin3d.layers[1].swin_blocks[0].attn
        self.assertNotIsInstance(untouched.sn_q.spiking_neuron, ATLIFPSN)

        stats = apply_trainable_mode(model, {"trainable": "threshold_only"})
        self.assertEqual(stats["trainable_parameters"], 4)
        self.assertTrue(first.sn_q.spiking_neuron.thresh.requires_grad)
        self.assertFalse(first.sn_q.spiking_neuron.weight.requires_grad)

    def test_installer_replaces_explicit_target_paths(self):
        from models.STSwinNet_SNN.atlif_psn.atlif_psn import ATLIFPSN
        from models.STSwinNet_SNN.atlif_psn.installer import apply_trainable_mode, install_atlif_psn_qk

        model = DummyModel()
        installed = install_atlif_psn_qk(
            model,
            {
                "enabled": True,
                "target": "none",
                "target_paths": ["custom"],
                "threshold_eta": 1e-3,
            },
        )

        self.assertEqual(installed, ["custom"])
        self.assertIsInstance(model.custom.spiking_neuron, ATLIFPSN)
        stats = apply_trainable_mode(model, {"trainable": "threshold_only"})
        self.assertEqual(stats["trainable_parameters"], 1)


if __name__ == "__main__":
    unittest.main()
