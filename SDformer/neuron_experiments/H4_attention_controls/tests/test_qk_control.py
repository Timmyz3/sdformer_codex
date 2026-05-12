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
    def forward(self, x):
        return x + 1


class DummyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.spiking_neuron = DummyPSN()

    def forward(self, x):
        return self.spiking_neuron(x)


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
    def __init__(self):
        super().__init__()
        self.swin_blocks = nn.ModuleList([DummyBlock()])


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sttmultires_unet = nn.Module()
        self.sttmultires_unet.encoders = nn.Module()
        self.sttmultires_unet.encoders.swin3d = nn.Module()
        self.sttmultires_unet.encoders.swin3d.layers = nn.ModuleList([DummyStage(), DummyStage()])


class QKControlTest(unittest.TestCase):
    def test_install_qk_zero_control(self):
        from models.STSwinNet_SNN.qk_control import ZeroLike, install_qk_control

        model = DummyModel()
        installed = install_qk_control(model, {"enabled": True, "stage_selection": "all", "target": "qk"})

        self.assertEqual(len(installed), 4)
        first_attn = model.sttmultires_unet.encoders.swin3d.layers[0].swin_blocks[0].attn
        self.assertIsInstance(first_attn.sn_q.spiking_neuron, ZeroLike)
        self.assertTrue(torch.equal(first_attn.sn_q(torch.ones(2, 3)), torch.zeros(2, 3)))


if __name__ == "__main__":
    unittest.main()
