from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


EXP_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = EXP_ROOT / "overlay"
BASELINE_ROOT = EXP_ROOT.parents[1] / "third_party" / "SDformerFlow"
for path in (str(BASELINE_ROOT), str(OVERLAY_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


class DummyWrapper(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.spiking_neuron = base


class DummyAttn(nn.Module):
    def __init__(self, q: nn.Module, k: nn.Module) -> None:
        super().__init__()
        self.sn_q = DummyWrapper(q)
        self.sn_k = DummyWrapper(k)


class DummyBlock(nn.Module):
    def __init__(self, q: nn.Module, k: nn.Module) -> None:
        super().__init__()
        self.attn = DummyAttn(q, k)


class DummyStage(nn.Module):
    def __init__(self, blocks: list[DummyBlock]) -> None:
        super().__init__()
        self.swin_blocks = nn.ModuleList(blocks)


class DummySwin3D(nn.Module):
    def __init__(self, stages: list[DummyStage]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(stages)


class DummyEncoders(nn.Module):
    def __init__(self, swin3d: DummySwin3D) -> None:
        super().__init__()
        self.swin3d = swin3d


class DummyUNet(nn.Module):
    def __init__(self, swin3d: DummySwin3D) -> None:
        super().__init__()
        self.encoders = DummyEncoders(swin3d)


class DummyModel(nn.Module):
    def __init__(self, swin3d: DummySwin3D) -> None:
        super().__init__()
        self.sttmultires_unet = DummyUNet(swin3d)


class AdaptiveTernaryPSNTest(unittest.TestCase):
    def test_ternary_activation_contract(self) -> None:
        from models.STSwinNet_SNN.adaptive_ternary import ternary_spike_activation

        x = torch.tensor([-1.2, -0.6, -0.4, 0.0, 0.4, 0.6, 1.2], requires_grad=True)
        y = ternary_spike_activation(x)

        torch.testing.assert_close(
            y.detach(),
            torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        )
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))

    def test_forward_keeps_psn_temporal_weights_and_outputs_scaled_ternary(self) -> None:
        from models.STSwinNet_SNN.Spiking_submodules import PSN
        from models.STSwinNet_SNN.adaptive_ternary import AdaptiveTernaryPSN

        base = PSN(T=2)
        with torch.no_grad():
            base.weight.fill_(1.0)
            base.bias.zero_()
        node = AdaptiveTernaryPSN(T=2, base_psn=base, theta_init=2.0)

        torch.testing.assert_close(node.weight, base.weight)
        torch.testing.assert_close(node.bias, base.bias)
        y = node(torch.ones(2, 1, 3))

        self.assertEqual(y.shape, (2, 1, 3))
        values = {round(float(item), 4) for item in y.detach().flatten()}
        self.assertTrue(values.issubset({-2.0, 0.0, 2.0}))

    def test_installer_replaces_only_attention_qk(self) -> None:
        from models.STSwinNet_SNN.Spiking_submodules import PSN
        from models.STSwinNet_SNN.adaptive_ternary import AdaptiveTernaryPSN, install_adaptive_ternary_qk

        model = DummyModel(
            DummySwin3D(
                [
                    DummyStage([DummyBlock(PSN(T=2), PSN(T=2)), DummyBlock(PSN(T=2), PSN(T=2))]),
                    DummyStage([DummyBlock(PSN(T=2), PSN(T=2))]),
                ]
            )
        )
        installed = install_adaptive_ternary_qk(
            model,
            {
                "enabled": True,
                "target": "qk",
                "stage_selection": "layer0_only",
                "theta_init": 1.0,
            },
        )

        self.assertEqual(len(installed), 4)
        self.assertIsInstance(
            model.sttmultires_unet.encoders.swin3d.layers[0].swin_blocks[0].attn.sn_q.spiking_neuron,
            AdaptiveTernaryPSN,
        )
        self.assertIsInstance(
            model.sttmultires_unet.encoders.swin3d.layers[0].swin_blocks[0].attn.sn_k.spiking_neuron,
            AdaptiveTernaryPSN,
        )
        self.assertNotIsInstance(
            model.sttmultires_unet.encoders.swin3d.layers[1].swin_blocks[0].attn.sn_q.spiking_neuron,
            AdaptiveTernaryPSN,
        )


if __name__ == "__main__":
    unittest.main()
