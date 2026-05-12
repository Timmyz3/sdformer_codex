from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = EXPERIMENT_ROOT / "overlay"
if str(OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERLAY_ROOT))


class NASNTest(unittest.TestCase):
    def test_nasn_forward_uses_normalized_integer_window_and_residual_state(self) -> None:
        from models.STSwinNet_SNN.experimental_neurons.single.asn import NASNNode

        node = NASNNode(T=3, D=4, beta=0.5, alpha_init=0.0)
        x = torch.tensor([[[0.2]], [[1.7]], [[-0.4]]])

        out = node(x)

        expected = torch.tensor([[[0.0]], [[0.5]], [[0.0]]])
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(node.alpha.detach(), torch.tensor(0.0))

    def test_nasn_alpha_receives_gradient_from_clipped_regions(self) -> None:
        from models.STSwinNet_SNN.experimental_neurons.single.asn import NASNNode

        node = NASNNode(T=2, D=4, beta=0.5, alpha_init=0.0, alpha_grad_scale=0.25)
        x = torch.tensor([[[5.5]], [[-1.0]]], requires_grad=True)

        loss = node(x).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(node.alpha.grad)
        self.assertTrue(torch.isfinite(node.alpha.grad))
        self.assertGreater(float(node.alpha.grad.abs()), 0.0)

    def test_factory_builds_nasn_from_experiment_config(self) -> None:
        from models.STSwinNet_SNN.experimental_neurons.factory import build_experimental_neuron
        from models.STSwinNet_SNN.experimental_neurons.single.asn import NASNNode

        neuron = build_experimental_neuron(
            neuron_type="exp_nasn",
            num_steps=2,
            D=4,
            beta=0.25,
            alpha_init=-1.0,
        )

        self.assertIsInstance(neuron, NASNNode)
        self.assertEqual(neuron.T, 2)
        self.assertEqual(neuron.D, 4)
        self.assertEqual(neuron.N, 4)
        torch.testing.assert_close(neuron.alpha.detach(), torch.tensor(-1.0))


if __name__ == "__main__":
    unittest.main()
