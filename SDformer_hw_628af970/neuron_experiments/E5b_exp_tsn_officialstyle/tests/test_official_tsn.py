from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXP_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = EXP_ROOT / "overlay"
if str(OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERLAY_ROOT))


class OfficialTSNTest(unittest.TestCase):
    def test_official_ternary_activation_contract(self) -> None:
        from models.STSwinNet_SNN.experimental_neurons.single.tsn import official_ternary_spike_activation

        x = torch.tensor([-1.2, -0.6, -0.4, 0.0, 0.4, 0.6, 1.2], requires_grad=True)
        y = official_ternary_spike_activation(x)

        torch.testing.assert_close(y.detach(), torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))

    def test_tsn_node_uses_official_update(self) -> None:
        from models.STSwinNet_SNN.experimental_neurons.single.tsn import TSNNode

        node = TSNNode(T=3, v_threshold=1.0, decay=0.25)
        self.assertEqual(node.official_source_commit, "2aca58747f01d7960cb6f0284665bbb353d35aab")

        x = torch.full((3, 2, 4), 0.6, requires_grad=True)
        y = node(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(set(y.detach().flatten().tolist()).issubset({-1.0, 0.0, 1.0}))
        y.sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
