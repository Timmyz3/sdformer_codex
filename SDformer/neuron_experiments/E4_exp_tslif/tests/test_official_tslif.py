from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXP_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = EXP_ROOT / "overlay"
if str(OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERLAY_ROOT))


class OfficialTSLIFTest(unittest.TestCase):
    def test_tslif_uses_official_trainable_alpha_contract(self) -> None:
        from models.STSwinNet_SNN.experimental_neurons.single.tslif import TSLIFNode

        node = TSLIFNode(T=3, v_threshold=0.5)
        self.assertEqual(node.official_source_commit, "a59826a6c7f62d0f16edbafdbb28db65bebd9f69")
        self.assertTrue(any(param is node.alpha_s for param in node.parameters()))
        self.assertTrue(any(param is node.alpha_l for param in node.parameters()))

        x = torch.full((3, 2, 4), 1.0, requires_grad=True)
        y = node(x)

        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(node.alpha_s)
        self.assertIsNotNone(node.alpha_l)
        self.assertTrue(node.alpha_s.requires_grad)
        self.assertTrue(node.alpha_l.requires_grad)
        self.assertEqual(tuple(node.alpha_s.shape), (1,))
        self.assertEqual(tuple(node.alpha_l.shape), (1,))

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(node.decay_factor.grad)
        self.assertIsNotNone(node.kk.grad)
        self.assertIsNotNone(node.yy.grad)
        self.assertIsNotNone(node.alpha_s.grad)
        self.assertIsNotNone(node.alpha_l.grad)

    def test_tslif_resets_state_between_independent_forward_calls(self) -> None:
        from models.STSwinNet_SNN.experimental_neurons.single.tslif import TSLIFNode

        node = TSLIFNode(T=2, v_threshold=0.5)
        x = torch.ones(2, 1, 3)

        y1 = node(x)
        y2 = node(x)

        torch.testing.assert_close(y1, y2)


if __name__ == "__main__":
    unittest.main()
