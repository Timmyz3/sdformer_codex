import unittest

import torch

from src.models.modules.spiking_neurons.candidates import (
    ATLIFNode,
    LMHNode,
    SNNode,
    TSLIFNode,
    TSNNode,
    get_candidate_neuron,
)


class CandidateSpikingNeuronsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(4, 2, 3, 5, 5, requires_grad=True)

    def test_binary_candidate_neurons_preserve_shape_and_backpropagate(self):
        for cls in (LMHNode, TSLIFNode, ATLIFNode, SNNode):
            with self.subTest(cls=cls.__name__):
                node = cls(T=4, v_threshold=1.0)
                y = node(self.x)

                self.assertEqual(y.shape, self.x.shape)
                self.assertTrue(torch.isfinite(y).all())
                self.assertGreaterEqual(y.min().item(), 0.0)

                loss = y.mean()
                loss.backward(retain_graph=True)
                self.assertIsNotNone(self.x.grad)
                self.assertTrue(torch.isfinite(self.x.grad).all())
                self.x.grad.zero_()

    def test_tsn_outputs_ternary_spikes_and_backpropagates(self):
        node = TSNNode(T=4, v_threshold=1.0)

        y = node(self.x)

        self.assertEqual(y.shape, self.x.shape)
        self.assertTrue(set(y.detach().unique().tolist()).issubset({-1.0, 0.0, 1.0}))
        y.mean().backward()
        self.assertIsNotNone(self.x.grad)

    def test_candidate_factory_resolves_aliases(self):
        self.assertIs(get_candidate_neuron("lmh"), LMHNode)
        self.assertIs(get_candidate_neuron("tslif"), TSLIFNode)
        self.assertIs(get_candidate_neuron("at-lif"), ATLIFNode)
        self.assertIs(get_candidate_neuron("sn"), SNNode)
        self.assertIs(get_candidate_neuron("tsn"), TSNNode)

        with self.assertRaises(KeyError):
            get_candidate_neuron("missing")


if __name__ == "__main__":
    unittest.main()
