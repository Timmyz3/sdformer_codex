from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(REPO_ROOT / "neuron_experiments" / "H9_bipolar_self_attention" / "overlay"))


class DummyPSN(nn.Module):
    def __init__(self, T: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(T))
        self.bias = nn.Parameter(torch.zeros(T, 1))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

        self.ternary = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(3),
            thresh=0.5,
            sparsity_eta=1e-3,
            negative_threshold_scale=30.0,
            output_mode="ternary",
        )
        self.binary = ATLIFTernaryPSN(
            T=3,
            base_psn=DummyPSN(3),
            thresh=0.5,
            sparsity_eta=1e-3,
            output_mode="binary",
        )


class SymmetricTernaryPSNTest(unittest.TestCase):
    def test_symmetric_forward_outputs_threshold_constrained_values(self):
        from neuron_autoresearch.attention.symmetric_ternary_psn import install_symmetric_ternary

        model = DummyModel()
        installed = install_symmetric_ternary(model, {"symmetric_ternary": {"enabled": True}})
        self.assertEqual(installed, ["ternary"])

        x = torch.tensor(
            [
                [1.2, -1.2, 0.2, -0.2],
                [0.7, -0.7, 0.0, 0.1],
                [0.0, 0.6, -0.6, 0.2],
            ]
        )
        out = model.ternary(x)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(set(torch.unique(out.detach()).tolist()), {-0.5, 0.0, 0.5})
        self.assertGreater(model.ternary.neg_r, 0.0)
        self.assertGreater(model.ternary.pos_r, 0.0)

    def test_binary_modules_are_not_patched_by_default(self):
        from neuron_autoresearch.attention.symmetric_ternary_psn import install_symmetric_ternary

        model = DummyModel()
        install_symmetric_ternary(model, {"symmetric_ternary": {"enabled": True}})

        self.assertFalse(hasattr(model.binary, "_sym_original_forward"))
        self.assertEqual(model.binary.output_mode, "binary")


if __name__ == "__main__":
    unittest.main()
