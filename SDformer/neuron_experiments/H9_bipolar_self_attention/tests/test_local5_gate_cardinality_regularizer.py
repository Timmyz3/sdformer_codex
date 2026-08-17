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


class Local5GateCardinalityRegularizerTest(unittest.TestCase):
    def test_equal_source_gates_have_zero_proxy(self) -> None:
        from models.STSwinNet_SNN.bsa_attention import (
            _source_gate_cardinality_proxy,
        )

        gate = torch.tensor(
            [[[[0.5, 0.25], [0.5, 0.25], [0.75, 0.25]]]],
            requires_grad=True,
        )
        source_index = torch.tensor([[0, 1], [0, 1], [2, 1]])
        valid = torch.ones_like(source_index, dtype=torch.bool)
        source_k = torch.ones(1, 1, 3, 4)
        loss = _source_gate_cardinality_proxy(
            gate, source_index=source_index, valid=valid, source_k=source_k
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(gate.grad)

    def test_unequal_source_gates_are_penalized(self) -> None:
        from models.STSwinNet_SNN.bsa_attention import (
            _source_gate_cardinality_proxy,
        )

        gate = torch.tensor(
            [[[[0.5, 0.25], [0.75, 0.25], [0.75, 0.25]]]],
            requires_grad=True,
        )
        source_index = torch.tensor([[0, 1], [0, 1], [2, 1]])
        valid = torch.ones_like(source_index, dtype=torch.bool)
        source_k = torch.ones(1, 1, 3, 4)
        loss = _source_gate_cardinality_proxy(
            gate, source_index=source_index, valid=valid, source_k=source_k
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(gate.grad.abs().sum()), 0.0)

    def test_tail_gap_c2_ignores_two_classes(self) -> None:
        from models.STSwinNet_SNN.bsa_attention import (
            _source_gate_cardinality_proxy,
        )

        gate = torch.tensor(
            [[[[0.25, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.25]]]],
            requires_grad=True,
        )
        source_index = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        valid = torch.ones_like(source_index, dtype=torch.bool)
        source_k = torch.ones(1, 1, 3, 4)
        loss = _source_gate_cardinality_proxy(
            gate,
            source_index=source_index,
            valid=valid,
            source_k=source_k,
            mode="tail_gap_c2",
        )
        self.assertEqual(float(loss.detach()), 0.0)

    def test_tail_gap_c2_penalizes_third_class(self) -> None:
        from models.STSwinNet_SNN.bsa_attention import (
            _source_gate_cardinality_proxy,
        )

        gate = torch.tensor(
            [[[[0.25, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.75]]]],
            requires_grad=True,
        )
        source_index = torch.zeros(3, 3, dtype=torch.long)
        valid = torch.eye(3, dtype=torch.bool)
        source_k = torch.ones(1, 1, 3, 4)
        loss = _source_gate_cardinality_proxy(
            gate,
            source_index=source_index,
            valid=valid,
            source_k=source_k,
            mode="tail_gap_c2",
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(gate.grad.abs().sum()), 0.0)

    def test_model_collector_is_default_off_and_fail_closed(self) -> None:
        from models.STSwinNet_SNN.bsa_attention import (
            regularize_source_gate_cardinality,
        )

        model = nn.Sequential(nn.Linear(1, 1))
        self.assertIsNone(regularize_source_gate_cardinality(model, {}))
        with self.assertRaisesRegex(RuntimeError, "no Local5 proxy"):
            regularize_source_gate_cardinality(
                model,
                {
                    "mode": "binary_axnor_local5_shiftmax",
                    "source_gate_cardinality_regularization_weight": 0.01,
                },
            )

    def test_model_collector_scales_mean_loss(self) -> None:
        from models.STSwinNet_SNN.bsa_attention import (
            regularize_source_gate_cardinality,
        )

        model = nn.Sequential(nn.Identity(), nn.Identity())
        model[0]._h9_source_gate_cardinality_proxy = torch.tensor(0.25)
        model[1]._h9_source_gate_cardinality_proxy = torch.tensor(0.75)
        result = regularize_source_gate_cardinality(
            model,
            {
                "mode": "binary_axnor_local5_shiftmax",
                "source_gate_cardinality_regularization_weight": 0.02,
            },
        )
        self.assertAlmostEqual(float(result), 0.01, places=7)


if __name__ == "__main__":
    unittest.main()
