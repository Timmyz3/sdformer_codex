from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


OVERLAY = Path(__file__).resolve().parents[1] / "overlay"
if str(OVERLAY) not in sys.path:
    sys.path.insert(0, str(OVERLAY))

from models.STSwinNet_SNN.sparse_gate import (  # noqa: E402
    DEFAULT_TARGET_LAYERS,
    HardSparseGate,
    install_sparse_gates,
    iter_sparse_gates,
    sparse_gate_regularization,
    sparse_gate_summary,
)


class DummySpikingNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.spiking_neuron = nn.Identity()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.spiking_neuron(x * self.weight)


def make_nested_model() -> nn.Module:
    root = nn.Module()
    cursor = root
    for name in DEFAULT_TARGET_LAYERS:
        parts = name.split(".")
        cursor = root
        for part in parts[:-1]:
            if not hasattr(cursor, part):
                setattr(cursor, part, nn.Module())
            cursor = getattr(cursor, part)
        setattr(cursor, parts[-1], DummySpikingNeuron())
    return root


def test_install_sparse_gates_wraps_all_targets():
    model = make_nested_model()

    installed = install_sparse_gates(
        model,
        {
            "enabled": True,
            "target_layers": DEFAULT_TARGET_LAYERS,
            "init_logit": -2.0,
            "threshold": 0.5,
        },
    )

    assert tuple(installed) == DEFAULT_TARGET_LAYERS
    gates = list(iter_sparse_gates(model))
    assert len(gates) == len(DEFAULT_TARGET_LAYERS)
    assert all(isinstance(gate, HardSparseGate) for _, gate in gates)


def test_closed_gate_outputs_zero_with_st_gradient_path():
    gate = HardSparseGate(nn.Identity(), init_logit=-2.0, threshold=0.5)
    x = torch.ones(4, requires_grad=True)

    y = gate(x)
    assert torch.count_nonzero(y).item() == 0

    y.sum().backward()
    assert gate.gate_logit.grad is not None


def test_freeze_backbone_keeps_only_gate_logits_trainable():
    model = make_nested_model()
    install_sparse_gates(
        model,
        {
            "enabled": True,
            "target_layers": DEFAULT_TARGET_LAYERS,
            "init_logit": -2.0,
            "threshold": 0.5,
            "freeze_backbone": True,
        },
    )

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert len(trainable) == len(DEFAULT_TARGET_LAYERS)
    assert all(name.endswith("gate_logit") for name in trainable)


def test_regularization_and_summary():
    model = make_nested_model()
    install_sparse_gates(
        model,
        {
            "enabled": True,
            "target_layers": DEFAULT_TARGET_LAYERS,
            "init_logit": -2.0,
            "threshold": 0.5,
        },
    )

    penalty = sparse_gate_regularization(model, {"enabled": True, "reg_lambda": 0.02})
    summary = sparse_gate_summary(model)

    assert penalty is not None
    assert penalty.item() > 0
    assert summary["num_gates"] == len(DEFAULT_TARGET_LAYERS)
    assert summary["open_gates"] == 0


if __name__ == "__main__":
    test_install_sparse_gates_wraps_all_targets()
    test_closed_gate_outputs_zero_with_st_gradient_path()
    test_freeze_backbone_keeps_only_gate_logits_trainable()
    test_regularization_and_summary()
    print("G1 sparse gate tests passed")
