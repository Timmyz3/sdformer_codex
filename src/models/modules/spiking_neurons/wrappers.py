"""Metadata wrappers for neuron selection."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.registry import register_module


@register_module("spiking_neurons", "spec")
@dataclass
class NeuronSpec:
    neuron_type: str
    v_th: float
    tau: float

