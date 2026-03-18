"""Spiking-neuron helpers shared by the SDformerFlow adapter."""

from __future__ import annotations

from typing import Type


def resolve_upstream_neuron_type(neuron_name: str):
    from spikingjelly.activation_based import neuron
    from models.STSwinNet_SNN.Spiking_submodules import GatedLIFNode, PSN, SLTTLIFNode

    mapping = {
        "if": neuron.IFNode,
        "lif": neuron.LIFNode,
        "plif": neuron.ParametricLIFNode,
        "glif": GatedLIFNode,
        "psn": PSN,
        "SLTTlif": SLTTLIFNode,
    }
    if neuron_name not in mapping:
        raise KeyError(f"unsupported upstream neuron type: {neuron_name}")
    return mapping[neuron_name]

