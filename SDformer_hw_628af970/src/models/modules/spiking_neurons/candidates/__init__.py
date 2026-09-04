"""Candidate spiking neurons for SDformerFlow ablations.

These modules are intentionally kept independent from SpikingJelly so they can
be smoke-tested in this repo before wiring them into the upstream wrapper.
"""

from __future__ import annotations

from .atlif import ATLIFNode
from .lmh import LMHNode
from .sn import SNNode
from .tslif import TSLIFNode
from .tsn import TSNNode

_CANDIDATES = {
    "lmh": LMHNode,
    "lm-h": LMHNode,
    "lmht": LMHNode,
    "tslif": TSLIFNode,
    "ts-lif": TSLIFNode,
    "atlif": ATLIFNode,
    "at-lif": ATLIFNode,
    "sn": SNNode,
    "simple": SNNode,
    "tsn": TSNNode,
    "ternary": TSNNode,
}


def get_candidate_neuron(name: str):
    key = name.lower()
    if key not in _CANDIDATES:
        raise KeyError(f"unsupported candidate neuron: {name}")
    return _CANDIDATES[key]


__all__ = [
    "ATLIFNode",
    "LMHNode",
    "SNNode",
    "TSLIFNode",
    "TSNNode",
    "get_candidate_neuron",
]
