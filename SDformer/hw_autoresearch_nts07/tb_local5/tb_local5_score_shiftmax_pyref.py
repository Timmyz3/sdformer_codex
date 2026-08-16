#!/usr/bin/env python3
"""Python reference vectors for Local-5 score+Shiftmax5 RTL (no full cocotb)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Import overlay attention helpers
import sys

REPO = Path(__file__).resolve().parents[2]
# Overlay must precede SDformerFlow so models.STSwinNet_SNN resolves to H9 overlay.
sys.path.insert(0, str(REPO / "third_party/SDformerFlow"))
sys.path.insert(0, str(REPO / "neuron_experiments/H9_bipolar_self_attention/overlay"))

from models.STSwinNet_SNN.bsa_attention import (  # type: ignore
    _apply_hardware_gate_quant,
    _apply_hardware_score_quant,
    _normalize_consensus_score,
    _rtl_shiftmax_gate_q17,
    config_from_dict,
)


def axnor_score(q: torch.Tensor, k: torch.Tensor, alpha0: float, head_dim: int) -> float:
    same_spike = (q * k).sum().item()
    same_silent = ((1 - q) * (1 - k)).sum().item()
    raw = same_spike + alpha0 * same_silent
    return raw / float(head_dim)


def main() -> int:
    cfg = config_from_dict(
        {
            "hardware_quant_enabled": True,
            "hardware_rtl_shiftmax_enabled": True,
            "hardware_score_step": 1 / 128,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
            "hardware_gate_step": 1 / 128,
            "hardware_gate_min": 0.0,
            "hardware_gate_max": 2.0,
            "alpha0": 1 / 64,
            "consensus_score_norm": "head_dim",
            "preserve_mean": False,
        }
    )
    head_dim = 32
    torch.manual_seed(66)
    vectors = []
    for n in range(32):
        q = torch.randint(0, 2, (head_dim,)).float()
        ks = [torch.randint(0, 2, (head_dim,)).float() for _ in range(5)]
        valid = [1, 1, 1, 1, 1]
        if n % 4 == 0:
            valid[1] = 0  # up invalid
            valid[3] = 0  # left invalid
        scores = torch.tensor(
            [axnor_score(q, k, 1 / 64, head_dim) for k in ks], dtype=torch.float32
        ).view(1, 1, 1, 5)
        scores = _apply_hardware_score_quant(scores, cfg)
        for i, v in enumerate(valid):
            if not v:
                scores[..., i] = -2.0
        gate = _rtl_shiftmax_gate_q17(scores, dim=-1, preserve_mean=False)
        gate = _apply_hardware_gate_quant(gate, cfg)
        vectors.append(
            {
                "q_bits": int("".join(str(int(x)) for x in q.tolist())[::-1], 2),
                "k_bits": [
                    int("".join(str(int(x)) for x in k.tolist())[::-1], 2) for k in ks
                ],
                "valid": valid,
                "score_q7": [int(round(float(s) * 128)) for s in scores.view(-1).tolist()],
                "gate_q17": [int(round(float(g) * 128)) for g in gate.view(-1).tolist()],
            }
        )
    out = Path(__file__).resolve().parent / "local5_ref_vectors.json"
    out.write_text(json.dumps({"head_dim": head_dim, "vectors": vectors}, indent=2) + "\n")
    print(out, "n=", len(vectors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
