#!/usr/bin/env python3
"""Deterministic Class File golden rows for a later RTL miter. No GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.h82_class_file_reference import (  # noqa: E402
    TOKENS,
    build_class_file,
    c7_multiplicity_weighted_gates,
    file_as_json,
    integer_c7_gates,
    integer_class_major_gates,
    q7_codes,
    token_shiftmax_gates,
)


def rows() -> dict[str, np.ndarray]:
    unequal = np.zeros(TOKENS, dtype=np.float64)
    unequal[:3] = 0.0
    unequal[3] = 1.0
    two_class = np.zeros(TOKENS, dtype=np.float64)
    two_class[225:] = 1.0
    singletons = np.asarray(
        [-2.0 + (1.0 / 128.0) * index for index in range(TOKENS)], dtype=np.float64
    )
    mixed = np.zeros(TOKENS, dtype=np.float64)
    mixed[0:20] = 0.25
    mixed[20:50] = -0.5
    mixed[50] = 1.25
    mixed[225] = 0.25
    return {
        "unequal_mult": unequal,
        "equal_mult_225": two_class,
        "singletons": singletons,
        "mixed_pair_mask": mixed,
    }


def pack(name: str, scores: np.ndarray) -> dict:
    class_file = build_class_file(scores, preserve_mean=True)
    codes = q7_codes(scores)
    return {
        "name": name,
        "n_occupied": class_file.n_occupied,
        "h82_gate_q17_int": integer_class_major_gates(codes, preserve_mean=True).tolist(),
        "c7_gate_q17_int": integer_c7_gates(codes, preserve_mean=True).tolist(),
        "h82_gate_float": class_file.gate_tokens().tolist(),
        "c7_gate_float": c7_multiplicity_weighted_gates(scores, preserve_mean=True).tolist(),
        "token_gate_float": token_shiftmax_gates(scores, preserve_mean=True).tolist(),
        "class_file": file_as_json(class_file),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "h82_class_file_isa_20260817"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "h82_class_file_golden_v1",
        "rows": [pack(name, scores) for name, scores in rows().items()],
    }
    (out / "golden_vectors.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "wrote": str(out / "golden_vectors.json"),
        "rows": [
            {"name": item["name"], "n_occupied": item["n_occupied"]}
            for item in payload["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
