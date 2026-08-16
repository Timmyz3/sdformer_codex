#!/usr/bin/env python3
"""Build the frozen CICC/Spike-FlowNet MVSEC train, validation, and test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "neuron_experiments/H9_bipolar_self_attention/manifests/mvsec_cicc_dt1_v1.json"
)
VALID_RANGES = {
    "indoor_flying1": (314, 2197),
    "indoor_flying2": (314, 2199),
    "indoor_flying3": (314, 2199),
    "outdoor_day1": (245, 3000),
    "outdoor_day2": (4375, 7002),
}


def evenly_spaced_indices(start: int, end: int, count: int) -> list[int]:
    indices = np.linspace(start, end - 1, num=count, dtype=np.int64).tolist()
    if len(indices) != count or len(set(indices)) != count:
        raise RuntimeError(f"Cannot select {count} unique indices from [{start}, {end})")
    return [int(index) for index in indices]


def build_manifest() -> dict[str, object]:
    train_start, train_end = VALID_RANGES["outdoor_day2"]
    validation_count = 263
    validation_start = train_end - validation_count
    gap_index = validation_start - 1
    train_indices = list(range(train_start, gap_index))
    validation_indices = list(range(validation_start, train_end))
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("MVSEC train and validation indices overlap")
    if max(train_indices) + 2 > min(validation_indices):
        raise RuntimeError("MVSEC split does not isolate adjacent dt1 event pairs")

    splits: dict[str, object] = {
        "train": {"sequence": "outdoor_day2", "indices": train_indices},
        "validation": {
            "sequence": "outdoor_day2",
            "indices": validation_indices,
        },
    }
    for sequence in ("outdoor_day1", "indoor_flying1", "indoor_flying2", "indoor_flying3"):
        start, end = VALID_RANGES[sequence]
        splits[f"test_fixed800_{sequence}"] = {
            "sequence": sequence,
            "indices": evenly_spaced_indices(start, end, 800),
        }
    return {
        "schema": "mvsec_cicc_split_manifest_v1",
        "protocol": "outdoor_day2_train_heldout_tail_validation_four_sequence_fixed800_dt1",
        "selection": {
            "train_validation": "chronological_90_10_with_one_sample_gap",
            "test_fixed800": "deterministic_uniform_coverage_of_valid_interval",
            "crop": [256, 256],
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "evaluation_mask": "event_and_valid_flow",
        },
        "excluded_gap": {"sequence": "outdoor_day2", "indices": [gap_index]},
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
