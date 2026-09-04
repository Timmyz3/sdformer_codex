#!/usr/bin/env python3
"""Fail-closed audit for encoded MVSEC files referenced by the frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MVSEC_TEST = REPO_ROOT / "third_party/SDformerFlow/data/Datasets/MVSEC/MVSEC_test"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "neuron_experiments/H9_bipolar_self_attention/manifests/mvsec_cicc_dt1_v1.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "neuron_experiments/H9_bipolar_self_attention/results/mvsec_cicc_dataset_audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_paths(sequence: str, index: int) -> tuple[Path, Path, Path]:
    root = MVSEC_TEST / sequence
    return (
        root / "flowgt_dt1" / f"{index}.npy",
        root / "event" / f"{index + 1:06d}.h5",
        root / "event" / f"{index + 2:06d}.h5",
    )


def inspect_sample(sequence: str, index: int) -> dict[str, object]:
    flow_path, event_old, event_new = sample_paths(sequence, index)
    flow = np.load(flow_path)
    event_rows = []
    for event_path in (event_old, event_new):
        with h5py.File(event_path, "r") as handle:
            if "myDataset" not in handle:
                raise RuntimeError(f"Missing myDataset in {event_path}")
            values = handle["myDataset"].get("block0_values")
            if values is None or len(values.shape) != 2 or values.shape[1] != 4:
                raise RuntimeError(f"Invalid pandas event table in {event_path}")
            event_rows.append(int(values.shape[0]))
    if flow.shape not in {(260, 346, 2), (2, 260, 346)}:
        raise RuntimeError(f"Unexpected MVSEC flow shape {flow.shape}: {flow_path}")
    return {
        "sequence": sequence,
        "index": index,
        "flow_shape": list(flow.shape),
        "flow_dtype": str(flow.dtype),
        "flow_finite_fraction": float(np.isfinite(flow).mean()),
        "event_rows": event_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = {}
    for role, split in manifest["splits"].items():
        sequence = split["sequence"]
        indices = [int(index) for index in split["indices"]]
        missing = []
        for index in indices:
            for path in sample_paths(sequence, index):
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path))
        if missing:
            raise RuntimeError(f"MVSEC split {role} has missing files: {missing[:8]}")
        probes = sorted({indices[0], indices[len(indices) // 2], indices[-1]})
        rows[role] = {
            "sequence": sequence,
            "count": len(indices),
            "first": indices[0],
            "last": indices[-1],
            "probes": [inspect_sample(sequence, index) for index in probes],
        }
    train = set(manifest["splits"]["train"]["indices"])
    validation = set(manifest["splits"]["validation"]["indices"])
    output = {
        "schema": "mvsec_cicc_dataset_audit_v1",
        "status": "PASS",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "train_validation_disjoint": not bool(train & validation),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
