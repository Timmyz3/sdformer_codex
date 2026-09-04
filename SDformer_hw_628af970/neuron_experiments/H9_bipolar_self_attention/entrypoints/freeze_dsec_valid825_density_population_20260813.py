#!/usr/bin/env python3
"""Freeze DATE Table G event-density quartiles from the valid825 voxel population.

This is model-independent. It does not use the GPU and does not tune cuts on AEE.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/Datasets/DSEC/saved_flow_data"
LIST_PATH = DATA / "sequence_lists/valid_split_seq.csv"
VOXEL_ROOT = DATA / "event_tensors/10bins/left"
OUTPUT_JSON = REPO / "neuron_autoresearch/DSEC_VALID825_DENSITY_POPULATION_20260813.json"
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
EXPECTED_LIST_SHA = "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0"
SEQ_RE = re.compile(r"^(.*)_(\d+)\.npy$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_name(filename: str) -> str:
    match = SEQ_RE.match(filename)
    if not match:
        raise RuntimeError(f"unrecognized validation filename: {filename}")
    return match.group(1)


def voxel_path(filename: str) -> Path:
    return VOXEL_ROOT / sequence_name(filename) / filename


def main() -> int:
    if sha256(LIST_PATH) != EXPECTED_LIST_SHA:
        raise RuntimeError(f"validation list SHA drift: {LIST_PATH}")
    names = [line.strip() for line in LIST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 825:
        raise RuntimeError(f"expected 825 validation frames, got {len(names)}")

    rows = []
    for name in names:
        path = voxel_path(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        voxel = np.load(path, mmap_mode="r")
        if tuple(voxel.shape) != (10, 480, 640):
            raise RuntimeError(f"unexpected voxel shape {voxel.shape}: {path}")
        l1 = float(np.abs(voxel).sum(dtype=np.float64))
        active = int(np.count_nonzero(np.any(voxel != 0, axis=0)))
        rows.append(
            {
                "file": name,
                "sequence": sequence_name(name),
                "voxel_l1": l1,
                "active_pixels": active,
                "active_pixel_ratio": active / float(480 * 640),
            }
        )

    l1_values = np.asarray([row["voxel_l1"] for row in rows], dtype=np.float64)
    cuts = {
        "q25": float(np.quantile(l1_values, 0.25)),
        "q50": float(np.quantile(l1_values, 0.50)),
        "q75": float(np.quantile(l1_values, 0.75)),
    }
    counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for row in rows:
        value = float(row["voxel_l1"])
        if value <= cuts["q25"]:
            quartile = "Q1"
        elif value <= cuts["q50"]:
            quartile = "Q2"
        elif value <= cuts["q75"]:
            quartile = "Q3"
        else:
            quartile = "Q4"
        row["quartile"] = quartile
        counts[quartile] += 1

    payload = {
        "schema": "dsec_valid825_density_population_v1",
        "status": "PASS_POPULATION_FROZEN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "population": "DSEC local valid825",
        "density_definition": "sum of absolute 10-bin voxel values at 480x640; no model, no AEE",
        "quartile_assignment": "Q1<=q25, Q2<=q50, Q3<=q75, Q4>q75 on voxel_l1",
        "validation_list": {
            "path": str(LIST_PATH.resolve()),
            "sha256": EXPECTED_LIST_SHA,
            "frames": 825,
        },
        "cuts": cuts,
        "quartile_counts": counts,
        "voxel_l1_summary": {
            "min": float(l1_values.min()),
            "mean": float(l1_values.mean()),
            "max": float(l1_values.max()),
        },
        "frames": rows,
        "pending_algorithm_columns": ["AEE", "Fl", "spikes_per_frame"],
        "pending_hardware_columns": ["active_relations", "memo_hit_rate", "cycles_per_frame"],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DSEC valid825 event-density population",
        "",
        f"Status: `{payload['status']}`; frames=`825`; density=`voxel L1`.",
        "",
        "| cut | voxel L1 |",
        "|---|---:|",
        f"| Q25 | {cuts['q25']:.3f} |",
        f"| Q50 | {cuts['q50']:.3f} |",
        f"| Q75 | {cuts['q75']:.3f} |",
        "",
        "| quartile | frames |",
        "|---|---:|",
        f"| Q1 | {counts['Q1']} |",
        f"| Q2 | {counts['Q2']} |",
        f"| Q3 | {counts['Q3']} |",
        f"| Q4 | {counts['Q4']} |",
        "",
        "Cuts are frozen on the validation voxels. Do not retune them after seeing AEE.",
        "AEE / Fl / spikes remain pending until a later same-checkpoint per-frame eval.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
