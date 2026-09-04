#!/usr/bin/env python3
"""生成仅用于协议回归的 ET3 synthetic ordered trace。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from et3_ordered_trace_replay import canonical_item_hash, file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        {
            "tag": 0x11,
            "sample": 0,
            "stage": 0,
            "block": 0,
            "window": 0,
            "head": 0,
            "items": [
                (0, 3, 0, 1, 0),
                (0, 3, 0, 1, 2),
                (0, 3, 0, 1, 4),
                (0, 2, 1, 1, 1),
                (0, 1, 3, 1, 8),
            ],
        },
        {
            "tag": 0x22,
            "sample": 0,
            "stage": 0,
            "block": 0,
            "window": 1,
            "head": 0,
            "items": [
                (1, 4, 2, 2, 0),
                (1, 4, 2, 2, 2),
                (1, 4, 2, 2, 4),
                (1, 4, 2, 2, 6),
                (1, 1, 3, 3, 1),
                (1, 2, 1, 4, 8),
            ],
        },
        {
            "tag": 0x33,
            "sample": 0,
            "stage": 0,
            "block": 0,
            "window": 2,
            "head": 0,
            "items": [],
        },
    ]
    flat = [item for group in groups for item in group["items"]]
    offsets = [0]
    for group in groups:
        offsets.append(offsets[-1] + len(group["items"]))
    arrays = {
        "group_offsets": np.asarray(offsets, dtype=np.int64),
        "group_tags": np.asarray(
            [group["tag"] for group in groups], dtype=np.uint64
        ),
        "item_mode_multiset": np.asarray(
            [item[0] for item in flat], dtype=np.uint8
        ),
        "item_gate_code": np.asarray(
            [item[1] for item in flat], dtype=np.uint16
        ),
        "item_lane_id": np.asarray(
            [item[2] for item in flat], dtype=np.uint16
        ),
        "item_multiplicity": np.asarray(
            [item[3] for item in flat], dtype=np.uint8
        ),
        "item_destination": np.asarray(
            [item[4] for item in flat], dtype=np.uint16
        ),
    }
    payload_path = args.output_dir / "ordered_items.npz"
    np.savez_compressed(payload_path, **arrays)
    metadata = []
    for index, group in enumerate(groups):
        start = offsets[index]
        end = offsets[index + 1]
        metadata.append(
            {
                "tag": group["tag"],
                "empty": start == end,
                "sample": group["sample"],
                "stage": group["stage"],
                "block": group["block"],
                "window": group["window"],
                "head": group["head"],
                "ordered_item_sha256": (
                    canonical_item_hash(arrays, start, end)
                    if start != end
                    else hashlib.sha256(b"").hexdigest()
                ),
            }
        )
    manifest = {
        "schema": "et3_ordered_term_trace_v1",
        "evidence_level": "synthetic",
        "payload_file": payload_path.name,
        "payload_sha256": file_sha256(payload_path),
        "config_sha256": "synthetic-protocol-only",
        "checkpoint_sha256": "synthetic-protocol-only",
        "cohort_sha256": "synthetic-protocol-only",
        "resolution": {
            "tokens": 16,
            "full_resolution": False,
            "note": "不是 DSEC workload",
        },
        "groups": metadata,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
