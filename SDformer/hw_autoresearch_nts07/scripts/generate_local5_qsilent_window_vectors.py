#!/usr/bin/env python3
"""Extract one complete Local5 S0 window (3 heads) for scheduler+Q-silent TB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_local5_active_projection_postg0_vectors import write_memh
from generate_local5_checkpoint_score_vectors import reconstruct_destination_rows
from generate_local5_score_projection_vectors import pack_fields

SOURCES = 450


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "results/local5_fullres_bb1e4_joint_heads_profile100_20260809/"
            "ordered_term_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    groups = [
        (index, row)
        for index, row in enumerate(manifest["groups"])
        if int(row["sample"]) == args.sample
        and int(row["stage"]) == args.stage
        and int(row["block"]) == args.block
    ]
    if not groups:
        raise ValueError("no matching groups")
    window = int(groups[0][1]["window"])
    selected = [
        (index, row)
        for index, row in groups
        if int(row["window"]) == window
    ]
    selected.sort(key=lambda item: int(item[1]["head"]))
    if [int(row["head"]) for _, row in selected] != [0, 1, 2]:
        raise ValueError(f"window {window} heads incomplete")

    payload = np.load(args.source_manifest.parent / manifest["payload_file"], mmap_mode="r")
    offsets = payload["descriptor_group_offsets"]
    heads = len(selected)
    input_q = np.zeros((heads, SOURCES), dtype=np.uint32)
    input_k = np.empty((heads, SOURCES), dtype=object)
    input_valid = np.zeros((heads, SOURCES), dtype=np.uint8)
    expected_scores = np.empty((heads, SOURCES), dtype=object)
    expected_gates = np.empty((heads, SOURCES), dtype=object)

    for out_index, (input_group, _row) in enumerate(selected):
        start = int(offsets[input_group])
        stop = int(offsets[input_group + 1])
        rows = reconstruct_destination_rows(
            q_bitmap=np.asarray(payload["descriptor_q_bitmap"][start:stop]),
            k_bitmap=np.asarray(payload["descriptor_k_bitmap"][start:stop]),
            planes=np.asarray(payload["descriptor_source_plane"][start:stop]),
            source_y=np.asarray(payload["descriptor_source_y"][start:stop]),
            source_x=np.asarray(payload["descriptor_source_x"][start:stop]),
            incoming_gates=np.asarray(payload["descriptor_incoming_gates"][start:stop]),
            incoming_valid_mask=np.asarray(payload["descriptor_valid_mask"][start:stop]),
        )
        for dest, (q_value, k_values, valid_mask, scores, gates) in enumerate(rows):
            input_q[out_index, dest] = np.uint32(q_value)
            input_k[out_index, dest] = pack_fields(k_values, 32)
            input_valid[out_index, dest] = np.uint8(valid_mask)
            leaf_scores = [
                score if (valid_mask >> role) & 1 else -256
                for role, score in enumerate(scores)
            ]
            expected_scores[out_index, dest] = pack_fields(leaf_scores, 16)
            expected_gates[out_index, dest] = pack_fields(gates, 9)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_memh(args.output_dir / "input_q.memh", input_q, 32)
    write_memh(args.output_dir / "input_candidate_k.memh", input_k, 160)
    write_memh(args.output_dir / "input_valid.memh", input_valid, 5)
    write_memh(args.output_dir / "expected_scores.memh", expected_scores, 80)
    write_memh(args.output_dir / "expected_gates.memh", expected_gates, 45)
    identity = {
        "schema": "local5_qsilent_window_vectors_v1",
        "sample": args.sample,
        "stage": args.stage,
        "block": args.block,
        "profile_window": window,
        "heads": [0, 1, 2],
        "group_indices": [index for index, _ in selected],
        "source_manifest": str(args.source_manifest.resolve()),
        "payload_sha256": manifest["payload_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "note": (
            "Scheduler first S0.B0 window index is 0; this payload is the only "
            "captured S0.B0 window for sample 0 (profile window 94). Topology "
            "matches; window id is remapped and must stay labeled as such."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS window vectors heads={heads} profile_window={window}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
