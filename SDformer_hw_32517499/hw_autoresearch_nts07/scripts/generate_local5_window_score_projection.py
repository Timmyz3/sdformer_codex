#!/usr/bin/env python3
"""Complete-window Local5 vectors for Q-silent score→Acc32."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_local5_active_projection_postg0_vectors import (  # noqa: E402
    GATE_W,
    HEAD_DIM,
    HEIGHT,
    ROLE_DX,
    ROLE_DY,
    WIDTH,
    load_checkpoint_projection_contract,
    write_memh,
)
from generate_local5_checkpoint_score_vectors import (  # noqa: E402
    reconstruct_destination_rows,
)
from generate_local5_score_projection_vectors import pack_fields  # noqa: E402

SOURCES = 450
OUT_DIM = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/local5_fullres_bb1e4_joint_heads_profile100_20260809"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--window", type=int, default=-1)
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "ordered_term_manifest.json").read_text())
    groups = [
        (index, row)
        for index, row in enumerate(manifest["groups"])
        if int(row["sample"]) == args.sample
        and int(row["stage"]) == args.stage
        and int(row["block"]) == args.block
    ]
    if not groups:
        raise ValueError("no matching groups")
    window = int(groups[0][1]["window"]) if args.window < 0 else args.window
    selected = sorted(
        [(i, r) for i, r in groups if int(r["window"]) == window],
        key=lambda item: int(item[1]["head"]),
    )
    heads_found = [int(r["head"]) for _, r in selected]
    expected_heads = list(range(len(heads_found)))
    if heads_found != expected_heads or not heads_found:
        raise ValueError(f"incomplete window {window} heads={heads_found}")

    payload = np.load(args.input_dir / manifest["payload_file"], mmap_mode="r")
    offsets = payload["descriptor_group_offsets"]
    projection_rows, projection_payload, _binding = load_checkpoint_projection_contract(
        args.input_dir, manifest
    )

    heads = len(selected)
    input_q = np.zeros((heads, SOURCES), dtype=np.uint32)
    input_k = np.empty((heads, SOURCES), dtype=object)
    input_valid = np.zeros((heads, SOURCES), dtype=np.uint8)
    expected_scores = np.empty((heads, SOURCES), dtype=object)
    expected_gates = np.empty((heads, SOURCES), dtype=object)
    input_weights = np.zeros((heads, HEAD_DIM, OUT_DIM), dtype=np.int8)
    expected_acc = np.zeros((heads, SOURCES, OUT_DIM), dtype=np.int64)
    expected_active = np.zeros(heads, dtype=np.uint16)
    expected_terms = np.zeros(heads, dtype=np.uint32)
    expected_updates = np.zeros(heads, dtype=np.uint32)
    selection = []

    for out_i, (input_group, meta) in enumerate(selected):
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
        stage = int(meta["stage"])
        block = int(meta["block"])
        head = int(meta["head"])
        prefix = str(projection_rows[(stage, block)]["prefix"])
        matrix = np.asarray(projection_payload[f"{prefix}_weight_int8"])
        dim = int(matrix.shape[0])
        channels = [(out_i * OUT_DIM + out) % dim for out in range(OUT_DIM)]
        input_weights[out_i] = matrix[head * HEAD_DIM:(head + 1) * HEAD_DIM][:, channels]
        for dest, (q_value, k_values, valid_mask, scores, gates) in enumerate(rows):
            input_q[out_i, dest] = np.uint32(q_value)
            input_k[out_i, dest] = pack_fields(k_values, 32)
            input_valid[out_i, dest] = np.uint8(valid_mask)
            leaf_scores = [
                score if (valid_mask >> role) & 1 else -256
                for role, score in enumerate(scores)
            ]
            expected_scores[out_i, dest] = pack_fields(leaf_scores, 16)
            expected_gates[out_i, dest] = pack_fields(gates, 9)
        planes = np.asarray(payload["descriptor_source_plane"][start:stop])
        source_y = np.asarray(payload["descriptor_source_y"][start:stop])
        source_x = np.asarray(payload["descriptor_source_x"][start:stop])
        k_bitmap = np.asarray(payload["descriptor_k_bitmap"][start:stop])
        src_gates = np.asarray(payload["descriptor_incoming_gates"][start:stop])
        src_valid = np.asarray(payload["descriptor_valid_mask"][start:stop])
        for source in range(SOURCES):
            k_value = int(k_bitmap[source])
            lane_count = k_value.bit_count()
            unique_gates: set[int] = set()
            role_count = 0
            sy = int(source_y[source])
            sx = int(source_x[source])
            plane = int(planes[source])
            for role in range(5):
                gate = int(src_gates[source, role])
                if not ((int(src_valid[source]) >> role) & 1) or gate == 0:
                    continue
                destination = plane * HEIGHT * WIDTH + (sy + ROLE_DY[role]) * WIDTH + (
                    sx + ROLE_DX[role]
                )
                unique_gates.add(gate)
                role_count += 1
                for lane in range(HEAD_DIM):
                    if (k_value >> lane) & 1:
                        for out in range(OUT_DIM):
                            expected_acc[out_i, destination, out] += (
                                gate * int(input_weights[out_i, lane, out])
                            )
            if lane_count and unique_gates:
                expected_active[out_i] += 1
                expected_terms[out_i] += lane_count * len(unique_gates)
                expected_updates[out_i] += lane_count * role_count
        row = dict(meta)
        row["input_group_index"] = input_group
        row["vector_group_index"] = out_i
        selection.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "input_q": write_memh(args.output_dir / "input_q.memh", input_q, 32),
        "input_candidate_k": write_memh(
            args.output_dir / "input_candidate_k.memh", input_k, 160
        ),
        "input_valid": write_memh(args.output_dir / "input_valid.memh", input_valid, 5),
        "expected_scores": write_memh(
            args.output_dir / "expected_scores.memh", expected_scores, 80
        ),
        "expected_gates": write_memh(
            args.output_dir / "expected_gates.memh", expected_gates, 45
        ),
        "input_weights": write_memh(
            args.output_dir / "input_weights.memh", input_weights, 8
        ),
        "expected_acc": write_memh(
            args.output_dir / "expected_acc.memh", expected_acc, 32
        ),
        "expected_active": write_memh(
            args.output_dir / "expected_active.memh", expected_active, 16
        ),
        "expected_terms": write_memh(
            args.output_dir / "expected_terms.memh", expected_terms, 32
        ),
        "expected_updates": write_memh(
            args.output_dir / "expected_updates.memh", expected_updates, 32
        ),
    }
    output = {
        "schema": "local5_score_projection_vectors_v1",
        "evidence": (
            f"sample{args.sample} S{args.stage}B{args.block} window{window} "
            f"{heads}-head Q/K through score to Acc32"
        ),
        "sample": args.sample,
        "stage": args.stage,
        "block": args.block,
        "selection": {"groups": heads, "rows": selection},
        "shape": {
            "height": 15,
            "width": 15,
            "planes": 2,
            "sources": SOURCES,
            "head_dim": HEAD_DIM,
            "out_dim": OUT_DIM,
        },
        "profile_window": window,
        "artifacts": artifacts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS window projection vectors heads={heads} window={window}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
