#!/usr/bin/env python3
"""Generate checkpoint-bound Local5 score/Shiftmax vectors from post-G0 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .generate_local5_masked_integer_vectors import (
        masked_shiftmax_q17,
        score_q7,
    )
else:
    from generate_local5_masked_integer_vectors import (
        masked_shiftmax_q17,
        score_q7,
    )


HEIGHT = 15
WIDTH = 15
PLANES = 2
TOKENS = PLANES * HEIGHT * WIDTH
ROLES = 5
ROLE_DY = (0, 1, -1, 0, 0)
ROLE_DX = (0, 0, 0, 1, -1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_groups(groups: list[dict[str, object]], per_stage: int) -> list[int]:
    selected: list[int] = []
    for stage in range(4):
        candidates = [
            index for index, row in enumerate(groups) if int(row["stage"]) == stage
        ]
        if len(candidates) < per_stage:
            raise ValueError(f"stage {stage}只有{len(candidates)}组")
        positions = np.linspace(0, len(candidates) - 1, per_stage, dtype=np.int64)
        selected.extend(candidates[int(position)] for position in positions)
    return selected


def reconstruct_destination_rows(
    *,
    q_bitmap: np.ndarray,
    k_bitmap: np.ndarray,
    planes: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
    incoming_gates: np.ndarray,
    incoming_valid_mask: np.ndarray,
) -> list[tuple[int, list[int], int, list[int], list[int]]]:
    valid_by_dest = np.zeros(TOKENS, dtype=np.uint8)
    k_by_dest = np.zeros((TOKENS, ROLES), dtype=np.uint32)
    gate_by_dest = np.zeros((TOKENS, ROLES), dtype=np.uint16)
    for source in range(TOKENS):
        plane = int(planes[source])
        sy = int(source_y[source])
        sx = int(source_x[source])
        for role in range(ROLES):
            if not ((int(incoming_valid_mask[source]) >> role) & 1):
                continue
            dy = sy + ROLE_DY[role]
            dx = sx + ROLE_DX[role]
            if not 0 <= dy < HEIGHT or not 0 <= dx < WIDTH:
                raise ValueError("有效source-consumer relation越界")
            destination = plane * HEIGHT * WIDTH + dy * WIDTH + dx
            if (int(valid_by_dest[destination]) >> role) & 1:
                raise ValueError("destination role重构冲突")
            valid_by_dest[destination] |= np.uint8(1 << role)
            k_by_dest[destination, role] = np.uint32(k_bitmap[source])
            gate_by_dest[destination, role] = np.uint16(incoming_gates[source, role])

    rows: list[tuple[int, list[int], int, list[int], list[int]]] = []
    for destination in range(TOKENS):
        q_value = int(q_bitmap[destination])
        k_values = [int(value) for value in k_by_dest[destination]]
        valid_mask = int(valid_by_dest[destination])
        valid = [(valid_mask >> role) & 1 for role in range(ROLES)]
        scores = [score_q7(q_value, value) for value in k_values]
        gates = masked_shiftmax_q17(scores, valid)
        recorded = [int(value) for value in gate_by_dest[destination]]
        if gates != recorded:
            raise ValueError(
                f"真实gate与独立整数参考不一致 destination={destination}: "
                f"trace={recorded} reference={gates}"
            )
        rows.append((q_value, k_values, valid_mask, scores, gates))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-stage", type=int, default=25)
    args = parser.parse_args()

    manifest_path = args.input_dir / "ordered_term_manifest.json"
    payload_path = args.input_dir / "ordered_term_items.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "et3_ordered_term_trace_v2":
        raise ValueError("输入不是ordered term trace v2")
    if manifest.get("evidence_level") != "post_g0":
        raise ValueError("输入不是post-G0 trace")
    if not manifest.get("qualification", {}).get("qualified"):
        raise ValueError("输入trace未通过qualification")
    contract = manifest.get("attention_score_trace_contract", {})
    if contract.get("id") != "local5_qk_score_shiftmax_trace_v1":
        raise ValueError("输入缺少Local5 Q/K score trace合同")

    payload = np.load(payload_path, mmap_mode="r")
    if "descriptor_q_bitmap" not in payload.files:
        raise ValueError("输入payload缺少descriptor_q_bitmap")
    offsets = payload["descriptor_group_offsets"]
    selected = select_groups(manifest["groups"], args.per_stage)
    vector_path = args.output_dir / "local5_checkpoint_score_vectors.txt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_count = 0
    selection_rows: list[dict[str, object]] = []
    with vector_path.open("w", encoding="ascii") as handle:
        for selected_index, input_group in enumerate(selected):
            start = int(offsets[input_group])
            stop = int(offsets[input_group + 1])
            if stop - start != TOKENS:
                raise ValueError("每组必须有450个descriptor")
            source_ids = np.asarray(payload["descriptor_source_id"][start:stop])
            if not np.array_equal(source_ids, np.arange(TOKENS, dtype=source_ids.dtype)):
                raise ValueError("source id不是0..449")
            rows = reconstruct_destination_rows(
                q_bitmap=np.asarray(payload["descriptor_q_bitmap"][start:stop]),
                k_bitmap=np.asarray(payload["descriptor_k_bitmap"][start:stop]),
                planes=np.asarray(payload["descriptor_source_plane"][start:stop]),
                source_y=np.asarray(payload["descriptor_source_y"][start:stop]),
                source_x=np.asarray(payload["descriptor_source_x"][start:stop]),
                incoming_gates=np.asarray(
                    payload["descriptor_incoming_gates"][start:stop]
                ),
                incoming_valid_mask=np.asarray(
                    payload["descriptor_valid_mask"][start:stop]
                ),
            )
            for q_value, k_values, valid_mask, scores, gates in rows:
                fields = [f"{q_value:08x}"]
                fields.extend(f"{value:08x}" for value in k_values)
                fields.append(f"{valid_mask:02x}")
                fields.extend(f"{value & 0xffff:04x}" for value in scores)
                fields.extend(f"{value:03x}" for value in gates)
                handle.write(" ".join(fields) + "\n")
            metadata = dict(manifest["groups"][input_group])
            metadata.update(
                {
                    "input_group_index": input_group,
                    "selected_group_index": selected_index,
                    "vector_start": vector_count,
                    "vector_count": len(rows),
                }
            )
            selection_rows.append(metadata)
            vector_count += len(rows)

    output = {
        "schema": "local5_checkpoint_score_vectors_v1",
        "evidence_scope": "checkpoint_bound_post_g0_qk_score_shiftmax",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "source_payload": str(payload_path.resolve()),
        "source_payload_sha256": sha256(payload_path),
        "vector_file": vector_path.name,
        "vector_sha256": sha256(vector_path),
        "vector_count": vector_count,
        "selection": {
            "method": "per-stage evenly spaced deterministic groups",
            "per_stage": args.per_stage,
            "groups": len(selected),
            "rows": selection_rows,
        },
        "geometry": {
            "planes": PLANES,
            "height": HEIGHT,
            "width": WIDTH,
            "tokens": TOKENS,
            "roles": ["self", "up", "down", "left", "right"],
        },
        "independent_reference": {
            "score": "alpha-XNOR Q7 alpha0=1/64 RNE",
            "shiftmax": "masked Q8 LUT ceil-pow2 Q1.7 RNE",
            "trace_gate_zero_mismatch": True,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"vectors": vector_count, "groups": len(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
