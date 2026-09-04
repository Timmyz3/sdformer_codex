#!/usr/bin/env python3
"""Build Local5 raw-Q/K-to-projection vectors from a frozen post-score cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

if __package__:
    from .generate_local5_checkpoint_score_vectors import (
        reconstruct_destination_rows,
    )
    from .generate_local5_active_projection_postg0_vectors import write_memh
else:
    from generate_local5_checkpoint_score_vectors import (
        reconstruct_destination_rows,
    )
    from generate_local5_active_projection_postg0_vectors import write_memh


HEIGHT = 15
WIDTH = 15
PLANES = 2
SOURCES = HEIGHT * WIDTH * PLANES
ROLES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_memh(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = [int(line.strip(), 16) for line in path.read_text().splitlines()]
    expected = int(np.prod(shape))
    if len(values) != expected:
        raise ValueError(f"{path} entries={len(values)} expected={expected}")
    return np.asarray(values, dtype=object).reshape(shape)


def pack_fields(values: list[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for index, value in enumerate(values):
        packed |= (int(value) & mask) << (index * width)
    return packed


def verify_source_binding(path: Path, expected_sha: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在: {path}")
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{label} SHA漂移: expected={expected_sha} actual={actual_sha}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postscore-vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parent_manifest_path = args.postscore_vector_dir / "manifest.json"
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent.get("schema") != "local5_active_projection_postg0_vectors_v1":
        raise ValueError("parent不是Local5 active-projection vector v1")
    shape = parent.get("shape", {})
    if (
        int(shape.get("height", -1)) != HEIGHT
        or int(shape.get("width", -1)) != WIDTH
        or int(shape.get("planes", -1)) != PLANES
        or int(shape.get("sources", -1)) != SOURCES
    ):
        raise ValueError("parent shape不是T450 Local5")

    source_manifest_path = Path(parent["source_manifest"])
    source_payload_path = Path(parent["source_payload"])
    verify_source_binding(
        source_manifest_path,
        parent["source_manifest_sha256"],
        "ordered manifest",
    )
    verify_source_binding(
        source_payload_path,
        parent["source_payload_sha256"],
        "ordered payload",
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema") != "et3_ordered_term_trace_v2":
        raise ValueError("source manifest不是ordered term trace v2")
    score_contract = source_manifest.get("attention_score_trace_contract", {})
    if score_contract.get("id") != "local5_qk_score_shiftmax_trace_v1":
        raise ValueError("source缺少Local5 Q/K score合同")

    selected = parent["selection"]["rows"]
    groups = len(selected)
    payload = np.load(source_payload_path, mmap_mode="r")
    offsets = payload["descriptor_group_offsets"]

    input_q = np.zeros((groups, SOURCES), dtype=np.uint32)
    input_candidate_k = np.empty((groups, SOURCES), dtype=object)
    input_valid = np.zeros((groups, SOURCES), dtype=np.uint8)
    expected_scores = np.empty((groups, SOURCES), dtype=object)
    expected_gates = np.empty((groups, SOURCES), dtype=object)

    parent_k = read_memh(
        args.postscore_vector_dir / parent["artifacts"]["input_k"]["file"],
        (groups, SOURCES),
    )
    parent_valid = read_memh(
        args.postscore_vector_dir / parent["artifacts"]["input_valid"]["file"],
        (groups, SOURCES),
    )
    parent_gates = read_memh(
        args.postscore_vector_dir / parent["artifacts"]["input_gates"]["file"],
        (groups, SOURCES),
    )

    for output_group, metadata in enumerate(selected):
        input_group = int(metadata["input_group_index"])
        start = int(offsets[input_group])
        stop = int(offsets[input_group + 1])
        if stop - start != SOURCES:
            raise ValueError(f"group {input_group}不是450 descriptors")
        source_ids = np.asarray(payload["descriptor_source_id"][start:stop])
        if not np.array_equal(source_ids, np.arange(SOURCES, dtype=source_ids.dtype)):
            raise ValueError(f"group {input_group} source id不是0..449")
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
        for destination, (q_value, k_values, valid_mask, scores, gates) in enumerate(rows):
            input_q[output_group, destination] = np.uint32(q_value)
            input_candidate_k[output_group, destination] = pack_fields(k_values, 32)
            input_valid[output_group, destination] = np.uint8(valid_mask)
            leaf_scores = [
                score if (valid_mask >> role) & 1 else -256
                for role, score in enumerate(scores)
            ]
            expected_scores[output_group, destination] = pack_fields(
                leaf_scores, 16
            )
            expected_gates[output_group, destination] = pack_fields(gates, 9)
            if int(parent_k[output_group, destination]) != int(k_values[0]):
                raise AssertionError("self K与post-score向量不一致")
            if int(parent_valid[output_group, destination]) != int(valid_mask):
                raise AssertionError("candidate valid与post-score向量不一致")
            if int(parent_gates[output_group, destination]) != pack_fields(gates, 9):
                raise AssertionError("gate与post-score向量不一致")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "input_q": write_memh(args.output_dir / "input_q.memh", input_q, 32),
        "input_candidate_k": write_memh(
            args.output_dir / "input_candidate_k.memh", input_candidate_k, 160
        ),
        "input_valid": write_memh(
            args.output_dir / "input_valid.memh", input_valid, 5
        ),
        "expected_scores": write_memh(
            args.output_dir / "expected_scores.memh", expected_scores, 80
        ),
        "expected_gates": write_memh(
            args.output_dir / "expected_gates.memh", expected_gates, 45
        ),
    }
    for logical_name in (
        "input_weights",
        "expected_active",
        "expected_terms",
        "expected_updates",
        "expected_acc",
    ):
        source_name = parent["artifacts"][logical_name]["file"]
        source_path = args.postscore_vector_dir / source_name
        if sha256(source_path) != parent["artifacts"][logical_name]["sha256"]:
            raise ValueError(f"parent artifact漂移: {logical_name}")
        target_path = args.output_dir / source_name
        shutil.copy2(source_path, target_path)
        artifacts[logical_name] = dict(parent["artifacts"][logical_name])

    output = {
        "schema": "local5_score_projection_vectors_v1",
        "evidence": "profile-qualified raw Q/K through score/Shiftmax5 to Acc",
        "parent_vector_manifest": str(parent_manifest_path.resolve()),
        "parent_vector_manifest_sha256": sha256(parent_manifest_path),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_payload": str(source_payload_path.resolve()),
        "source_payload_sha256": sha256(source_payload_path),
        "selection": parent["selection"],
        "shape": parent["shape"],
        "weight_mode": parent["weight_mode"],
        "weight_contract": parent["weight_contract"],
        "projection_contract_binding": parent["projection_contract_binding"],
        "score_contract": {
            "score": "alpha-XNOR Q7 alpha0=1/64 RNE",
            "shiftmax": "masked Q8 LUT ceil-pow2 Q1.7 RNE",
            "candidate_order": ["self", "up", "down", "left", "right"],
            "postscore_gate_zero_mismatch": True,
        },
        "artifacts": artifacts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"groups": groups, "rows": groups * SOURCES}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
