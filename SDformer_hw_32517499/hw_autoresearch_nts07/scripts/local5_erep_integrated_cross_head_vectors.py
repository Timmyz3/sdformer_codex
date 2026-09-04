#!/usr/bin/env python3
"""为 Local5 集成跨头 formal canary 生成真实 Q/K 与投影权重。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .generate_local5_checkpoint_score_vectors import (
        reconstruct_destination_rows,
    )
else:
    from generate_local5_checkpoint_score_vectors import (
        reconstruct_destination_rows,
    )


TOKENS = 450
HEAD_DIM = 32
HELPERS = (
    Path(__file__).with_name("generate_local5_checkpoint_score_vectors.py"),
    Path(__file__).with_name("generate_local5_masked_integer_vectors.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plan(
    plan: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    if plan.get("schema") != "local5_projection_task_plan_v1":
        raise ValueError("task plan schema不合法")
    heads = int(plan.get("heads", 0))
    tasks = plan.get("tasks")
    if heads <= 0 or not isinstance(tasks, list) or len(tasks) != heads * heads:
        raise ValueError("task plan必须完整覆盖HxH")
    expected = [
        (head, tile) for tile in range(heads) for head in range(heads)
    ]
    observed: list[tuple[int, int]] = []
    groups_by_head: dict[int, tuple[int, dict[str, Any]]] = {}
    for task in tasks:
        group_index = int(task["input_group_index"])
        tile = int(task["output_tile"])
        group = manifest["groups"][group_index]
        head = int(group["head"])
        observed.append((head, tile))
        previous = groups_by_head.setdefault(head, (group_index, group))
        if previous[0] != group_index:
            raise ValueError("同一input head映射到多个group")
    if observed != expected or sorted(groups_by_head) != list(range(heads)):
        raise ValueError("task plan不是tile-major/head-major完整HxH顺序")
    groups = [groups_by_head[head] for head in range(heads)]
    identities = {
        (int(group["sample"]), int(group["stage"]), int(group["block"]),
         int(group["window"]))
        for _, group in groups
    }
    if len(identities) != 1:
        raise ValueError("input heads不属于同一个window")
    return groups, heads


def write_head_inputs(
    path: Path,
    payload: np.lib.npyio.NpzFile,
    offsets: np.ndarray,
    group_index: int,
) -> int:
    start = int(offsets[group_index])
    stop = int(offsets[group_index + 1])
    if stop - start != TOKENS:
        raise ValueError("formal group不是T450")
    source_ids = np.asarray(payload["descriptor_source_id"][start:stop])
    if not np.array_equal(source_ids, np.arange(TOKENS, dtype=source_ids.dtype)):
        raise ValueError("descriptor source id不是0..449")
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
    with path.open("w", encoding="ascii") as handle:
        for token, (q_value, k_values, valid_mask, _, _) in enumerate(rows):
            plane, spatial = divmod(token, 225)
            y, x = divmod(spatial, 15)
            fields = [str(plane), str(y), str(x), f"{q_value:08x}"]
            fields.extend(f"{value:08x}" for value in k_values)
            fields.append(f"{valid_mask:02x}")
            handle.write(" ".join(fields) + "\n")
    return len(rows)


def write_weights(
    path: Path,
    matrix: np.ndarray,
    heads: int,
) -> int:
    expected_shape = (heads * HEAD_DIM, heads * HEAD_DIM)
    if matrix.shape != expected_shape:
        raise ValueError(f"projection矩阵shape错误: {matrix.shape} != {expected_shape}")
    count = 0
    with path.open("w", encoding="ascii") as handle:
        for head in range(heads):
            input_base = head * HEAD_DIM
            for tile in range(heads):
                output_base = tile * HEAD_DIM
                for lane in range(HEAD_DIM):
                    for out_index in range(HEAD_DIM):
                        value = int(matrix[output_base + out_index, input_base + lane])
                        if not -128 <= value <= 127:
                            raise ValueError("projection权重越过INT8")
                        handle.write(
                            f"{head} {tile} {lane} {out_index} "
                            f"{value & 0xff:02x}\n"
                        )
                        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    profile = args.profile.resolve()
    plan_path = args.task_plan.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = profile / "ordered_term_manifest.json"
    payload_path = profile / "ordered_term_items.npz"
    projection_json = profile / "checkpoint_projection_contract.json"
    projection_npz = profile / "checkpoint_projection_contract.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    contract = json.loads(projection_json.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "et3_ordered_term_trace_v2"
        or (manifest.get("qualification") or {}).get("qualified") is not True
        or contract.get("status") != "THETA_FOLDED_WEIGHT_CONTRACT"
        or sha256(manifest_path) != plan.get("source_manifest_sha256")
        or sha256(payload_path) != plan.get("source_payload_sha256")
        or sha256(projection_json) != plan.get("projection_contract_sha256")
        or sha256(projection_npz) != plan.get("projection_payload_sha256")
    ):
        raise ValueError("formal输入或task plan来源绑定失效")
    groups, heads = validate_plan(plan, manifest)
    stage = int(groups[0][1]["stage"])
    block = int(groups[0][1]["block"])
    block_rows = [
        row for row in contract["blocks"]
        if (int(row["stage"]), int(row["block"])) == (stage, block)
    ]
    if len(block_rows) != 1 or int(block_rows[0]["heads"]) != heads:
        raise ValueError("projection block/head合同不一致")
    prefix = str(block_rows[0]["prefix"])

    files: dict[str, dict[str, Any]] = {}
    with np.load(payload_path, allow_pickle=False) as payload:
        offsets = payload["descriptor_group_offsets"]
        for head, (group_index, _) in enumerate(groups):
            path = out / f"head{head}_inputs.txt"
            entries = write_head_inputs(path, payload, offsets, group_index)
            files[f"head{head}_inputs"] = {
                "file": path.name,
                "entries": entries,
                "sha256": sha256(path),
                "input_group_index": group_index,
            }
    combined_path = out / "combined_head_inputs.txt"
    combined_entries = 0
    with combined_path.open("w", encoding="ascii") as combined:
        for head in range(heads):
            head_path = out / str(files[f"head{head}_inputs"]["file"])
            for line in head_path.read_text(encoding="ascii").splitlines():
                combined.write(f"{head} {line}\n")
                combined_entries += 1
    files["combined_head_inputs"] = {
        "file": combined_path.name,
        "entries": combined_entries,
        "sha256": sha256(combined_path),
    }
    with np.load(projection_npz, allow_pickle=False) as projection:
        matrix = np.asarray(projection[f"{prefix}_weight_int8"], dtype=np.int16)
        weight_path = out / "projection_weights.txt"
        weight_count = write_weights(weight_path, matrix, heads)
    files["projection_weights"] = {
        "file": weight_path.name,
        "entries": weight_count,
        "sha256": sha256(weight_path),
    }

    output = {
        "schema": "local5_erep_integrated_cross_head_vectors_v1",
        "status": "PASS_CANARY_INPUTS_NOT_G0",
        "evidence": "[prof]+[软件整数重构]",
        "formal_g0": "DENY",
        "task_plan": str(plan_path),
        "task_plan_sha256": sha256(plan_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_payload": str(payload_path),
        "source_payload_sha256": sha256(payload_path),
        "projection_contract_sha256": sha256(projection_json),
        "projection_payload_sha256": sha256(projection_npz),
        "identity": {
            "sample": int(groups[0][1]["sample"]),
            "stage": stage,
            "block": block,
            "window": int(groups[0][1]["window"]),
            "heads": heads,
            "tokens": TOKENS,
            "out_dim": HEAD_DIM,
        },
        "files": files,
        "generator_binding": {
            "file": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
            "helpers": [
                {"file": str(path.resolve()), "sha256": sha256(path)}
                for path in HELPERS
            ],
            "numpy_version": np.__version__,
        },
        "numeric_boundary": (
            "Q/K重构同时逐token验证trace gate等于独立整数score/Shiftmax5；"
            "权重为theta-folded dyadic INT8 checkpoint合同"
        ),
        "forbidden_claims": [
            "不是1200-window formal archive",
            "不是Local5 full-encoder性能结果",
            "不是ASIC PPA",
        ],
    }
    manifest_out = out / "manifest.json"
    manifest_out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
