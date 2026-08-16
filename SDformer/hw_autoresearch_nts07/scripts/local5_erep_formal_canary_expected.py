#!/usr/bin/env python3
"""为 Local5 正式来源隔离 canary 独立生成软件 Acc32 金参考。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
TOKENS = 450
HEAD_DIM = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_window_groups(
    manifest: dict[str, Any], sample: int, stage: int, block: int
) -> tuple[int, list[tuple[int, dict[str, Any]]]]:
    matches = [
        (index, group)
        for index, group in enumerate(manifest["groups"])
        if (group["sample"], group["stage"], group["block"])
        == (sample, stage, block)
    ]
    if not matches:
        raise ValueError("canary坐标在正式manifest中不存在")
    heads = int(matches[0][1]["heads"])
    windows = {int(group["window"]) for _, group in matches}
    by_head: dict[int, tuple[int, dict[str, Any]]] = {}
    for group_index, group in matches:
        head = int(group["head"])
        if int(group["heads"]) != heads or head in by_head:
            raise ValueError("canary input-head重复或heads字段不一致")
        by_head[head] = (group_index, group)
    if (
        len(matches) != heads
        or len(windows) != 1
        or sorted(by_head) != list(range(heads))
    ):
        raise ValueError("canary必须是同窗全部input-head且按head顺序")
    return windows.pop(), [by_head[head] for head in range(heads)]


def destination_item_coefficients(
    payload: np.lib.npyio.NpzFile,
    start: int,
    stop: int,
) -> np.ndarray:
    modes = np.asarray(payload["item_mode_multiset"][start:stop])
    gates = np.asarray(payload["item_gate_code"][start:stop])
    lanes = np.asarray(payload["item_lane_id"][start:stop])
    multiplicities = np.asarray(payload["item_multiplicity"][start:stop])
    destinations = np.asarray(payload["item_destination"][start:stop])
    if (
        np.any(modes != 1)
        or np.any(gates <= 0)
        or np.any(gates >= 1 << 9)
        or np.any(lanes >= HEAD_DIM)
        or np.any(multiplicities < 1)
        or np.any(multiplicities > 5)
        or np.any(destinations >= TOKENS)
    ):
        raise ValueError("destination-major producer item越界")
    coefficient = np.zeros((TOKENS, HEAD_DIM), dtype=np.int64)
    np.add.at(
        coefficient,
        (destinations.astype(np.int64), lanes.astype(np.int64)),
        gates.astype(np.int64) * multiplicities.astype(np.int64),
    )
    return coefficient


def destination_item_accumulate(
    payload: np.lib.npyio.NpzFile,
    start: int,
    stop: int,
    weight: np.ndarray,
) -> np.ndarray:
    if weight.shape != (HEAD_DIM, HEAD_DIM):
        raise ValueError("weight形状不满足32x32合同")
    acc = destination_item_coefficients(payload, start, stop) @ weight.T
    if np.any(acc < np.iinfo(np.int32).min) or np.any(
        acc > np.iinfo(np.int32).max
    ):
        raise OverflowError("canary Acc32溢出")
    return acc


def build_expected(
    profile: Path, sample: int, stage: int, block: int
) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = profile / "ordered_term_manifest.json"
    payload_path = profile / "ordered_term_items.npz"
    projection_json = profile / "checkpoint_projection_contract.json"
    projection_npz = profile / "checkpoint_projection_contract.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(projection_json.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "et3_ordered_term_trace_v2"
        or (manifest.get("qualification") or {}).get("qualified") is not True
        or contract.get("status") != "THETA_FOLDED_WEIGHT_CONTRACT"
    ):
        raise ValueError("正式manifest/projection合同未准入")
    window, groups = select_window_groups(manifest, sample, stage, block)
    heads = len(groups)
    block_rows = [
        row
        for row in contract["blocks"]
        if (row["stage"], row["block"]) == (stage, block)
    ]
    if len(block_rows) != 1 or int(block_rows[0]["heads"]) != heads:
        raise ValueError("projection block/head合同不一致")
    prefix = str(block_rows[0]["prefix"])
    tasks = [
        {"input_group_index": group_index, "output_tile": tile}
        for tile in range(heads)
        for group_index, _ in groups
    ]
    with np.load(payload_path, allow_pickle=False) as payload, np.load(
        projection_npz, allow_pickle=False
    ) as projection:
        offsets = payload["group_offsets"]
        matrix = np.asarray(projection[f"{prefix}_weight_int8"], dtype=np.int64)
        if matrix.shape != (heads * HEAD_DIM, heads * HEAD_DIM):
            raise ValueError("projection矩阵shape不匹配")
        coefficient = np.zeros(
            (TOKENS, heads * HEAD_DIM), dtype=np.int64
        )
        for group_index, group in groups:
            input_head = int(group["head"])
            input_slice = slice(
                input_head * HEAD_DIM, (input_head + 1) * HEAD_DIM
            )
            coefficient[:, input_slice] = destination_item_coefficients(
                payload,
                int(offsets[group_index]),
                int(offsets[group_index + 1]),
            )
        expected_channels = coefficient @ matrix.T
        expected = expected_channels.reshape(
            TOKENS, heads, HEAD_DIM
        ).transpose(1, 0, 2)
    if np.any(expected < np.iinfo(np.int32).min) or np.any(
        expected > np.iinfo(np.int32).max
    ):
        raise OverflowError("跨head canary Acc32溢出")
    plan = {
        "schema": "local5_projection_task_plan_v1",
        "scope": "formal_source_isolation_canary_not_g0",
        "sample": sample,
        "stage": stage,
        "block": block,
        "window": window,
        "heads": heads,
        "out_dim": HEAD_DIM,
        "tasks": tasks,
        "source_manifest_sha256": sha256(manifest_path),
        "source_payload_sha256": sha256(payload_path),
        "projection_contract_sha256": sha256(projection_json),
        "projection_payload_sha256": sha256(projection_npz),
    }
    plan["task_sha256"] = canonical_sha(tasks)
    return plan, expected.astype(np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan, expected = build_expected(
        args.profile.resolve(), args.sample, args.stage, args.block
    )
    plan_path = args.output_dir / "task_plan.json"
    expected_path = args.output_dir / "software_expected.npz"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez(
        expected_path,
        schema_version=np.asarray([1], dtype=np.uint16),
        expected_acc32=expected.reshape(-1),
    )
    receipt = {
        "schema": "local5_erep_formal_canary_software_expected_v1",
        "status": "PASS_CANARY_NOT_G0",
        "evidence": "[软件整数金参考]",
        "task_plan_sha256": sha256(plan_path),
        "software_expected_sha256": sha256(expected_path),
        "expected_shape": list(expected.shape),
        "expected_scalar_count": int(expected.size),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "formal_g0": "DENY",
        "oracle_path": (
            "producer destination-major item_*直接累加；不使用descriptor方向映射"
        ),
    }
    (args.output_dir / "software_expected_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
