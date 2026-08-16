#!/usr/bin/env python3
"""从qualified Local5 post-G0 descriptor生成统一投影顶层RTL向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


HEIGHT = 15
WIDTH = 15
PLANES = 2
SOURCES = HEIGHT * WIDTH * PLANES
ROLES = 5
GATE_W = 9
HEAD_DIM = 32
ROLE_DY = (0, 1, -1, 0, 0)
ROLE_DX = (0, 0, 0, 1, -1)


def weight(lane: int, out_index: int) -> int:
    return (lane % 5 + 1) * (1 if out_index == 0 else -2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_memh(path: Path, values: np.ndarray, width: int) -> dict[str, object]:
    mask = (1 << width) - 1
    digits = (width + 3) // 4
    flattened = np.asarray(values).reshape(-1)
    with path.open("w", encoding="ascii") as handle:
        for value in flattened:
            handle.write(f"{int(value) & mask:0{digits}x}\n")
    return {
        "file": path.name,
        "entries": int(flattened.size),
        "width": width,
        "sha256": sha256(path),
    }


def select_groups(
    groups: list[dict[str, object]], per_stage: int, all_groups: bool
) -> list[int]:
    if all_groups:
        return list(range(len(groups)))
    selected: list[int] = []
    for stage in range(4):
        candidates = [index for index, row in enumerate(groups) if row["stage"] == stage]
        if len(candidates) < per_stage:
            raise ValueError(f"stage {stage}只有{len(candidates)}组")
        positions = np.linspace(0, len(candidates) - 1, per_stage, dtype=np.int64)
        selected.extend(candidates[int(position)] for position in positions)
    return selected


def select_sample_disjoint_stage_groups(
    groups: list[dict[str, object]], per_stage: int
) -> list[int]:
    """Select one outcome-independent group per sample, rotating the stage."""

    samples = sorted({int(row["sample"]) for row in groups})
    if len(samples) != per_stage * 4:
        raise ValueError(
            "sample-disjoint-stage要求sample数恰为4*per-stage"
        )
    selected: list[int] = []
    for sample in samples:
        stage = sample % 4
        candidates = [
            index
            for index, row in enumerate(groups)
            if int(row["sample"]) == sample and int(row["stage"]) == stage
        ]
        if not candidates:
            raise ValueError(f"sample {sample} stage {stage}没有候选group")
        # Rotate block/head positions without consulting activity or cycle fields.
        selected.append(candidates[(sample // 4) % len(candidates)])
    return selected


def select_sample_population_weighted_groups(
    groups: list[dict[str, object]], sample_count: int
) -> list[int]:
    """Select one group per sample with stage quotas matching the source cohort."""

    samples = sorted({int(row["sample"]) for row in groups})
    if len(samples) != sample_count:
        raise ValueError("population-weighted selection要求sample-count覆盖全部sample")
    stage_population = Counter(int(row["stage"]) for row in groups)
    if set(stage_population) != set(range(4)):
        raise ValueError("来源group必须覆盖四个stage")
    population_total = sum(stage_population.values())
    exact = {
        stage: sample_count * stage_population[stage] / population_total
        for stage in range(4)
    }
    quotas = {stage: int(exact[stage]) for stage in range(4)}
    remaining = sample_count - sum(quotas.values())
    for stage in sorted(
        range(4), key=lambda item: (exact[item] - quotas[item], -item), reverse=True
    )[:remaining]:
        quotas[stage] += 1

    # A fixed permutation prevents stage assignment from tracking sample order.
    sample_order = sorted(samples, key=lambda sample: ((sample * 37) % 101, sample))
    assigned_stage: dict[int, int] = {}
    cursor = 0
    for stage in range(4):
        for sample in sample_order[cursor : cursor + quotas[stage]]:
            assigned_stage[sample] = stage
        cursor += quotas[stage]
    if len(assigned_stage) != sample_count:
        raise AssertionError("population-weighted stage quota未覆盖全部sample")

    selected: list[int] = []
    for sample in samples:
        stage = assigned_stage[sample]
        candidates = [
            index
            for index, row in enumerate(groups)
            if int(row["sample"]) == sample and int(row["stage"]) == stage
        ]
        if not candidates:
            raise ValueError(f"sample {sample} stage {stage}没有候选group")
        selected.append(candidates[sample % len(candidates)])
    return selected


def load_task_plan(
    path: Path,
    groups: list[dict[str, object]],
    out_dim: int,
) -> tuple[list[int], list[int], dict[str, object]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        plan.get("schema") != "local5_projection_task_plan_v1"
        or not isinstance(plan.get("tasks"), list)
        or not plan["tasks"]
    ):
        raise ValueError("projection task plan schema/tasks不合法")
    selected: list[int] = []
    output_tiles: list[int] = []
    observed: set[tuple[int, int]] = set()
    for index, task in enumerate(plan["tasks"]):
        if not isinstance(task, dict) or set(task) != {
            "input_group_index", "output_tile"
        }:
            raise ValueError(f"task {index}字段集合不合法")
        group_index = task["input_group_index"]
        output_tile = task["output_tile"]
        if (
            type(group_index) is not int
            or type(output_tile) is not int
            or not 0 <= group_index < len(groups)
        ):
            raise ValueError(f"task {index}索引不合法")
        heads = int(groups[group_index]["heads"])
        if not 0 <= output_tile < heads:
            raise ValueError(f"task {index} output tile越界")
        if out_dim != HEAD_DIM:
            raise ValueError("显式output-tile任务当前只允许OUT_DIM=HEAD_DIM=32")
        identity = (group_index, output_tile)
        if identity in observed:
            raise ValueError(f"task {index}重复")
        observed.add(identity)
        selected.append(group_index)
        output_tiles.append(output_tile)
    return selected, output_tiles, {
        "file": str(path.resolve()),
        "sha256": sha256(path),
        "schema": plan["schema"],
        "task_count": len(selected),
    }


def load_checkpoint_projection_contract(
    input_dir: Path,
    trace_manifest: dict[str, object],
    manifest_override: Path | None = None,
) -> tuple[dict[tuple[int, int], dict[str, object]], np.lib.npyio.NpzFile, dict[str, object]]:
    manifest_path = (
        manifest_override.resolve()
        if manifest_override is not None
        else input_dir / str(trace_manifest.get("projection_contract_file", ""))
    )
    contract = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = (
        manifest_path.parent / str(contract.get("payload_file", ""))
        if manifest_override is not None
        else input_dir / str(trace_manifest.get("projection_contract_payload", ""))
    )
    if not manifest_path.is_file() or not payload_path.is_file():
        raise ValueError("trace缺少checkpoint projection contract")
    if manifest_override is None and (
        sha256(manifest_path)
        != trace_manifest.get("projection_contract_file_sha256")
        or sha256(payload_path)
        != trace_manifest.get("projection_contract_payload_sha256")
    ):
        raise ValueError("checkpoint projection contract外层SHA绑定失效")
    if (
        contract.get("schema")
        not in {
            "local5_checkpoint_projection_contract_v1",
            "local5_checkpoint_projection_contract_v2",
        }
        or contract.get("checkpoint") != trace_manifest.get("checkpoint")
        or contract.get("checkpoint_sha256")
        != trace_manifest.get("checkpoint_sha256")
        or contract.get("payload_file") != payload_path.name
        or contract.get("payload_sha256") != sha256(payload_path)
    ):
        raise ValueError("checkpoint projection contract provenance不匹配")
    if contract.get("schema") == "local5_checkpoint_projection_contract_v2":
        if (
            contract.get("status") != "THETA_FOLDED_WEIGHT_CONTRACT"
            or contract.get("quantization_order")
            != "W_eff=theta_K*W_float; quantize_dyadic_int8(W_eff)"
            or any(
                not np.isfinite(float(row.get("theta", float("nan"))))
                or float(row.get("theta", 0.0)) <= 0.0
                for row in contract.get("blocks", [])
            )
        ):
            raise ValueError("theta-folded projection contract语义不完整")
    rows: dict[tuple[int, int], dict[str, object]] = {}
    for row in contract.get("blocks", []):
        key = (int(row["stage"]), int(row["block"]))
        if key in rows:
            raise ValueError(f"projection contract重复block: {key}")
        rows[key] = row
    if len(rows) != 12:
        raise ValueError("projection contract必须覆盖12个block")
    return rows, np.load(payload_path, mmap_mode="r"), {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "payload": str(payload_path.resolve()),
        "payload_sha256": sha256(payload_path),
        "numeric_scope": contract.get("numeric_scope"),
        "bn_folding": contract.get("bn_folding"),
        "schema": contract.get("schema"),
        "status": contract.get("status"),
        "value_contract": contract.get("value_contract"),
        "quantization_order": contract.get("quantization_order"),
        "runtime_datapath": contract.get("runtime_datapath"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/local5_fullres_postg0_qfsa_profile100_20260730"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tb_qfit/vectors/local5_active_projection_postg0_100"),
    )
    parser.add_argument("--per-stage", type=int, default=25)
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="保持 manifest 原顺序导出全部 group，不再做 per-stage 子采样",
    )
    parser.add_argument(
        "--sample-disjoint-stage",
        action="store_true",
        help="每个sample固定选一组，sample mod 4决定stage；不读取活动度或周期",
    )
    parser.add_argument(
        "--sample-population-weighted",
        action="store_true",
        help="每sample一组，stage配额按来源group总体比例做最大余数分配",
    )
    parser.add_argument("--out-dim", type=int, default=2)
    parser.add_argument(
        "--task-plan",
        type=Path,
        help="显式(input_group_index, output_tile)任务；用于正式来源隔离回放",
    )
    parser.add_argument(
        "--omit-expected-acc",
        action="store_true",
        help="NO_ACC_CHECK来源隔离回放不落盘同源expected_acc",
    )
    parser.add_argument(
        "--weight-mode",
        choices=(
            "synthetic",
            "checkpoint_dyadic_int8_head_slice",
            "checkpoint_theta_folded_dyadic_int8_head_slice",
        ),
        default="synthetic",
    )
    parser.add_argument(
        "--projection-contract-manifest",
        type=Path,
        help="显式指定checkpoint合同；仅用于可审计旁路或重放",
    )
    args = parser.parse_args()

    manifest_path = args.input_dir / "ordered_term_manifest.json"
    payload_path = args.input_dir / "ordered_term_items.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "et3_ordered_term_trace_v2":
        raise ValueError("输入不是ordered term trace v2")
    if not manifest.get("qualification", {}).get("qualified"):
        raise ValueError("输入trace未通过qualification")
    payload = np.load(payload_path, mmap_mode="r")
    projection_rows = None
    projection_payload = None
    projection_binding = None
    if args.weight_mode != "synthetic":
        (
            projection_rows,
            projection_payload,
            projection_binding,
        ) = load_checkpoint_projection_contract(
            args.input_dir,
            manifest,
            args.projection_contract_manifest,
        )
        expected_schema = (
            "local5_checkpoint_projection_contract_v2"
            if args.weight_mode
            == "checkpoint_theta_folded_dyadic_int8_head_slice"
            else "local5_checkpoint_projection_contract_v1"
        )
        if projection_binding["schema"] != expected_schema:
            raise ValueError(
                f"weight mode要求{expected_schema}，"
                f"实际{projection_binding['schema']}"
            )
    offsets = payload["descriptor_group_offsets"]
    task_plan_binding = None
    if args.task_plan is not None:
        if (
            args.sample_disjoint_stage
            or args.sample_population_weighted
            or args.all_groups
        ):
            raise ValueError("task-plan不能与其他selection模式并用")
        selected, explicit_output_tiles, task_plan_binding = load_task_plan(
            args.task_plan, manifest["groups"], args.out_dim
        )
    else:
        if (
            int(args.sample_disjoint_stage)
            + int(args.sample_population_weighted)
            + int(args.all_groups)
            > 1
        ):
            raise ValueError("selection模式不能并用")
        if args.sample_population_weighted:
            selected = select_sample_population_weighted_groups(
                manifest["groups"], args.per_stage * 4
            )
        elif args.sample_disjoint_stage:
            selected = select_sample_disjoint_stage_groups(
                manifest["groups"], args.per_stage
            )
        else:
            selected = select_groups(
                manifest["groups"], args.per_stage, args.all_groups
            )
        explicit_output_tiles = [None] * len(selected)
    group_count = len(selected)

    input_valid = np.zeros((group_count, SOURCES), dtype=np.uint8)
    input_active = np.zeros_like(input_valid)
    input_k = np.zeros((group_count, SOURCES), dtype=np.uint32)
    input_gates = np.zeros((group_count, SOURCES), dtype=np.uint64)
    input_weights = np.zeros(
        (group_count, HEAD_DIM, args.out_dim), dtype=np.int8
    )
    expected_acc = np.zeros(
        (group_count, SOURCES, args.out_dim), dtype=np.int64
    )
    expected_active = np.zeros(group_count, dtype=np.uint16)
    expected_terms = np.zeros(group_count, dtype=np.uint32)
    expected_updates = np.zeros(group_count, dtype=np.uint32)

    selection_rows: list[dict[str, object]] = []
    for output_group, (input_group, explicit_output_tile) in enumerate(
        zip(selected, explicit_output_tiles, strict=True)
    ):
        start = int(offsets[input_group])
        stop = int(offsets[input_group + 1])
        if stop - start != SOURCES:
            raise ValueError("每组必须有450个source descriptor")
        source_id = np.asarray(payload["descriptor_source_id"][start:stop])
        if not np.array_equal(source_id, np.arange(SOURCES, dtype=source_id.dtype)):
            raise ValueError(f"group {input_group}的source id不是0..449")
        planes = np.asarray(payload["descriptor_source_plane"][start:stop])
        source_y = np.asarray(payload["descriptor_source_y"][start:stop])
        source_x = np.asarray(payload["descriptor_source_x"][start:stop])
        k_bitmap = np.asarray(payload["descriptor_k_bitmap"][start:stop])
        gates = np.asarray(payload["descriptor_incoming_gates"][start:stop])
        valid_mask = np.asarray(payload["descriptor_valid_mask"][start:stop])
        input_k[output_group] = k_bitmap.astype(np.uint32)
        group_metadata = manifest["groups"][input_group]
        output_channels: list[int] = []
        if args.weight_mode != "synthetic":
            assert projection_rows is not None and projection_payload is not None
            stage = int(group_metadata["stage"])
            block = int(group_metadata["block"])
            head = int(group_metadata["head"])
            projection_row = projection_rows[(stage, block)]
            if int(projection_row["head_dim"]) != HEAD_DIM:
                raise ValueError("checkpoint projection head_dim不是32")
            heads = int(projection_row["heads"])
            if not 0 <= head < heads:
                raise ValueError("trace head超出checkpoint projection范围")
            prefix = str(projection_row["prefix"])
            matrix = np.asarray(projection_payload[f"{prefix}_weight_int8"])
            dim = int(matrix.shape[0])
            if matrix.shape != (dim, dim) or dim != heads * HEAD_DIM:
                raise ValueError("checkpoint projection矩阵shape与head合同不符")
            if explicit_output_tile is None:
                output_channels = [
                    (output_group * args.out_dim + out_index) % dim
                    for out_index in range(args.out_dim)
                ]
            else:
                output_channels = [
                    explicit_output_tile * HEAD_DIM + out_index
                    for out_index in range(args.out_dim)
                ]
            input_start = head * HEAD_DIM
            input_weights[output_group] = matrix[
                np.asarray(output_channels),
                input_start : input_start + HEAD_DIM,
            ].T
        else:
            for lane in range(HEAD_DIM):
                for out_index in range(args.out_dim):
                    input_weights[output_group, lane, out_index] = weight(
                        lane, out_index
                    )

        for source in range(SOURCES):
            plane = int(planes[source])
            sy = int(source_y[source])
            sx = int(source_x[source])
            k_value = int(k_bitmap[source])
            lane_count = k_value.bit_count()
            unique_gates: set[int] = set()
            role_count = 0
            for role in range(ROLES):
                gate = int(gates[source, role])
                if not ((int(valid_mask[source]) >> role) & 1) or gate == 0:
                    continue
                dy = sy + ROLE_DY[role]
                dx = sx + ROLE_DX[role]
                if not (0 <= dy < HEIGHT and 0 <= dx < WIDTH):
                    raise ValueError("有效consumer越界")
                destination = plane * HEIGHT * WIDTH + dy * WIDTH + dx
                if (input_valid[output_group, destination] >> role) & 1:
                    raise ValueError("destination role重构冲突")
                input_valid[output_group, destination] |= np.uint8(1 << role)
                input_gates[output_group, destination] |= np.uint64(
                    gate << (role * GATE_W)
                )
                if k_value != 0:
                    input_active[output_group, destination] |= np.uint8(1 << role)
                unique_gates.add(gate)
                role_count += 1
                for lane in range(HEAD_DIM):
                    if (k_value >> lane) & 1:
                        for out_index in range(args.out_dim):
                            expected_acc[output_group, destination, out_index] += (
                                gate
                                * int(input_weights[output_group, lane, out_index])
                            )
            if lane_count and unique_gates:
                expected_active[output_group] += 1
                expected_terms[output_group] += lane_count * len(unique_gates)
                expected_updates[output_group] += lane_count * role_count

        # Ensure relation inversion reproduces the original descriptor exactly.
        for source in range(SOURCES):
            plane = int(planes[source])
            sy = int(source_y[source])
            sx = int(source_x[source])
            for role in range(ROLES):
                dy = sy + ROLE_DY[role]
                dx = sx + ROLE_DX[role]
                reconstructed_gate = 0
                reconstructed_valid = 0
                if 0 <= dy < HEIGHT and 0 <= dx < WIDTH:
                    destination = plane * HEIGHT * WIDTH + dy * WIDTH + dx
                    reconstructed_valid = (
                        int(input_valid[output_group, destination]) >> role
                    ) & 1
                    reconstructed_gate = (
                        int(input_gates[output_group, destination])
                        >> (role * GATE_W)
                    ) & ((1 << GATE_W) - 1)
                original_valid = (int(valid_mask[source]) >> role) & 1
                original_gate = int(gates[source, role]) if original_valid else 0
                if reconstructed_valid != original_valid or reconstructed_gate != original_gate:
                    raise AssertionError("source/destination关系逆变换不等价")

        metadata = dict(group_metadata)
        metadata.update(
            {
                "input_group_index": input_group,
                "vector_group_index": output_group,
                "active_sources": int(expected_active[output_group]),
                "terms": int(expected_terms[output_group]),
                "updates": int(expected_updates[output_group]),
                "weight_mode": args.weight_mode,
                "projection_output_channels": output_channels,
                "projection_output_tile": explicit_output_tile,
            }
        )
        selection_rows.append(metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "input_valid": write_memh(args.output_dir / "input_valid.memh", input_valid, 5),
        "input_active": write_memh(args.output_dir / "input_active.memh", input_active, 5),
        "input_k": write_memh(args.output_dir / "input_k.memh", input_k, 32),
        "input_gates": write_memh(args.output_dir / "input_gates.memh", input_gates, 45),
        "input_weights": write_memh(
            args.output_dir / "input_weights.memh", input_weights, 8
        ),
        "expected_active": write_memh(args.output_dir / "expected_active.memh", expected_active, 16),
        "expected_terms": write_memh(args.output_dir / "expected_terms.memh", expected_terms, 32),
        "expected_updates": write_memh(args.output_dir / "expected_updates.memh", expected_updates, 32),
    }
    if not args.omit_expected_acc:
        artifacts["expected_acc"] = write_memh(
            args.output_dir / "expected_acc.memh", expected_acc, 32
        )
    output_manifest = {
        "schema": "local5_active_projection_postg0_vectors_v1",
        "evidence": "qualified post-G0 descriptor inversion",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "source_payload": str(payload_path.resolve()),
        "source_payload_sha256": sha256(payload_path),
        "selection": {
            "method": (
                "explicit_projection_task_plan_v1"
                if args.task_plan is not None
                else (
                    "manifest_order_all_groups"
                    if args.all_groups
                    else (
                        "sample-disjoint population-stage-weighted deterministic groups"
                        if args.sample_population_weighted
                        else (
                            "sample-disjoint stage-rotating deterministic groups"
                            if args.sample_disjoint_stage
                            else "per-stage evenly spaced deterministic groups"
                        )
                    )
                )
            ),
            "per_stage": (
                args.per_stage
                if not args.all_groups
                and not args.sample_population_weighted
                and args.task_plan is None
                else None
            ),
            "requested_groups": group_count,
            "stage_counts": {
                str(stage): sum(
                    int(row["stage"]) == stage for row in selection_rows
                )
                for stage in range(4)
            },
            "groups": group_count,
            "rows": selection_rows,
        },
        "task_plan_binding": task_plan_binding,
        "shape": {
            "height": HEIGHT,
            "width": WIDTH,
            "planes": PLANES,
            "sources": SOURCES,
            "head_dim": HEAD_DIM,
            "out_dim": args.out_dim,
        },
        "weight_mode": args.weight_mode,
        "weight_contract": (
            "per-group real checkpoint dyadic INT8 projection head slice; "
            "partial accumulator before cross-head sum/bias/BN/requant"
            if args.weight_mode == "checkpoint_dyadic_int8_head_slice"
            else (
                "per-group theta-folded real checkpoint dyadic INT8 projection "
                "head slice; binary K event; partial accumulator before "
                "cross-head sum/bias/BN/requant"
                if args.weight_mode
                == "checkpoint_theta_folded_dyadic_int8_head_slice"
                else "(lane%5+1)*(out0?1:-2)"
            )
        ),
        "projection_contract_binding": projection_binding,
        "artifacts": artifacts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "groups": group_count,
                "active_sources": int(expected_active.sum()),
                "terms": int(expected_terms.sum()),
                "updates": int(expected_updates.sum()),
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
