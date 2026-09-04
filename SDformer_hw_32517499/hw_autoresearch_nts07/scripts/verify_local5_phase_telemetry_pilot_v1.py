#!/usr/bin/env python3
"""独立验证 Local5 H3 phase telemetry pilot，并生成中文机器报告。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "local5_phase_semantic_telemetry_v1"
RECEIPT_SCHEMA = "local5_phase_telemetry_pilot_run_receipt_v1"
EXPECTED_IDENTITY_EVENTS = {
    "group_start": 1,
    "group_done": 1,
    "tile_start": 3,
    "tile_done": 3,
    "head_start": 9,
    "head_done": 9,
    "relation_accept": 4050,
    "relation_response_accept": 4050,
    "weight_accept": 9216,
    "weight_response_accept": 9216,
    "final_accept": 43200,
}
ALIGNED_RESOURCE = {
    "RELATION_REQ_ACCEPT",
    "RELATION_RSP_ACCEPT",
    "WEIGHT_REQ_ACCEPT",
    "WEIGHT_RSP_ACCEPT",
    "FINAL_ACCEPT",
}
DIRECT_ONLY_RESOURCE = {"CROSS_ACC_CMD", "TCFM5_BANK_UPDATE_MASK"}
REQUIRED_PHASE_ROLES = {
    "GROUP_TRANSACTION",
    "TILE_TRANSACTION",
    "HEAD_TRANSACTION",
    "HEAD_WEIGHT",
    "HEAD_FRONTEND",
    "HEAD_READOUT",
    "HEAD_RELEASE",
    "TILE_DRAIN",
}
REQUIRED_BINDINGS = {
    "telemetry", "identity_trace", "actual_acc32", "software_expected",
    "task_plan", "table_manifest", "table_receipt", "vector_manifest",
    "selection_plan", "profile_manifest",
    "release_manifest", "compile_argv", "executable", "run_argv",
    "verilator_log", "monitor_source", "bind_source", "verifier_source",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(rows: Iterable[tuple[Any, ...]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(("|".join(map(str, row)) + "\n").encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Phase:
    sequence: int
    stage: int
    block: int
    window: int
    tile: int
    head: int
    role: str
    start: int
    end: int
    duration: int
    origin: str


@dataclass(frozen=True)
class Resource:
    sequence: int
    stage: int
    block: int
    window: int
    tile: int
    head: int
    resource: str
    cycle: int
    identity0: int
    identity1: int
    identity2: int
    origin: str


def _ints(fields: list[str], indexes: Iterable[int]) -> list[int]:
    try:
        return [int(fields[index]) for index in indexes]
    except (ValueError, IndexError) as exc:
        raise ValueError("telemetry 数字字段非法") from exc


def parse_telemetry(
    path: Path, expected: dict[str, int], *, strict_roles: bool = True
) -> tuple[list[Phase], list[Resource], dict[str, int]]:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 5 or lines[0] != f"SCHEMA,{SCHEMA}":
        raise ValueError("telemetry schema 缺失或错误")
    if lines[1] != "ORIGIN,RTL_DIRECT":
        raise ValueError("telemetry 顶层 origin 不是 RTL_DIRECT")
    if not lines[2].startswith("COLUMNS_P,") or not lines[3].startswith("COLUMNS_R,"):
        raise ValueError("telemetry 冻结列定义缺失")
    phases: list[Phase] = []
    resources: list[Resource] = []
    end_row: list[str] | None = None
    for line_number, line in enumerate(lines[4:], start=5):
        fields = line.split(",")
        if not fields or fields[0] not in {"P", "R", "END"}:
            raise ValueError(f"telemetry 第{line_number}行记录类型非法")
        if end_row is not None:
            raise ValueError("END 后仍有 telemetry 记录")
        if fields[0] == "P":
            if len(fields) != 12:
                raise ValueError("phase 列数非法")
            values = _ints(fields, (1, 2, 3, 4, 5, 6, 8, 9, 10))
            row = Phase(
                values[0], values[1], values[2], values[3], values[4],
                values[5], fields[7], values[6], values[7], values[8], fields[11]
            )
            if row.sequence != len(phases):
                raise ValueError("phase 缺失、重复或乱序")
            if row.origin != "RTL_DIRECT":
                raise ValueError("phase origin 伪造或降级")
            if row.end < row.start or row.duration != row.end - row.start + 1:
                raise ValueError("phase cycle/duration 不一致")
            if phases and row.end < phases[-1].end:
                raise ValueError("phase end cycle 乱序")
            if (row.stage, row.block, row.window) != (
                expected["stage"], expected["block"], expected["window"]
            ):
                raise ValueError("phase 身份 tuple 不一致")
            phases.append(row)
        elif fields[0] == "R":
            if len(fields) != 13:
                raise ValueError("resource 列数非法")
            values = _ints(fields, (1, 2, 3, 4, 5, 6, 8, 9, 10, 11))
            row = Resource(
                values[0], values[1], values[2], values[3], values[4],
                values[5], fields[7], values[6], values[7], values[8],
                values[9], fields[12]
            )
            if row.sequence != len(resources):
                raise ValueError("resource 缺失、重复或乱序")
            if row.origin != "RTL_DIRECT":
                raise ValueError("resource origin 伪造或降级")
            if row.cycle < 0:
                raise ValueError("resource cycle 非法")
            if resources and row.cycle < resources[-1].cycle:
                raise ValueError("resource cycle 乱序")
            if (row.stage, row.block, row.window) != (
                expected["stage"], expected["block"], expected["window"]
            ):
                raise ValueError("resource 身份 tuple 不一致")
            if row.resource not in ALIGNED_RESOURCE | DIRECT_ONLY_RESOURCE:
                raise ValueError("resource code 不在冻结枚举中")
            resources.append(row)
        else:
            if len(fields) != 5 or fields[4] != "RTL_DIRECT":
                raise ValueError("END 记录非法")
            _ints(fields, (1, 2, 3))
            end_row = fields
    if end_row is None:
        raise ValueError("telemetry 缺少 END")
    end_cycle, phase_count, resource_count = map(int, end_row[1:4])
    if phase_count != len(phases) or resource_count != len(resources):
        raise ValueError("END count 与 telemetry 记录不一致")
    if strict_roles:
        observed_roles = {row.role for row in phases}
        if not REQUIRED_PHASE_ROLES <= observed_roles:
            raise ValueError("telemetry 缺少必要 phase role")
    return phases, resources, {
        "end_cycle": end_cycle,
        "phase_count": phase_count,
        "resource_count": resource_count,
    }


def parse_identity_trace(path: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        required = {
            "cycle", "event", "tile", "head", "source", "lane", "out",
            "delay", "index", "origin", "payload",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError("IDENTITY_TRACE schema 不匹配")
        previous_cycle = -1
        for row in reader:
            cycle = int(row["cycle"])
            if cycle < previous_cycle:
                raise ValueError("IDENTITY_TRACE cycle 逆序")
            previous_cycle = cycle
            rows.append(row)
            counts[row["event"]] += 1
    for event, expected_count in EXPECTED_IDENTITY_EVENTS.items():
        if counts[event] != expected_count:
            raise ValueError(f"IDENTITY_TRACE {event} 数量不匹配")
    return rows, counts


def classify_head_state(state: int) -> str | None:
    if state in {1, 2}:
        return "HEAD_WEIGHT"
    if state in {3, 4, 5, 6, 7, 8, 9}:
        return "HEAD_FRONTEND"
    if state in {10, 11, 12}:
        return "HEAD_READOUT"
    if state in {13, 14}:
        return "HEAD_RELEASE"
    if state == 15:
        return "HEAD_ERROR"
    return None


def derive_trace_phases(rows: list[dict[str, str]]) -> list[tuple[Any, ...]]:
    phases: list[tuple[Any, ...]] = []
    group_start: int | None = None
    tile_starts: dict[int, int] = {}
    head_starts: dict[tuple[int, int], int] = {}
    active_head_role: str | None = None
    active_head_start = -1
    active_head_tile = -1
    active_head = -1
    drain_start: int | None = None
    drain_tile = -1
    for row in rows:
        event = row["event"]
        cycle = int(row["cycle"])
        tile = int(row["tile"])
        head = int(row["head"])
        if event == "group_start":
            group_start = cycle
        elif event == "group_done":
            if group_start is None:
                raise ValueError("trace group_done 无 start")
            phases.append((-1, -1, "GROUP_TRANSACTION", group_start, cycle))
        elif event == "tile_start":
            tile_starts[tile] = cycle
        elif event == "tile_done":
            if tile not in tile_starts:
                raise ValueError("trace tile_done 无 start")
            phases.append((tile, -1, "TILE_TRANSACTION", tile_starts.pop(tile), cycle))
        elif event == "head_start":
            head_starts[(tile, head)] = cycle
        elif event == "head_done":
            key = (tile, head)
            if key not in head_starts:
                raise ValueError("trace head_done 无 start")
            phases.append((tile, head, "HEAD_TRANSACTION", head_starts.pop(key), cycle))
        elif event == "head_state":
            role = classify_head_state(int(row["index"]))
            if role != active_head_role:
                if active_head_role is not None:
                    phases.append((active_head_tile, active_head, active_head_role,
                                   active_head_start, cycle - 1))
                active_head_role = role
                if role is not None:
                    active_head_start = cycle
                    active_head_tile = tile
                    active_head = head
        elif event == "tx_state":
            state = int(row["index"])
            in_drain = 4 <= state <= 6
            if in_drain and drain_start is None:
                drain_start = cycle
                drain_tile = tile
            elif not in_drain and drain_start is not None:
                phases.append((drain_tile, -1, "TILE_DRAIN", drain_start, cycle - 1))
                drain_start = None
    if group_start is None or tile_starts or head_starts or active_head_role is not None:
        raise ValueError("trace boundary/state phase 未闭合")
    if drain_start is not None:
        raise ValueError("trace drain phase 未闭合")
    return phases


def _trace_resource(rows: list[dict[str, str]], event: str) -> list[tuple[Any, ...]]:
    output: list[tuple[Any, ...]] = []
    for row in rows:
        if row["event"] != event:
            continue
        cycle = int(row["cycle"])
        tile = int(row["tile"])
        head = int(row["head"])
        source = int(row["source"])
        lane = int(row["lane"])
        out = int(row["out"])
        if event == "relation_accept":
            output.append((cycle, tile, head, source))
        elif event == "relation_response_accept":
            output.append((cycle, head, source))
        elif event in {"weight_accept", "weight_response_accept"}:
            output.append((cycle, tile, head, lane, out))
        elif event == "final_accept":
            output.append((cycle, tile, source, out))
    return output


def align_evidence(
    phases: list[Phase], resources: list[Resource], trace_rows: list[dict[str, str]]
) -> dict[str, Any]:
    telemetry_phases = sorted(
        (p.tile, p.head, p.role, p.start, p.end) for p in phases
    )
    trace_phases = sorted(derive_trace_phases(trace_rows))
    if telemetry_phases != trace_phases:
        raise ValueError("compact phase 与完整 IDENTITY_TRACE 边界不一致")

    mapping = {
        "RELATION_REQ_ACCEPT": "relation_accept",
        "RELATION_RSP_ACCEPT": "relation_response_accept",
        "WEIGHT_REQ_ACCEPT": "weight_accept",
        "WEIGHT_RSP_ACCEPT": "weight_response_accept",
        "FINAL_ACCEPT": "final_accept",
    }
    aligned: dict[str, dict[str, Any]] = {}
    for resource_name, trace_event in mapping.items():
        selected = [r for r in resources if r.resource == resource_name]
        if resource_name == "RELATION_RSP_ACCEPT":
            compact = [(r.cycle, r.head, r.identity0) for r in selected]
        elif resource_name.startswith("RELATION"):
            compact = [(r.cycle, r.tile, r.head, r.identity0) for r in selected]
        elif resource_name.startswith("WEIGHT"):
            compact = [
                (r.cycle, r.tile, r.head, r.identity0, r.identity1)
                for r in selected
            ]
        else:
            compact = [
                (r.cycle, r.tile, r.identity0, r.identity1) for r in selected
            ]
        reference = _trace_resource(trace_rows, trace_event)
        if compact != reference:
            raise ValueError(f"{resource_name} 与 IDENTITY_TRACE 逐事件不一致")
        aligned[resource_name] = {
            "event_count": len(compact),
            "ordered_cycle_identity_sha256": canonical_digest(compact),
        }
    return {
        "phase_count": len(phases),
        "phase_order_independent_sha256": canonical_digest(telemetry_phases),
        "aligned_resources": aligned,
    }


def resolve_binding(package: Path, row: dict[str, Any]) -> Path:
    path = Path(str(row.get("path", "")))
    return path if path.is_absolute() else package / path


def verify_bindings(package: Path, receipt: dict[str, Any]) -> dict[str, Path]:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("pilot receipt schema 错误")
    bindings = receipt.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("pilot receipt bindings 缺失")
    if not REQUIRED_BINDINGS <= set(bindings):
        raise ValueError("pilot receipt 缺少必要 binding")
    output: dict[str, Path] = {}
    for name, row in bindings.items():
        if not isinstance(row, dict) or not isinstance(row.get("sha256"), str):
            raise ValueError("pilot receipt binding 格式错误")
        path = resolve_binding(package, row)
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"pilot receipt digest/路径重绑: {name}")
        output[name] = path
    return output


def verify_compile_contract(paths: dict[str, Path]) -> dict[str, Any]:
    release = json.loads(paths["release_manifest"].read_text(encoding="utf-8"))
    compiled = json.loads(paths["compile_argv"].read_text(encoding="utf-8"))
    base = release.get("builds", {}).get("3")
    if not isinstance(base, dict) or base.get("service_mode") != "identity_derived":
        raise ValueError("release H3 不是 identity-derived")
    base_argv = base.get("compile_argv")
    if not isinstance(base_argv, list) or not isinstance(compiled, list):
        raise ValueError("compile argv schema 非法")
    release_dir = paths["release_manifest"].parent
    resolved_base = [
        str((release_dir / item).resolve()) if str(item).startswith("source/")
        else str(item)
        for item in base_argv
    ]
    if len(compiled) != len(resolved_base) + 2:
        raise ValueError("pilot compile argv 未严格旁路扩展 v10")
    mdir_index = resolved_base.index("build/h3/obj")
    for index, item in enumerate(resolved_base):
        if index == mdir_index:
            continue
        if compiled[index] != item:
            raise ValueError("pilot compile argv 改写了 v10 基线参数/源文件")
    if (
        Path(compiled[-2]).resolve() != paths["monitor_source"].resolve()
        or Path(compiled[-1]).resolve() != paths["bind_source"].resolve()
        or "-GUSE_MEMO=0" not in compiled
        or "-GIDENTITY_DERIVED_SERVICE=1" not in compiled
    ):
        raise ValueError("pilot monitor/bind 或 Direct service 模式未冻结")
    return {
        "baseline_release_manifest_sha256": sha256(paths["release_manifest"]),
        "compile_argv_sha256": sha256(paths["compile_argv"]),
        "executable_sha256": sha256(paths["executable"]),
        "passive_extension_files": [compiled[-2], compiled[-1]],
    }


def verify_acc32(actual_path: Path, expected_path: Path) -> dict[str, Any]:
    values: list[int] = []
    for line_number, line in enumerate(
        actual_path.read_text(encoding="ascii").splitlines(), start=1
    ):
        text = line.strip()
        if not text:
            continue
        try:
            raw = int(text, 16)
        except ValueError as exc:
            raise ValueError(f"actual.memh 第{line_number}行非法") from exc
        if not 0 <= raw <= 0xFFFF_FFFF:
            raise ValueError("actual.memh 越过 uint32")
        values.append(raw - (1 << 32) if raw & (1 << 31) else raw)
    actual = np.asarray(values, dtype=np.int64)
    with np.load(expected_path, allow_pickle=False) as archive:
        if "expected_acc32" not in archive.files:
            raise ValueError("software expected 缺少 expected_acc32")
        expected = np.asarray(archive["expected_acc32"], dtype=np.int64).reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError("Acc32 actual/expected shape 不一致")
    delta = actual - expected
    mismatch = int(np.count_nonzero(delta))
    if mismatch:
        raise ValueError(f"Acc32 miter mismatch={mismatch}")
    return {
        "scalars": int(actual.size),
        "mismatch": mismatch,
        "max_abs_error": int(np.max(np.abs(delta))) if delta.size else 0,
        "actual_sha256": sha256(actual_path),
        "expected_sha256": sha256(expected_path),
    }


def verify_identity_contract(
    receipt: dict[str, Any], task_plan: dict[str, Any], run_argv: list[str]
) -> tuple[dict[str, int], dict[str, Any]]:
    actual = receipt.get("actual_identity")
    requested = receipt.get("requested_identity")
    if not isinstance(actual, dict) or not isinstance(requested, dict):
        raise ValueError("pilot identity 合同缺失")
    keys = ("sample", "stage", "block", "window", "heads")
    actual_int = {key: int(actual[key]) for key in keys}
    task_int = {key: int(task_plan[key]) for key in keys}
    if actual_int != task_int:
        raise ValueError("actual identity 与 task plan 不一致")
    plusargs = {arg.split("=", 1)[0]: arg.split("=", 1)[1]
                for arg in run_argv if arg.startswith("+") and "=" in arg}
    for name, key in (
        ("+STAGE_ID", "stage"), ("+BLOCK_ID", "block"),
        ("+WINDOW_ID", "window"), ("+TELEMETRY_STAGE", "stage"),
        ("+TELEMETRY_BLOCK", "block"), ("+TELEMETRY_WINDOW", "window"),
    ):
        if int(plusargs.get(name, -1)) != actual_int[key]:
            raise ValueError(f"run argv {name} 未绑定实际 identity")
    requested_int = {key: int(requested[key]) for key in keys}
    if requested_int != actual_int:
        mismatch_fields = [key for key in keys if requested_int[key] != actual_int[key]]
        raise ValueError(
            "身份 P0: requested identity 与 actual identity 不一致: "
            + ",".join(mismatch_fields)
        )
    if receipt.get("requested_tuple_status") != "MATCH":
        raise ValueError("requested tuple 必须为 MATCH")
    return actual_int, {
        "requested": requested_int,
        "actual": actual_int,
        "status": "MATCH",
        "mismatch_fields": [],
    }


def verify_canonical_provenance(
    paths: dict[str, Path], actual: dict[str, int], task_plan: dict[str, Any]
) -> dict[str, Any]:
    selection = json.loads(paths["selection_plan"].read_text(encoding="utf-8"))
    profile = json.loads(paths["profile_manifest"].read_text(encoding="utf-8"))
    records = [
        row for row in selection.get("records", [])
        if all(int(row.get(key, -1)) == actual[key]
               for key in ("sample", "stage", "block", "window", "heads"))
    ]
    group_indices = sorted({
        int(row["input_group_index"]) for row in task_plan.get("tasks", [])
    })
    groups = profile.get("groups", [])
    if (
        selection.get("schema") != "local5_uniform_joint_window_plan_v1"
        or len(records) != 1
        or task_plan.get("source_manifest_sha256") != sha256(paths["profile_manifest"])
        or len(group_indices) != 3
        or any(index < 0 or index >= len(groups) for index in group_indices)
    ):
        raise ValueError("canonical selection/profile/task provenance 不一致")
    selected = [groups[index] for index in group_indices]
    if sorted(int(row.get("head", -1)) for row in selected) != [0, 1, 2] or any(
        any(int(row.get(key, -1)) != actual[key]
            for key in ("sample", "stage", "block", "window", "heads"))
        for row in selected
    ):
        raise ValueError("profile task groups 与 canonical identity 不一致")
    return {
        "selection_plan_sha256": sha256(paths["selection_plan"]),
        "profile_manifest_sha256": sha256(paths["profile_manifest"]),
        "task_group_indices": group_indices,
        "selected_heads": [0, 1, 2],
    }


def verify_package(package: Path, output_json: Path, output_md: Path) -> dict[str, Any]:
    package = package.resolve()
    receipt_path = package / "run_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("run_receipt.json 缺失")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    paths = verify_bindings(package, receipt)
    compile_contract = verify_compile_contract(paths)
    task_plan = json.loads(paths["task_plan"].read_text(encoding="utf-8"))
    run_argv = json.loads(paths["run_argv"].read_text(encoding="utf-8"))
    actual_identity, requested_audit = verify_identity_contract(
        receipt, task_plan, run_argv
    )
    canonical_provenance = verify_canonical_provenance(
        paths, actual_identity, task_plan
    )
    phases, resources, telemetry_end = parse_telemetry(
        paths["telemetry"], actual_identity
    )
    trace_rows, trace_counts = parse_identity_trace(paths["identity_trace"])
    alignment = align_evidence(phases, resources, trace_rows)
    acc32 = verify_acc32(paths["actual_acc32"], paths["software_expected"])

    phase_counts = Counter(row.role for row in phases)
    resource_counts = Counter(row.resource for row in resources)
    if (
        phase_counts["GROUP_TRANSACTION"] != 1
        or phase_counts["TILE_TRANSACTION"] != 3
        or phase_counts["HEAD_TRANSACTION"] != 9
    ):
        raise ValueError("H3 phase 拓扑不完整")
    if resource_counts["CROSS_ACC_CMD"] <= 0:
        raise ValueError("cross-head 1RW Acc direct telemetry 为空")
    if resource_counts["TCFM5_BANK_UPDATE_MASK"] <= 0:
        raise ValueError("TCFM5 五 bank direct telemetry 为空")
    for row in resources:
        if row.resource == "TCFM5_BANK_UPDATE_MASK" and not 1 <= row.identity2 <= 31:
            raise ValueError("TCFM5 bank update mask 非法")
        if row.resource == "CROSS_ACC_CMD" and not 0 <= row.identity0 < 14400:
            raise ValueError("cross-head Acc 地址越界")

    table_manifest = json.loads(paths["table_manifest"].read_text(encoding="utf-8"))
    table_receipt = json.loads(paths["table_receipt"].read_text(encoding="utf-8"))
    vector_manifest = json.loads(paths["vector_manifest"].read_text(encoding="utf-8"))
    if (
        table_manifest.get("formal_g0") != "DENY"
        or table_receipt.get("formal_g0") != "DENY"
        or table_manifest.get("identity") != actual_identity
        or vector_manifest.get("identity", {}).get("sample") != actual_identity["sample"]
        or vector_manifest.get("identity", {}).get("stage") != actual_identity["stage"]
        or vector_manifest.get("identity", {}).get("block") != actual_identity["block"]
        or vector_manifest.get("identity", {}).get("window") != actual_identity["window"]
        or vector_manifest.get("identity", {}).get("heads") != actual_identity["heads"]
    ):
        raise ValueError("identity service table 与 pilot 身份/G0 边界不一致")

    missing = [
        "epoch_slot_1rw accepted command 未在 Direct cross-head DUT 暴露",
        "FIFO2 push/pop 未在 Direct cross-head DUT 暴露",
        "EREP fill/execute primitive 不属于本 Direct pilot",
        "五个 Acc bank 的逐 bank 地址未导出；仅实测 term_commit 的五位 bank update mask",
        "prepare/drain 的 formal scheduler resource code 尚未映射",
        "sample 不是 RTL 信号；仅由 task-plan、vector 与 table 哈希绑定",
    ]
    report = {
        "schema": "local5_phase_telemetry_pilot_verify_v1",
        "status": "PASS_H3_PHASE_TELEMETRY_PILOT_NOT_G0",
        "evidence": "[rtl-direct]+[完整identity-trace逐事件对齐]",
        "formal_g0": "DENY",
        "identity_audit": requested_audit,
        "canonical_provenance": canonical_provenance,
        "telemetry": {
            "sha256": sha256(paths["telemetry"]),
            "phase_count": len(phases),
            "resource_event_count": len(resources),
            "end_cycle_validation_only": telemetry_end["end_cycle"],
            "phase_role_counts": dict(sorted(phase_counts.items())),
            "resource_counts": dict(sorted(resource_counts.items())),
            "phase_schema_scope": "H3_DIRECT_PILOT_SEMANTIC_RECORDS",
            "formal_phase_schema_records": 462600,
            "equivalent_to_formal_phase_schema": False,
        },
        "identity_trace": {
            "sha256": sha256(paths["identity_trace"]),
            "rows": len(trace_rows),
            "event_counts": dict(sorted(trace_counts.items())),
        },
        "alignment": alignment,
        "acc32_miter": acc32,
        "compile_contract": compile_contract,
        "archive_size": {
            "telemetry_bytes": paths["telemetry"].stat().st_size,
            "identity_trace_bytes": paths["identity_trace"].stat().st_size,
            "telemetry_to_trace_ratio": (
                paths["telemetry"].stat().st_size
                / paths["identity_trace"].stat().st_size
            ),
        },
        "coverage": {
            "covered_rtl_direct": [
                "group/tile/head transaction 边界",
                "head weight/frontend/readout/release 状态 phase",
                "tile drain 状态 phase",
                "relation request/response accepted cycle 与 identity",
                "weight request/response accepted cycle 与 identity",
                "final accepted cycle 与 identity",
                "cross-head 1RW Acc command cycle/address/read-write",
                "TCFM5 term commit cycle/source/lane/五位 bank update mask",
                "43,200 个 Acc32 与独立软件 expected 零失配",
            ],
            "explicitly_missing": missing,
        },
        "negative_test_contract": [
            "phase 缺失", "phase 乱序", "cycle 篡改", "origin 伪造",
            "digest/receipt 重绑",
            "requested/actual identity mismatch",
        ],
        "boundary": [
            "pilot 周期仅为验证环境延迟，不是架构性能",
            "不是 1200-window formal archive",
            "phase_count=52 不是 formal 462600 phase schema",
            "不是 candidate RTL 性能、ASIC PPA、吞吐或能耗结果",
            "formal G0 保持 DENY",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_name(f"{output_json.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_json)
    md = [
        "# Local5 Phase Telemetry Pilot 机器报告",
        "",
        f"- 状态：`{report['status']}`",
        "- 证据：`[rtl-direct]+[完整 identity trace 逐事件对齐]`",
        "- formal G0：`DENY`",
        f"- 实际身份：`sample{actual_identity['sample']}/stage{actual_identity['stage']}/block{actual_identity['block']}/window{actual_identity['window']}/H{actual_identity['heads']}`",
        f"- 请求身份裁决：`{requested_audit['status']}`；不一致字段：`{','.join(requested_audit['mismatch_fields']) or '无'}`",
        f"- phase：`{len(phases)}`；资源 accepted/direct event：`{len(resources)}`",
        f"- phase schema：H3 Direct pilot 仅 `{len(phases)}` 条；formal schema 为 `462600` 条，二者不等价",
        f"- 完整 IDENTITY_TRACE 行数：`{len(trace_rows)}`",
        f"- Acc32 miter：`{acc32['scalars']}` 个，mismatch=`{acc32['mismatch']}`",
        f"- telemetry/完整 trace 字节比：`{report['archive_size']['telemetry_to_trace_ratio']:.4f}`",
        "",
        "## 已真实覆盖",
        "",
    ]
    md.extend(f"- {item}" for item in report["coverage"]["covered_rtl_direct"])
    md.extend(["", "## 显式缺口", ""])
    md.extend(f"- {item}" for item in missing)
    md.extend([
        "", "## 口径边界", "",
        "- 验证周期不能作为架构性能。",
        "- 本 pilot 不能替代 1200-window formal archive。",
        "- `phase_count=52` 仅是局部语义 phase 记录，不是 formal `462600` phase schema。",
        "- 本结果不构成 candidate RTL、ASIC PPA、吞吐或能耗结论。",
        "- formal G0 未改变，保持 `DENY`。",
        "",
    ])
    output_md.write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine-report-md", type=Path, required=True)
    args = parser.parse_args()
    report = verify_package(args.package_dir, args.output, args.machine_report_md)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
