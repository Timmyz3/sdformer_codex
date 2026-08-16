#!/usr/bin/env python3
"""严格解析 Local5 EREP T450 Direct RTL 相序探针日志。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any


PHASE_PREFIX = "EREP_PHASE_V4 "
GROUP_PREFIX = "GROUP "
PHASE_SCHEMA = "local5_erep_t450_phase_v4"
OUTPUT_SCHEMA = "local5_erep_t450_phase_evidence_v4"
PHASE_FIELDS = frozenset(
    {
        "schema",
        "group",
        "first_relation_accept_cycle",
        "last_relation_accept_cycle",
        "execute_begin_cycle",
        "execute_end_cycle",
        "done_cycle",
        "prepare",
        "relation_fill",
        "relation_commit",
        "execute",
        "compute_drain",
        "total",
        "active",
        "terms",
        "updates",
        "term_stall",
        "sram_reads",
        "sram_writes",
    }
)
GROUP_FIELDS = frozenset(
    {
        "backend",
        "new1rw",
        "mode",
        "latency",
        "group",
        "cycles",
        "active",
        "avoided",
        "memory_wait",
        "terms",
        "updates",
        "term_stall",
        "sram_reads",
        "sram_writes",
    }
)
PASS_RE = re.compile(
    r"PASS post-G0 active projection backend=(\d+) latency=(\d+) "
    r"groups=(\d+) total_cycles=(\d+) descriptors=(\d+)"
)
FINISH_RE = re.compile(r"- .+:\d+: Verilog \$finish")
SHAPE = {
    "height": 15,
    "width": 15,
    "planes": 2,
    "sources": 450,
    "head_dim": 32,
    "out_dim": 2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def parse_fields(body: str, *, expected: frozenset[str], line_number: int) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in shlex.split(body):
        if "=" not in token:
            raise ValueError(f"line {line_number}: malformed token {token!r}")
        key, value = token.split("=", 1)
        if key in fields:
            raise ValueError(f"line {line_number}: duplicate key {key!r}")
        fields[key] = value
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(
            f"line {line_number}: exact field schema mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return fields


def decimal_fields(
    fields: dict[str, str], *, exclude: frozenset[str] = frozenset()
) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in fields.items():
        if key in exclude:
            continue
        try:
            result[key] = int(value, 10)
        except ValueError as exc:
            raise ValueError(f"{key} is not a decimal integer: {value!r}") from exc
    return result


def validate_phase(row: dict[str, int], group: int) -> None:
    if row["group"] != group:
        raise ValueError(f"phase group order mismatch: {row['group']} != {group}")
    if any(value < 0 for key, value in row.items() if key != "group"):
        raise ValueError(f"group {group}: negative phase/counter value")
    boundaries = (
        row["first_relation_accept_cycle"],
        row["last_relation_accept_cycle"],
        row["execute_begin_cycle"],
        row["execute_end_cycle"],
        row["done_cycle"],
    )
    if list(boundaries) != sorted(boundaries):
        raise ValueError(f"group {group}: observed interface boundaries are not ordered")
    derived = {
        "prepare": row["first_relation_accept_cycle"] - 1,
        "relation_fill": (
            row["last_relation_accept_cycle"]
            - row["first_relation_accept_cycle"]
            + 1
        ),
        "relation_commit": (
            row["execute_begin_cycle"] - row["last_relation_accept_cycle"]
        ),
        "execute": row["execute_end_cycle"] - row["execute_begin_cycle"],
        "compute_drain": row["done_cycle"] - row["execute_end_cycle"],
        "total": row["done_cycle"],
    }
    for key, value in derived.items():
        if row[key] != value:
            raise ValueError(
                f"group {group}: {key} does not match observed interface boundaries"
            )
    if row["relation_fill"] != SHAPE["sources"]:
        raise ValueError(f"group {group}: fixed T450 relation-fill contract mismatch")
    if row["total"] <= 0 or row["active"] > SHAPE["sources"]:
        raise ValueError(f"group {group}: total/active range mismatch")
    phase_sum = (
        row["prepare"]
        + row["relation_fill"]
        + row["relation_commit"]
        + row["execute"]
        + row["compute_drain"]
    )
    if phase_sum != row["total"]:
        raise ValueError(f"group {group}: phase sum {phase_sum} != total {row['total']}")
    if row["updates"] != row["sram_writes"]:
        raise ValueError(f"group {group}: update/write ledger mismatch")


def parse_log(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    pending: dict[str, int] | None = None
    terminal: tuple[int, int, int, int, int] | None = None
    finish_seen = False

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        if line.startswith(PHASE_PREFIX):
            if terminal is not None or pending is not None:
                raise ValueError(f"line {line_number}: phase row out of order")
            fields = parse_fields(
                line[len(PHASE_PREFIX) :],
                expected=PHASE_FIELDS,
                line_number=line_number,
            )
            if fields.pop("schema") != PHASE_SCHEMA:
                raise ValueError(f"line {line_number}: phase schema mismatch")
            pending = decimal_fields(fields)
            validate_phase(pending, len(rows))
            continue
        if line.startswith(GROUP_PREFIX):
            if terminal is not None or pending is None:
                raise ValueError(f"line {line_number}: GROUP row without preceding phase")
            fields = parse_fields(
                line[len(GROUP_PREFIX) :],
                expected=GROUP_FIELDS,
                line_number=line_number,
            )
            group_row = decimal_fields(fields)
            group = len(rows)
            if any(value < 0 for value in group_row.values()):
                raise ValueError(f"group {group}: GROUP contains a negative counter")
            if (
                group_row["backend"] != 0
                or group_row["new1rw"] != 1
                or group_row["mode"] != 0
                or group_row["latency"] != 1
                or group_row["group"] != group
            ):
                raise ValueError(f"group {group}: Direct/1RW/latency identity mismatch")
            correspondence = {
                "cycles": "total",
                "active": "active",
                "terms": "terms",
                "updates": "updates",
                "term_stall": "term_stall",
                "sram_reads": "sram_reads",
                "sram_writes": "sram_writes",
            }
            for group_key, phase_key in correspondence.items():
                if group_row[group_key] != pending[phase_key]:
                    raise ValueError(
                        f"group {group}: {group_key}/{phase_key} cross-row mismatch"
                    )
            if group_row["avoided"] != SHAPE["sources"] - pending["active"]:
                raise ValueError(f"group {group}: active/avoided ledger mismatch")
            rows.append({**pending, "memory_wait": group_row["memory_wait"]})
            pending = None
            continue
        match = PASS_RE.fullmatch(line)
        if match:
            if terminal is not None or pending is not None or not rows:
                raise ValueError(f"line {line_number}: invalid or duplicate terminal PASS")
            terminal = tuple(int(value) for value in match.groups())
            continue
        if FINISH_RE.fullmatch(line):
            if terminal is None or finish_seen:
                raise ValueError(f"line {line_number}: misplaced or duplicate finish")
            finish_seen = True
            continue
        raise ValueError(f"line {line_number}: unknown output {line!r}")

    if pending is not None or terminal is None or not finish_seen:
        raise ValueError(
            "incomplete phase/GROUP pair, missing terminal PASS, or missing simulator finish"
        )
    backend, latency, groups, total_cycles, descriptors = terminal
    expected_terminal = (
        0,
        1,
        len(rows),
        sum(row["total"] for row in rows),
        sum(row["active"] for row in rows),
    )
    if terminal != expected_terminal:
        raise ValueError(f"terminal ledger mismatch: {terminal} != {expected_terminal}")
    return rows


def validate_manifest(path: Path, rows: list[dict[str, int]]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "local5_active_projection_postg0_vectors_v1"
        or manifest.get("shape") != SHAPE
        or manifest.get("selection", {}).get("method") != "manifest_order_all_groups"
    ):
        raise ValueError("vector manifest schema/shape/selection contract mismatch")
    metadata = manifest.get("selection", {}).get("rows") or []
    manifest_groups = int(manifest.get("selection", {}).get("groups", 0))
    if manifest_groups != len(metadata) or len(metadata) < len(rows):
        raise ValueError("vector manifest has fewer groups than RTL trace")
    for group, (row, meta) in enumerate(zip(rows, metadata, strict=False)):
        if int(meta.get("vector_group_index", -1)) != group:
            raise ValueError(f"group {group}: vector identity mismatch")
        for rtl_key, meta_key in (
            ("active", "active_sources"),
            ("terms", "terms"),
            ("updates", "updates"),
        ):
            if row[rtl_key] != int(meta.get(meta_key, -1)):
                raise ValueError(f"group {group}: manifest {meta_key} mismatch")
    for path_key, hash_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("source_payload", "source_payload_sha256"),
    ):
        source = Path(str(manifest.get(path_key, "")))
        if not source.is_file() or sha256(source) != manifest.get(hash_key):
            raise ValueError(f"vector manifest source binding mismatch: {path_key}")
    source_manifest = json.loads(Path(manifest["source_manifest"]).read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema") != "et3_ordered_term_trace_v2"
        or not source_manifest.get("qualification", {}).get("qualified")
    ):
        raise ValueError("vector manifest source trace is not qualified")

    expected_artifacts = {
        "input_valid": (manifest_groups * 450, 5),
        "input_active": (manifest_groups * 450, 5),
        "input_k": (manifest_groups * 450, 32),
        "input_gates": (manifest_groups * 450, 45),
        "input_weights": (manifest_groups * 32 * 2, 8),
        "expected_acc": (manifest_groups * 450 * 2, 32),
        "expected_active": (manifest_groups, 16),
        "expected_terms": (manifest_groups, 32),
        "expected_updates": (manifest_groups, 32),
    }
    artifacts = manifest.get("artifacts") or {}
    if set(artifacts) != set(expected_artifacts):
        raise ValueError("vector manifest artifact set is not exact")
    for name, (entries, width) in expected_artifacts.items():
        contract = artifacts[name]
        artifact = path.parent / str(contract.get("file", ""))
        if (
            not artifact.is_file()
            or int(contract.get("entries", -1)) != entries
            or int(contract.get("width", -1)) != width
            or line_count(artifact) != entries
            or sha256(artifact) != contract.get("sha256")
        ):
            raise ValueError(f"vector artifact binding mismatch: {name}")
    return manifest


def build_evidence(trace_path: Path, manifest_path: Path) -> dict[str, Any]:
    rows = parse_log(trace_path)
    manifest = validate_manifest(manifest_path, rows)
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "PASS_RTL_CALIBRATION_ONLY",
        "formal_adapter_status": "DENY",
        "evidence": "[rtl校准]",
        "scope": "sampled T450 Direct phase decomposition; not formal profile or EREP admission",
        "measurement_method": "testbench_observation_of_rtl_interface_boundaries",
        "independent_cycle_predictor": False,
        "group_row_is_independent_cross_evidence": False,
        "configuration": {
            **SHAPE,
            "backend": "direct_1rw",
            "relation_read_latency": 1,
            "run_groups": len(rows),
            "manifest_groups": len(manifest["selection"]["rows"]),
        },
        "trace": str(trace_path.resolve()),
        "trace_sha256": sha256(trace_path),
        "vector_manifest": str(manifest_path.resolve()),
        "vector_manifest_sha256": sha256(manifest_path),
        "source_manifest": manifest["source_manifest"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "source_payload": manifest["source_payload"],
        "source_payload_sha256": manifest["source_payload_sha256"],
        "totals": {
            key: sum(row[key] for row in rows)
            for key in (
                "prepare",
                "relation_fill",
                "relation_commit",
                "execute",
                "compute_drain",
                "total",
                "active",
                "terms",
                "updates",
                "term_stall",
                "sram_reads",
                "sram_writes",
                "memory_wait",
            )
        },
        "rows": rows,
        "limits": [
            "仅回放manifest前缀组；当前校准为OUT_DIM=2，不是正式OUT_DIM=32。",
            "相序是TB对RTL接口边界的周期观测，不是RTL内部相位计数器或独立周期预测。",
            "GROUP与phase行来自同一TB/同一perf信号，只作格式兼容和一致性检查。",
            "不含EREP epoch capture/replay、共同向量drain或目标工艺PPA。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = build_evidence(args.trace, args.vector_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
