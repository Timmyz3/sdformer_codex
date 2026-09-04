#!/usr/bin/env python3
"""Build a fail-closed source-issue cycle envelope from ordered traces.

The model assumes one resident weight segment is read per selected source and
updates ``lanes`` Acc32 destinations in parallel.  It includes a configurable
per-command pipeline cost, but deliberately excludes SRAM conflicts, state
store traffic, DMA, and clock-frequency degradation.  Results are envelopes,
not measured latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PASS = "PASS_EXACT_SOURCE_WORK"


def integer(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing {key} for {row.get('name')}")
    return int(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_exact_row(row: dict[str, str]) -> None:
    fanout = integer(row, "output_channel_fanout")
    current = integer(row, "current_source_count")
    positive = integer(row, "positive_transition_source_count")
    negative = integer(row, "negative_transition_source_count")
    local = integer(row, "local_work")
    motion = integer(row, "motion_work")
    selected = integer(row, "selected_work")
    saved = integer(row, "selector_saved_work")
    selector_rows = integer(row, "selector_rows")
    local_rows = integer(row, "local_selected_rows")
    motion_rows = integer(row, "motion_selected_rows")
    valid = integer(row, "valid_source_work")
    if fanout <= 0:
        raise ValueError("fanout must be positive")
    if local != current * fanout:
        raise ValueError(f"local conservation failed: {row['name']}")
    if motion != (positive + negative) * fanout:
        raise ValueError(f"motion conservation failed: {row['name']}")
    if selected + saved != local or selected > local:
        raise ValueError(f"selector conservation failed: {row['name']}")
    if local_rows + motion_rows != selector_rows:
        raise ValueError(f"selector row partition failed: {row['name']}")
    if valid < local or any(value % fanout for value in (valid, local, motion, selected)):
        raise ValueError(f"source/fanout divisibility failed: {row['name']}")


def infer_linear_geometry(row: dict[str, str]) -> tuple[int, int, int]:
    input_elements = integer(row, "input_elements")
    output_elements = integer(row, "output_elements")
    weights = integer(row, "weight_elements")
    rows_squared = input_elements * output_elements
    if weights <= 0 or rows_squared % weights:
        raise ValueError(f"cannot infer Linear geometry: {row['name']}")
    rows = math.isqrt(rows_squared // weights)
    if rows * rows * weights != rows_squared:
        raise ValueError(f"non-square inferred row count: {row['name']}")
    if input_elements % rows or output_elements % rows:
        raise ValueError(f"non-integral Linear dimensions: {row['name']}")
    return rows, input_elements // rows, output_elements // rows


def build_identity(
    dual_rows: list[dict[str, str]],
    operator_rows: list[dict[str, str]],
    lanes: list[int],
    command_overhead: int,
) -> dict[str, Any]:
    exact = [row for row in dual_rows if row.get("status") == PASS]
    for row in exact:
        validate_exact_row(row)
    qk = [
        row for row in operator_rows
        if ".attn.linear_q" in row.get("name", "")
        or ".attn.linear_k" in row.get("name", "")
    ]
    comparable_operators = len({row["name"] for row in exact})
    sample_ids = {row.get("sample_id", "") for row in exact}
    if comparable_operators == 0:
        raise ValueError("no Motion-comparable operators")
    if not sample_ids or "" in sample_ids:
        raise ValueError("Motion-comparable trace has no complete sample identity")
    if len({row["name"] for row in qk}) != 24:
        raise ValueError("expected 24 Local-only Q/K operators")
    sample_count = len(sample_ids)
    if any(integer(row, "calls") != sample_count for row in qk):
        raise ValueError("Q/K aggregate call count does not match dual-trace sample count")

    configurations = []
    for lane_count in lanes:
        if lane_count <= 0:
            raise ValueError("lane count must be positive")
        dense_cycles = local_cycles = selected_cycles = commands = 0
        for row in exact:
            fanout = integer(row, "output_channel_fanout")
            segments = math.ceil(fanout / lane_count)
            command_count = integer(row, "selector_rows") * segments
            dense_sources = integer(row, "valid_source_work") // fanout
            local_sources = integer(row, "local_work") // fanout
            selected_sources = integer(row, "selected_work") // fanout
            dense_cycles += dense_sources * segments + command_overhead * command_count
            local_cycles += local_sources * segments + command_overhead * command_count
            selected_cycles += selected_sources * segments + command_overhead * command_count
            commands += command_count
        for row in qk:
            linear_rows, _input_width, output_width = infer_linear_geometry(row)
            segments = math.ceil(output_width / lane_count)
            command_count = linear_rows * segments
            dense_cycles += integer(row, "input_elements") * segments + command_overhead * command_count
            qk_local = integer(row, "input_active") * segments + command_overhead * command_count
            local_cycles += qk_local
            selected_cycles += qk_local
            commands += command_count
        configurations.append(
            {
                "lanes": lane_count,
                "commands": commands,
                "dense_cycles": dense_cycles,
                "local_cycles": local_cycles,
                "selected_cycles": selected_cycles,
                "dense_over_local": dense_cycles / local_cycles,
                "dense_over_selected": dense_cycles / selected_cycles,
                "motion_increment_over_local": local_cycles / selected_cycles,
            }
        )
    return {
        "sample_count": sample_count,
        "motion_comparable_operators": comparable_operators,
        "local_only_qk_operators": 24,
        "configurations": configurations,
    }


def add_system_envelope(result: dict[str, Any], ledger_dir: Path) -> None:
    summary = json.loads((ledger_dir / "system_summary.json").read_text(encoding="utf-8"))
    operators = read_csv(ledger_dir / "operator_transactions.csv")
    eligible = [
        row for row in operators
        if row["operator"] in {"Linear", "Conv2d"}
        and row["input_binary_packed_eligible"] == "True"
        and row.get("replaced_by_attention_rtl_anchor", "False") != "True"
    ]
    if len(eligible) != 55:
        raise ValueError("expected 55 Local-eligible ledger operators")
    if result["motion_comparable_operators"] + result["local_only_qk_operators"] != len(eligible):
        raise ValueError("trace/ledger Local-eligible operator coverage mismatch")
    eligible_cycles = sum(integer(row, "activity_cycles_at_config_lanes") for row in eligible)
    fixed = int(summary["cycles_per_frame_model"]["fixed_total"])
    fixed_attention = int(summary["attention"]["fixed_cycles_per_frame"])
    rqtb_attention = int(summary["attention"]["rqtb_cycles_per_frame"])
    frozen = fixed - fixed_attention - eligible_cycles + rqtb_attention
    baseline_lanes = int(summary["config"]["mac_lanes"])
    result["system_mapping"] = {
        "fixed_cycles": fixed,
        "activity_weighted_eligible_cycles_at_baseline_lanes": eligible_cycles,
        "baseline_mac_lanes": baseline_lanes,
        "frozen_cycles_with_rqtb": frozen,
        "mapping_rule": "frozen + ceil(real s10 candidate source-issue cycles / sample_count)",
    }
    for row in result["configurations"]:
        candidate = math.ceil(row["selected_cycles"] / result["sample_count"])
        cycles = frozen + candidate
        row["candidate_eligible_cycles_per_frame"] = candidate
        row["candidate_eligible_speedup_vs_activity_ledger"] = eligible_cycles / candidate
        row["h67_system_cycles_direct_source_issue"] = cycles
        row["h67_system_speedup_direct_source_issue"] = fixed / cycles


def write_report(path: Path, results: dict[str, dict[str, Any]], overhead: int) -> None:
    lines = [
        "# Dual-line source-issue cycle envelope\n\n",
        f"Per command pipeline overhead: `{overhead}` cycles.\n\n",
        "| identity | lanes | dense/local | dense/selected | Motion increment | eligible vs ledger | H67 direct system ratio |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for label, result in results.items():
        for row in result["configurations"]:
            system = row.get("h67_system_speedup_direct_source_issue")
            eligible = row.get("candidate_eligible_speedup_vs_activity_ledger")
            lines.append(
                f"| {label} | {row['lanes']} | {row['dense_over_local']:.6f}x | "
                f"{row['dense_over_selected']:.6f}x | {row['motion_increment_over_local']:.6f}x | "
                f"{eligible:.6f}x | {system:.6f}x |\n" if system is not None else
                f"| {label} | {row['lanes']} | {row['dense_over_local']:.6f}x | "
                f"{row['dense_over_selected']:.6f}x | {row['motion_increment_over_local']:.6f}x | n/a | n/a |\n"
            )
    lines.extend([
        "\nThe H67 system column directly replaces the ledger's already activity-weighted eligible cycles "
        "with real-trace candidate cycles per frame; it never divides those cycles by a sparsity ratio again. "
        "A ratio below 1.0 means the candidate is slower. This is still a source-issue lower bound, not measured latency. "
        "It assumes resident weights, "
        "one segment read per selected source, parallel Acc32 lane updates, and no bank conflict. "
        "The trace makes one command per complete row/output segment; it does not yet multiply commands "
        "for input rows wider than the 256-bit RTL tile, so it remains optimistic. SRAM/DRAM timing, "
        "state-store traffic, and post-DC frequency must still be applied.\n"
    ])
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="append", nargs=3, metavar=("LABEL", "DUAL_TRACE", "OPERATOR_RUNTIME"), required=True)
    parser.add_argument("--lanes", type=int, nargs="+", default=[8, 16, 32, 64, 96, 128, 256, 512, 1024])
    parser.add_argument("--command-overhead", type=int, default=5)
    parser.add_argument("--h67-ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command_overhead < 0:
        raise ValueError("command overhead must be nonnegative")
    results = {}
    for label, dual_path, operator_path in args.identity:
        result = build_identity(read_csv(Path(dual_path)), read_csv(Path(operator_path)), args.lanes, args.command_overhead)
        if args.h67_ledger and label.lower().startswith("h67"):
            add_system_envelope(result, args.h67_ledger)
        results[label] = result
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dual_line_source_issue_envelope_v1",
        "status": "PASS_DIRECT_SOURCE_ISSUE_MAPPING_NOT_MEMORY_TIMED",
        "command_overhead_cycles": args.command_overhead,
        "identities": results,
    }
    (args.output / "source_issue_envelope.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(args.output / "REPORT.md", results, args.command_overhead)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
