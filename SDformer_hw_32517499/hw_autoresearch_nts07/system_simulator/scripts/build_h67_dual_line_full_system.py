#!/usr/bin/env python3
"""Build a fail-closed Local/Motion full-system execution contract.

This script does not claim cycle accuracy.  It maps real ep35 aggregate
operators to two exact arithmetic paths and states which ordered data are
still required before the paths can be timed:

* Local: source-owned accumulation of the weight columns selected by the
  current binary activation tile.
* Motion: signed updates from activation transitions, preserving the prior
  output accumulator for the same stream and tile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LEDGER_STATUS = "PASS_TRANSACTION_LEDGER_MODEL_NOT_CYCLE_ACCURATE"
CONTRACT_STATUS = "PASS_ARCHITECTURE_CONTRACT_TRACE_TIMING_BLOCKED"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value == "True"


def eligible(row: dict[str, str]) -> bool:
    return (
        row["operator"] in {"Linear", "Conv2d"}
        and truth(row["input_binary_packed_eligible"])
        and not truth(row.get("replaced_by_attention_rtl_anchor", "False"))
    )


def dense_binary_accumulate(
    weights: list[list[int]], activation: list[int]
) -> list[int]:
    """Integer reference for y = W*x with x in {0,1}."""

    if any(value not in {0, 1} for value in activation):
        raise ValueError("activation must be binary")
    if any(len(row) != len(activation) for row in weights):
        raise ValueError("weight/input shape mismatch")
    return [sum(weight * bit for weight, bit in zip(row, activation)) for row in weights]


def motion_delta_accumulate(
    weights: list[list[int]],
    previous_activation: list[int],
    current_activation: list[int],
    previous_output: list[int],
) -> list[int]:
    """Exact update y_t = y_(t-1) + W*(x_t-x_(t-1))."""

    if len(previous_activation) != len(current_activation):
        raise ValueError("activation shape changed within a state stream")
    if len(previous_output) != len(weights):
        raise ValueError("output/weight shape mismatch")
    delta = [current - previous for previous, current in zip(previous_activation, current_activation)]
    if any(value not in {-1, 0, 1} for value in delta):
        raise ValueError("activation pair must be binary")
    return [
        old + sum(weight * change for weight, change in zip(row, delta))
        for row, old in zip(weights, previous_output)
    ]


def audit_artifacts(root: Path, candidates: dict[str, list[str]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for artifact, paths in candidates.items():
        resolved = [root / path for path in paths]
        found = next((path for path in resolved if path.exists()), None)
        rows[artifact] = {
            "available": found is not None,
            "selected": str(found) if found is not None else None,
            "candidates": [str(path) for path in resolved],
        }
    return rows


def route_operator(row: dict[str, str]) -> dict[str, Any]:
    is_eligible = eligible(row)
    # Window partitioning moves the temporal axis before attention Q/K
    # projections.  They remain valid Local binary Linear work, but the
    # ordered trace cannot key an adjacent-timestep Motion state there.
    motion_eligible = is_eligible and row["category"] not in {
        "attention_q_projection",
        "attention_k_projection",
    }
    return {
        "name": row["name"],
        "operator": row["operator"],
        "category": row["category"],
        "baseline_activity_cycles": int(row["activity_cycles_at_config_lanes"]),
        "input_activity": float(row["input_activity"]),
        "eligible": is_eligible,
        "local_eligible": is_eligible,
        "motion_eligible": motion_eligible,
        "local_path": (
            "SOURCE_OWNED_SELECTED_WEIGHT_ACCUMULATION" if is_eligible else "BYPASS"
        ),
        "motion_path": (
            "SIGNED_TEMPORAL_DELTA_ACCUMULATION"
            if motion_eligible
            else ("LOCAL_ONLY_WINDOW_RESHAPED" if is_eligible else "BYPASS")
        ),
        "selector_metric": (
            "min(current_nonzero_count, positive_plus_negative_transition_count)"
            if motion_eligible
            else "not_applicable"
        ),
        "measurement_status": (
            "ORDERED_BINARY_TILE_TRACE_AVAILABLE"
            if motion_eligible
            else (
                "LOCAL_ONLY_ATTENTION_WINDOW_LAYOUT"
                if is_eligible
                else "EXPLICIT_BYPASS"
            )
        ),
    }


def minimum_engine_speedup(
    fixed_total: int, frozen_remainder: int, eligible_cycles: int, target: float
) -> float | None:
    budget = fixed_total / target - frozen_remainder
    return eligible_cycles / budget if budget > 0 else None


def build(
    summary: dict[str, Any],
    operators: list[dict[str, str]],
    config: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    if summary.get("status") != LEDGER_STATUS:
        raise RuntimeError("ledger status is not admitted")
    routes = [route_operator(row) for row in operators]
    eligible_rows = [row for row in routes if row["eligible"]]
    eligible_cycles = sum(row["baseline_activity_cycles"] for row in eligible_rows)

    fixed_total = int(summary["cycles_per_frame_model"]["fixed_total"])
    fixed_attention = int(summary["attention"]["fixed_cycles_per_frame"])
    rqtb_attention = int(summary["attention"]["rqtb_cycles_per_frame"])
    frozen_remainder = fixed_total - fixed_attention - eligible_cycles + rqtb_attention

    local_anchor = float(config["local_component_anchor"]["speedup"])
    local_anchor_total = frozen_remainder + math.ceil(eligible_cycles / local_anchor)
    targets = []
    for target in config["targets"]:
        required = minimum_engine_speedup(
            fixed_total, frozen_remainder, eligible_cycles, float(target)
        )
        targets.append(
            {
                "target_system_speedup": float(target),
                "minimum_dual_line_eligible_engine_speedup": required,
                "maximum_effective_work_fraction": (1.0 / required if required else None),
                "additional_reduction_vs_local_anchor": (
                    local_anchor / required if required else None
                ),
            }
        )

    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"operators": 0, "eligible_operators": 0, "cycles": 0, "eligible_cycles": 0}
    )
    for row in routes:
        bucket = categories[row["category"]]
        bucket["operators"] += 1
        bucket["cycles"] += row["baseline_activity_cycles"]
        if row["eligible"]:
            bucket["eligible_operators"] += 1
            bucket["eligible_cycles"] += row["baseline_activity_cycles"]

    trace_ready = all(
        artifacts[name]["available"]
        for name in (
            "full_network_execution_trace",
            "full_network_dual_line_operator_trace",
        )
    )
    return {
        "schema": "h67_dual_line_full_system_contract_v0",
        "status": CONTRACT_STATUS if not trace_ready else "PASS_TRACE_PRESENT_TIMING_PENDING",
        "claim_boundary": [
            "The arithmetic identities are exact for binary Linear/Conv2d tiles.",
            "Aggregate ep35 activity establishes coverage but not transition counts or cycles.",
            "Attention Q/K projections are Local-eligible but Motion-ineligible after window-axis reshaping.",
            "The Local VCS ratio is an architecture anchor, not a full-network measurement.",
            "Bias, BN/requant, residual, ATLIF, attention, DMA, and memory timing remain explicit shared stages.",
        ],
        "architecture": {
            "local": {
                "operation": "For each active source bit, read its weight segment once and multicast signed INT8 weights into bank-local Acc32 destinations.",
                "reused_mechanisms": [
                    "Q-silent source suppression",
                    "source-owned work issue",
                    "identical descriptor/weight-segment reuse",
                    "resident output accumulators",
                ],
            },
            "motion": {
                "operation": "Retain the previous exact accumulator and apply +W columns for 0->1 transitions and -W columns for 1->0 transitions.",
                "invariants": [
                    "state is keyed by sequence/operator/call/output tile",
                    "state is invalidated at sequence or shape boundaries",
                    "periodic refresh recomputes through the Local path",
                    "selector may use Motion only when signed-transition work is lower",
                ],
            },
            "shared_selector": config["selector"],
        },
        "coverage": {
            "operators": len(routes),
            "eligible_operators": len(eligible_rows),
            "eligible_cycles": eligible_cycles,
            "eligible_fraction_of_fixed_system": eligible_cycles / fixed_total,
            "frozen_remainder_with_rqtb_attention": frozen_remainder,
            "maximum_system_speedup_if_eligible_work_is_free": fixed_total / frozen_remainder,
            "categories": dict(sorted(categories.items())),
        },
        "envelopes": {
            "fixed_total_cycles_model": fixed_total,
            "rqtb_attention_cycles_model": rqtb_attention,
            "local_anchor_component_speedup": local_anchor,
            "local_anchor_system_cycles_model": local_anchor_total,
            "local_anchor_system_speedup_model": fixed_total / local_anchor_total,
            "targets": targets,
        },
        "artifacts": artifacts,
        "required_trace_schema": {
            "execution_trace.csv": [
                "sequence_id", "sample_id", "frame_id", "timestep", "operator_index",
                "operator_name", "call_index", "tile_id", "input_shape", "output_shape",
                "input_payload_key", "weight_object", "output_state_object", "reset_state"
            ],
            "dual_line_operator_trace.csv": [
                "operator_name", "operator_call_index", "temporal_step",
                "selector_rows", "current_source_count",
                "positive_transition_source_count",
                "negative_transition_source_count", "local_work",
                "motion_work", "selected_work", "selector_saved_work"
            ],
            "selected_binary_tile_vectors.npz": [
                "packed_current_bits", "packed_previous_bits",
                "expected_positive_bits", "expected_negative_bits",
                "expected_selector_mode"
            ],
            "memory trace next stage": [
                "cycle", "read_or_write", "address", "bytes", "object_id", "bank", "source"
            ],
        },
        "operators": routes,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, result: dict[str, Any]) -> None:
    coverage = result["coverage"]
    envelopes = result["envelopes"]
    missing = [name for name, row in result["artifacts"].items() if not row["available"]]
    lines = [
        "# Local + Motion full-system contract v0\n",
        f"Status: `{result['status']}`\n",
        "## Coverage\n",
        f"- Eligible operators: {coverage['eligible_operators']}/{coverage['operators']}\n",
        f"- Eligible fixed-system cycle fraction: {coverage['eligible_fraction_of_fixed_system']:.4%}\n",
        f"- Free-eligible-work system upper bound: {coverage['maximum_system_speedup_if_eligible_work_is_free']:.6f}x\n",
        f"- Local 1.783485x component-anchor system envelope: {envelopes['local_anchor_system_speedup_model']:.6f}x\n",
        "\n## Target thresholds\n",
        "| system target | required eligible-engine speedup | maximum effective work | residual vs Local anchor |\n",
        "|---:|---:|---:|---:|\n",
    ]
    for row in envelopes["targets"]:
        required = row["minimum_dual_line_eligible_engine_speedup"]
        if required is None:
            lines.append(f"| {row['target_system_speedup']:.1f}x | unreachable | n/a | n/a |\n")
        else:
            lines.append(
                f"| {row['target_system_speedup']:.1f}x | {required:.6f}x | "
                f"{row['maximum_effective_work_fraction']:.4%} | "
                f"{row['additional_reduction_vs_local_anchor']:.4%} |\n"
            )
    lines.extend(["\n## Missing artifacts\n"])
    lines.extend(f"- `{name}`\n" for name in missing)
    if result["status"] == "PASS_TRACE_PRESENT_TIMING_PENDING":
        lines.append(
            "\nThe ordered trace is present and transition work has been measured. "
            "The remaining artifacts block bit-exact datapath replay and cycle/memory timing, "
            "not RTL microarchitecture work.\n"
        )
    else:
        lines.append(
            "\nThe missing ordered artifacts block measured transition reuse, cycle timing, "
            "and memory timing; they do not block RTL microarchitecture work.\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.ledger / "system_summary.json").read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifacts = audit_artifacts(args.workspace_root, config["artifact_candidates"])
    result = build(
        summary,
        read_csv(args.ledger / "operator_transactions.csv"),
        config,
        artifacts,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dual_line_contract.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "operator_routes.csv", result["operators"])
    write_report(args.output / "REPORT.md", result)
    print(json.dumps({"status": result["status"], **result["coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
