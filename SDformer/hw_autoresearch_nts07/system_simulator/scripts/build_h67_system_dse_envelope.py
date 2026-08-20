#!/usr/bin/env python3
"""Build resource and memory envelopes around the H67 full-network ledger.

This script intentionally does not claim cycle accuracy or memory scheduling.
It quantifies the conditions a later cycle simulator must satisfy and emits
machine-readable memory requests for CACTI/DRAMsim3 integration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


LEDGER_STATUS = "PASS_TRANSACTION_LEDGER_MODEL_NOT_CYCLE_ACCURATE"
OUTPUT_STATUS = "PASS_SYSTEM_DSE_ENVELOPE_NOT_CYCLE_ACCURATE"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def integer(row: dict[str, str], key: str) -> int:
    return int(row[key])


def minimum_non_attention_scale(
    non_attention_cycles: int,
    fixed_attention_cycles: int,
    proposed_attention_cycles: int,
    target_speedup: float,
) -> float | None:
    """Return the required uniform non-attention speed scale.

    The fixed and proposed systems share the scaled non-attention term. A
    finite solution only exists below the attention-only speedup limit.
    """

    if target_speedup <= 1.0:
        return 1.0
    denominator = fixed_attention_cycles - target_speedup * proposed_attention_cycles
    if denominator <= 0:
        return None
    required = (
        (target_speedup - 1.0) * non_attention_cycles / denominator
    )
    return max(1.0, required)


def resource_sensitivity(
    summary: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    cycles = summary["cycles_per_frame_model"]
    attention = summary["attention"]
    fixed_attention = int(attention["fixed_cycles_per_frame"])
    proposed_attention = int(attention["rqtb_cycles_per_frame"])
    non_attention = int(cycles["operator_activity_weighted"]) + int(
        cycles["atlif_non_dead"]
    )
    rows = []
    for scale in config["non_attention_parallelism_scale"]:
        scaled_non_attention = math.ceil(non_attention / float(scale))
        fixed_total = scaled_non_attention + fixed_attention
        proposed_total = scaled_non_attention + proposed_attention
        rows.append(
            {
                "non_attention_parallelism_scale": scale,
                "scaled_non_attention_cycles": scaled_non_attention,
                "fixed_cycles": fixed_total,
                "rqtb_cycles": proposed_total,
                "speedup": fixed_total / proposed_total,
                "fixed_attention_share": fixed_attention / fixed_total,
            }
        )
    targets = []
    for target in config["target_end_to_end_speedup"]:
        required = minimum_non_attention_scale(
            non_attention,
            fixed_attention,
            proposed_attention,
            float(target),
        )
        targets.append(
            {
                "target_speedup": target,
                "finite": required is not None,
                "minimum_non_attention_parallelism_scale": required,
            }
        )
    return {
        "contract": (
            "Uniformly scale the v0 non-attention cycle term while holding the "
            "RTL-calibrated attention anchor fixed; sensitivity only."
        ),
        "non_attention_cycles_v0": non_attention,
        "fixed_attention_cycles": fixed_attention,
        "rqtb_attention_cycles": proposed_attention,
        "attention_only_speedup_limit": fixed_attention / proposed_attention,
        "sweep": rows,
        "targets": targets,
    }


def object_fit_sweep(
    objects: list[tuple[str, int]], capacities: list[int]
) -> list[dict[str, Any]]:
    """Optimistic object-fit spill envelope.

    An object no larger than the capacity is individually eligible for
    residency. Concurrent live objects and bank conflicts are not modeled.
    Objects that do not individually fit incur one write and one read in the
    spill envelope.
    """

    rows = []
    for capacity in capacities:
        fit = [(name, size) for name, size in objects if size <= capacity]
        spill = [(name, size) for name, size in objects if size > capacity]
        rows.append(
            {
                "capacity_bytes": capacity,
                "fit_objects": len(fit),
                "spill_objects": len(spill),
                "individually_fitting_payload_bytes": sum(size for _, size in fit),
                "spill_payload_bytes": sum(size for _, size in spill),
                "spill_read_write_bytes": 2 * sum(size for _, size in spill),
                "largest_spill_object_bytes": max(
                    (size for _, size in spill), default=0
                ),
            }
        )
    return rows


def memory_envelope(
    ledger_dir: Path, summary: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    activations = read_csv(ledger_dir / "activation_objects.csv")
    atlif = read_csv(ledger_dir / "atlif_transactions.csv")
    operators = read_csv(ledger_dir / "operator_transactions.csv")

    activation_objects = [
        (f"{row['name']}:{row['kind']}", integer(row, "bytes_int8"))
        for row in activations
    ]
    streaming_accumulator_objects = [
        (row["name"], integer(row, "minimum_streaming_accumulator_bytes_per_call"))
        for row in atlif
        if row["deployment_dead_result"] != "True"
    ]
    full_output_buffer_objects = [
        (row["name"], integer(row, "full_temporal_output_buffer_bytes_per_frame"))
        for row in atlif
        if row["deployment_dead_result"] != "True"
    ]
    weight_objects = [
        (row["name"], integer(row, "weight_bytes_int8"))
        for row in operators
        if integer(row, "weight_bytes_int8") > 0
    ]
    total_weights = sum(size for _, size in weight_objects)
    weight_rows = []
    for capacity in config["weight_sram_bytes"]:
        persistent_bytes = min(capacity, total_weights)
        weight_rows.append(
            {
                "capacity_bytes": capacity,
                "cold_frame_dram_bytes": total_weights,
                "steady_frame_dram_lower_bytes": max(
                    0, total_weights - persistent_bytes
                ),
                "persistent_weight_bytes": persistent_bytes,
                "individual_weight_objects_fit": sum(
                    1 for _, size in weight_objects if size <= capacity
                ),
                "weight_objects": len(weight_objects),
            }
        )

    activation_sweep = object_fit_sweep(
        activation_objects, config["activation_sram_bytes"]
    )
    streaming_accumulator_sweep = object_fit_sweep(
        streaming_accumulator_objects, config["accumulator_sram_bytes"]
    )
    full_output_buffer_sweep = object_fit_sweep(
        full_output_buffer_objects, config["accumulator_sram_bytes"]
    )
    upper = summary["traffic_per_frame_proxy"]
    return {
        "claim_boundary": [
            "Object-fit rows are optimistic lower envelopes, not a schedule.",
            "Concurrent live ranges, bank conflicts, DMA overlap, and frame IO are pending.",
            "Weight persistence is byte-granular across frames; cold-frame traffic is reported separately.",
        ],
        "object_counts": {
            "activation": len(activation_objects),
            "atlif_streaming_accumulator": len(streaming_accumulator_objects),
            "atlif_full_output_buffer": len(full_output_buffer_objects),
            "weight": len(weight_objects),
        },
        "logical_upper_proxy": upper,
        "activation_object_fit": activation_sweep,
        "atlif_streaming_accumulator_object_fit": streaming_accumulator_sweep,
        "atlif_full_output_buffer_object_fit": full_output_buffer_sweep,
        "weight_persistence": weight_rows,
    }


def memory_requests(config: dict[str, Any]) -> list[dict[str, Any]]:
    requests = []
    mapping = (
        ("activation", config["activation_sram_bytes"]),
        ("atlif_accumulator", config["accumulator_sram_bytes"]),
        ("weight", config["weight_sram_bytes"]),
    )
    for kind, capacities in mapping:
        contract = config["memory_requests"][kind]
        for capacity in capacities:
            requests.append(
                {
                    "memory_kind": kind,
                    "capacity_bytes": capacity,
                    "word_bits": int(contract["word_bits"]),
                    "ports": contract["ports"],
                    "cacti_status": "PENDING_EXTERNAL_CACTI",
                    "dramsim3_status": "PENDING_ADDRESS_TIMED_TRACE",
                }
            )
    return requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.ledger / "system_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != LEDGER_STATUS:
        raise RuntimeError(
            f"ledger is not admitted: {summary.get('status')} != {LEDGER_STATUS}"
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resource = resource_sensitivity(summary, config)
    memory = memory_envelope(args.ledger, summary, config)
    requests = memory_requests(config)
    output = {
        "schema": "h67_ep35_system_dse_envelope_v1",
        "status": OUTPUT_STATUS,
        "claim_boundary": [
            "Full-network transactions are measured from the frozen ep35 profile.",
            "Resource scaling and residency are analytical envelopes.",
            "No CACTI, DRAMsim3, bank-conflict, or cycle-accurate result is claimed.",
        ],
        "resource_sensitivity": resource,
        "memory_envelope": memory,
        "external_model_readiness": {
            "cacti": "REQUESTS_GENERATED_RESULTS_PENDING",
            "dramsim3": "BLOCKED_UNTIL_ADDRESS_TIMED_TRACE",
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "system_dse_envelope.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "memory_model_requests.json").write_text(
        json.dumps(requests, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# H67 ep35 system DSE envelope v1",
        "",
        f"Status: `{OUTPUT_STATUS}`.",
        "",
        "## Resource sensitivity",
        "",
        "| non-attention scale | Fixed cycles | RQTB cycles | speedup | attention share |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in resource["sweep"]:
        report.append(
            f"| {row['non_attention_parallelism_scale']} | {row['fixed_cycles']} | "
            f"{row['rqtb_cycles']} | {row['speedup']:.6f}x | "
            f"{row['fixed_attention_share']:.4%} |"
        )
    report.extend(
        [
            "",
            "## Required non-attention scale",
            "",
            "| target system speedup | finite | minimum scale |",
            "|---:|:---:|---:|",
        ]
    )
    for row in resource["targets"]:
        scale = row["minimum_non_attention_parallelism_scale"]
        report.append(
            f"| {row['target_speedup']:.3f}x | {str(row['finite']).lower()} | "
            f"{scale if scale is not None else 'unreachable'} |"
        )
    report.extend(
        [
            "",
            "The table is a sensitivity envelope. It is not an area-fair architecture or a cycle-accurate simulation.",
        ]
    )
    (args.output / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": OUTPUT_STATUS,
                "attention_only_speedup_limit": resource[
                    "attention_only_speedup_limit"
                ],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
