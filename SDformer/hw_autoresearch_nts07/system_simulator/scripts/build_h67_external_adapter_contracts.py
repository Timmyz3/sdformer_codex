#!/usr/bin/env python3
"""Map H67 operators to Prosperity- and Phi-like spiking-GEMM contracts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LEDGER_STATUS = "PASS_TRANSACTION_LEDGER_MODEL_NOT_CYCLE_ACCURATE"
ADAPTER_STATUS = "PASS_STRUCTURAL_MAPPING_CYCLES_BLOCKED"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value == "True"


def map_operator(row: dict[str, str]) -> dict[str, Any]:
    spiking_gemm = row["operator"] in {"Linear", "Conv2d"} and truth(
        row["input_binary_packed_eligible"]
    )
    return {
        "name": row["name"],
        "operator": row["operator"],
        "category": row["category"],
        "input_binary01": truth(row["input_binary_packed_eligible"]),
        "activity_weighted_macs_per_frame": int(
            row["activity_weighted_macs_per_frame"]
        ),
        "activity_cycles_at_config_lanes": int(row["activity_cycles_at_config_lanes"]),
        "prosperity_structurally_eligible": spiking_gemm,
        "phi_like_structurally_eligible": spiking_gemm,
        "mapping_reason": (
            "binary Linear/Conv2d maps to paper-defined spiking GeMM"
            if spiking_gemm
            else "non-binary activation or unsupported operator"
        ),
    }


def build(summary: dict[str, Any], operators: list[dict[str, str]]) -> dict[str, Any]:
    if summary.get("status") != LEDGER_STATUS:
        raise RuntimeError("ledger status is not admitted")
    mapped = [map_operator(row) for row in operators]
    total_macs = sum(row["activity_weighted_macs_per_frame"] for row in mapped)
    eligible_macs = sum(
        row["activity_weighted_macs_per_frame"]
        for row in mapped
        if row["prosperity_structurally_eligible"]
    )
    eligible_cycles = sum(
        row["activity_cycles_at_config_lanes"]
        for row in mapped
        if row["prosperity_structurally_eligible"]
    )
    cycles = summary["cycles_per_frame_model"]
    fixed_total = int(cycles["fixed_total"])
    fixed_attention = int(summary["attention"]["fixed_cycles_per_frame"])
    proposed_attention = int(summary["attention"]["rqtb_cycles_per_frame"])
    proposed_unaccelerated_remainder = (
        fixed_total - fixed_attention - eligible_cycles + proposed_attention
    )
    targets = []
    for target in (1.5, 2.0, 3.0):
        eligible_budget = fixed_total / target - proposed_unaccelerated_remainder
        required = eligible_cycles / eligible_budget if eligible_budget > 0 else None
        targets.append(
            {
                "target_system_speedup": target,
                "minimum_eligible_engine_speedup": required,
            }
        )
    return {
        "schema": "h67_external_adapter_contracts_v1",
        "status": ADAPTER_STATUS,
        "claim_boundary": [
            "Structural eligibility only; no Prosperity or Phi paper speedup is imported.",
            "One OP must mean one accumulation for a binary-one activation for all rows.",
            "Non-binary operators, ATLIF, the native attention core, memory timing, and adapter cycles remain explicit.",
        ],
        "coverage": {
            "operators": len(mapped),
            "structurally_eligible_operators": sum(
                row["prosperity_structurally_eligible"] for row in mapped
            ),
            "activity_weighted_macs_per_frame": total_macs,
            "eligible_activity_weighted_macs_per_frame": eligible_macs,
            "eligible_mac_fraction": eligible_macs / total_macs if total_macs else 0.0,
            "eligible_cycles_per_frame_model": eligible_cycles,
            "eligible_fraction_of_fixed_system_cycles": eligible_cycles / fixed_total,
            "max_system_speedup_if_eligible_cycles_are_free": (
                fixed_total / proposed_unaccelerated_remainder
            ),
            "eligible_engine_targets": targets,
        },
        "adapters": {
            "Prosperity": {
                "paper_mechanism": "runtime product sparsity over spiking GeMM",
                "status": "BLOCKED_ORDERED_SPIKE_ROWS_AND_CYCLE_MEMORY_MODEL",
                "required_next": [
                    "ordered per-operator binary activation rows",
                    "m=256 k=16 n=128 matched-resource cycle adapter",
                    "TCAM/product-table and 128-PE synthesis under local libraries",
                    "shared SRAM/DRAM address-timed trace",
                ],
            },
            "Phi-like": {
                "paper_mechanism": "offline pattern products plus sparse residual",
                "status": "BLOCKED_PATTERN_CALIBRATION_AND_CYCLE_MEMORY_MODEL",
                "required_next": [
                    "train/calibration split and per-tile pattern catalog",
                    "accuracy audit with and without pattern-aware fine-tuning",
                    "m=256 k=16 n=32 matched-resource cycle adapter",
                    "PWP/pattern-ID traffic and shared SRAM/DRAM address-timed trace",
                ],
            },
        },
        "operators": mapped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(
        (args.ledger / "system_summary.json").read_text(encoding="utf-8")
    )
    result = build(summary, read_csv(args.ledger / "operator_transactions.csv"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "external_adapter_contracts.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], **result["coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
