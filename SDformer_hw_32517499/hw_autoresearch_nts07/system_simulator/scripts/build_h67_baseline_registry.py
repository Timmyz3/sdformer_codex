#!/usr/bin/env python3
"""Build a fail-closed system-baseline comparison registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LEDGER_STATUS = "PASS_TRANSACTION_LEDGER_MODEL_NOT_CYCLE_ACCURATE"
NATIVE_STATUS = "ADMITTED_SYSTEM_ENVELOPE"


def build_registry(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != LEDGER_STATUS:
        raise RuntimeError(f"ledger is not admitted: {summary.get('status')}")
    cycles = summary["cycles_per_frame_model"]
    rows = []
    for baseline in config["baselines"]:
        row = dict(baseline)
        if row["kind"] == "native":
            if row.get("status") != NATIVE_STATUS:
                raise RuntimeError(f"native baseline is not admitted: {row['name']}")
            cycle_key = row.pop("cycle_key")
            row["cycles_per_frame_envelope"] = int(cycles[cycle_key])
            row["paper_comparison_ready"] = False
            row["blocked_by"] = [
                "non-attention cycle calibration",
                "shared CACTI and DRAMsim3 contract",
            ]
        else:
            if not str(row.get("status", "")).startswith("BLOCKED_"):
                raise RuntimeError(
                    f"external baseline must fail closed until matched: {row['name']}"
                )
            if not row.get("missing"):
                raise RuntimeError(f"external baseline lacks missing list: {row['name']}")
            row["cycles_per_frame_envelope"] = None
            row["paper_comparison_ready"] = False
            row["blocked_by"] = list(row.pop("missing"))
        rows.append(row)
    return {
        "schema": "h67_system_baseline_registry_v0",
        "status": "PASS_FAIL_CLOSED_BASELINE_REGISTRY",
        "paper_comparison_ready": False,
        "claim_boundary": [
            "Only native Fixed2S and RQTB2S consume the same local system ledger.",
            "No published Prosperity or Phi performance number is imported.",
            "All rows remain non-paper until cycles and memory are calibrated under one contract.",
        ],
        "common_contract": config["common_contract"],
        "baselines": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.ledger / "system_summary.json").read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    registry = build_registry(summary, config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": registry["status"],
        "paper_comparison_ready": registry["paper_comparison_ready"],
        "baselines": [row["name"] for row in registry["baselines"]],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
