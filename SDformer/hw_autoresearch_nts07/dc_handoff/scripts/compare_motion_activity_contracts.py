#!/usr/bin/env python3
"""Ensure Fixed2S and RQTB2S activity contracts use the same Motion workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--rqtb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixed = json.loads(args.fixed.read_text(encoding="utf-8"))
    rqtb = json.loads(args.rqtb.read_text(encoding="utf-8"))
    fixed_rows = fixed.get("workload_rows", [])
    rqtb_rows = rqtb.get("workload_rows", [])
    fixed_identity = [
        (row.get("index"), row.get("equal"), row.get("emitted"))
        for row in fixed_rows
    ]
    rqtb_identity = [
        (row.get("index"), row.get("equal"), row.get("emitted"))
        for row in rqtb_rows
    ]
    checks = {
        "contracts_pass": fixed.get("status") == "PASS" and rqtb.get("status") == "PASS",
        "design_pair": fixed.get("design_name") == "h67_fixed2s_mssb5_dc_top"
        and rqtb.get("design_name") == "h67_rqtb2s_mssb5_dc_top",
        "same_trace_sha256": fixed.get("trace_sha256") == rqtb.get("trace_sha256"),
        "same_trace_scope": fixed.get("trace_scope") == rqtb.get("trace_scope"),
        "same_measurement_scope": fixed.get("measurement_scope")
        == rqtb.get("measurement_scope"),
        "same_row_boundary_overhead": fixed.get("measurement_overhead_cycles") == 552
        and rqtb.get("measurement_overhead_cycles") == 552,
        "same_power_purpose": fixed.get("activity_purpose") == "paper_power_compute"
        and rqtb.get("activity_purpose") == "paper_power_compute",
        "same_row_identity": fixed_identity == rqtb_identity and bool(fixed_identity),
        "paper_power_eligible": fixed.get("paper_power_eligible") is True
        and rqtb.get("paper_power_eligible") is True,
        "frozen_fair_cycle_anchor": fixed.get("busy_cycles") == 112589
        and rqtb.get("busy_cycles") == 94891,
        "complete_138_row_population": len(fixed_rows) == 138
        and len(rqtb_rows) == 138,
        "frozen_slot_anchor": sum(row.get("slots", 0) for row in fixed_rows) == 62100
        and sum(row.get("slots", 0) for row in rqtb_rows) == 34099,
        "frozen_equal_anchor": sum(row.get("equal", 0) for row in fixed_rows) == 28001
        and sum(row.get("equal", 0) for row in rqtb_rows) == 28001,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": len(fixed_rows),
        "fixed_busy_cycles": fixed.get("busy_cycles"),
        "rqtb_busy_cycles": rqtb.get("busy_cycles"),
        "boundary": (
            "Fixed2S and RQTB2S activity use the same 138-row fair-LFSR workload and "
            "match the frozen 112589/94891 cycle and 62100/34099 slot anchors."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
