#!/usr/bin/env python3
"""Receipt-blind audit of raw M479r2 and M477 DC artifacts."""

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "dc_handoff/runs/m479r2_lane_local_dc_3p000ns_r1_20260827"
BASE = ROOT / "dc_handoff/runs/m477_m476r2_backpressure_safe_parent_queue_dc_3p000ns_r1_20260826"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(pattern: str, body: str, cast=float):
    match = re.search(pattern, body, re.MULTILINE)
    if not match:
        raise AssertionError(f"missing pattern: {pattern}")
    return cast(match.group(1))


def normalized_netlist(body: str) -> str:
    body = body.replace(
        "m476_dual_slot_parent_queue_pipeline_LANES96_ROW_BITS6",
        "m479_lane_local_parent_queue_pipeline_LANES96_ROW_BITS6",
    )
    body = body.replace(
        "m476r2_backpressure_safe_parent_queue_pipeline",
        "m479_lane_local_backpressure_safe_parent_queue_pipeline",
    )
    # Module-name length changes DC's line wrapping.  Compare Verilog tokens,
    # not generated comments or cosmetic whitespace.
    body = re.sub(r"//.*$", "", body, flags=re.MULTILINE)
    return re.sub(r"\s+", "", body)


area = text(RUN / "reports/area.rpt")
qor = text(RUN / "reports/qor.rpt")
setup = text(RUN / "reports/timing_setup.rpt")
hold = text(RUN / "reports/timing_hold.rpt")
constraints = text(RUN / "reports/constraint_violators.rpt")
dc_log = text(RUN / "dc.log")

base_area = text(BASE / "reports/area.rpt")
base_net = text(BASE / "netlist/m476r2_backpressure_safe_parent_queue_pipeline_mapped.v")
run_net = text(RUN / "netlist/m479_lane_local_backpressure_safe_parent_queue_pipeline_mapped.v")

measured = {
    "design": one(r"^Design\s*:\s*(\S+)", area, str),
    "ports": one(r"Number of ports:\s+(\d+)", area, int),
    "nets": one(r"Number of nets:\s+(\d+)", area, int),
    "cells": one(r"Number of cells:\s+(\d+)", area, int),
    "combinational_cells": one(r"Number of combinational cells:\s+(\d+)", area, int),
    "sequential_cells": one(r"Number of sequential cells:\s+(\d+)", area, int),
    "cell_area_um2": one(r"Total cell area:\s+([0-9.]+)", area),
    "setup_worst_slack_ns": one(r"slack \(MET\)\s+([0-9.]+)", setup),
    "hold_worst_slack_ns": one(r"slack \(MET\)\s+([0-9.]+)", hold),
    "logic_levels": one(r"Levels of Logic:\s+([0-9.]+)", qor),
    "qor_violating_nets": one(r"Nets With Violations:\s+(\d+)", qor, int),
    "max_transition_violating_nets": one(r"Max Trans Violations:\s+(\d+)", qor, int),
    "max_capacitance_violating_nets": one(r"Max Cap Violations:\s+(\d+)", qor, int),
    "max_fanout_violating_nets": one(r"Max Fanout Violations:\s+(\d+)", qor, int),
    "no_violated_constraint_sections": constraints.count("This design has no violated constraints."),
}

expected_identity = "m479_lane_local_backpressure_safe_parent_queue_pipeline"
assert measured["design"] == expected_identity
assert "Current design is 'm479_lane_local_backpressure_safe_parent_queue_pipeline'." in dc_log
assert "Current design is 'm476r2_backpressure_safe_parent_queue_pipeline'." not in dc_log
assert measured["setup_worst_slack_ns"] >= 0.0
assert measured["hold_worst_slack_ns"] >= 0.0
assert measured["no_violated_constraint_sections"] == 2
assert measured["qor_violating_nets"] == 3
assert measured["max_transition_violating_nets"] == 1
assert measured["max_capacitance_violating_nets"] == 2
assert measured["max_fanout_violating_nets"] == 3
assert "u_core/n17470" in constraints
assert "u_core/n16011" in constraints
assert "u_core/n1" in constraints

base_metrics = {
    "ports": one(r"Number of ports:\s+(\d+)", base_area, int),
    "nets": one(r"Number of nets:\s+(\d+)", base_area, int),
    "cells": one(r"Number of cells:\s+(\d+)", base_area, int),
    "combinational_cells": one(r"Number of combinational cells:\s+(\d+)", base_area, int),
    "sequential_cells": one(r"Number of sequential cells:\s+(\d+)", base_area, int),
    "cell_area_um2": one(r"Total cell area:\s+([0-9.]+)", base_area),
}
for key, value in base_metrics.items():
    assert measured[key] == value, (key, measured[key], value)

run_normalized = normalized_netlist(run_net)
base_normalized = normalized_netlist(base_net)
assert run_normalized == base_normalized

input_sha_lines = {}
for line in text(RUN / "input_sha256.txt").splitlines():
    digest, path = line.split(maxsplit=1)
    input_sha_lines[path.strip()] = digest
for path, digest in input_sha_lines.items():
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    assert candidate.is_file(), candidate
    assert sha(candidate) == digest, candidate

out = {
    "schema": "m479r2_failed_dc_receipt_blind_audit_v1",
    "status": "PASS_AUDIT_CONFIRMS_M479R2_DC_NO_GO",
    "receipt_blind": True,
    "measured": measured,
    "m477_raw_report_comparison": {
        "metrics_equal": True,
        "cell_area_delta_um2": measured["cell_area_um2"] - base_metrics["cell_area_um2"],
        "normalized_mapped_netlist_equal": True,
        "normalized_mapped_netlist_sha256": hashlib.sha256(run_normalized.encode()).hexdigest(),
    },
    "five_constraint_classes": {
        "max_delay": "PASS",
        "min_delay": "PASS",
        "max_capacitance": "FAIL_2_NETS",
        "max_transition": "FAIL_1_NET",
        "max_fanout": "FAIL_3_NETS",
    },
    "verdict": "NO_GO_M479R2_CURRENT_RTL",
}
(Path(__file__).parent / "m479r2_failed_dc_receipt_blind_audit_r1.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(out, indent=2, sort_keys=True))
