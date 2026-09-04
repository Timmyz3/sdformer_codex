#!/usr/bin/python3.12
"""Fresh read-only result hammer for the M872/M803 R16 three-axis DC run.

This script never invokes EDA.  It admits a result only after the canonical
three-axis directory has been atomically published and its recursive double
seal, per-axis artifact receipts, runtime gates, and setup/area reports all
recompute cleanly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
CANON = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
ATTEMPT = HW / "dc_handoff/runs/.m872_m803_c2_r16_channel_split_three_axis_dc_attempt_consumed"
VCS_REVIEW_DIR = HW / "reviews/m867_m859_c2_r25_shared_whitelist_vcs_result_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED_DOC359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_CYCLES = {
    "k8": [51, 131, 486, 1231, 14],
    "k1x8": [53, 133, 499, 1246, 14],
}
AXES = ("k1", "k8", "k1x8")
DESIGN = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
EXPECTED_ARTIFACTS = {
    "mapped_verilog": f"netlist/{DESIGN}_mapped.v",
    "mapped_sdc": f"netlist/{DESIGN}_mapped.sdc",
    "ddc": f"netlist/{DESIGN}.ddc",
    "svf": f"netlist/{DESIGN}.svf",
    "area_report": "reports/area.rpt",
    "qor_report": "reports/qor.rpt",
    "setup_timing_report": "reports/timing_setup.rpt",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(cond: bool, message: str, errors: list[str]) -> None:
    if not cond:
        errors.append(message)


def regular_nonsymlink(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def verify_manifest(directory: Path, errors: list[str], label: str) -> dict:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    fail(regular_nonsymlink(manifest), f"{label}: missing regular SHA256SUMS", errors)
    fail(regular_nonsymlink(outer), f"{label}: missing regular outer seal", errors)
    if not regular_nonsymlink(manifest) or not regular_nonsymlink(outer):
        return {}
    outer_fields = outer.read_text().strip().split()
    fail(len(outer_fields) == 2 and outer_fields[1] == "SHA256SUMS",
         f"{label}: malformed outer seal", errors)
    if len(outer_fields) >= 1:
        fail(outer_fields[0] == sha(manifest), f"{label}: outer seal mismatch", errors)
    listed: dict[str, str] = {}
    for line_no, line in enumerate(manifest.read_text().splitlines(), 1):
        parts = line.split(None, 1)
        if len(parts) != 2:
            errors.append(f"{label}: malformed manifest line {line_no}")
            continue
        digest, rel = parts
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        fail(bool(rel) and not Path(rel).is_absolute() and ".." not in Path(rel).parts,
             f"{label}: unsafe manifest path {rel}", errors)
        fail(rel not in listed, f"{label}: duplicate manifest path {rel}", errors)
        listed[rel] = digest
        target = directory / rel
        fail(regular_nonsymlink(target), f"{label}: non-regular/symlink payload {rel}", errors)
        if regular_nonsymlink(target):
            fail(sha(target) == digest, f"{label}: digest mismatch {rel}", errors)
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected = set(listed) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    fail(actual == expected,
         f"{label}: recursive population mismatch missing={sorted(expected-actual)} extra={sorted(actual-expected)}",
         errors)
    symlinks = [str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_symlink()]
    fail(not symlinks, f"{label}: symlinks present {symlinks}", errors)
    return {
        "manifest_sha256": sha(manifest),
        "outer_seal_file_sha256": sha(outer),
        "manifest_entries": len(listed),
        "regular_files_including_seals": len(actual),
        "symlinks": len(symlinks),
    }


def exact_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_axis(axis: str, errors: list[str]) -> dict:
    point = CANON / axis
    fail(point.is_dir() and not point.is_symlink(), f"{axis}: directory missing/symlink", errors)
    run_complete = exact_kv(point / "RUN_COMPLETE.txt")
    fail(run_complete.get("status") ==
         f"PASS_M872_M803_DC_{axis.upper()}_SETUP_AREA_LOGIC_ONLY_DC_3NS_PENDING_RECEIPT_REVIEW",
         f"{axis}: RUN_COMPLETE status", errors)
    for key, expected in {
        "macro_count": "0", "logic_only_pre_macro": "true",
        "hold_not_closed_at_dc": "true", "hold_diagnostic_only": "true",
        "power": "false", "energy": "false", "ppa": "false",
        "paper_ppa_ready": "false", "system": "false",
        "system_speedup": "false", "headline": "false",
    }.items():
        fail(run_complete.get(key) == expected, f"{axis}: RUN_COMPLETE {key}", errors)

    gate = exact_kv(point / "reports/precompile_loop_gate.rpt")
    fail(gate.get("TIM-209") == "0", f"{axis}: TIM-209 is not zero", errors)
    fail(gate.get("OPT-150") == "0", f"{axis}: OPT-150 is not zero", errors)
    fail(gate.get("status") == "PASS_PRECOMPILE_LOOP_GATE", f"{axis}: precompile gate status", errors)

    terminal = exact_kv(point / "TCL_PASS_TERMINAL.txt")
    fail(terminal.get("status") == "PASS_M519_R8_SETUP_AREA_DC_TCL_TERMINAL",
         f"{axis}: TCL terminal", errors)
    fail(terminal.get("compile_ultra_count") == "1", f"{axis}: compile count", errors)
    fail(terminal.get("incremental_compile_count") == "0", f"{axis}: incremental compile", errors)
    fail(terminal.get("hold_optimization_count") == "0", f"{axis}: hold optimization count", errors)
    fail((point / "dc.rc").read_text().strip() == "0", f"{axis}: dc rc", errors)
    fail((point / "runtime_monitor.rc").read_text().strip() == "0", f"{axis}: runtime monitor rc", errors)

    ack_text = (point / "runtime_final_gate_ack.txt").read_text()
    fail("final_gate_applied=true" in ack_text and "status=PASS_FINAL_GATE_ACK" in ack_text,
         f"{axis}: runtime final gate ack", errors)
    faults = (point / "descendant_identity_faults.log").read_bytes()
    fail(len(faults) == 0, f"{axis}: descendant identity faults", errors)
    collision_lines = (point / "resource_runtime_external_collisions.tsv").read_text().splitlines()
    fail(len(collision_lines) == 1, f"{axis}: external resource collision rows", errors)
    runtime_samples = [line for line in (point / "runtime_gate_every_snapshot.log").read_text().splitlines()
                       if line.strip()]
    fail(len(runtime_samples) >= 3, f"{axis}: fewer than three resource samples", errors)
    fail(all("gate_reason=none" in line for line in runtime_samples),
         f"{axis}: runtime gate failure", errors)

    inventory_path = point / "artifact_receipts/artifact_inventory.tsv"
    terminal_receipt = exact_kv(point / "artifact_receipts/artifact_terminal_receipt.txt")
    fail(terminal_receipt.get("artifact_count") == "7", f"{axis}: artifact count", errors)
    fail(terminal_receipt.get("status") == "PASS_M872_M803_DC_ATOMIC_RECEIPT_AND_LIVE_SEVEN_ARTIFACT_TUPLE",
         f"{axis}: artifact terminal status", errors)
    inv_lines = inventory_path.read_text().splitlines()
    fail(inv_lines[:1] == ["artifact\tpath\tsize_bytes\tsha256"], f"{axis}: inventory header", errors)
    inventory: dict[str, dict] = {}
    for line in inv_lines[1:]:
        fields = line.split("\t")
        fail(len(fields) == 4, f"{axis}: malformed inventory row", errors)
        if len(fields) != 4:
            continue
        label, rel, size_text, digest = fields
        inventory[label] = {"path": rel, "size_bytes": int(size_text), "sha256": digest}
    fail(set(inventory) == set(EXPECTED_ARTIFACTS), f"{axis}: artifact labels", errors)
    for label, expected_rel in EXPECTED_ARTIFACTS.items():
        if label not in inventory:
            continue
        row = inventory[label]
        path = point / row["path"]
        fail(row["path"] == expected_rel, f"{axis}: {label} path", errors)
        fail(regular_nonsymlink(path), f"{axis}: {label} nonregular", errors)
        if regular_nonsymlink(path):
            fail(path.stat().st_size == row["size_bytes"], f"{axis}: {label} size", errors)
            fail(sha(path) == row["sha256"], f"{axis}: {label} digest", errors)

    area_text = (point / "reports/area.rpt").read_text()
    area_match = re.search(r"^Total cell area:\s+([0-9.]+)\s*$", area_text, re.M)
    fail(area_match is not None, f"{axis}: total cell area missing", errors)
    area = float(area_match.group(1)) if area_match else math.nan
    qor_text = (point / "reports/qor.rpt").read_text()
    wns_match = re.search(r"^\s*Design\s+WNS:\s*([-0-9.]+)\s+TNS:\s*([-0-9.]+)\s+Number of Violating Paths:\s*([0-9.]+)", qor_text, re.M)
    fail(wns_match is not None, f"{axis}: QoR setup tuple missing", errors)
    if wns_match:
        wns, tns, violations = map(float, wns_match.groups())
        fail(wns >= 0.0 and tns >= 0.0 and violations == 0.0,
             f"{axis}: setup QoR not met", errors)
    else:
        wns = tns = violations = math.nan
    timing_text = (point / "reports/timing_setup.rpt").read_text()
    fail("slack (VIOLATED)" not in timing_text, f"{axis}: setup violated", errors)
    slacks = [float(x) for x in re.findall(r"slack \(MET\)\s+([-0-9.]+)", timing_text)]
    fail(bool(slacks) and min(slacks) >= 0.0, f"{axis}: setup slack parsing/failure", errors)

    launch = (point / "launch_pid_tree_root.txt").read_text().splitlines()[0]
    runner_match = re.search(r"\brunner_pid=(\d+)\b", launch)
    fail(runner_match is not None, f"{axis}: runner pid missing", errors)
    return {
        "area_um2": area,
        "setup_wns_ns_qor_rounded": wns,
        "setup_tns_ns_qor_rounded": tns,
        "setup_violating_paths": int(violations) if math.isfinite(violations) else None,
        "minimum_reported_setup_slack_ns": min(slacks) if slacks else None,
        "tim209": int(gate.get("TIM-209", "-1")),
        "opt150": int(gate.get("OPT-150", "-1")),
        "artifact_count": len(inventory),
        "runtime_gate_samples": len(runtime_samples),
        "runner_pid": int(runner_match.group(1)) if runner_match else None,
    }


def main() -> int:
    errors: list[str] = []
    fail(CANON.is_dir() and not CANON.is_symlink(), "canonical result absent or symlink", errors)
    if errors:
        print(json.dumps({"status": "WAIT_OR_FAIL", "errors": errors}, indent=2))
        return 2
    canonical_seal = verify_manifest(CANON, errors, "canonical")
    attempt_seal = verify_manifest(ATTEMPT, errors, "attempt")
    vcs_seal = verify_manifest(VCS_REVIEW_DIR, errors, "M867 VCS review")

    root_complete = exact_kv(CANON / "RUN_COMPLETE.txt")
    fail(root_complete.get("status") ==
         "PASS_M872_M803_DC_THREE_AXIS_SETUP_AREA_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RECEIPT_REVIEW",
         "enclosing RUN_COMPLETE status", errors)
    for key, expected in {
        "logic_only_pre_macro": "true", "hold_not_closed_at_dc": "true",
        "hold_diagnostic_only": "true", "power": "false", "energy": "false",
        "ppa": "false", "paper_ppa_ready": "false", "system": "false",
        "system_speedup": "false", "headline": "false",
    }.items():
        fail(root_complete.get(key) == expected, f"enclosing RUN_COMPLETE {key}", errors)

    attempt = exact_kv(ATTEMPT / "ATTEMPT_CONSUMED.txt")
    fail(attempt.get("status") == "CONSUMED_AT_FIRST_DC_LAUNCH", "attempt status", errors)
    fail(attempt.get("canonical") == str(CANON), "attempt canonical binding", errors)
    fail(attempt.get("license_query_is_reservation") == "false", "attempt license label", errors)

    fail(sha(DOC359) == EXPECTED_DOC359, "docs/359 identity changed", errors)
    vcs_review = json.loads((VCS_REVIEW_DIR / "review.json").read_text())
    fail(vcs_review.get("status") == "PASS100_M859_R25_DIRECTED_COMPONENT_VCS_E3_RESULT_ADMITTED",
         "M867 VCS review status", errors)
    got_cycles = vcs_review.get("vcs_evidence", {}).get("equal_bandwidth", {}).get("exact_cycles")
    fail(got_cycles == EXPECTED_CYCLES, "frozen VCS cycle arrays drifted", errors)

    axes = {axis: parse_axis(axis, errors) for axis in AXES}
    runner_pids = {axes[a]["runner_pid"] for a in AXES}
    fail(len(runner_pids) == 1 and None not in runner_pids,
         f"axes do not share one runner attempt: {runner_pids}", errors)

    per_case = []
    k8_area = axes["k8"]["area_um2"]
    k1x8_area = axes["k1x8"]["area_um2"]
    for index, (c8, c1x8) in enumerate(zip(EXPECTED_CYCLES["k8"], EXPECTED_CYCLES["k1x8"]), 1):
        cycle_speedup = c1x8 / c8
        throughput_per_area_ratio = (c1x8 * k1x8_area) / (c8 * k8_area)
        per_case.append({
            "case": index,
            "k8_cycles": c8,
            "k1x8_cycles": c1x8,
            "equal_bandwidth_cycle_speedup_k8_vs_k1x8": cycle_speedup,
            "equal_bandwidth_throughput_per_mm2_ratio_k8_vs_k1x8": throughput_per_area_ratio,
        })
    sum_k8 = sum(EXPECTED_CYCLES["k8"])
    sum_k1x8 = sum(EXPECTED_CYCLES["k1x8"])
    aggregate_cycle_speedup = sum_k1x8 / sum_k8
    aggregate_tpa = (sum_k1x8 * k1x8_area) / (sum_k8 * k8_area)

    status = "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED" if not errors else "FAIL_M872_M803_C2_R16_THREE_AXIS_DC_RESULT_NOT_ADMITTED"
    review = {
        "schema": "m903_m872_m803_c2_r16_three_axis_dc_result_hammer_v1",
        "date": "2026-08-29",
        "status": status,
        "verdict": "PASS" if not errors else "FAIL",
        "score_out_of_100": 100 if not errors else 0,
        "p0": errors,
        "p0_count": len(errors),
        "p1": [], "p1_count": 0, "p2": [], "p2_count": 0,
        "identity": {
            "canonical_result": str(CANON.relative_to(HW)),
            "canonical_seal": canonical_seal,
            "attempt_seal": attempt_seal,
            "m867_vcs_review_seal": vcs_seal,
            "docs359_sha256": sha(DOC359),
            "single_runner_pid_all_axes": sorted(runner_pids)[0] if len(runner_pids) == 1 and None not in runner_pids else None,
        },
        "dc_evidence": {
            "tool_flow": "Synopsys Design Compiler setup/area-only, 3.000 ns ideal clock, ZeroWireload",
            "same_attempt_axis_order": list(AXES),
            "axes": axes,
            "all_axes_tim209_zero": all(axes[a]["tim209"] == 0 for a in AXES),
            "all_axes_opt150_zero": all(axes[a]["opt150"] == 0 for a in AXES),
            "all_axes_seven_artifacts": all(axes[a]["artifact_count"] == 7 for a in AXES),
        },
        "fair_equal_bandwidth_metrics": {
            "comparison": "K8 candidate versus equal-bandwidth K1x8 baseline only",
            "frozen_directed_vcs_cycles": EXPECTED_CYCLES,
            "per_case": per_case,
            "aggregate_sum_cycles": {"k8": sum_k8, "k1x8": sum_k1x8},
            "aggregate_equal_bandwidth_cycle_speedup_k8_vs_k1x8": aggregate_cycle_speedup,
            "aggregate_equal_bandwidth_throughput_per_mm2_ratio_k8_vs_k1x8": aggregate_tpa,
            "aggregation_scope": "sum over five frozen directed component workloads; not a full-network or trace-weighted workload",
        },
        "claim_boundary": {
            "logic_only_pre_macro": True,
            "macro_count": 0,
            "setup_area_citable": not errors,
            "directed_component_equal_bandwidth_cycle_and_throughput_per_area_citable": not errors,
            "hold_not_closed_at_dc": True,
            "hold_diagnostic_only": True,
            "power": False, "energy": False, "ppa": False,
            "paper_ppa_ready": False, "system": False, "system_speedup": False,
            "headline": False,
            "k8_vs_single_k1_headline_forbidden": True,
            "paper_usage": "Citable only as logic-only pre-macro DC setup/area plus frozen directed component VCS equal-bandwidth K8-vs-K1x8 metrics; never as system/PPA/energy or K8-vs-single-K1 headline.",
        },
        "execution_receipt": {
            "eda_runs_by_reviewer": 0, "dc_runs_by_reviewer": 0,
            "license_queries_by_reviewer": 0, "canonical_result_modified": False,
            "docs359_modified": False,
        },
    }
    out = Path(__file__).resolve().parent / "review.json"
    out.write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
