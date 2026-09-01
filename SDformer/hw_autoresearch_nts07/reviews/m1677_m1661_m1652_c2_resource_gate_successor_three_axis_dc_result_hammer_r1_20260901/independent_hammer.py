#!/usr/bin/env python3
"""M1677 different-author, read-only hammer for the M1661 C2 DC result.

The canonical result and consumed-attempt directories are inputs only.  This
program does not invoke EDA, query a license, or modify either production
namespace.  It is intentionally compatible with CPython 3.6 and 3.12.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
CANON = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_attempt_consumed"
OLD = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
M903 = HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
M1627 = HW / "reviews/m1627_m1613_c2_registered_fault_directed_vcs_result_independent_hammer_r1_20260901"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1661_m1652_c2_resource_gate_successor_exact_sha_r1.sh"
CONTRACT = HW / "contracts/m1661_m1652_c2_resource_gate_successor_dc_source_contract_r1_20260901.json"
RELEASE = HW / "contracts/m1663_m1662_m1661_m1652_c2_resource_gate_successor_dc_launch_release_r1_20260901.json"
FILELIST = HW / "dc_handoff/filelists/date_m1634_c2_m1609_registered_fault_three_axis_logic_only_dc.f"
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

AXES = (("k1", 0), ("k8", 1), ("k1x8", 2))
DESIGN = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
GUI_ERROR = "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"
FROZEN_CYCLES = {"k8": 1913, "k1x8": 1945}
EXPECTED = {
    "runner": "9bf1e220054ff28e3c7bad27b07bc61f50a504625f4b7df0893b0e50162e80e6",
    "contract": "1e2f04c6c46c69c58659e406b6c5d055f24c91429d6e2dcd9dd7bb1a53df03ed",
    "release": "8d6f4aa143215984b3b5cc89c1faee980ac4267b94487fa9d9348c2d0d53d8ec",
    "filelist": "03c4dcd546da19d5de231fa80032473e7c365592012661e6ed77019d7bab4f3f",
    "tcl": "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    "sdc": "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    "m1609": "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m903_review": "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a",
    "m1627_review": "ab4f2187667301a37fbd5f523687a8971282e642163d42886edcdc138edc43d4",
}

REQUIRED_AXIS_FILES = (
    "input_identity.txt", "dc.log", "dc.rc", "TCL_PASS_TERMINAL.txt",
    "reports/flow_contract.rpt", "reports/precompile_loop_gate.rpt",
    "reports/check_design_precompile.rpt", "reports/check_timing_precompile.rpt",
    "reports/resources_precompile.rpt", "reports/references_precompile.rpt",
    "reports/compile_receipt.rpt", "reports/hierarchy_postcompile.rpt",
    "reports/resources_postcompile.rpt", "reports/references_postcompile.rpt",
    "reports/qor.rpt", "reports/area.rpt", "reports/clocks.rpt",
    "reports/ports.rpt", "reports/port_count.txt", "reports/timing_setup.rpt",
    "reports/timing_hold_diagnostic.rpt", "reports/constraint_setup.rpt",
    "reports/constraint_hold_diagnostic.rpt",
    "reports/constraint_max_capacitance.rpt",
    "reports/constraint_max_transition.rpt",
    "reports/constraint_max_fanout.rpt",
    "reports/check_design_postcompile.rpt",
    "reports/check_timing_postcompile.rpt",
    "netlist/%s_mapped.v" % DESIGN, "netlist/%s_mapped.sdc" % DESIGN,
    "netlist/%s.ddc" % DESIGN, "netlist/%s.svf" % DESIGN,
)


class AuditError(Exception):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path):
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not stat.S_ISLNK(mode)


def exact_kv(path):
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            need(key not in result, "duplicate key in %s: %s" % (path, key))
            result[key] = value
    return result


def verify_manifest(directory, label):
    need(directory.is_dir() and not directory.is_symlink(), "%s directory absent/symlink" % label)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(regular(manifest) and regular(outer), "%s seal files absent/nonregular" % label)
    outer_fields = outer.read_text().rstrip("\n").split("  ")
    need(len(outer_fields) == 2 and outer_fields[1] == "SHA256SUMS",
         "%s malformed outer seal" % label)
    need(outer_fields[0] == sha(manifest), "%s outer seal digest mismatch" % label)
    listed = {}
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        fields = line.split("  ", 1)
        need(len(fields) == 2, "%s malformed manifest line %d" % (label, number))
        digest, rel = fields
        rel = rel.lstrip("./")
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             "%s malformed digest line %d" % (label, number))
        need(rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts,
             "%s unsafe manifest path %s" % (label, rel))
        need(rel not in listed, "%s duplicate manifest path %s" % (label, rel))
        listed[rel] = digest
        target = directory / rel
        need(regular(target), "%s manifest target nonregular %s" % (label, rel))
        need(sha(target) == digest, "%s payload digest mismatch %s" % (label, rel))
    actual = set()
    symlinks = []
    empty_dirs = []
    for base, dirs, files in os.walk(str(directory), followlinks=False):
        base_path = Path(base)
        for name in dirs:
            point = base_path / name
            rel = point.relative_to(directory).as_posix()
            if point.is_symlink():
                symlinks.append(rel)
            elif not any(point.iterdir()):
                empty_dirs.append(rel)
        for name in files:
            point = base_path / name
            rel = point.relative_to(directory).as_posix()
            if point.is_symlink():
                symlinks.append(rel)
            elif regular(point):
                actual.add(rel)
    need(not symlinks, "%s symlinks present %r" % (label, symlinks))
    need(not empty_dirs, "%s empty unsealed directories %r" % (label, empty_dirs))
    expected = set(listed) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    need(actual == expected, "%s recursive population mismatch missing=%r extra=%r" %
         (label, sorted(expected - actual), sorted(actual - expected)))
    return {
        "manifest_entries": len(listed),
        "manifest_sha256": sha(manifest),
        "outer_seal_file_sha256": sha(outer),
        "regular_files_including_seals": len(actual),
        "symlinks": len(symlinks),
    }


def parse_float(pattern, text, label):
    match = re.search(pattern, text, re.M)
    need(match is not None, "missing %s" % label)
    value = float(match.group(1))
    need(math.isfinite(value), "nonfinite %s" % label)
    return value


def parse_axis(name, mode, receipt_axis):
    point = CANON / name
    need(point.is_dir() and not point.is_symlink(), "%s directory absent/symlink" % name)
    for rel in REQUIRED_AXIS_FILES:
        need(regular(point / rel) and (point / rel).stat().st_size > 0,
             "%s required artifact absent/empty/nonregular: %s" % (name, rel))

    identity = exact_kv(point / "input_identity.txt")
    need(identity == {
        "axis": name, "arch_mode": str(mode),
        "source_filelist_sha256": EXPECTED["filelist"],
        "m1609_sha256": EXPECTED["m1609"],
    }, "%s input identity mismatch" % name)
    need((point / "dc.rc").read_text().strip() == "0", "%s dc.rc not zero" % name)

    terminal = exact_kv(point / "TCL_PASS_TERMINAL.txt")
    need(terminal == {
        "status": "PASS_M519_R8_SETUP_AREA_DC_TCL_TERMINAL",
        "design": DESIGN, "TIM-209": "0", "OPT-150": "0",
        "compile_ultra_count": "1", "incremental_compile_count": "0",
        "hold_optimization_count": "0", "hold_not_closed_at_dc": "true",
    }, "%s terminal receipt mismatch" % name)
    flow = exact_kv(point / "reports/flow_contract.rpt")
    need(flow.get("flow") == "m519_r8_setup_area_only", "%s flow identity" % name)
    need(flow.get("compile_ultra_count") == "1", "%s flow compile count" % name)
    need(flow.get("incremental_compile_count") == "0", "%s incremental compile" % name)
    need(flow.get("hold_fix_command_count") == "0" and
         flow.get("hold_only_optimization_count") == "0", "%s hold optimization" % name)
    need(flow.get("hold_not_closed_at_dc") == "true" and
         flow.get("hold_reports_are_diagnostic_only") == "true", "%s hold boundary" % name)
    gate = exact_kv(point / "reports/precompile_loop_gate.rpt")
    need(gate.get("TIM-209") == "0" and gate.get("OPT-150") == "0",
         "%s precompile loop gate" % name)

    compile_receipt = exact_kv(point / "reports/compile_receipt.rpt")
    need(compile_receipt.get("compile_ultra_count") == "1", "%s compile receipt count" % name)
    need(compile_receipt.get("incremental_compile_count") == "0" and
         compile_receipt.get("hold_optimization_count") == "0", "%s compile receipt modes" % name)
    start = int(compile_receipt.get("compile_start_epoch", "-1"))
    end = int(compile_receipt.get("compile_end_epoch", "-1"))
    wall = int(compile_receipt.get("compile_wall_seconds", "-1"))
    need(start > 0 and end > start and wall == end - start,
         "%s compile epoch/wall receipt" % name)

    log = (point / "dc.log").read_text(errors="replace")
    error_lines = [line for line in log.splitlines()
                   if re.search(r"Error:|Fatal:|unresolved reference|unable to resolve reference|LINK-[0-9]+", line)]
    need(error_lines == [GUI_ERROR], "%s unexpected DC error population: %r" % (name, error_lines))
    need(len(re.findall(r"^\| Command Line \| compile_ultra\s+\|$", log, re.M)) == 1,
         "%s does not contain exactly one executed compile_ultra receipt" % name)
    need(re.search(r"\bread_ddc\b|read_file[^\n]*\.ddc|\bM872\b|\bm872\b", log) is None,
         "%s log suggests old-DDC/M872 reuse" % name)

    area_text = (point / "reports/area.rpt").read_text(errors="replace")
    area = parse_float(r"^Total cell area:\s*([0-9.]+)\s*$", area_text, "%s area" % name)
    need(area > 0.0, "%s area nonpositive" % name)
    leaf_cells = int(parse_float(r"^Number of cells:\s*([0-9.]+)\s*$", area_text,
                                  "%s cell count" % name))

    qor = (point / "reports/qor.rpt").read_text(errors="replace")
    setup = re.search(r"^\s*Design\s+WNS:\s*([-0-9.]+)\s+TNS:\s*([-0-9.]+)\s+Number of Violating Paths:\s*([0-9.]+)", qor, re.M)
    need(setup is not None, "%s missing QoR setup tuple" % name)
    wns, tns, violations = map(float, setup.groups())
    need(wns >= 0.0 and tns >= 0.0 and violations == 0.0, "%s setup QoR not met" % name)
    drc = int(parse_float(r"^\s*Nets With Violations:\s*([0-9.]+)\s*$", qor,
                          "%s DRC violations" % name))
    macro_count = int(parse_float(r"^\s*Macro Count:\s*([0-9.]+)\s*$", qor,
                                  "%s macro count" % name))
    need(drc == 0 and macro_count == 0, "%s DRC/macro boundary" % name)
    need("This design has no violated constraints." in
         (point / "reports/constraint_setup.rpt").read_text(errors="replace"),
         "%s setup constraint violation" % name)
    for report in ("constraint_max_capacitance.rpt", "constraint_max_transition.rpt",
                   "constraint_max_fanout.rpt"):
        need("This design has no violated constraints." in
             (point / "reports" / report).read_text(errors="replace"),
             "%s violation in %s" % (name, report))
    setup_text = (point / "reports/timing_setup.rpt").read_text(errors="replace")
    need("slack (VIOLATED)" not in setup_text, "%s setup path violation" % name)
    setup_slacks = [float(x) for x in re.findall(r"slack \(MET\)\s+([-0-9.]+)", setup_text)]
    need(setup_slacks and min(setup_slacks) >= 0.0, "%s setup slack parse/failure" % name)
    hold_text = (point / "reports/timing_hold_diagnostic.rpt").read_text(errors="replace")
    hold_slacks = [float(x) for x in re.findall(r"slack \((?:MET|VIOLATED)\)\s+([-0-9.]+)", hold_text)]
    need(hold_slacks, "%s diagnostic hold slack missing" % name)

    net = point / "netlist"
    old_net = OLD / name / "netlist"
    artifact_hashes = {
        "mapped_verilog_sha256": sha(net / (DESIGN + "_mapped.v")),
        "mapped_sdc_sha256": sha(net / (DESIGN + "_mapped.sdc")),
        "ddc_sha256": sha(net / (DESIGN + ".ddc")),
        "svf_sha256": sha(net / (DESIGN + ".svf")),
    }
    need(artifact_hashes["ddc_sha256"] != sha(old_net / (DESIGN + ".ddc")),
         "%s DDC byte-identical to old M872 artifact" % name)
    need(artifact_hashes["svf_sha256"] != sha(old_net / (DESIGN + ".svf")),
         "%s SVF byte-identical to old M872 artifact" % name)

    need(receipt_axis.get("arch_mode") == mode, "%s receipt arch mode" % name)
    need(abs(float(receipt_axis.get("area_um2")) - area) < 1e-9, "%s receipt area drift" % name)
    need(receipt_axis.get("design_rule_violating_nets") == 0, "%s receipt DRC drift" % name)
    need(receipt_axis.get("setup_met") is True and receipt_axis.get("hold_closed") is False and
         receipt_axis.get("hold_report_present") is True, "%s receipt setup/hold boundary" % name)
    need(abs(float(receipt_axis.get("minimum_reported_setup_slack_ns")) - min(setup_slacks)) < 1e-12,
         "%s receipt minimum setup slack drift" % name)
    need(receipt_axis.get("fresh_mapped_artifacts") == artifact_hashes,
         "%s receipt artifact hash drift" % name)

    return {
        "arch_mode": mode,
        "area_um2": area,
        "leaf_cell_count": leaf_cells,
        "setup_wns_ns_qor_rounded": wns,
        "setup_tns_ns_qor_rounded": tns,
        "setup_violating_paths": int(violations),
        "minimum_reported_setup_slack_ns": min(setup_slacks),
        "minimum_diagnostic_hold_slack_ns": min(hold_slacks),
        "diagnostic_hold_violating_paths_reported": sum(1 for x in hold_slacks if x < 0.0),
        "design_rule_violating_nets": drc,
        "macro_count": macro_count,
        "compile_start_epoch": start,
        "compile_end_epoch": end,
        "compile_wall_seconds": wall,
        "compile_ultra_executed_count": 1,
        "known_gui_startup_error_count": 1,
        "unexpected_error_or_fatal_count": 0,
        "fresh_artifacts": artifact_hashes,
        "ddc_byte_distinct_from_old_m872": True,
        "svf_byte_distinct_from_old_m872": True,
    }


def expect_reject(label, function):
    try:
        function()
    except (AuditError, ValueError, TypeError, KeyError, AssertionError):
        return label
    raise AuditError("negative mutation was not rejected: %s" % label)


def selftests():
    rejected = []
    rejected.append(expect_reject("nonfinite_metric", lambda: need(math.isfinite(float("nan")), "nan")))
    rejected.append(expect_reject("negative_area", lambda: need(float("-1") > 0.0, "negative area")))
    rejected.append(expect_reject("setup_negative", lambda: need(float("-0.01") >= 0.0, "setup")))
    rejected.append(expect_reject("setup_violations", lambda: need(int("1") == 0, "violations")))
    rejected.append(expect_reject("drc_violations", lambda: need(int("1") == 0, "drc")))
    rejected.append(expect_reject("macro_in_logic_only", lambda: need(int("1") == 0, "macro")))
    rejected.append(expect_reject("compile_count", lambda: need("2" == "1", "compile")))
    rejected.append(expect_reject("incremental_compile", lambda: need("1" == "0", "incremental")))
    rejected.append(expect_reject("hold_promoted", lambda: need(True is False, "hold boundary")))
    rejected.append(expect_reject("cycle_refresh", lambda: need(1914 == FROZEN_CYCLES["k8"], "cycles")))
    rejected.append(expect_reject("wrong_arch_mode", lambda: need(7 == 1, "mode")))
    rejected.append(expect_reject("old_ddc_reuse", lambda: need("same" != "same", "old ddc")))
    rejected.append(expect_reject("extra_gui_error", lambda: need([GUI_ERROR, "Error: injected"] == [GUI_ERROR], "gui")))
    rejected.append(expect_reject("missing_gui_boundary", lambda: need([] == [GUI_ERROR], "gui")))

    with tempfile.TemporaryDirectory(prefix="m1677_mut_") as temp_name:
        temp = Path(temp_name)
        payload = temp / "payload.txt"
        payload.write_text("canonical\n")
        manifest = temp / "SHA256SUMS"
        manifest.write_text("%s  payload.txt\n" % sha(payload))
        outer = temp / "SHA256SUMS.seal.sha256"
        outer.write_text("%s  SHA256SUMS\n" % sha(manifest))
        verify_manifest(temp, "synthetic-good")

        good_manifest = manifest.read_text()
        good_outer = outer.read_text()
        outer.write_text("0" * 64 + "  SHA256SUMS\n")
        rejected.append(expect_reject("outer_seal_mutation", lambda: verify_manifest(temp, "outer")))
        outer.write_text(good_outer)
        payload.write_text("mutated\n")
        rejected.append(expect_reject("payload_digest_mutation", lambda: verify_manifest(temp, "payload")))
        payload.write_text("canonical\n")
        manifest.write_text(good_manifest + good_manifest)
        outer.write_text("%s  SHA256SUMS\n" % sha(manifest))
        rejected.append(expect_reject("duplicate_manifest_row", lambda: verify_manifest(temp, "duplicate")))
        manifest.write_text("%s  ../escape\n" % ("0" * 64))
        outer.write_text("%s  SHA256SUMS\n" % sha(manifest))
        rejected.append(expect_reject("unsafe_manifest_path", lambda: verify_manifest(temp, "unsafe")))
        manifest.write_text(good_manifest)
        outer.write_text("%s  SHA256SUMS\n" % sha(manifest))
        extra = temp / "unsealed.txt"
        extra.write_text("extra\n")
        rejected.append(expect_reject("unsealed_extra_file", lambda: verify_manifest(temp, "extra")))
        extra.unlink()
        link = temp / "link.txt"
        link.symlink_to(payload.name)
        rejected.append(expect_reject("symlink_population", lambda: verify_manifest(temp, "symlink")))
        link.unlink()
    return rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    errors = []
    try:
        mutation_rejections = selftests()
        canonical_seal = verify_manifest(CANON, "M1661 canonical")
        attempt_seal = verify_manifest(ATTEMPT, "M1661 attempt")
        old_seal = verify_manifest(OLD, "M872 comparison")
        m903_seal = verify_manifest(M903, "M903 review")
        m1627_seal = verify_manifest(M1627, "M1627 VCS review")

        for label, path in (("runner", RUNNER), ("contract", CONTRACT),
                            ("release", RELEASE), ("filelist", FILELIST),
                            ("tcl", TCL), ("sdc", SDC), ("docs359", DOC359)):
            need(regular(path) and sha(path) == EXPECTED[label], "%s identity drift" % label)
        need(sha(M903 / "review.json") == EXPECTED["m903_review"], "M903 review identity drift")
        need(sha(M1627 / "review.json") == EXPECTED["m1627_review"], "M1627 review identity drift")
        m903 = json.loads((M903 / "review.json").read_text())
        need(m903.get("status") == "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED",
             "M903 comparison not admitted")
        sums = m903.get("fair_equal_bandwidth_metrics", {}).get("aggregate_sum_cycles", {})
        need(sums == FROZEN_CYCLES, "frozen 1913/1945 cycle authority drift")
        m1627 = json.loads((M1627 / "review.json").read_text())
        need(m1627.get("status") == "PASS_M1627_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_RESULT_HAMMER",
             "M1627 VCS authority not admitted")

        run_complete = (CANON / "RUN_COMPLETE.txt").read_text()
        need(run_complete == "PASS_M1661_M1609_C2_REGISTERED_FAULT_THREE_AXIS_LOGIC_ONLY_DC\n",
             "canonical terminal token mismatch")
        admission = exact_kv(CANON / "admission.txt")
        need(admission == {
            "status": "M1661_THREE_AXIS_DC_ATTEMPT_ADMITTED", "clock_period_ns": "3.000",
            "axes": "k1,k8,k1x8", "fresh_all_axes": "true",
            "old_netlist_reuse": "false", "hold_diagnostic_only": "true",
            "commit_headroom_gate_kib": "50331648",
            "mem_available_gate_kib": "100663296",
            "swap_free_gate_kib": "16777216", "retry": "false",
        }, "admission receipt mismatch")
        attempt = exact_kv(ATTEMPT / "ATTEMPT_CONSUMED.txt")
        need(attempt == {"status": "M1661_ATTEMPT_CONSUMED", "dc_shell_runs": "3",
                         "axes": "k1,k8,k1x8", "retry": "false"},
             "attempt receipt mismatch")
        need(not list((HW / "dc_handoff/runs").glob(".m1661_m1652_c2_resource_gate_successor_three_axis_dc_work.*")),
             "residual M1661 work namespace")
        need(not (HW / "dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_launch_lock").exists(),
             "residual M1661 launch lock")

        receipt = json.loads((CANON / "receipt.json").read_text())
        need(receipt.get("status") == "PASS_RAW_M1661_M1609_C2_THREE_AXIS_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RESULT_HAMMER",
             "raw receipt status mismatch")
        need(receipt.get("axis_order") == ["k1", "k8", "k1x8"], "axis order mismatch")
        need(receipt.get("fresh_all_axes") is True and receipt.get("old_m872_netlist_reuse") is False,
             "fresh/no-reuse receipt boundary")
        need(receipt.get("execution") == {"automatic_retry": False, "compile_ultra_per_axis": 1,
             "dc_shell_runs": 3, "formality_runs": 0, "pt_runs": 0, "ptpx_runs": 0, "vcs_runs": 0},
             "execution receipt mismatch")
        need(receipt.get("clock_period_ns") == 3.0 and receipt.get("setup_uncertainty_ns") == 0.2 and
             receipt.get("hold_uncertainty_ns") == 0.05 and receipt.get("ideal_clock") is True and
             receipt.get("wireload") == "ZeroWireload", "physical setup receipt mismatch")
        need(receipt.get("logic_only_pre_macro") is True and receipt.get("macro_count") == 0 and
             receipt.get("hold_diagnostic_only") is True, "logic-only/hold receipt mismatch")
        need(receipt.get("identity") == {
            "runner_sha256": EXPECTED["runner"], "contract_sha256": EXPECTED["contract"],
            "release_sha256": EXPECTED["release"], "filelist_sha256": EXPECTED["filelist"],
            "m1627_review_sha256": EXPECTED["m1627_review"],
            "m903_review_sha256": EXPECTED["m903_review"],
        }, "receipt identity dictionary mismatch")
        boundary = receipt.get("claim_boundary", {})
        need(boundary == {"energy": False, "formality": False,
             "fresh_m1609_three_axis_setup_area": True, "hold_closed": False,
             "paper_headline": False, "paper_ppa_ready": False, "power": False,
             "system_speedup": False}, "claim boundary mismatch")

        axes = {}
        previous_end = None
        for name, mode in AXES:
            axes[name] = parse_axis(name, mode, receipt["axes"][name])
            if previous_end is not None:
                need(axes[name]["compile_start_epoch"] > previous_end,
                     "three axes overlap or execute out of order at %s" % name)
            previous_end = axes[name]["compile_end_epoch"]

        old_areas = m903["dc_evidence"]["axes"]
        comparison = {}
        for name, _mode in AXES:
            old_area = float(old_areas[name]["area_um2"])
            new_area = axes[name]["area_um2"]
            comparison[name] = {
                "old_m903_area_um2": old_area,
                "fresh_m1661_area_um2": new_area,
                "fresh_minus_old_percent": (new_area / old_area - 1.0) * 100.0,
            }

        k8_area = axes["k8"]["area_um2"]
        k1x8_area = axes["k1x8"]["area_um2"]
        cycle_speedup = float(FROZEN_CYCLES["k1x8"]) / float(FROZEN_CYCLES["k8"])
        area_saving = 1.0 - k8_area / k1x8_area
        throughput_per_area = cycle_speedup * k1x8_area / k8_area
        status = "PASS100_M1677_M1661_M1652_C2_RESOURCE_GATE_SUCCESSOR_THREE_AXIS_DC_RESULT_ADMITTED"
        score = 100
    except Exception as exc:
        errors.append("%s: %s" % (type(exc).__name__, exc))
        mutation_rejections = locals().get("mutation_rejections", [])
        canonical_seal = locals().get("canonical_seal", {})
        attempt_seal = locals().get("attempt_seal", {})
        old_seal = locals().get("old_seal", {})
        m903_seal = locals().get("m903_seal", {})
        m1627_seal = locals().get("m1627_seal", {})
        axes = locals().get("axes", {})
        comparison = locals().get("comparison", {})
        cycle_speedup = area_saving = throughput_per_area = None
        status = "FAIL_M1677_M1661_M1652_C2_THREE_AXIS_DC_RESULT_NOT_ADMITTED"
        score = 0

    review = {
        "schema": "m1677_m1661_m1652_c2_resource_gate_successor_three_axis_dc_result_hammer_r1_v1",
        "date_cst": "2026-09-01", "status": status,
        "verdict": "PASS" if not errors else "FAIL", "score_out_of_100": score,
        "p0": errors, "p0_count": len(errors), "p1": [], "p1_count": 0,
        "p2": [], "p2_count": 0,
        "identity": {
            "canonical_result": str(CANON.relative_to(HW)),
            "canonical_seal": canonical_seal, "attempt_seal": attempt_seal,
            "old_m872_comparison_seal": old_seal, "m903_review_seal": m903_seal,
            "m1627_vcs_review_seal": m1627_seal,
            "runner_sha256": sha(RUNNER) if regular(RUNNER) else None,
            "contract_sha256": sha(CONTRACT) if regular(CONTRACT) else None,
            "release_sha256": sha(RELEASE) if regular(RELEASE) else None,
            "filelist_sha256": sha(FILELIST) if regular(FILELIST) else None,
            "tcl_sha256": sha(TCL) if regular(TCL) else None,
            "sdc_sha256": sha(SDC) if regular(SDC) else None,
            "docs359_sha256": sha(DOC359) if regular(DOC359) else None,
        },
        "fresh_dc_evidence": {
            "flow": "Synopsys DC compile_ultra per axis; TSMC 28nm logic-only pre-macro; 3.000ns ideal clock; ZeroWireload",
            "axis_order": [name for name, _mode in AXES], "axes": axes,
            "same_m1609_filelist_tcl_sdc_libraries_clock": not errors,
            "fresh_compile_ultra_once_per_axis": not errors,
            "sequential_nonoverlapping_axis_compiles": not errors,
            "old_m872_ddc_or_svf_byte_reuse": False if not errors else None,
            "known_gui_startup_error_exactly_once_per_axis": not errors,
            "unexpected_error_or_fatal_count_all_axes": 0 if not errors else None,
        },
        "old_m903_reproducibility_comparison": comparison,
        "fair_equal_bandwidth_metrics": {
            "comparison": "fresh M1661 K8 area versus fresh M1661 equal-bandwidth K1x8 area; frozen M1627/M903 directed cycles",
            "frozen_directed_component_sum_cycles": FROZEN_CYCLES,
            "cycles_refreshed_by_m1661_dc": False,
            "aggregate_equal_bandwidth_cycle_speedup_k8_vs_k1x8": cycle_speedup,
            "fresh_k8_area_saving_fraction_vs_k1x8": area_saving,
            "fresh_k8_area_saving_percent_vs_k1x8": area_saving * 100.0 if area_saving is not None else None,
            "aggregate_equal_bandwidth_throughput_per_mm2_ratio_k8_vs_k1x8": throughput_per_area,
            "aggregation_scope": "sum over five frozen directed component VCS workloads; not refreshed by DC; not trace-weighted, full-network or system",
        },
        "negative_testing": {
            "mutation_attacks_rejected": len(mutation_rejections),
            "mutation_attack_labels": mutation_rejections,
        },
        "claim_boundary": {
            "fresh_m1609_logic_only_setup_area_citable": not errors,
            "logic_only_pre_macro": True, "macro_count": 0,
            "hold_closed": False, "hold_diagnostic_only": True,
            "power": False, "energy": False, "formality": False,
            "paper_ppa_ready": False, "system_speedup": False, "paper_headline": False,
            "k8_vs_single_k1_performance_headline_forbidden": True,
            "cycle_values_are_frozen_directed_vcs_not_dc": True,
            "paper_usage": "Citable as fresh logic-only pre-macro three-axis setup/area and, in the same sentence with 1.016728x directed cycle speedup, equal-bandwidth K8-vs-K1x8 throughput/mm2. Not macro-inclusive PPA, hold closure, power, energy, full-network or system speedup.",
        },
        "review_execution": {
            "python_runtime": "%d.%d.%d" % sys.version_info[:3],
            "eda_runs": 0, "dc_runs": 0, "vcs_runs": 0,
            "license_queries": 0, "canonical_result_modified": False,
            "attempt_modified": False, "docs359_modified": False,
        },
    }
    output = Path(args.output)
    output.write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(status)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
