#!/usr/bin/env python3
"""Read-only result hammer for the canonical M917/M518-r5 DC point."""

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


HW = Path("/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07")
RUN = HW / "dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829"
ATTEMPT = HW / "dc_handoff/runs/.m917_m518_r5_fixed_descendant_safe_setup_area_attempt_consumed"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sealed_dir(path, expected_entries, expected_files):
    assert path.is_dir() and not path.is_symlink()
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    rows = []
    for raw in manifest.read_text().splitlines():
        expected, rel = raw.split("  ", 1)
        member = path / rel
        assert member.is_file() and not member.is_symlink()
        assert str(member.resolve()).startswith(str(path.resolve()) + os.sep)
        assert sha(member) == expected
        rows.append(rel)
    assert len(rows) == expected_entries and len(rows) == len(set(rows))
    expected_manifest, named = outer.read_text().strip().split(None, 1)
    assert named == "SHA256SUMS" and sha(manifest) == expected_manifest
    regular = [p for p in path.rglob("*") if p.is_file()]
    symlinks = [p for p in path.rglob("*") if p.is_symlink()]
    assert not symlinks and len(regular) == expected_files
    nested_seals = {
        candidate for candidate in regular
        if candidate.parent != path and candidate.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    assert set(regular) == {path / rel for rel in rows} | {manifest, outer} | nested_seals
    return {
        "manifest_entries": len(rows),
        "regular_files_including_seals": len(regular),
        "symlinks": len(symlinks),
        "manifest_sha256": sha(manifest),
        "outer_seal_file_sha256": sha(outer),
    }


def parse_kv(path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_resource_log(path):
    rows = []
    for line in path.read_text().splitlines():
        row = dict(token.split("=", 1) for token in line.split() if "=" in token)
        assert row
        rows.append(row)
    return rows


def parse_source_ports(path):
    text = path.read_text()
    start = text.index("module m518_matched_fixed_t10_atlif")
    body = text[text.index(") (", start) + 3:text.index(");", start)]
    ports = []
    bits = 0
    skip = False
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("`ifdef"):
            skip = True
            continue
        if line.startswith("`endif"):
            skip = False
            continue
        if skip:
            continue
        line = line.split("//", 1)[0].strip().lstrip(",").rstrip(",").strip()
        if not line:
            continue
        match = re.fullmatch(
            r"(input|output)\s+logic\s*(\[[^]]+\])?\s*([A-Za-z_][A-Za-z0-9_]*)",
            line,
        )
        assert match, line
        direction, width_expr, name = match.groups()
        ports.append((direction, re.sub(r"\s+", "", width_expr or ""), name))
        if width_expr is None:
            bits += 1
        else:
            high, low = (term.strip() for term in width_expr[1:-1].split(":"))
            values = {"TAG_W": 48, "FIFO_COUNT_W": 5}
            for symbol, value in values.items():
                high = high.replace(symbol, str(value))
                low = low.replace(symbol, str(value))
            assert re.fullmatch(r"[0-9()+*/ -]+", high)
            assert re.fullmatch(r"[0-9()+*/ -]+", low)
            bits += abs(eval(high, {"__builtins__": {}}, {}) - eval(low, {"__builtins__": {}}, {})) + 1
    return ports, bits


def main():
    canonical = verify_sealed_dir(RUN, 46, 50)
    attempt = verify_sealed_dir(ATTEMPT, 2, 4)
    verify_sealed_dir(RUN / "preflight/fixed", 3, 5)

    expected_input_hashes = {}
    for raw in (RUN / "input_sha256.txt").read_text().splitlines():
        expected, named = raw.split(None, 1)
        target = Path(named) if named.startswith("/") else HW / named
        assert sha(target) == expected
        expected_input_hashes[named] = expected

    contract = json.loads((RUN / "contract.json").read_text())
    admission = json.loads((RUN / "launch_admission.json").read_text())
    assert contract["schema"] == "m916_m518_r5_fixed_descendant_safe_setup_area_dc_contract_v1"
    assert admission["status"] == "AUTHORIZED_ONE_M917_M518_R5_FIXED_DESCENDANT_SAFE_SETUP_AREA_DC_ATTEMPT"
    assert sha(RUN / "contract.json") == expected_input_hashes["contracts/m916_m518_r5_fixed_descendant_safe_setup_area_dc_contract_r1_20260829.json"]
    assert sha(RUN / "launch_admission.json") == expected_input_hashes["contracts/m917_m518_r5_fixed_descendant_safe_setup_area_dc_launch_admission_r1_20260829.json"]
    assert sha(DOC359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

    run_complete = parse_kv(RUN / "RUN_COMPLETE.txt")
    assert run_complete["status"] == "PASS_M917_M518_R5_FIXED_RAW_SETUP_AREA_DC__AWAITING_INDEPENDENT_RESULT_REVIEW"
    assert run_complete["result_reviewed"] == "false"
    assert (RUN / "fixed/dc.rc").read_text().strip() == "0"
    assert (RUN / "fixed/runtime_monitor.rc").read_text().strip() == "0"
    assert not list(RUN.rglob("*TCL_EXPLICIT_FAILURE*"))
    terminal = parse_kv(RUN / "fixed/TCL_PASS_TERMINAL.txt")
    assert terminal == {
        "status": "PASS_M518_R3_PER_POINT_SETUP_AREA_DC_TCL_TERMINAL",
        "design": "m518_matched_fixed_t10_atlif",
        "check_design_ok": "1",
        "check_timing_ok": "1",
        "dc_bit_level_port_count": "1175",
        "compile_ultra_count": "1",
        "incremental_compile_count": "0",
        "hold_optimization_count": "0",
        "hold_not_closed_at_dc": "true",
    }

    rtl = HW / "rtl_m518/m518_matched_fixed_t10_atlif.sv"
    ports, source_bits = parse_source_ports(rtl)
    assert len(ports) == 50 and source_bits == 1175
    assert (RUN / "fixed/reports/dc_bit_port_count.txt").read_text().strip() == "1175"
    structured = parse_kv(RUN / "fixed/reports/structured_postcompile_gate.rpt")
    assert structured["check_design_ok"] == structured["check_timing_ok"] == "1"
    assert structured["dc_bit_level_port_count"] == "1175"
    assert structured["expected_source_declaration_tuple_count"] == "50"

    area_text = (RUN / "fixed/reports/area.rpt").read_text()
    def area_value(label: str, integer: bool = False):
        match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)$", area_text, re.M)
        assert match
        return int(match.group(1)) if integer else float(match.group(1))
    area = area_value("Total cell area")
    cells = area_value("Number of cells", True)
    comb = area_value("Number of combinational cells", True)
    sequential = area_value("Number of sequential cells", True)
    macros = area_value("Number of macros/black boxes", True)
    assert (area, cells, comb, sequential, macros) == (62433.503388, 71898, 61325, 10573, 0)

    setup_text = (RUN / "fixed/reports/timing_setup.rpt").read_text()
    setup_slacks = [float(value) for value in re.findall(r"slack \(MET\)\s+([0-9.-]+)", setup_text)]
    assert len(setup_slacks) == 100 and min(setup_slacks) == 0.0003
    assert "slack (VIOLATED)" not in setup_text
    qor = (RUN / "fixed/reports/qor.rpt").read_text()
    assert "Total Negative Slack:          0.00" in qor
    assert "No. of Violating Paths:        0.00" in qor
    assert "Worst Hold Violation:         -0.02" in qor
    assert "Total Hold Violation:        -58.19" in qor
    assert "No. of Hold Violations:     9741.00" in qor

    constraint_reports = [
        "constraint_setup.rpt",
        "constraint_max_capacitance.rpt",
        "constraint_max_transition.rpt",
        "constraint_max_fanout.rpt",
    ]
    for name in constraint_reports:
        report = (RUN / "fixed/reports" / name).read_text()
        assert "This design has no violated constraints." in report
    assert (RUN / "fixed/reports/check_design_postcompile.rpt").read_text().strip().endswith("1")
    check_timing = (RUN / "fixed/reports/check_timing_postcompile.rpt").read_text()
    assert "Checking unconstrained_endpoints" in check_timing and check_timing.strip().endswith("1")

    flow = parse_kv(RUN / "fixed/reports/flow_contract.rpt")
    receipt = parse_kv(RUN / "fixed/reports/compile_receipt.rpt")
    assert flow["compile_ultra_count"] == receipt["compile_ultra_count"] == "1"
    assert flow["incremental_compile_count"] == receipt["incremental_compile_count"] == "0"
    assert flow["hold_fix_command_count"] == "0"
    assert flow["hold_only_optimization_count"] == "0"
    assert receipt["hold_optimization_count"] == "0"
    assert flow["hold_not_closed_at_dc"] == "true"

    preflight = parse_resource_log(RUN / "preflight/fixed/resource_preflight.log")
    runtime = parse_resource_log(RUN / "fixed/resource_runtime.log")
    assert len(preflight) == 3 and len(runtime) == 34
    for rows in (preflight, runtime):
        assert all(row["external_eda_collision"] == "none" for row in rows)
        assert all(row["cgroup_failcnt"] == row["cgroup_under_oom"] == row["cgroup_oom_kill"] == "0" for row in rows)
    assert min(int(row["commit_headroom_kib"]) for row in runtime) == 117719968
    assert min(int(row["mem_available_kib"]) for row in runtime) == 413454520
    assert min(int(row["swap_free_kib"]) for row in runtime) == 54219772
    assert (RUN / "preflight/fixed/resource_preflight_external_collisions.tsv").stat().st_size == 0
    assert (RUN / "fixed/resource_runtime_external_collisions.tsv").stat().st_size == 0
    gate_lines = (RUN / "fixed/runtime_gate_every_snapshot.log").read_text().splitlines()
    assert len(gate_lines) == 33 and all("gate=none" in line for line in gate_lines)
    final_gate = dict(
        token.split("=", 1)
        for token in (RUN / "fixed/runtime_final_gate_ack.txt").read_text().split()
        if "=" in token
    )
    assert final_gate["samples_including_final"] == "34"
    assert final_gate["runtime_resource_latch"] == "0"
    assert final_gate["job_tree_empty_before_ack"] == "true"
    assert final_gate["status"] == "PASS_FINAL_GATE_ACK"

    child = parse_kv(RUN / "fixed/dc_child_identity.txt")
    assert child["pid"] == child["pgrp"] == child["session"]
    assert child["exe"].endswith("/common_shell_exec")
    cmdline = bytes.fromhex(child["cmdline_nul_hex"]).split(b"\0")
    assert cmdline[-2].endswith(b"run_dc_m518_r3_per_point_setup_area.tcl")
    assert (RUN / "fixed/runtime_descendant_exclusions.tsv").stat().st_size == 0

    home = RUN / "safe_home"
    assert home.is_dir() and not home.is_symlink() and (home.stat().st_mode & 0o777) == 0o700
    safe_home_record = parse_kv(RUN / "safe_home_contract.txt")
    assert safe_home_record["mode"] == "0700" and safe_home_record["inside_work"] == "true"

    log = (RUN / "fixed/dc.log").read_text()
    assert not re.search(r"^(Error|Fatal):", log, re.M)
    assert "no such variable" not in log and "::env(HOME)" not in log
    warning_codes = Counter(re.findall(r"^Warning:.*\(([A-Z][A-Z0-9-]*-\d+)\)", log, re.M))
    assert warning_codes == Counter({
        "LINT-1": 482, "LINT-31": 202, "LINT-52": 200, "VER-318": 32,
        "TIM-134": 4, "VER-281": 2, "UISN-40": 2, "UID-228": 1,
    })
    netlist = (RUN / "fixed/netlist/m518_matched_fixed_t10_atlif_mapped.v").read_text()
    assert len(re.findall(r"^module ", netlist, re.M)) == 1
    assert len(re.findall(r"^endmodule", netlist, re.M)) == 1
    assert "DW_" not in netlist and "blackbox" not in netlist.lower()

    output = {
        "status": "PASS_M928_M917_M518_R5_FIXED_LOGIC_ONLY_DC_RESULT_ADMITTED",
        "canonical_seal": canonical,
        "attempt_seal": attempt,
        "area_um2": area,
        "cell_count": cells,
        "combinational_cell_count": comb,
        "sequential_cell_count": sequential,
        "macro_count": macros,
        "setup_paths_checked": len(setup_slacks),
        "minimum_setup_slack_ns": min(setup_slacks),
        "source_declaration_tuple_count": len(ports),
        "expanded_bit_port_count": source_bits,
        "runtime_samples_including_final": len(runtime),
        "runtime_min_commit_headroom_kib": min(int(row["commit_headroom_kib"]) for row in runtime),
        "runtime_min_mem_available_kib": min(int(row["mem_available_kib"]) for row in runtime),
        "runtime_min_swap_free_kib": min(int(row["swap_free_kib"]) for row in runtime),
        "warning_codes": dict(sorted(warning_codes.items())),
        "docs359_sha256": sha(DOC359),
        "eda_runs_by_reviewer": 0,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
