#!/usr/bin/env python3
"""Read-only M913 forensic verifier for the immutable M892 quarantine."""

from __future__ import print_function

import hashlib
import pathlib
import re


HW_ROOT = pathlib.Path(__file__).resolve().parents[2]
QREL = pathlib.Path(
    "dc_handoff/runs/"
    "m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829."
    "failed_or_incomplete.2411700.quarantine"
)
Q = HW_ROOT / QREL
ATTEMPT = HW_ROOT / (
    "dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_attempt_consumed"
)
CANONICAL = HW_ROOT / (
    "dc_handoff/runs/m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829"
)
RUNNER = HW_ROOT / (
    "dc_handoff/scripts/"
    "run_dc_m892_m528_r21_macro_aware_product_schema_repair_exact_sha_r1.sh"
)
TCL = HW_ROOT / (
    "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_candidate.tcl"
)
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path):
    entries = []
    for line in path.read_text().splitlines():
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        assert match, "malformed manifest line: %r" % line
        entries.append((match.group(1), match.group(2)))
    return entries


def verify_seal(directory):
    assert directory.is_dir() and not directory.is_symlink()
    assert not any(path.is_symlink() for path in directory.rglob("*"))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    for expected, rel in parse_manifest(manifest):
        item = directory / rel[2:] if rel.startswith("./") else directory / rel
        assert item.is_file() and sha(item) == expected, rel
    outer_entries = parse_manifest(outer)
    assert outer_entries == [(sha(manifest), "SHA256SUMS")]


def first_path(report):
    start = re.search(r"^  Startpoint:\s*(\S+)", report, re.M).group(1)
    end = re.search(r"^  Endpoint:\s*(\S+)", report, re.M).group(1)
    slack = float(
        re.search(r"slack \((?:MET|VIOLATED)\)\s+([-+]?\d+(?:\.\d+)?)", report).group(1)
    )
    arrivals = re.findall(r"data arrival time\s+([-+]?\d+(?:\.\d+)?)", report)
    required = re.findall(r"data required time\s+([-+]?\d+(?:\.\d+)?)", report)
    # The full-path report repeats the arrival with a minus sign in the final
    # slack equation.  The first occurrence is the physical arrival value.
    return start, end, slack, float(arrivals[0]), float(required[0])


def main():
    verify_seal(Q)
    verify_seal(ATTEMPT)
    assert not CANONICAL.exists()
    assert (Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text() == (
        "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n"
        "exit_code=9\n"
        "fair_K_zero_bit=false\n"
        "paper_ppa_ready=false\n"
    )
    assert (Q / "dc.rc").read_text() == "0\n"
    terminal = (Q / "TCL_PASS_TERMINAL.txt").read_text()
    assert "status=PASS_M884_M528_R21_MACRO_AWARE_PRODUCT_DC_TCL_TERMINAL" in terminal
    assert "TIM-209=0" in terminal and "OPT-150=0" in terminal
    assert "macro_count_pre=9" in terminal and "macro_count_post=9" in terminal

    required = [
        "reports/link.rpt", "reports/macro_binding_audit.txt",
        "reports/check_design_precompile.rpt", "reports/check_design_postcompile.rpt",
        "reports/check_timing_precompile.rpt", "reports/check_timing_postcompile.rpt",
        "reports/resources_precompile.rpt", "reports/resources_postcompile.rpt",
        "reports/references_precompile.rpt", "reports/references_postcompile.rpt",
        "reports/hierarchy_postcompile.rpt", "reports/qor.rpt",
        "reports/area_hierarchy.rpt", "reports/timing_setup.rpt",
        "reports/timing_hold_diagnostic.rpt", "reports/constraint_setup.rpt",
        "reports/constraint_hold_diagnostic.rpt",
        "reports/constraint_max_capacitance.rpt",
        "reports/constraint_max_transition.rpt",
        "reports/constraint_max_fanout.rpt", "reports/flow_contract.rpt",
        "reports/precompile_loop_gate.rpt",
        "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v",
        "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.sdc",
        "netlist/m528_dead_write_only_1rw_product_capture_island_r2.ddc",
        "netlist/m528_dead_write_only_1rw_product_capture_island_r2.svf",
        "TCL_PASS_TERMINAL.txt",
    ]
    assert all((Q / rel).is_file() and (Q / rel).stat().st_size > 0 for rel in required)

    macro = (Q / "reports/macro_binding_audit.txt").read_text()
    assert "status=PASS_M884_RESOLVED_LIBRARY_MACRO_STRUCTURE" in macro
    assert "macro_count_pre=9" in macro and "macro_count_post=9" in macro
    netlist = (Q / "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v").read_text()
    assert netlist.count("TS1N28HPCPHVTB128X128M4S") == 9

    log = (Q / "dc.log").read_text(errors="replace")
    error_lines = re.findall(r"^Error:.*$", log, re.M)
    assert error_lines == [
        "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"
    ]
    assert 'no such variable\n    (read trace on "::env(HOME)")' in log
    assert not re.search(r"^Fatal:", log, re.M)
    assert not re.search(r"unresolved reference|unable to resolve reference", log, re.I)

    area = (Q / "reports/area_hierarchy.rpt").read_text()
    total = float(re.search(r"Total cell area:\s*([0-9.]+)", area).group(1))
    comb = float(re.search(r"Combinational area:\s*([0-9.]+)", area).group(1))
    noncomb = float(re.search(r"Noncombinational area:\s*([0-9.]+)", area).group(1))
    macro_area = float(re.search(r"Macro/Black Box area:\s*([0-9.]+)", area).group(1))
    assert abs(total - 156394.874050) < 1e-6
    assert abs(comb - 51769.367721) < 1e-6
    assert abs(noncomb - 25800.263165) < 1e-6
    assert abs(macro_area - 78825.243164) < 1e-6

    qor = (Q / "reports/qor.rpt").read_text()
    assert "Critical Path Slack:          -7.05" in qor
    assert "Total Negative Slack:     -73958.98" in qor
    assert "No. of Violating Paths:    12553.00" in qor
    assert "Worst Hold Violation:         -0.08" in qor
    assert "Total Hold Violation:       -121.60" in qor
    assert "No. of Hold Violations:    12481.00" in qor

    setup = first_path((Q / "reports/timing_setup.rpt").read_text())
    hold = first_path((Q / "reports/timing_hold_diagnostic.rpt").read_text())
    assert setup == ("exec_bank_q_reg", "psum_write_valid", -7.0468, 9.5968, 2.55)
    assert hold == (
        "slot0_data_q_reg[0]",
        "u_parent_scratch/g_slice[0].u_parent_sram",
        -0.0799,
        0.0661,
        0.146,
    )
    assert "(VIOLATED)" in (Q / "reports/constraint_setup.rpt").read_text()
    for rel in (
        "reports/constraint_max_capacitance.rpt",
        "reports/constraint_max_transition.rpt",
        "reports/constraint_max_fanout.rpt",
    ):
        assert "This design has no violated constraints." in (Q / rel).read_text()

    runner = RUNNER.read_text()
    tcl = TCL.read_text()
    assert "env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp" in runner
    assert "HOME=" not in runner[runner.index("env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp"):runner.index('"${m892_dc}" -f')]
    assert "status=PASS_M892_RESOLVED_LIBRARY_MACRO_STRUCTURE" in runner
    assert "status=PASS_M884_RESOLVED_LIBRARY_MACRO_STRUCTURE" in tcl
    assert sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

    print("PASS M913 read-only quarantine forensic checks=34")
    print("quarantine_double_seal=true attempt_double_seal=true canonical_absent=true")
    print("dc_rc=0 tcl_terminal=true macro_pre_post=9/9 netlist_macros=9")
    print("runner_exit9_first_trigger=HOME_GUI_STARTUP_ERROR_FALSE_POSITIVE")
    print("real_physical_failure=SETUP_AND_HOLD_VIOLATIONS")
    print("setup_wns_ns=-7.05 setup_tns_ns=-73958.98 setup_violating_paths=12553")
    print("hold_diag_wns_ns=-0.08 hold_diag_tns_ns=-121.60 hold_violating_paths=12481")
    print("area_total_um2=156394.874050 stdcell_um2=77569.630886 macro_um2=78825.243164")
    print("docs359_unchanged=true")


if __name__ == "__main__":
    main()
