#!/usr/bin/env python3
"""Independent, read-only M2030 result hammer. Never invokes EDA/license/GPU."""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNS = HW / "dc_handoff/runs"
RESULT = RUNS / "m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
ATTEMPT = RUNS / ".m2029_m2018_c2_tsbg_b4_divfree_matched_dc_attempt_consumed"
SOURCE_REVIEW = HW / "reviews/m2028_m2027_m2018_c2_tsbg_b4_divfree_matched_dc_source_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RESULT / "receipt.json": "4dd17885266217df2613ed6911b8f2827f640bf4ac88ab36c6509e22441e8f19",
    RESULT / "SHA256SUMS": "67533560afbcf46db9ad8c51c06a2adbe5981a0998345e6bd59791219c5fa887",
    RESULT / "SHA256SUMS.seal.sha256": "15132a7b8a47603a1f26eed4aad35baa6b36eeb0d1b7b70e4aa00d2210dcf0ff",
    ATTEMPT / "SHA256SUMS": "809796faecb7eb4007bbced4db4e309b6f3ef187be36a58812e2f663228e3832",
    ATTEMPT / "SHA256SUMS.seal.sha256": "1a81132cdc16bd4f1c2aaa0f104f9a87c7d9619ebff98dc495ceec0640bfff6e",
    SOURCE_REVIEW / "review.json": "a6d2bfae6dba7a33f2f846dfc057d2a4f27e6bda590e3ada14f778d3750a0ef2",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dir_seal(directory):
    assert directory.is_dir() and not directory.is_symlink()
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    digest, name = outer.read_text().strip().split(maxsplit=1)
    assert name.lstrip(" *") == manifest.name and digest == sha(manifest)
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, relative = row.split(maxsplit=1)
        relative = relative.lstrip(" *")
        target = directory / relative
        assert target.is_file() and not target.is_symlink() and sha(target) == digest
        listed.add(relative)
    actual = set(str(p.relative_to(directory)) for p in directory.rglob("*")
                 if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    assert listed == actual, (listed - actual, actual - listed)


def slacks(path):
    values = re.findall(r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)",
                        path.read_text(errors="replace"))
    assert len(values) == 100
    return [float(value) for value in values]


def verify_log(axis):
    log = axis / "dc.log"
    rows = log.read_text(errors="replace").splitlines()
    hits = [(i + 1, row) for i, row in enumerate(rows)
            if re.match(r"^(Error:|Fatal:)", row)]
    expected = "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"
    assert hits == [(32, expected)]
    block = "\n".join(rows[31:47]) + "\n"
    assert hashlib.sha256(block.encode()).hexdigest() == "3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1"
    assert rows[30] == "Initializing..." and rows[47].startswith("Current time:")
    filtered = rows[:31] + rows[47:]
    assert not any(re.match(r"^(Error:|Fatal:)", row) for row in filtered)
    assert not any(re.match(r"^(Warning|Information):.*\((TIM-209|OPT-150)\)", row)
                   for row in filtered)
    receipt = (axis / "bootstrap_log_whitelist_receipt.txt").read_text()
    for token in ("status=PASS_EXACT_SINGLE_BOOTSTRAP_BLOCK_WHITELIST",
                  "block_start_line=32", "block_end_line=47",
                  "block_sha256=3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1",
                  "other_error_fatal_tim209_opt150_count=0"):
        assert token in receipt


def verify_axis(name, expected):
    axis = RESULT / name
    assert (axis / "dc.rc").read_text() == "0\n"
    verify_log(axis)
    terminal = (axis / "TCL_PASS_TERMINAL.txt").read_text()
    for token in ("status=PASS_M519_R8_SETUP_AREA_DC_TCL_TERMINAL",
                  "TIM-209=0", "OPT-150=0", "compile_ultra_count=1",
                  "incremental_compile_count=0", "hold_optimization_count=0",
                  "hold_not_closed_at_dc=true"):
        assert token in terminal
    compile_receipt = (axis / "reports/compile_receipt.rpt").read_text()
    assert compile_receipt.count("compile_ultra_count=1") == 1
    assert "incremental_compile_count=0" in compile_receipt
    assert "hold_optimization_count=0" in compile_receipt
    assert (axis / "reports/port_count.txt").read_text().strip() == "4551"
    area_text = (axis / "reports/area.rpt").read_text(errors="replace")
    areas = re.findall(r"Total cell area:\s*([0-9.]+)", area_text)
    assert len(areas) == 1 and float(areas[0]) == expected["area_um2"]
    setup = slacks(axis / "reports/timing_setup.rpt")
    hold = slacks(axis / "reports/timing_hold_diagnostic.rpt")
    assert min(setup) == expected["setup_wns_ns"]
    assert min(hold) == expected["hold_diagnostic_wns_ns"]
    assert setup[0] == min(setup) and hold[0] == min(hold)
    assert sha(axis / "dc.log") == expected["dc_log_sha256"]
    mapped = axis / "netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v"
    assert sha(mapped) == expected["mapped_netlist_sha256"]
    assert (axis / "reports/precompile_loop_gate.rpt").read_text().splitlines() == [
        "TIM-209=0", "OPT-150=0", "status=PASS_PRECOMPILE_LOOP_GATE"]
    for report in ("constraint_setup.rpt", "constraint_max_capacitance.rpt",
                   "constraint_max_transition.rpt", "constraint_max_fanout.rpt"):
        text = (axis / "reports" / report).read_text(errors="replace")
        assert text.count("This design has no violated constraints.") == 1


def main():
    for path, expected in EXPECTED.items():
        assert path.is_file() and not path.is_symlink() and sha(path) == expected, path
    verify_dir_seal(RESULT)
    verify_dir_seal(ATTEMPT)
    verify_dir_seal(SOURCE_REVIEW)
    matches = sorted(p.name for p in RUNS.iterdir()
                     if "m2029_m2018_c2_tsbg_b4_divfree_matched" in p.name)
    assert matches == [
        ".m2029_m2018_c2_tsbg_b4_divfree_matched_dc_attempt_consumed",
        "m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902",
    ]
    assert (ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text().splitlines() == [
        "status=M2029_ATTEMPT_CONSUMED", "license_queries=1", "dc_shell_runs=2",
        "axes=ordinary_lru4,tsbg_b4", "retry=false"]
    assert (RESULT / "RUN_COMPLETE.txt").read_text() == \
        "RAW_PASS_M2029_M2018_TSBG_DIVFREE_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW\n"
    license_text = (RESULT / "license_preflight.log").read_text(errors="replace")
    assert license_text.count("Users of Design-Compiler:") == 1
    assert "license server UP" in license_text and "snpslmd: UP" in license_text
    assert "Total of 99 licenses issued;  Total of 0 licenses in use" in license_text

    receipt = json.loads((RESULT / "receipt.json").read_text())
    assert receipt["status"] == "PASS_RAW_M2029_M2018_TSBG_DIVFREE_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW"
    assert receipt["execution"] == {"automatic_retry": False, "dc_shell_runs": 2,
                                      "license_queries": 1}
    expected_axes = {
        "ordinary_lru4": {"area_um2": 249710.451846, "setup_wns_ns": 0.0264,
                          "hold_diagnostic_wns_ns": -0.0164, "public_port_count": 4551,
                          "schedule_mode": 0,
                          "dc_log_sha256": "495a4c8b77c3d08fe66ae7ec897b22734f92aea50eb27e3d24b96d87db4c1cab",
                          "mapped_netlist_sha256": "f5847f355329a52511ab044ef458284a19ae424ac778418a4bc4778bb2d3a2b0"},
        "tsbg_b4": {"area_um2": 249739.809848, "setup_wns_ns": 0.0688,
                    "hold_diagnostic_wns_ns": -0.0164, "public_port_count": 4551,
                    "schedule_mode": 1,
                    "dc_log_sha256": "d3eb059f9625d23f88bf09dcfc2e90c9f929fb50d0090e54bf7b73123b960ac4",
                    "mapped_netlist_sha256": "739eb76dcb732ec0c66b75392c768cbe36027ecc5d458bd4b088f8488f67c9af"},
    }
    assert receipt["axes"] == expected_axes
    for name, expected in expected_axes.items():
        verify_axis(name, expected)
    ratio = expected_axes["tsbg_b4"]["area_um2"] / expected_axes["ordinary_lru4"]["area_um2"]
    comparison = receipt["comparison"]
    assert math.isclose(comparison["tsbg_over_ordinary_logic_area_ratio"], ratio,
                        rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(comparison["tsbg_logic_area_overhead_fraction"], ratio - 1.0,
                        rel_tol=0.0, abs_tol=1e-15)
    assert comparison["public_port_count_equal"] is True
    assert comparison["both_setup_met"] is True
    assert comparison["m2026_directed_bundle_request_reduction_fraction"] == 0.75
    assert comparison["m2026_directed_scalar_request_reduction_fraction"] == 0.75
    assert comparison["m1866_cpu_premodel_speedup_not_upgraded_to_rtl"] is True
    assert all(receipt["candidate_gate"].values())
    boundary = receipt["claim_boundary"]
    for key in ("hold_closed", "power", "energy", "exact_rtl_cycle_speedup",
                "same_area", "system_speedup", "paper_ppa_ready",
                "production_g48_dynamically_verified",
                "cpu_premodel_2p533808x_upgraded_by_dc"):
        assert boundary[key] is False, key
    assert boundary["logic_only_pre_macro"] is True
    assert boundary["ideal_clock"] is True
    assert boundary["wireload"] == "ZeroWireload"
    assert boundary["physical_schedule_ablation_not_full_conventional_baseline"] is True
    assert boundary["state_arrays_synthesized_as_standard_cells"] is True
    print("PASS_M2030_INDEPENDENT_RESULT_HAMMER axes=2 ports=4551 ratio={0:.12f} setup_met=2 hold_closed=0 p0=0 p1=0 p2=0 no_eda=1 no_license=1".format(ratio))


if __name__ == "__main__":
    main()
