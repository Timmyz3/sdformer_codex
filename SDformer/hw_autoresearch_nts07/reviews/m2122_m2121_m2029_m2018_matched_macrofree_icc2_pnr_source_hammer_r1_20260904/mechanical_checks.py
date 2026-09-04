#!/usr/bin/env python3
"""Independent, no-EDA M2122 source hammer for the repaired ICC2 campaign."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2121_m2122_m2029_m2018_matched_macrofree_icc2_pnr_one_shot.sh"
TCL = HW / "dc_handoff/scripts/run_icc2_m2121_m2029_m2018_matched_macrofree_axis.tcl"
PARSER = HW / "system_simulator/scripts/parse_m2121_m2029_m2018_matched_macrofree_icc2_pnr.py"
TESTS = HW / "tests/test_m2121_m2029_m2018_matched_macrofree_icc2_pnr.py"
CONTRACT = HW / "contracts/m2121_m2029_m2018_matched_macrofree_icc2_pnr_source_contract_r1_20260904.json"
MANIFEST = HW / "dc_handoff/manifests/m2121_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
AUTHOR = HW / "reviews/m2121_m2029_m2018_matched_macrofree_icc2_pnr_source_author_receipt_r1_20260904"
PREVIOUS = HW / "reviews/m2111_m2110_m2029_m2018_matched_macrofree_icc2_pnr_source_hammer_r1_20260904"
M2029 = HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
ADDENDUM = HW / "reviews/tcasii2027_m2018_icc2_physical_tech_readonly_addendum_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
MW = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
ICC2 = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
RESULT = HW / "dc_handoff/runs/m2123_m2029_m2018_matched_macrofree_icc2_pnr_raw_r1_20260904"
ATTEMPT = HW / "dc_handoff/runs/.m2123_m2029_m2018_matched_macrofree_icc2_pnr_attempt_consumed"
LOCK = HW / "dc_handoff/runs/.m2123_m2029_m2018_matched_macrofree_icc2_pnr_launch_lock"
DOC_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise AssertionError("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite JSON token: " + token)))


def need(condition: bool, label: str, checks: dict) -> None:
    checks[label] = bool(condition)
    if not condition:
        raise AssertionError(label)


def reject(callback, label: str, checks: dict) -> None:
    try:
        callback()
    except Exception:
        checks[label] = True
        return
    checks[label] = False
    raise AssertionError(label)


def verify_sealed_dir(root: Path) -> bool:
    if not root.is_dir() or root.is_symlink():
        return False
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if not manifest.is_file() or manifest.is_symlink() or not outer.is_file() or outer.is_symlink():
        return False
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        return False
    expected = set()
    for raw in manifest.read_text().splitlines():
        fields = raw.split(maxsplit=1)
        if len(fields) != 2:
            return False
        rel = Path(fields[1].lstrip("*"))
        path = root / rel
        key = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts or key in expected:
            return False
        if not path.is_file() or path.is_symlink() or sha(path) != fields[0]:
            return False
        expected.add(key)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    return expected == actual


def replace_fact(path: Path, key: str, value: str) -> None:
    text = path.read_text()
    text, count = re.subn(r"^" + re.escape(key) + r"=.*$", key + "=" + value,
                          text, count=1, flags=re.M)
    if count != 1:
        raise AssertionError("missing fact: " + key)
    path.write_text(text)


def main() -> None:
    checks = {}
    frozen = [RUNNER, TCL, PARSER, TESTS, CONTRACT, MANIFEST, DOC359]
    frozen_before = {str(p): sha(p) for p in frozen}
    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "source_receipt.json")

    need(contract["schema"] == "m2121_m2029_m2018_matched_macrofree_icc2_pnr_source_contract_r1_v2",
         "contract_schema_v2_exact", checks)
    need(contract["status"] == "SOURCE_ONLY_PENDING_M2122_INDEPENDENT_HAMMER__NO_EDA_AUTHORIZED",
         "source_only_status_exact", checks)
    need(contract["authorization"] == {
        "license_queries": 0, "icc2_shell_runs": 0, "all_other_eda_runs": 0,
        "gpu_runs": 0, "automatic_retry": False,
        "release_condition": "M2122 independent source hammer must be double-sealed with score >=95 and P0/P1/P2=0/0/0; actual runner then creates only M2123 raw result with exactly one license query and two sequential ICC2 shells; M2124 independently reviews the raw result",
    }, "contract_pre_review_authorization_zero", checks)
    need(author["status"].startswith("PASS_M2121_REPAIRED_SOURCE_ONLY_PENDING_M2122"),
         "author_receipt_source_only", checks)

    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract_inner_seal_exact", checks)
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract_outer_seal_exact", checks)
    need(verify_sealed_dir(AUTHOR), "author_receipt_double_seal_exhaustive", checks)
    need(verify_sealed_dir(PREVIOUS), "m2111_failure_double_seal_exhaustive", checks)
    need(verify_sealed_dir(M2029), "m2029_input_double_seal_exhaustive", checks)
    need(verify_sealed_dir(ADDENDUM), "physical_addendum_double_seal_exhaustive", checks)
    need(sha(DOC359) == DOC_SHA, "docs359_identity_preserved", checks)

    expected_tools = contract["frozen_execution_identity"]
    for label, path, expected in (
        ("icc2", ICC2, expected_tools["icc2_shell"]),
        ("lmutil", LMUTIL, expected_tools["lmutil_lmstat"]),
    ):
        mode = path.stat().st_mode
        need(path.is_file() and not path.is_symlink() and stat.S_ISREG(mode)
             and bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
             label + "_regular_nonsymlink_executable", checks)
        need(str(path) == expected["path"] and sha(path) == expected["sha256"],
             label + "_path_sha_exact", checks)

    entries = []
    for raw in MANIFEST.read_text().splitlines():
        digest, rel = raw.split(maxsplit=1)
        rel = rel.lstrip("*")
        need(rel not in {x[1] for x in entries} and not Path(rel).is_absolute()
             and ".." not in Path(rel).parts, "mw_manifest_member_safe_" + str(len(entries)), checks)
        entries.append((digest, rel))
    actual = sorted(p.relative_to(MW).as_posix() for p in MW.rglob("*") if p.is_file())
    need(len(entries) == 1051 and len(actual) == 1051,
         "mw_inventory_1051_files", checks)
    need(sorted(rel for _, rel in entries) == actual, "mw_manifest_exhaustive_exact_tree", checks)
    need(all((MW / rel).is_file() and not (MW / rel).is_symlink() and sha(MW / rel) == digest
             for digest, rel in entries), "mw_manifest_1051_hashes_exact", checks)
    need(sum(rel.startswith("FRAM/") for _, rel in entries) == 1044,
         "mw_fram_count_1044", checks)
    need(sum(rel.startswith("CEL/") for _, rel in entries) == 2,
         "mw_cel_count_2", checks)
    path_hash = hashlib.sha256(("\n".join(actual) + "\n").encode()).hexdigest()
    need(path_hash == expected_tools["milkyway_manifest"]["sorted_path_inventory_sha256"],
         "mw_sorted_path_hash_exact", checks)

    runner = RUNNER.read_text()
    tcl = TCL.read_text()
    parser = PARSER.read_text()
    completed = subprocess.run(["bash", "-n", str(RUNNER)], cwd=REPO, check=False)
    need(completed.returncode == 0, "runner_bash_n_pass", checks)
    unit = subprocess.run(["python3.12", str(TESTS)], cwd=REPO, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    need(unit.returncode == 0 and "Ran 6 tests" in unit.stdout and "OK" in unit.stdout,
         "parser_unit_tests_6_of_6_pass", checks)

    blocked_match = re.search(r"blocked = \{(.*?)\}", runner, flags=re.S)
    need(blocked_match is not None, "same_uid_guard_literal_found", checks)
    blocked = set(re.findall(r"'([^']+)'", blocked_match.group(1)))
    required_blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
                        "pt_shell", "fm_shell", "icc2_shell", "icc2_lm_shell",
                        "common_shell_exec", "common_shell_exe", "lmutil", "lmstat"}
    need(blocked == required_blocked, "same_uid_guard_exact_14_names", checks)
    need(runner.count('"${ICC2}" -f "${TCL}"') == 1
         and "for index in 0 1" in runner, "icc2_two_sequential_axes_one_callsite", checks)
    need(not any(p.exists() for p in (RESULT, ATTEMPT, LOCK)),
         "m2123_result_attempt_lock_fresh", checks)

    order_tokens = [
        "create_lib", "read_verilog", "link_block -force", "set tt_covered 0",
        "set ss_covered 0", "set ff_covered 0", "set physical_covered 0",
        "read_parasitic_tech", "initialize_floorplan", "place_opt", "clock_opt",
        "route_auto", "route_opt",
    ]
    order = [tcl.index(token) for token in order_tokens]
    need(order == sorted(order), "library_corner_94_before_nxtgrd_then_pnr", checks)
    for key in ("tt_master_coverage", "ss_master_coverage", "ff_master_coverage",
                "physical_master_coverage"):
        stem = key.replace("_master_coverage", "")
        need('puts $fh "' + key + '=$' + stem + '_covered/94"' in tcl,
             key + "_emitted_measured_over_94", checks)
    need("m2121_parse_route_report" in tcl
         and 'if {$value != 0} { error "M2121 nonzero routed open/DRC count $value" }' in tcl,
         "route_real_count_zero_gate_present", checks)
    need("-repair_status accepted -quiet" in tcl and "accepted_mismatch_count" in tcl,
         "accepted_mismatch_explicit_query_zero_gate", checks)

    # The old P0 asked for direct unbound/unmapped object queries as applicable.
    direct_unbound = bool(re.search(r"get_cells[^\n]*(?:is_unbound|is_unmapped)", tcl))
    checks["direct_unbound_unmapped_tool_query_present"] = direct_unbound
    need(not direct_unbound, "confirmed_p0_direct_unbound_unmapped_query_absent", checks)
    need("get_mismatch_objects -repair_status not_repaired" in tcl,
         "not_repaired_mismatch_query_present_but_not_reference_query", checks)

    need("routed.spef_scenario" not in parser
         and 'root / "output" / "routed.spef"' in parser
         and 'root / "output" / "routed.spef.gz"' in parser,
         "strict_spef_exact_names_only", checks)
    need("operating_conditions, wireload_headers, wireload_continuations) == (1, 1, 1)" in runner,
         "sdc_removal_cardinality_exact", checks)
    for token in ("ports_sorted.txt", "actual_floorplan.txt", "actual_routing_layers.rpt",
                  "actual_cts_cells.txt", "actual_hold_cells.txt", "actual_scenarios.rpt"):
        need(token in runner and token in parser, "matched_policy_" + token, checks)

    # Parser fixture attacks: demonstrate fixed P0s and the remaining report-inventory P2.
    spec = importlib.util.spec_from_file_location("m2121_tests_hammer", TESTS)
    tests_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tests_mod)
    mod = tests_mod.MOD
    with tempfile.TemporaryDirectory(prefix="m2122.fixture.") as temp_name:
        root = Path(temp_name)
        ordinary = tests_mod.make_axis(root / "ordinary", "ordinary_lru4", 260000.0)
        tsbg = tests_mod.make_axis(root / "tsbg", "tsbg_b4", 260100.0)
        need(mod.parse_pair(ordinary, tsbg)["comparison"]["both_hold_met"],
             "positive_pair_fixture_accepts", checks)
        route = tsbg / "reports/route_check.rpt"
        clean_route = route.read_text()
        route.write_text(clean_route.replace("open nets = 0", "open nets = 999"))
        reject(lambda: mod.parse_pair(ordinary, tsbg), "route_open_999_rejected", checks)
        route.write_text(clean_route.replace("DRC violations = 0", "DRC violations = 777"))
        reject(lambda: mod.parse_pair(ordinary, tsbg), "route_drc_777_rejected", checks)
        route.write_text(clean_route)
        spef = tsbg / "output/routed.spef"
        spef.unlink()
        (tsbg / "output/routed.spef_scenario").write_text("fake\n")
        reject(lambda: mod.parse_pair(ordinary, tsbg), "spef_scenario_only_rejected", checks)
        (tsbg / "output/routed.spef_scenario").unlink()
        reject(lambda: mod.parse_pair(ordinary, tsbg), "missing_all_exact_spef_rejected", checks)
        (tsbg / "output/routed.spef.gz").write_bytes(b"gzip-name-fixture\n")
        need(mod.parse_pair(ordinary, tsbg)["comparison"]["both_hold_met"],
             "exact_spef_gz_name_accepted", checks)
        facts = tsbg / "machine_facts.txt"
        replace_fact(facts, "accepted_mismatch_count", "1")
        reject(lambda: mod.parse_pair(ordinary, tsbg), "accepted_mismatch_fact_one_rejected", checks)
        replace_fact(facts, "accepted_mismatch_count", "0")
        replace_fact(facts, "unresolved_reference_count", "1")
        reject(lambda: mod.parse_pair(ordinary, tsbg), "unresolved_reference_fact_one_rejected", checks)
        replace_fact(facts, "unresolved_reference_count", "0")
        need(mod.parse_pair(ordinary, tsbg)["comparison"]["both_hold_met"],
             "fixture_restored", checks)
        for name in ("design_library.rpt", "final_design.rpt", "vectorless_power_diagnostic.rpt"):
            need(not (tsbg / "reports" / name).exists(), "fixture_omits_" + name, checks)
        need(mod.parse_pair(ordinary, tsbg)["comparison"]["both_hold_met"],
             "confirmed_p2_parser_accepts_three_generated_reports_absent", checks)
        (tsbg / "reports/timing_setup.rpt").write_text("slack (VIOLATED) -999.000\n")
        need(mod.parse_pair(ordinary, tsbg)["comparison"]["both_hold_met"],
             "confirmed_p2_parser_ignores_contradictory_timing_report", checks)

    need("design_library.rpt" not in parser and "final_design.rpt" not in parser
         and "vectorless_power_diagnostic.rpt" not in parser,
         "confirmed_p2_three_generated_reports_not_required", checks)

    need({str(p): sha(p) for p in frozen} == frozen_before,
         "all_frozen_sources_unchanged", checks)
    output = {
        "schema": "m2122_mechanical_checks_r1_v1",
        "status": "FAIL_M2122_MECHANICAL_CHECKS__P0_AND_P2_REMAIN__NO_EDA_AUTHORIZED",
        "date_cst": "2026-09-04",
        "eda_invoked": False,
        "license_query_invoked": False,
        "checks": checks,
        "finding_summary": {
            "p0": "No direct post-link is_unbound/is_unmapped reference-object query; not_repaired mismatch objects are not a substitute.",
            "p2": "Three generated reports are not required, and contradictory timing report text is accepted.",
        },
        "identity": {p.name: sha(p) for p in frozen},
    }
    (HERE / "mechanical_checks.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(output["status"])


if __name__ == "__main__":
    main()
