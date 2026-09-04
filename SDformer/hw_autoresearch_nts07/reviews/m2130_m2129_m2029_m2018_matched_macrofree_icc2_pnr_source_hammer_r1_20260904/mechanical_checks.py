#!/usr/bin/env python3
"""Independent read-only/no-EDA hammer for the M2129 ICC2 source package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2129_m2130_m2029_m2018_matched_macrofree_icc2_pnr_one_shot.sh"
TCL = HW / "dc_handoff/scripts/run_icc2_m2129_m2029_m2018_matched_macrofree_axis.tcl"
PARSER = HW / "system_simulator/scripts/parse_m2129_m2029_m2018_matched_macrofree_icc2_pnr.py"
TESTS = HW / "tests/test_m2129_m2029_m2018_matched_macrofree_icc2_pnr.py"
CONTRACT = HW / "contracts/m2129_m2029_m2018_matched_macrofree_icc2_pnr_source_contract_r1_20260904.json"
MANIFEST = HW / "dc_handoff/manifests/m2129_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
AUTHOR = HW / "reviews/m2129_m2029_m2018_matched_macrofree_icc2_pnr_source_author_receipt_r1_20260904"
M2121_AUTHOR = HW / "reviews/m2121_m2029_m2018_matched_macrofree_icc2_pnr_source_author_receipt_r1_20260904"
M2122 = HW / "reviews/m2122_m2121_m2029_m2018_matched_macrofree_icc2_pnr_source_hammer_r1_20260904"
M2029 = HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
ADDENDUM = HW / "reviews/tcasii2027_m2018_icc2_physical_tech_readonly_addendum_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
MW = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
ICC2 = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
WRITE_PARASITICS_DOC = Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/ICC2/man/cat2/write_parasitics.2")
RESULT = HW / "dc_handoff/runs/m2131_m2029_m2018_matched_macrofree_icc2_pnr_raw_r1_20260904"
ATTEMPT = HW / "dc_handoff/runs/.m2131_m2029_m2018_matched_macrofree_icc2_pnr_attempt_consumed"
LOCK = HW / "dc_handoff/runs/.m2131_m2029_m2018_matched_macrofree_icc2_pnr_launch_lock"
DOC_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AssertionError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite JSON token: " + token)))


def need(value: bool, label: str, checks: dict) -> None:
    checks[label] = bool(value)
    if not value:
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
        key = rel.as_posix()
        path = root / rel
        if rel.is_absolute() or ".." in rel.parts or key in expected:
            return False
        if not path.is_file() or path.is_symlink() or sha(path) != fields[0]:
            return False
        expected.add(key)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    return expected == actual


def replace_fact(path: Path, key: str, value: str) -> None:
    text, count = re.subn(r"^" + re.escape(key) + r"=.*$", key + "=" + value,
                          path.read_text(), count=1, flags=re.M)
    if count != 1:
        raise AssertionError("missing fact " + key)
    path.write_text(text)


def main() -> None:
    checks = {}
    frozen = [RUNNER, TCL, PARSER, TESTS, CONTRACT, MANIFEST, DOC359]
    frozen_before = {str(path): sha(path) for path in frozen}
    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "source_receipt.json")
    previous = strict_json(M2122 / "review.json")

    need(contract["schema"] == "m2129_m2029_m2018_matched_macrofree_icc2_pnr_source_contract_r1_v2",
         "contract_schema_exact", checks)
    need(contract["status"] == "SOURCE_ONLY_PENDING_M2130_INDEPENDENT_HAMMER__NO_EDA_AUTHORIZED",
         "source_only_status_exact", checks)
    auth = contract["authorization"]
    need(auth["license_queries"] == 0 and auth["icc2_shell_runs"] == 0
         and auth["all_other_eda_runs"] == 0 and auth["gpu_runs"] == 0
         and auth["automatic_retry"] is False, "contract_prehammer_authorization_zero", checks)
    need(author["status"].startswith("PASS_M2129_REPAIRED_SOURCE_ONLY_PENDING_M2130"),
         "author_receipt_source_only", checks)

    identities = {
        "tcl_sha256": TCL, "runner_sha256": RUNNER, "parser_sha256": PARSER,
        "tests_sha256": TESTS, "contract_sha256": CONTRACT,
        "milkyway_manifest_sha256": MANIFEST, "docs359_sha256": DOC359,
    }
    for key, path in identities.items():
        need(author["identity"][key] == sha(path), "author_identity_" + key, checks)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract_inner_seal_exact", checks)
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract_outer_seal_exact", checks)
    for label, path in (("m2129_author", AUTHOR), ("m2121_author", M2121_AUTHOR),
                        ("m2122_failure", M2122), ("m2029_input", M2029),
                        ("physical_addendum", ADDENDUM)):
        need(verify_sealed_dir(path), label + "_double_seal_exhaustive", checks)
    need(previous["identity"]["runner_sha256"] == sha(HW / "dc_handoff/scripts/run_m2121_m2122_m2029_m2018_matched_macrofree_icc2_pnr_one_shot.sh"),
         "m2121_runner_immutable_since_m2122", checks)
    need(previous["identity"]["tcl_sha256"] == sha(HW / "dc_handoff/scripts/run_icc2_m2121_m2029_m2018_matched_macrofree_axis.tcl"),
         "m2121_tcl_immutable_since_m2122", checks)
    need(previous["identity"]["parser_sha256"] == sha(HW / "system_simulator/scripts/parse_m2121_m2029_m2018_matched_macrofree_icc2_pnr.py"),
         "m2121_parser_immutable_since_m2122", checks)
    need(previous["identity"]["tests_sha256"] == sha(HW / "tests/test_m2121_m2029_m2018_matched_macrofree_icc2_pnr.py"),
         "m2121_tests_immutable_since_m2122", checks)
    need(sha(DOC359) == DOC_SHA, "docs359_identity_preserved", checks)

    expected_tools = contract["frozen_execution_identity"]
    for label, path, expected in (("icc2", ICC2, expected_tools["icc2_shell"]),
                                  ("lmutil_lmstat", LMUTIL, expected_tools["lmutil_lmstat"])):
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
        need(not Path(rel).is_absolute() and ".." not in Path(rel).parts,
             "mw_manifest_members_paths_safe", checks)
        entries.append((digest, rel))
    actual = sorted(p.relative_to(MW).as_posix() for p in MW.rglob("*") if p.is_file())
    need(len(entries) == len(set(rel for _, rel in entries)) == len(actual) == 1051,
         "mw_inventory_1051_unique_files", checks)
    need(sorted(rel for _, rel in entries) == actual, "mw_manifest_exhaustive_exact_tree", checks)
    need(all((MW / rel).is_file() and not (MW / rel).is_symlink() and sha(MW / rel) == digest
             for digest, rel in entries), "mw_manifest_all_hashes_exact", checks)
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
    compile(PARSER.read_text(), str(PARSER), "exec")
    compile(TESTS.read_text(), str(TESTS), "exec")
    need(subprocess.run(["bash", "-n", str(RUNNER)], cwd=REPO, check=False).returncode == 0,
         "runner_bash_n_pass", checks)
    unit = subprocess.run(["python3.12", str(TESTS)], cwd=REPO, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    need(unit.returncode == 0 and "Ran 10 tests" in unit.stdout and "OK" in unit.stdout,
         "parser_unit_tests_10_of_10_pass", checks)

    blocked_match = re.search(r"blocked = \{(.*?)\}", runner, flags=re.S)
    need(blocked_match is not None, "same_uid_guard_literal_found", checks)
    blocked = set(re.findall(r"'([^']+)'", blocked_match.group(1)))
    required_blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
                        "pt_shell", "fm_shell", "icc2_shell", "icc2_lm_shell",
                        "common_shell_exec", "common_shell_exe", "lmutil", "lmstat"}
    need(blocked == required_blocked, "same_uid_guard_exact_14_names", checks)
    need(runner.count('"${LMUTIL}" lmstat') == 1, "exactly_one_license_query_callsite", checks)
    need(runner.count('"${ICC2}" -f "${TCL}"') == 1 and "for index in 0 1" in runner,
         "two_sequential_icc2_axes_one_callsite", checks)
    need(runner.index('mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"') < runner.index('"${LMUTIL}" lmstat'),
         "attempt_consumed_before_license_query", checks)
    need("retry=false" in runner and not re.search(r"^\s*(?:while|until)\b", runner, re.M),
         "no_automatic_retry", checks)
    need(not any(path.exists() for path in (RESULT, ATTEMPT, LOCK)),
         "m2131_result_attempt_lock_fresh", checks)
    need("operating_conditions, wireload_headers, wireload_continuations) == (1, 1, 1)" in runner,
         "sdc_removal_cardinality_exact", checks)

    link_at = tcl.index("link_block -force -verbose")
    direct_at = tcl.index("set direct_unbound_count", link_at)
    master_at = tcl.index("set input_leaf", direct_at)
    nxt_at = tcl.index("read_parasitic_tech", master_at)
    need(link_at < direct_at < master_at < nxt_at,
         "postlink_direct_census_immediate_before_master_and_nxtgrd", checks)
    for attr in ("is_unbound", "is_unmapped", "is_black_box"):
        need(re.search(r"get_cells -hierarchical -quiet -filter \"" + attr + r" == true\"", tcl) is not None,
             "direct_query_" + attr, checks)
    need("get_attribute -quiet $direct_black_box_cells ref_name" in tcl,
         "black_box_ref_name_census_direct", checks)
    need("direct_unbound_cell_count" in parser and "direct_unmapped_cell_count" in parser
         and "direct_black_box_cell_count" in parser
         and "direct_black_box_ref_name_count" in parser,
         "direct_census_facts_and_parser_present", checks)
    for key in ("tt_master_coverage", "ss_master_coverage", "ff_master_coverage",
                "physical_master_coverage"):
        stem = key.replace("_master_coverage", "")
        need('puts $fh "' + key + '=$' + stem + '_covered/94"' in tcl,
             key + "_measured_emitted", checks)
    order = [tcl.index(token) for token in ("create_lib", "read_verilog", "link_block -force",
             "set tt_covered 0", "set ss_covered 0", "set ff_covered 0",
             "set physical_covered 0", "read_parasitic_tech", "initialize_floorplan",
             "place_opt", "clock_opt", "route_auto", "route_opt")]
    need(order == sorted(order), "four_view_94_before_nxtgrd_then_pnr", checks)
    need("-repair_status accepted -quiet" in tcl,
         "accepted_mismatch_explicit_query", checks)
    need("m2129_parse_route_report" in tcl and "if {$value != 0}" in tcl,
         "route_real_numeric_zero_gate", checks)
    for report in ("design_library.rpt", "final_design.rpt", "vectorless_power_diagnostic.rpt",
                   "timing_setup.rpt", "timing_hold.rpt", "postroute_metrics.txt"):
        need(report in tcl and report in parser, "required_report_" + report, checks)
    need("parse_timing_report" in parser and "math.isclose(setup_report, setup" in parser
         and "math.isclose(hold_report, hold" in parser,
         "timing_reports_cross_checked_to_facts", checks)
    need("machine/live-query report mismatch" in parser,
         "live_query_area_cell_census_cross_checked", checks)
    for token in ("ports_sorted.txt", "actual_floorplan.txt", "actual_routing_layers.rpt",
                  "actual_cts_cells.txt", "actual_hold_cells.txt", "actual_scenarios.rpt"):
        need(token in runner and token in parser, "matched_policy_" + token, checks)

    spec = importlib.util.spec_from_file_location("m2129_tests_hammer", TESTS)
    tests_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tests_mod)
    mod = tests_mod.MOD
    with tempfile.TemporaryDirectory(prefix="m2130.fixture.") as temp_name:
        root = Path(temp_name)
        ordinary = tests_mod.make_axis(root / "ordinary", "ordinary_lru4", 260000.0)
        tsbg = tests_mod.make_axis(root / "tsbg", "tsbg_b4", 260100.0)
        need(mod.parse_pair(ordinary, tsbg)["comparison"]["both_hold_met"],
             "positive_pair_fixture_accepts", checks)
        route = tsbg / "reports/route_check.rpt"
        clean_route = route.read_text()
        for old, new, label in (("open nets = 0", "open nets = 999", "route_open_999_rejected"),
                                ("DRC violations = 0", "DRC violations = 777", "route_drc_777_rejected")):
            route.write_text(clean_route.replace(old, new))
            reject(lambda: mod.parse_pair(ordinary, tsbg), label, checks)
        route.write_text(clean_route)
        spef = tsbg / "output/routed.spef"
        spef.unlink()
        (tsbg / "output/routed.spef_scenario").write_text("scenario-not-spef\n")
        reject(lambda: mod.parse_pair(ordinary, tsbg), "scenario_only_spef_rejected", checks)
        (tsbg / "output/routed.spef_scenario").unlink()
        reject(lambda: mod.parse_pair(ordinary, tsbg), "deleted_real_spef_rejected", checks)
        spef.write_text("fixture\n")
        for key in ("direct_unbound_cell_count", "direct_unmapped_cell_count",
                    "direct_black_box_cell_count", "direct_black_box_ref_name_count"):
            facts = tsbg / "machine_facts.txt"
            replace_fact(facts, key, "1")
            reject(lambda: mod.parse_pair(ordinary, tsbg), key + "_fact_one_rejected", checks)
            replace_fact(facts, key, "0")
            census = tsbg / "reports/postlink_reference_census.txt"
            replace_fact(census, key, "1")
            reject(lambda: mod.parse_pair(ordinary, tsbg), key + "_report_one_rejected", checks)
            replace_fact(census, key, "0")
        for report_name in ("design_library.rpt", "final_design.rpt", "vectorless_power_diagnostic.rpt"):
            report = tsbg / "reports" / report_name
            payload = report.read_bytes()
            report.unlink()
            reject(lambda: mod.parse_pair(ordinary, tsbg), "missing_" + report_name + "_rejected", checks)
            report.write_bytes(payload)
        timing = tsbg / "reports/timing_setup.rpt"
        clean_timing = timing.read_text()
        timing.write_text("slack (VIOLATED) -999.000\n")
        reject(lambda: mod.parse_pair(ordinary, tsbg), "violated_minus999_rejected", checks)
        timing.write_text("slack (MET) 0.002000\n")
        reject(lambda: mod.parse_pair(ordinary, tsbg), "setup_fact_report_mismatch_rejected", checks)
        timing.write_text(clean_timing)
        metrics = tsbg / "reports/postroute_metrics.txt"
        clean_metrics = metrics.read_text()
        replace_fact(metrics, "routed_standard_cell_area_um2", "999.0")
        reject(lambda: mod.parse_pair(ordinary, tsbg), "area_live_fact_mismatch_rejected", checks)
        metrics.write_text(clean_metrics)
        replace_fact(metrics, "routed_leaf_cell_count", "999")
        reject(lambda: mod.parse_pair(ordinary, tsbg), "leaf_live_fact_mismatch_rejected", checks)

    # Guaranteed source/run incompatibility: write_parasitics treats -output as
    # a filename prefix and appends .<parasitic-tech>_<temperature>.spef.
    doc = WRITE_PARASITICS_DOC.read_text(errors="replace")
    doc = re.sub(r".\x08", "", doc)
    need("All corners are written to SPEF files with the following format:" in doc
         and "file_name" in doc and "parasitic_tech_name" in doc,
         "installed_write_parasitics_filename_rule_found", checks)
    need('write_parasitics -output "$axis_dir/output/routed.spef"' in tcl,
         "tcl_uses_routed_spef_as_write_parasitics_prefix", checks)
    need('[[ -s "${axis_dir}/output/routed.spef"' in runner,
         "runner_requires_literal_routed_spef", checks)
    need('root / "output" / "routed.spef"' in parser
         and 'root / "output" / "routed.spef.gz"' in parser,
         "parser_strict_literal_spef_names", checks)
    checks["p0_write_parasitics_output_can_satisfy_literal_spef_gate"] = False

    need({str(path): sha(path) for path in frozen} == frozen_before,
         "all_m2129_frozen_sources_unchanged_by_hammer", checks)
    output = {
        "schema": "m2130_m2129_source_mechanical_checks_r1_v1",
        "status": "FAIL_M2130_MECHANICAL_CHECKS__P0_SPEF_GENERATION_NAME_MISMATCH__NO_EDA_AUTHORIZED",
        "date_cst": "2026-09-04",
        "eda_invoked": False,
        "license_query_invoked": False,
        "checks": checks,
        "m2122_repairs": {
            "direct_postlink_unbound_unmapped_blackbox_and_refname_census": True,
            "facts_report_parser_and_negative_mutations": True,
            "complete_generated_report_inventory": True,
            "setup_hold_slack_cross_check_and_minus999_rejection": True,
            "live_query_area_and_cell_census_cross_check": True,
        },
        "inherited_m2121_regression": {
            "route_999_777_rejected": True,
            "strict_spef_parser_rejects_scenario_and_deleted_real_spef": True,
            "strict_spef_generator_runner_compatible": False,
            "four_views_94_of_94_before_nxtgrd": True,
            "def_and_policy_equality": True,
            "tool_kind_path_sha": True,
            "milkyway_1051_exhaustive": True,
            "one_shot_no_retry_same_uid_sdc_cardinality": True,
            "docs359_unchanged": True,
        },
        "finding": "ICC2 write_parasitics appends a corner/technology suffix to its -output prefix. M2129 supplies routed.spef as the prefix but then requires the literal routed.spef, so a real run cannot pass the strict SPEF gate.",
        "identity": {path.name: sha(path) for path in frozen},
    }
    (HERE / "mechanical_checks.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(output["status"])


if __name__ == "__main__":
    main()
