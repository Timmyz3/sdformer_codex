#!/usr/bin/env python3
"""Read-only, fail-closed M2118 source hammer for the M2117 campaign.

Only static Python execution and synthetic parser fixtures under /tmp are
allowed here.  This program never invokes a license query or any EDA tool.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2117_m2018_tsbg_rtl_saifmap_power_one_shot.py"
PARSER = HW / "system_simulator/scripts/parse_m2117_m2018_tsbg_rtl_saifmap_power.py"
CONTRACT = HW / "contracts/m2117_m2018_tsbg_rtl_saifmap_power_source_contract_r1_20260904.json"
SELFCHECK = HW / "reviews/m2117_m2018_tsbg_rtl_saifmap_power_source_selfcheck_r1_20260904"
M2113_REVIEW = HW / "reviews/m2114_m2113_m2018_tsbg_rtl_saifmap_power_source_hammer_r1_20260904"
M522_FAIL = HW / "reviews/m522_m514_dc_tool_invocation_failure_hammer_r1_20260827"
M522_GO = HW / "reviews/m522_m514_dc_static_hammer_r6_20260827"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "results/m2119_m2117_m2018_tsbg_rtl_saifmap_power_r1_20260904"
ATTEMPT = HW / "results/.m2119_m2117_tsbg_rtl_saifmap_power_attempt_consumed"
LOCK = HW / "results/.m2119_m2117_tsbg_rtl_saifmap_power_launch_lock"
OLD_RESULT = HW / "results/m2115_m2113_m2018_tsbg_rtl_saifmap_power_r1_20260904"
OLD_ATTEMPT = HW / "results/.m2115_m2113_tsbg_rtl_saifmap_power_attempt_consumed"
OLD_LOCK = HW / "results/.m2115_m2113_tsbg_rtl_saifmap_power_launch_lock"
DC_LINK = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
DC_TARGET = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
DC_SHA = "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2"
DOC_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise AssertionError(f"duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError(f"nonfinite JSON token: {token}")))


def need(value: bool, label: str, checks: dict[str, bool]) -> None:
    checks[label] = bool(value)
    if not value:
        raise AssertionError(label)


def expect_failure(callback, label: str, checks: dict[str, bool]) -> None:
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
    if not manifest.is_file() or manifest.is_symlink() \
            or not outer.is_file() or outer.is_symlink():
        return False
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        return False
    expected = set()
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            return False
        rel = Path(fields[1].lstrip("*"))
        path = root / rel
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() in expected \
                or not path.is_file() or path.is_symlink() or sha(path) != fields[0]:
            return False
        expected.add(rel.as_posix())
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file()
              and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    return actual == expected


def run_static(path: Path, *args: str) -> dict:
    completed = subprocess.run(
        ["python3.12", str(path), *args], cwd=REPO, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        check=False, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.strip())
    return json.loads(completed.stdout)


def write_saif(path: Path, duration: float, *, tx_bad: bool = False,
               conservation_bad: bool = False, toggled: int = 100,
               omit_critical: str | None = None) -> None:
    critical = [
        "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
        "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
    ]
    names = critical + [f"signal_{index}" for index in range(92)]
    if omit_critical:
        names[names.index(omit_critical)] = "replacement_signal"
    rows = []
    for index, name in enumerate(names):
        tx = 1.0 if tx_bad and index == 99 else 0.0
        t0, t1 = duration / 2.0 - tx, duration / 2.0
        if conservation_bad and index == 99:
            t1 += 2.0
        tc = 1 + index % 3 if index < toggled else 0
        rows.append(f"({name} (T0 {t0}) (T1 {t1}) (TX {tx}) (TC {tc}))")
    path.write_text("(SAIFILE\n(TIMESCALE 1 ns)\n"
                    f"(DURATION {duration})\n(INSTANCE dut\n(NET\n"
                    + "\n".join(rows) + "\n)))\n")


def main() -> None:
    checks: dict[str, bool] = {}
    contract = strict_json(CONTRACT)
    inventory = contract["source_inventory"]
    frozen_paths = sorted(set([RUNNER, PARSER, CONTRACT, DOC359]
                              + [REPO / rel for rel in inventory]))
    frozen_before = {path.as_posix(): sha(path) for path in frozen_paths}

    # Frozen identities and exhaustive seals.
    need(contract["schema"] ==
         "m2117_m2018_tsbg_rtl_saifmap_power_source_contract_r1_v1",
         "contract_schema_exact", checks)
    need(contract["status"] ==
         "SOURCE_ONLY__M2118_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "source_only_status_exact", checks)
    need(len(inventory) == 21, "source_inventory_exactly_21_members", checks)
    need(len(inventory) == len(set(inventory)), "source_inventory_unique", checks)
    need(all((REPO / rel).is_file() and not (REPO / rel).is_symlink()
             and sha(REPO / rel) == digest for rel, digest in inventory.items()),
         "source_inventory_21_of_21_exact_regular", checks)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.is_file() and not sidecar.is_symlink()
         and sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract_inner_seal_exact", checks)
    need(outer.is_file() and not outer.is_symlink()
         and outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract_outer_seal_exact", checks)
    need(verify_sealed_dir(SELFCHECK), "selfcheck_double_seal_exhaustive", checks)
    selfcheck = strict_json(SELFCHECK / "selfcheck.json")
    need(selfcheck["status"] ==
         "PASS_M2117_SOURCE_SELFCHECK__M2118_INDEPENDENT_HAMMER_REQUIRED__NO_EDA"
         and selfcheck["static_checks"]["source_inventory_count"] == 21
         and selfcheck["new_production_identity"]["all_absent_at_selfcheck"] is True,
         "selfcheck_identity_and_freshness_exact", checks)
    need(sha(DOC359) == DOC_SHA, "docs359_identity_preserved", checks)

    # Historical basename-dispatch evidence must remain exhaustive and frozen.
    need(verify_sealed_dir(M522_FAIL) and verify_sealed_dir(M522_GO),
         "m522_failure_and_positive_reviews_double_sealed", checks)
    fail = strict_json(M522_FAIL / "m522_m514_dc_tool_invocation_failure_hammer_r1.json")
    go = strict_json(M522_GO / "m522_m514_dc_static_hammer_r6.json")
    need(fail["root_cause"]["classification"] ==
         "WRONG_LAUNCHER_BASENAME__SYNOPSYS_WRAPPER_DISPATCH_NOT_ENTERED"
         and fail["root_cause"]["minimal_valid_repair"].startswith(
             "Invoke the frozen dc_shell symlink pathname"),
         "m522_failure_root_cause_pinned", checks)
    need(go["launcher_static_proof"]["positive_argv0"] == str(DC_LINK)
         and go["launcher_static_proof"]["resolved_target_execute_line_count"] == 0
         and go["launcher_static_proof"]["wrapper_selected_backend"] ==
         "common_shell_exec -shell dc_shell",
         "m522_positive_pattern_pinned", checks)
    need(verify_sealed_dir(M2113_REVIEW), "m2114_failed_review_double_sealed", checks)
    old_review = strict_json(M2113_REVIEW / "review.json")
    need(old_review["status"].startswith("FAIL_M2114")
         and old_review["severity_counts"] == {"p0": 1, "p1": 0, "p2": 0}
         and old_review["required_next_identity"]["m2115_must_not_run"] is True,
         "m2113_failure_and_m2115_prohibition_pinned", checks)

    # Launcher four-layer identity and installed backend semantics, without execution.
    need(DC_LINK.is_symlink(), "dc_launcher_is_symlink", checks)
    need(os.readlink(DC_LINK) == "snps_shell", "dc_launcher_raw_link_exact", checks)
    need(DC_LINK.resolve(strict=True) == DC_TARGET,
         "dc_launcher_resolved_target_exact", checks)
    need(DC_TARGET.is_file() and not DC_TARGET.is_symlink() and sha(DC_TARGET) == DC_SHA,
         "dc_resolved_target_regular_and_sha_exact", checks)
    wrapper = DC_TARGET.read_text()
    need('script_name=""' in wrapper
         and 'script_name=`expr "$cmd" : \'.*/\\([^/]*\\)$\'`' in wrapper
         and "dc_shell|dc_shell-t|dc_shell-xg-t)" in wrapper
         and "-shell dc_shell" in wrapper
         and 'echo "Error: The $script_name script is not supported."' in wrapper,
         "installed_wrapper_basename_dispatch_static_proof", checks)
    need(PT.is_file() and not PT.is_symlink(), "pt_path_correct_regular_nonsymlink", checks)

    runner_text = RUNNER.read_text()
    tree = ast.parse(runner_text)
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = ast.get_source_segment(runner_text, node.value)
    need(assignments.get("DC") ==
         'Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")',
         "runner_dc_argv0_constant_literal", checks)
    need(assignments.get("DC_TARGET") ==
         'Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")',
         "runner_dc_target_identity_constant_literal", checks)
    need(assignments.get("PT") ==
         'Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")',
         "runner_pt_constant_literal", checks)
    run_calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "run" and node.args]
    dc_calls = []
    direct_target_calls = []
    for node in run_calls:
        command = node.args[0]
        if not isinstance(command, ast.List) or not command.elts:
            continue
        first = ast.get_source_segment(runner_text, command.elts[0]) or ""
        whole = ast.get_source_segment(runner_text, command) or ""
        if first == "str(DC)":
            dc_calls.append(whole)
        if first == "str(DC_TARGET)" or whole.startswith(
                '["/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell"'):
            direct_target_calls.append(whole)
    need(dc_calls == ['[str(DC), "-f", str(DC_TCL)]'],
         "positive_dc_launch_site_exactly_one_and_dash_f", checks)
    need(not direct_target_calls, "direct_snps_shell_launch_site_zero", checks)
    need(contract["dc_launcher_identity"] == {
        "positive_argv0": str(DC_LINK), "positive_argument_prefix": ["-f"],
        "launcher_must_be_symlink": True, "raw_link_text": "snps_shell",
        "resolved_target": str(DC_TARGET),
        "resolved_target_must_be_regular_nonsymlink": True,
        "resolved_target_sha256": DC_SHA,
        "direct_resolved_target_execution_forbidden": True,
        "dispatch_backend": "common_shell_exec -shell dc_shell",
    }, "contract_dc_launcher_four_layer_identity_exact", checks)
    blocked_match = re.search(r'blocked = (\{[^}]+\})', runner_text, flags=re.DOTALL)
    blocked = ast.literal_eval(blocked_match.group(1)) if blocked_match else set()
    need(blocked == {"vcs", "simv", "snps_shell", "dc_shell",
                     "common_shell_exec", "common_shell_exe", "pt_shell", "lmstat"},
         "same_uid_guard_all_required_tool_names_exact", checks)

    # One-shot ordering and freshness.
    need(not any(path.exists() for path in
                 (RESULT, ATTEMPT, LOCK, OLD_RESULT, OLD_ATTEMPT, OLD_LOCK)),
         "m2115_and_m2119_result_attempt_lock_all_fresh", checks)
    budget = {"license_queries": 1, "vcs_compiles": 1, "simv_runs": 2,
              "dc_runs": 2, "ptpx_runs": 2, "saif_files": 2,
              "automatic_retry": False, "p1_serial": True,
              "reuse_old_artifacts": False}
    need(contract["execution_budget"] == budget, "execution_budget_exact", checks)
    production = runner_text[runner_text.index("def production()") :]
    ordered = [production.index(token) for token in (
        "source_validation(require_review=True)",
        "need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists()",
        "no_same_uid_eda()", "LOCK.mkdir()", "ATTEMPT.mkdir()",
        'run([str(LMUTIL), "lmstat"')]
    need(ordered == sorted(ordered),
         "review_freshness_collision_before_attempt_and_license", checks)
    for counter in ("license_queries", "vcs_compiles", "simv_runs",
                    "dc_runs", "ptpx_runs", "saif_files"):
        need(runner_text.count(f'counts["{counter}"] += 1') == 1,
             f"single_increment_site_{counter}", checks)
    need("for axis, cfg in AXES.items():" in production
         and production.index('counts["simv_runs"] += 1')
         < production.index('counts["dc_runs"] += 1')
         < production.index('counts["ptpx_runs"] += 1')
         and "counts == COUNTS" in production,
         "two_axes_serial_sim_dc_pt_and_final_count", checks)
    need("automatic_retry=false" in production
         and "shutil.rmtree(ATTEMPT" not in production
         and "ATTEMPT.rmdir" not in production,
         "no_retry_and_consumed_attempt_preserved", checks)

    runner_static = run_static(RUNNER, "--static")
    parser_static = run_static(PARSER, "static")
    need(runner_static["status"] == "PASS_M2117_STATIC_RUNNER"
         and runner_static["source_count"] == 21
         and runner_static["dc_launcher"]["positive_argv0"] == str(DC_LINK),
         "runner_static_pass", checks)
    need(parser_static["status"] == "PASS_M2117_STATIC_PARSER"
         and all(parser_static["checks"].values()), "parser_static_pass", checks)

    # Prove an absent review is rejected before persistent state or any subprocess.
    spec = importlib.util.spec_from_file_location("m2117_runner_hammer", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="m2118_prereview.") as tmp_name:
        tmp = Path(tmp_name)
        module.REVIEW = tmp / "missing_review"
        module.RESULT, module.ATTEMPT, module.LOCK = (
            tmp / "result", tmp / "attempt", tmp / "lock")
        expect_failure(module.production, "missing_review_rejected", checks)
        need(not any((tmp / name).exists() for name in ("result", "attempt", "lock")),
             "missing_review_created_no_persistent_state", checks)

    # DUT/window and map/power source boundaries.
    tb = (HW / "tb_m2018/tb_m2117_m2018_tsbg_rtl_saifmap_power.sv").read_text()
    for token in ("FROZEN_WORKLOAD_SLOT = 42", "FROZEN_PRELOAD_CYCLES = 383",
                  "FROZEN_BASE_CYCLES = 20292", "FROZEN_TSBG_CYCLES = 7569",
                  "FROZEN_BASE_SCALAR = 14304", "FROZEN_TSBG_SCALAR = 4608",
                  "core.sample_id != 0", "core.layer_id != 28", "core.is_fc2 != 0",
                  "core.token_start != 0", "core.real_source_groups != 48",
                  "requires exactly one axis plusarg", "completion ledger drift"):
        need(token in tb, f"testbench_contract_{token}", checks)
    ucli = [
        (HW / "dc_handoff/scripts/m2117_m2018_tsbg_ordinary_rtl_saif.ucli.tcl").read_text(),
        (HW / "dc_handoff/scripts/m2117_m2018_tsbg_tsbg_rtl_saif.ucli.tcl").read_text(),
    ]
    need("core.dut_base.implementation" in ucli[0]
         and "core.dut_tsbg.implementation" not in ucli[0]
         and "core.dut_tsbg.implementation" in ucli[1]
         and "core.dut_base.implementation" not in ucli[1],
         "ucli_each_axis_dut_only_scope", checks)
    need(all(text.count("power -enable") == 1
             and text.count("power -disable") == 1
             and text.count("run") == 2
             and "M2117_RTL_SAIF_FILE" in text for text in ucli),
         "ucli_single_measurement_window_each_axis", checks)
    dc = (HW / "dc_handoff/scripts/run_dc_m2117_m2018_tsbg_saifmap_axis.tcl").read_text()
    need('elaborate $design_name -parameters "SCHEDULE_MODE=>$mode"' in dc
         and 'if {$mode ne "0" && $mode ne "1"}' in dc,
         "dc_only_schedule_mode_axis", checks)
    need("saif_map -start" in dc and dc.count("saif_map -write_map") == 3
         and "-type ptpx -essential" in dc,
         "dc_fresh_native_maps_default_and_essential", checks)
    pt = (HW / "dc_handoff/scripts/run_ptpx_m2117_m2018_tsbg_rtl_saifmap_axis.tcl").read_text()
    need(pt.index("source $default_map") < pt.index("source $essential_map")
         < pt.index("read_saif"), "pt_default_then_essential_then_saif", checks)
    first_power = pt.index("\nreport_power")
    for token in ("M2117_FAIL_ANNOTATION_GATE_BEFORE_POWER",
                  "M2117_FAIL_NONZERO_TOGGLE_COVERAGE_BEFORE_POWER",
                  "M2117_FAIL_INCONSISTENT_ANNOTATION_BEFORE_POWER",
                  "M2117_FAIL_ZERO_CRITICAL_CONE_BEFORE_POWER",
                  "M2117_FAIL_CHECK_POWER"):
        need(token in pt and pt.index(token) < first_power,
             f"pt_prepower_gate_{token}", checks)
    need("weight_sram_capacity_bytes=294912" in pt
         and "weight_sram_dynamic_energy_in_ptpx=false" in pt
         and "weight_sram_area_in_ptpx=false" in pt,
         "common_288kib_sram_external_and_disclosed", checks)

    # Synthetic parser mutations exercise every publication-critical gate.
    parser_spec = importlib.util.spec_from_file_location("m2117_parser_hammer", PARSER)
    parser_module = importlib.util.module_from_spec(parser_spec)
    assert parser_spec.loader is not None
    parser_spec.loader.exec_module(parser_module)
    with tempfile.TemporaryDirectory(prefix="m2118_parser.") as tmp_name:
        tmp = Path(tmp_name)
        for axis, cycles in (("ordinary_lru4", 20292), ("tsbg_b4", 7569)):
            valid = tmp / f"{axis}.saif"
            write_saif(valid, cycles * 3.0)
            need(parser_module.parse_saif(valid, axis)["expected_cycles"] == cycles,
                 f"valid_saif_{axis}_pass", checks)
        for name, kwargs in (("tx", {"tx_bad": True}),
                             ("conservation", {"conservation_bad": True}),
                             ("low_toggle", {"toggled": 19}),
                             ("missing_critical", {"omit_critical": "commit_accept"})):
            bad = tmp / f"bad_{name}.saif"
            write_saif(bad, 20292 * 3.0, **kwargs)
            expect_failure(lambda p=bad: parser_module.parse_saif(
                p, "ordinary_lru4"), f"saif_mutation_{name}_rejected", checks)
        bad_duration = tmp / "bad_duration.saif"
        write_saif(bad_duration, 20292 * 3.0 + 1.0)
        expect_failure(lambda: parser_module.parse_saif(
            bad_duration, "ordinary_lru4"), "saif_duration_drift_rejected", checks)
        default_map, essential_map = tmp / "default.tcl", tmp / "essential.tcl"
        default_map.write_text("set_rtl_to_gate_name -rtl {state_q} -gate U0\n"
                               "set_rtl_to_gate_name -rtl {count_q} -gate U1\n")
        essential_map.write_text("set_rtl_to_gate_name -rtl {state_q} -gate U0/Q\n"
                                 "set_rtl_to_gate_name -rtl {valid_q} -gate U2/Q\n")
        maps = parser_module.classify_maps(default_map, essential_map)
        need(maps["intersection_entries"] == 1 and maps["union_entries"] == 3
             and maps["intersection_target_difference_entries"] == 1,
             "map_intersection_union_and_target_difference_retained", checks)
        conflict = tmp / "conflict.tcl"
        conflict.write_text("set_rtl_to_gate_name -rtl {state_q} -gate U0\n"
                            "set_rtl_to_gate_name -rtl {state_q} -gate U9\n")
        expect_failure(lambda: parser_module.map_rows(conflict),
                       "intra_class_map_conflict_rejected", checks)
        annotation = tmp / "annotation.rpt"
        annotation.write_text("Total number of nets = 100\n"
                              "Number of annotated nets = 95 (95.00%)\n"
                              "Total number of leaf cells = 100\n"
                              "Number of fully annotated leaf cells = 95 (95.00%)\n")
        need(parser_module.parse_annotation(annotation)["net_percent"] == 95.0,
             "annotation_95_boundary_pass", checks)
        annotation.write_text(annotation.read_text().replace("95.00%", "94.99%", 1))
        expect_failure(lambda: parser_module.parse_annotation(annotation),
                       "annotation_below_95_rejected", checks)
        coverage = tmp / "coverage.rpt"
        coverage.write_text("m2018_axis 20.00 20 100\n")
        need(parser_module.parse_switching_coverage(coverage)["percent"] == 20.0,
             "toggle_coverage_20_boundary_pass", checks)
        coverage.write_text("m2018_axis 19.99 20 100\n")
        expect_failure(lambda: parser_module.parse_switching_coverage(coverage),
                       "toggle_coverage_below_20_rejected", checks)
        critical = tmp / "critical.rpt"
        critical.write_text("mem_req_valid 0.25\n")
        need(parser_module.parse_critical(critical, "mem_req_valid")["has_nonzero_numeric"],
             "critical_cone_object_activity_pass", checks)
        critical.write_text("Report mem_req_valid 2026\n")
        expect_failure(lambda: parser_module.parse_critical(critical, "mem_req_valid"),
                       "critical_header_number_spoof_rejected", checks)
        power = tmp / "power.rpt"
        power.write_text("Net Switching Power = 1.00000000\n"
                         "Cell Internal Power = 2.00000000\n"
                         "Cell Leakage Power = 0.50000000\n"
                         "Total Power = 3.50000000\n")
        need(abs(parser_module.parse_power(power, 30.0)["energy_nj"] - 0.105) < 1e-12,
             "power_and_energy_arithmetic_pass", checks)
        power.write_text(power.read_text().replace("3.50000000", "9.50000000"))
        expect_failure(lambda: parser_module.parse_power(power, 30.0),
                       "power_component_mismatch_rejected", checks)

    frozen_after = {path.as_posix(): sha(path) for path in frozen_paths}
    need(frozen_before == frozen_after, "all_sources_unchanged_by_hammer", checks)
    value = {
        "schema": "m2118_m2117_source_mechanical_checks_r1_v1",
        "status": "PASS_M2118_M2117_SOURCE_MECHANICAL_CHECKS__NO_EDA",
        "eda_or_license_invoked": False,
        "check_count": len(checks),
        "checks": checks,
        "source_sha256": frozen_after,
    }
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
