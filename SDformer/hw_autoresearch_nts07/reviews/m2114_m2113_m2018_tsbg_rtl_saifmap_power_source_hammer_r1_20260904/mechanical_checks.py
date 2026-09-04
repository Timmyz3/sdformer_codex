#!/usr/bin/env python3
"""Independent source-only hammer for M2113.

This script may create synthetic parser fixtures only under /tmp.  It must
never invoke lmstat, VCS, simv, Design Compiler, or PrimeTime.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2113_m2018_tsbg_rtl_saifmap_power_one_shot.py"
PARSER = HW / "system_simulator/scripts/parse_m2113_m2018_tsbg_rtl_saifmap_power.py"
CONTRACT = HW / "contracts/m2113_m2018_tsbg_rtl_saifmap_power_source_contract_r1_20260904.json"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m2105_m2018_tsbg_rtl_saifmap_power_one_shot.py"
OLD_CONTRACT = HW / "contracts/m2105_m2018_tsbg_rtl_saifmap_power_source_contract_r1_20260904.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "results/m2115_m2113_m2018_tsbg_rtl_saifmap_power_r1_20260904"
ATTEMPT = HW / "results/.m2115_m2113_tsbg_rtl_saifmap_power_attempt_consumed"
LOCK = HW / "results/.m2115_m2113_tsbg_rtl_saifmap_power_launch_lock"
OLD_RESULT = HW / "results/m2107_m2105_m2018_tsbg_rtl_saifmap_power_r1_20260904"
OLD_ATTEMPT = HW / "results/.m2107_m2105_tsbg_rtl_saifmap_power_attempt_consumed"
OLD_LOCK = HW / "results/.m2107_m2105_tsbg_rtl_saifmap_power_launch_lock"
PREV_FAIL = HW / "reviews/m522_m514_dc_tool_invocation_failure_hammer_r1_20260827"
PREV_GO = HW / "reviews/m522_m514_dc_static_hammer_r6_20260827"

PAIR_SUFFIXES = (
    ("system_simulator/scripts/parse_m2105_m2018_tsbg_rtl_saifmap_power.py",
     "system_simulator/scripts/parse_m2113_m2018_tsbg_rtl_saifmap_power.py"),
    ("tb_m2018/tb_m2105_m2018_tsbg_rtl_saifmap_power.sv",
     "tb_m2018/tb_m2113_m2018_tsbg_rtl_saifmap_power.sv"),
    ("dc_handoff/filelists/tcasii_m2105_m2018_tsbg_rtl_saif_vcs.f",
     "dc_handoff/filelists/tcasii_m2113_m2018_tsbg_rtl_saif_vcs.f"),
    ("dc_handoff/filelists/tcasii_m2105_m2018_tsbg_saifmap_dc.f",
     "dc_handoff/filelists/tcasii_m2113_m2018_tsbg_saifmap_dc.f"),
    ("dc_handoff/scripts/m2105_m2018_tsbg_ordinary_rtl_saif.ucli.tcl",
     "dc_handoff/scripts/m2113_m2018_tsbg_ordinary_rtl_saif.ucli.tcl"),
    ("dc_handoff/scripts/m2105_m2018_tsbg_tsbg_rtl_saif.ucli.tcl",
     "dc_handoff/scripts/m2113_m2018_tsbg_tsbg_rtl_saif.ucli.tcl"),
    ("dc_handoff/scripts/run_dc_m2105_m2018_tsbg_saifmap_axis.tcl",
     "dc_handoff/scripts/run_dc_m2113_m2018_tsbg_saifmap_axis.tcl"),
    ("dc_handoff/scripts/run_ptpx_m2105_m2018_tsbg_rtl_saifmap_axis.tcl",
     "dc_handoff/scripts/run_ptpx_m2113_m2018_tsbg_rtl_saifmap_axis.tcl"),
)


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
                          AssertionError(f"nonfinite JSON: {token}")))


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


def verify_sealed_directory(root: Path) -> bool:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if not root.is_dir() or root.is_symlink():
        return False
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        return False
    expected = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            return False
        rel = Path(fields[1].lstrip("*"))
        path = root / rel
        if rel.is_absolute() or ".." in rel.parts or not path.is_file() \
                or path.is_symlink() or sha(path) != fields[0]:
            return False
        expected[rel.as_posix()] = fields[0]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file()
              and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    return actual == set(expected)


def run_static(path: Path, *args: str) -> dict:
    completed = subprocess.run(
        ["python3.12", str(path), *args], cwd=REPO, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        check=False)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.strip())
    return json.loads(completed.stdout)


def normalized_identity(text: str, new: bool) -> str:
    if new:
        substitutions = (("M2116", "M2108"), ("m2116", "m2108"),
                         ("M2115", "M2107"), ("m2115", "m2107"),
                         ("M2114", "M2106"), ("m2114", "m2106"),
                         ("M2113", "M2105"), ("m2113", "m2105"))
        for source, target in substitutions:
            text = text.replace(source, target)
    return text


def write_saif(path: Path, duration: float, *, tx_bad: bool = False,
               conservation_bad: bool = False, toggled: int = 100,
               omit_critical: str | None = None) -> None:
    critical = [
        "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
        "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
    ]
    names = critical + [f"signal_{index}" for index in range(92)]
    if omit_critical is not None:
        names[names.index(omit_critical)] = "replacement_signal"
    rows = []
    for index, name in enumerate(names):
        tx = 1.0 if tx_bad and index == 99 else 0.0
        t0 = duration / 2.0 - tx
        t1 = duration / 2.0
        if conservation_bad and index == 99:
            t1 += 2.0
        tc = 1 + index % 3 if index < toggled else 0
        rows.append(f"({name} (T0 {t0}) (T1 {t1}) (TX {tx}) (TC {tc}))")
    path.write_text(
        "(SAIFILE\n(TIMESCALE 1 ns)\n"
        f"(DURATION {duration})\n(INSTANCE dut\n(NET\n"
        + "\n".join(rows) + "\n)))\n")


def main() -> None:
    checks: dict[str, bool] = {}
    source_paths = [RUNNER, PARSER, CONTRACT, DOC359]
    contract = strict_json(CONTRACT)
    source_paths.extend(REPO / rel for rel in contract["source_inventory"])
    source_paths = sorted(set(source_paths))
    frozen_before = {path.as_posix(): sha(path) for path in source_paths}

    # Contract, inventory, and double seal.
    need(contract["schema"] ==
         "m2113_m2018_tsbg_rtl_saifmap_power_source_contract_r1_v1",
         "contract_schema_exact", checks)
    need(contract["status"] ==
         "SOURCE_ONLY__M2114_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "source_only_status_exact", checks)
    inventory = contract["source_inventory"]
    need(len(inventory) == 19, "source_inventory_has_19_members", checks)
    need(all((REPO / rel).is_file() and not (REPO / rel).is_symlink()
             and sha(REPO / rel) == digest for rel, digest in inventory.items()),
         "source_inventory_19_of_19_exact_nonlinks", checks)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract_inner_seal_exact", checks)
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract_outer_seal_exact", checks)
    need(sha(DOC359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359_identity_preserved", checks)
    need(verify_sealed_directory(PREV_FAIL)
         and verify_sealed_directory(PREV_GO),
         "historical_basename_reviews_double_sealed_exhaustive", checks)
    fail_review = strict_json(
        PREV_FAIL / "m522_m514_dc_tool_invocation_failure_hammer_r1.json")
    go_review = strict_json(PREV_GO / "m522_m514_dc_static_hammer_r6.json")
    need(fail_review["root_cause"]["classification"] ==
         "WRONG_LAUNCHER_BASENAME__SYNOPSYS_WRAPPER_DISPATCH_NOT_ENTERED"
         and fail_review["root_cause"]["minimal_valid_repair"].startswith(
             "Invoke the frozen dc_shell symlink pathname"),
         "historical_failure_root_cause_matches_current_p0", checks)
    need(go_review["launcher_static_proof"]["positive_argv0"] ==
         "/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
         and go_review["launcher_static_proof"]["resolved_target_execute_line_count"] == 0,
         "historical_positive_pattern_requires_dc_shell_argv0", checks)

    # M2105 -> M2113 must be an additive executable-path correction, not an
    # experiment, workload, DUT, parser, Tcl, or testbench mutation.
    old_contract = strict_json(OLD_CONTRACT)
    for key in ("objective", "axes", "matched_invariants", "fail_closed_gates",
                "external_sram_accounting", "execution_budget", "claim_boundary"):
        need(contract[key] == old_contract[key],
             f"predecessor_contract_semantics_unchanged_{key}", checks)
    need(contract["additive_correction"] == {
        "predecessor_source_identity": "M2105",
        "predecessor_raw_result_identity": "M2107",
        "predecessor_attempt_result_lock_absent_after_preflight_failure": True,
        "predecessor_sources_immutable": True,
        "corrections": [
            "DC invokes the real non-symlink snps_shell executable",
            "PrimeTime invokes the installed prime/W-2024.09-SP3 pt_shell executable",
            "same-UID collision guard includes snps_shell and dc_shell",
        ],
    }, "additive_correction_exact", checks)
    for old_rel, new_rel in PAIR_SUFFIXES:
        old_text = (HW / old_rel).read_text()
        new_text = normalized_identity((HW / new_rel).read_text(), True)
        need(new_text == old_text, f"identity_only_pair_{Path(new_rel).name}", checks)

    old_runner = OLD_RUNNER.read_text()
    new_runner = normalized_identity(RUNNER.read_text(), True)
    additive_doc = (
        "Additive correction of the preflight-only M2105 source identity: use the real\n"
        "non-symlink DC/PT executables and guard the actual snps_shell process name.\n"
        "M2105 sources and its unconsumed M2107 predecessor identity remain immutable.\n\n")
    need(additive_doc in new_runner, "runner_additive_explanation_present", checks)
    restored_runner = new_runner.replace(additive_doc, "")
    restored_runner = restored_runner.replace(
        'DC = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")',
        'DC = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")')
    restored_runner = restored_runner.replace(
        'PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")',
        'PT = Path("/opt/synopsys/pts/W-2024.09-SP3/bin/pt_shell")')
    restored_runner = restored_runner.replace(
        '{"vcs", "simv", "snps_shell", "dc_shell", "pt_shell", "lmstat"}',
        '{"vcs", "simv", "dc_shell", "pt_shell", "lmstat"}')
    need(restored_runner == old_runner,
         "runner_delta_only_identity_paths_and_snps_guard", checks)

    # Tool identity and launcher semantics.  The regular snps_shell target is
    # frozen, but it is *not* the legal positive argv[0]: this installation's
    # POSIX wrapper dispatches Design Compiler from the original dc_shell
    # symlink basename.
    dc_link = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
    dc_target = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")
    pt_tool = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
    need(dc_link.is_symlink() and dc_link.readlink().as_posix() == "snps_shell",
         "dc_shell_raw_symlink_text_is_snps_shell", checks)
    need(dc_link.resolve() == dc_target and dc_target.is_file()
         and not dc_target.is_symlink()
         and sha(dc_target) ==
         "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
         "dc_shell_resolved_target_identity_frozen", checks)
    need(pt_tool.is_file() and not pt_tool.is_symlink(),
         "corrected_pt_path_regular_nonlink", checks)
    wrapper = dc_target.read_text()
    need('script_name=""' in wrapper
         and 'script_name=`expr "$cmd" : \'.*/\\([^/]*\\)$\'`' in wrapper,
         "wrapper_captures_script_name_only_during_symlink_walk", checks)
    need("dc_shell|dc_shell-t|dc_shell-xg-t)" in wrapper
         and "-shell dc_shell" in wrapper
         and 'echo "Error: The $script_name script is not supported."' in wrapper,
         "wrapper_dc_arm_and_unsupported_default_proven", checks)
    runner_text = RUNNER.read_text()
    need('DC = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")' in runner_text
         and 'PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")' in runner_text,
         "m2113_direct_snps_and_correct_pt_paths_observed", checks)
    need('run([str(DC), "-f", str(DC_TCL)]' in runner_text
         and 'DC = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")'
         not in runner_text,
         "p0_direct_snps_shell_positive_launch_detected", checks)
    blocked_match = re.search(r'blocked = (\{[^\n]+\})', runner_text)
    need(blocked_match is not None and blocked_match.group(1) ==
         '{"vcs", "simv", "snps_shell", "dc_shell", "pt_shell", "lmstat"}',
         "same_uid_guard_exact_process_names", checks)
    need(not any(path.exists() for path in
                 (RESULT, ATTEMPT, LOCK, OLD_RESULT, OLD_ATTEMPT, OLD_LOCK)),
         "m2107_and_m2115_result_attempt_lock_all_fresh", checks)

    runner_static = run_static(RUNNER, "--static")
    parser_static = run_static(PARSER, "static")
    need(runner_static["status"] == "PASS_M2113_STATIC_RUNNER",
         "runner_static_pass", checks)
    need(parser_static["status"] == "PASS_M2113_STATIC_PARSER"
         and all(parser_static["checks"].values()),
         "parser_static_pass", checks)

    expected_budget = {
        "license_queries": 1, "vcs_compiles": 1, "simv_runs": 2,
        "dc_runs": 2, "ptpx_runs": 2, "saif_files": 2,
        "automatic_retry": False, "p1_serial": True,
        "reuse_old_artifacts": False,
    }
    need(contract["execution_budget"] == expected_budget,
         "execution_budget_exact", checks)
    for counter in ("license_queries", "vcs_compiles", "simv_runs",
                    "dc_runs", "ptpx_runs", "saif_files"):
        need(runner_text.count(f'counts["{counter}"] += 1') == 1,
             f"single_increment_site_{counter}", checks)
    production_text = runner_text[runner_text.index("def production()") :]
    positions = [production_text.index(token) for token in (
        "source_validation(require_review=True)",
        'need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists()',
        "no_same_uid_eda()", "LOCK.mkdir()", "ATTEMPT.mkdir()",
        'run([str(LMUTIL), "lmstat"',
    )]
    need(positions == sorted(positions),
         "review_freshness_collision_before_attempt_and_license", checks)
    need("for axis, cfg in AXES.items():" in production_text
         and production_text.index('counts["simv_runs"] += 1')
         < production_text.index('counts["dc_runs"] += 1')
         < production_text.index('counts["ptpx_runs"] += 1')
         and "counts == COUNTS" in production_text,
         "two_axes_strict_serial_sim_dc_pt_and_final_count", checks)
    need("automatic_retry=false" in runner_text
         and "reuse_old_artifacts" not in runner_text[runner_text.index("def production"):],
         "no_retry_no_old_artifact_reuse_path", checks)

    # Dynamically prove an unsealed/missing review fails before attempt
    # creation.  All mutable paths are redirected to /tmp; REVIEW is forced to
    # a nonexistent path, so no tool preflight or subprocess can be reached.
    runner_spec = importlib.util.spec_from_file_location("m2113_runner_hammer", RUNNER)
    runner_module = importlib.util.module_from_spec(runner_spec)
    assert runner_spec.loader is not None
    runner_spec.loader.exec_module(runner_module)
    with tempfile.TemporaryDirectory(prefix="m2114_prereview_hammer.") as tmp_name:
        tmp = Path(tmp_name)
        runner_module.REVIEW = tmp / "missing_review"
        runner_module.RESULT = tmp / "result"
        runner_module.ATTEMPT = tmp / "attempt"
        runner_module.LOCK = tmp / "lock"
        expect_failure(runner_module.production,
                       "missing_review_rejected_before_production", checks)
        need(not any((tmp / name).exists() for name in ("result", "attempt", "lock")),
             "missing_review_created_no_attempt_result_or_lock", checks)

    # Testbench/window/UCLI/DC/PT boundaries.
    tb = (HW / "tb_m2018/tb_m2113_m2018_tsbg_rtl_saifmap_power.sv").read_text()
    for token in (
        "FROZEN_WORKLOAD_SLOT = 42", "FROZEN_PRELOAD_CYCLES = 383",
        "FROZEN_BASE_CYCLES = 20292", "FROZEN_TSBG_CYCLES = 7569",
        "FROZEN_BASE_SCALAR = 14304", "FROZEN_TSBG_SCALAR = 4608",
        "core.sample_id != 0", "core.layer_id != 28", "core.is_fc2 != 0",
        "core.token_start != 0", "core.real_source_groups != 48",
        "requires exactly one axis plusarg", "completion ledger drift",
    ):
        need(token in tb, f"testbench_contract_{token}", checks)
    ucli_ord = (HW / "dc_handoff/scripts/m2113_m2018_tsbg_ordinary_rtl_saif.ucli.tcl").read_text()
    ucli_tsbg = (HW / "dc_handoff/scripts/m2113_m2018_tsbg_tsbg_rtl_saif.ucli.tcl").read_text()
    need("core.dut_base.implementation" in ucli_ord
         and "core.dut_tsbg.implementation" not in ucli_ord,
         "ordinary_ucli_dut_only", checks)
    need("core.dut_tsbg.implementation" in ucli_tsbg
         and "core.dut_base.implementation" not in ucli_tsbg,
         "tsbg_ucli_dut_only", checks)
    need(all(text.count("power -enable") == 1
             and text.count("power -disable") == 1
             and "M2113_RTL_SAIF_FILE" in text
             for text in (ucli_ord, ucli_tsbg)),
         "single_ucli_measurement_window_each_axis", checks)
    dc = (HW / "dc_handoff/scripts/run_dc_m2113_m2018_tsbg_saifmap_axis.tcl").read_text()
    need('elaborate $design_name -parameters "SCHEDULE_MODE=>$mode"' in dc
         and 'if {$mode ne "0" && $mode ne "1"}' in dc,
         "dc_only_schedule_mode_axis", checks)
    need("saif_map -start" in dc and "saif_map -write_map" in dc
         and "-type ptpx -essential" in dc,
         "dc_fresh_native_default_and_essential_maps", checks)
    pt = (HW / "dc_handoff/scripts/run_ptpx_m2113_m2018_tsbg_rtl_saifmap_axis.tcl").read_text()
    need(pt.index("source $default_map") < pt.index("source $essential_map")
         < pt.index("read_saif"), "pt_map_source_order", checks)
    first_power = pt.index("\nreport_power")
    for token in (
        "M2113_FAIL_ANNOTATION_GATE_BEFORE_POWER",
        "M2113_FAIL_NONZERO_TOGGLE_COVERAGE_BEFORE_POWER",
        "M2113_FAIL_INCONSISTENT_ANNOTATION_BEFORE_POWER",
        "M2113_FAIL_ZERO_CRITICAL_CONE_BEFORE_POWER",
        "M2113_FAIL_CHECK_POWER",
    ):
        need(token in pt and pt.index(token) < first_power,
             f"pt_prepower_gate_{token}", checks)
    need("weight_sram_capacity_bytes=294912" in pt
         and "weight_sram_dynamic_energy_in_ptpx=false" in pt
         and "weight_sram_area_in_ptpx=false" in pt,
         "common_288kib_sram_explicitly_external", checks)

    # Parser mutations exercise the claimed fail-closed gates without EDA.
    parser_spec = importlib.util.spec_from_file_location("m2113_parser_hammer", PARSER)
    parser_module = importlib.util.module_from_spec(parser_spec)
    assert parser_spec.loader is not None
    parser_spec.loader.exec_module(parser_module)
    with tempfile.TemporaryDirectory(prefix="m2114_parser_hammer.") as tmp_name:
        tmp = Path(tmp_name)
        valid_ord = tmp / "valid_ord.saif"
        write_saif(valid_ord, 20292 * 3.0)
        ord_result = parser_module.parse_saif(valid_ord, "ordinary_lru4")
        need(ord_result["tx_sum"] == 0.0 and ord_result["record_count"] == 100,
             "valid_ordinary_saif_pass", checks)
        valid_tsbg = tmp / "valid_tsbg.saif"
        write_saif(valid_tsbg, 7569 * 3.0)
        need(parser_module.parse_saif(valid_tsbg, "tsbg_b4")["expected_cycles"] == 7569,
             "valid_tsbg_saif_pass", checks)
        for name, kwargs in (
            ("tx", {"tx_bad": True}),
            ("conservation", {"conservation_bad": True}),
            ("low_toggle", {"toggled": 19}),
            ("missing_critical", {"omit_critical": "commit_accept"}),
        ):
            path = tmp / f"bad_{name}.saif"
            write_saif(path, 20292 * 3.0, **kwargs)
            expect_failure(lambda p=path: parser_module.parse_saif(
                p, "ordinary_lru4"), f"saif_mutation_{name}_rejected", checks)
        bad_duration = tmp / "bad_duration.saif"
        write_saif(bad_duration, 20292 * 3.0 + 1.0)
        expect_failure(lambda: parser_module.parse_saif(
            bad_duration, "ordinary_lru4"),
            "saif_mutation_duration_rejected", checks)

        default_map = tmp / "default.tcl"
        essential_map = tmp / "essential.tcl"
        default_map.write_text(
            "set_rtl_to_gate_name -rtl {state_q} -gate U0\n"
            "set_rtl_to_gate_name -rtl {count_q} -gate U1\n")
        essential_map.write_text(
            "set_rtl_to_gate_name -rtl {state_q} -gate U0/Q\n"
            "set_rtl_to_gate_name -rtl {valid_q} -gate U2/Q\n")
        maps = parser_module.classify_maps(default_map, essential_map)
        need(maps["intersection_entries"] == 1 and maps["union_entries"] == 3
             and maps["intersection_target_difference_entries"] == 1,
             "map_union_intersection_and_cross_class_difference_retained", checks)
        conflict_map = tmp / "conflict.tcl"
        conflict_map.write_text(
            "set_rtl_to_gate_name -rtl {state_q} -gate U0\n"
            "set_rtl_to_gate_name -rtl {state_q} -gate U9\n")
        expect_failure(lambda: parser_module.map_rows(conflict_map),
                       "intra_class_map_conflict_rejected", checks)

        annotation = tmp / "annotation.rpt"
        annotation.write_text(
            "Total number of nets = 100\n"
            "Number of annotated nets = 95 (95.00%)\n"
            "Total number of leaf cells = 100\n"
            "Number of fully annotated leaf cells = 95 (95.00%)\n")
        need(parser_module.parse_annotation(annotation)["net_percent"] == 95.0,
             "annotation_boundary_95_pass", checks)
        annotation.write_text(annotation.read_text().replace("95.00%", "94.99%", 1))
        expect_failure(lambda: parser_module.parse_annotation(annotation),
                       "annotation_below_95_rejected", checks)

        coverage = tmp / "coverage.rpt"
        coverage.write_text("m2018_axis 20.00 20 100\n")
        need(parser_module.parse_switching_coverage(coverage)["percent"] == 20.0,
             "nonzero_toggle_boundary_20_pass", checks)
        coverage.write_text("m2018_axis 19.99 20 100\n")
        expect_failure(lambda: parser_module.parse_switching_coverage(coverage),
                       "nonzero_toggle_below_20_rejected", checks)

        critical = tmp / "critical.rpt"
        critical.write_text("mem_req_valid 0.25\n")
        need(parser_module.parse_critical(
            critical, "mem_req_valid")["has_nonzero_numeric"],
            "critical_cone_object_activity_pass", checks)
        critical.write_text("Report mem_req_valid 2026\n")
        expect_failure(lambda: parser_module.parse_critical(
            critical, "mem_req_valid"),
            "critical_header_number_cannot_spoof_activity", checks)

        power = tmp / "power.rpt"
        power.write_text(
            "Net Switching Power = 1.00000000\n"
            "Cell Internal Power = 2.00000000\n"
            "Cell Leakage Power = 0.50000000\n"
            "Total Power = 3.50000000\n")
        parsed_power = parser_module.parse_power(power, 30.0)
        need(abs(parsed_power["energy_nj"] - 0.105) < 1.0e-12,
             "power_and_energy_arithmetic_pass", checks)
        power.write_text(power.read_text().replace(
            "Total Power = 3.50000000", "Total Power = 9.50000000"))
        expect_failure(lambda: parser_module.parse_power(power, 30.0),
                       "power_component_mismatch_rejected", checks)

    frozen_after = {path.as_posix(): sha(path) for path in source_paths}
    need(frozen_before == frozen_after,
         "all_source_identities_stable_during_hammer", checks)
    output = {
        "schema": "m2114_m2113_source_mechanical_checks_r1_v1",
        "status": "FAIL_M2114_P0_DIRECT_SNPS_SHELL_BASENAME_DISPATCH__NO_EDA",
        "eda_or_license_invoked": False,
        "check_count": len(checks),
        "checks": checks,
        "p0_findings": [{
            "id": "P0_DC_LAUNCHER_BASENAME",
            "runner_positive_argv0":
                "/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell",
            "required_positive_argv0":
                "/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell",
            "reason": "installed wrapper selects the dc_shell backend from the original symlink basename",
            "required_repair": "invoke dc_shell -f while independently checking raw link text, resolved target, and target SHA256",
        }],
        "source_sha256": frozen_after,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
