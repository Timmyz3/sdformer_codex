#!/usr/bin/env python3
"""Independent no-EDA source hammer for M2125/M2127."""

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
RUNNER = HW / "dc_handoff/scripts/run_m2125_m2018_tsbg_rtl_saif_window_diagnostic_one_shot.py"
PARSER = HW / "system_simulator/scripts/parse_m2125_m2018_tsbg_rtl_saif_window_diagnostic.py"
TB = HW / "tb_m2018/tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.sv"
CONTRACT = HW / "contracts/m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_20260904.json"
SELFCHECK = HW / "reviews/m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_selfcheck_r1_20260904"
REQUEST = SELFCHECK / "M2126_REVIEW_REQUEST.md"
M2117_CONTRACT = HW / "contracts/m2117_m2018_tsbg_rtl_saifmap_power_source_contract_r1_20260904.json"
M2119 = HW / "results/m2119_m2117_m2018_tsbg_rtl_saifmap_power_r1_20260904.failed.1771297.quarantine"
M2120 = HW / "reviews/m2120_m2119_m2117_tsbg_saifmap_power_failure_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
RESULT = HW / "results/m2127_m2125_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904"
ATTEMPT = HW / "results/.m2127_m2125_tsbg_rtl_saif_window_diagnostic_attempt_consumed"
LOCK = HW / "results/.m2127_m2125_tsbg_rtl_saif_window_diagnostic_launch_lock"
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
                          AssertionError("nonfinite token: " + token)))


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


def verify_seal(root: Path, expected_manifest: str | None = None,
                expected_outer: str | None = None) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise AssertionError("sealed directory absent: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if expected_manifest:
        assert sha(manifest) == expected_manifest
    if expected_outer:
        assert sha(outer) == expected_outer
    assert outer.read_text().split() == [sha(manifest), "SHA256SUMS"]
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, rel_text = line.split(maxsplit=1)
        rel = Path(rel_text.lstrip("*"))
        assert not rel.is_absolute() and ".." not in rel.parts
        path = root / rel
        assert path.is_file() and not path.is_symlink() and sha(path) == digest
        assert rel.as_posix() not in rows
        rows[rel.as_posix()] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    assert set(rows) == actual
    return rows


def write_runtime(path: Path, axis: str, cycles: int, reads: int) -> None:
    path.write_text(
        "M2125_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 "
        "sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 "
        "preload_cycles=383 time_ns=1150.51\n"
        f"M2125_RTL_SAIF_WINDOW_END axis={axis} sampling=settled_negedge "
        f"measurement_cycles={cycles} scalar_weight_reads={reads} "
        f"duration_ns={cycles * 3.0:.2f}\n"
        "PASS_M2125_RTL_SAIF_WINDOW_DIAGNOSTIC_AXIS ledger_exact=1 "
        "initreg_diagnostic_only=1 paper_citable=0\n")


def write_saif(path: Path, duration: float, *, record_count: int = 93971,
               tx_bad: bool = False, conservation_bad: bool = False,
               critical_zero: str | None = None) -> None:
    critical = ["mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
                "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept"]
    rows = []
    for index in range(record_count):
        name = critical[index] if index < len(critical) else f"signal_{index:05d}"
        tx = 1.0 if tx_bad and index == 100 else 0.0
        t0 = duration / 2.0
        t1 = duration / 2.0 - tx
        if conservation_bad and index == 100:
            t1 -= 1.0
        tc = 0 if name == critical_zero else (1 + index % 3)
        rows.append(f"({name} (T0 {t0}) (T1 {t1}) (TX {tx}) (TC {tc}))")
    path.write_text("(SAIFILE\n(TIMESCALE 1 ns)\n"
                    f"(DURATION {duration})\n(INSTANCE implementation\n(NET\n"
                    + "\n".join(rows) + "\n)))\n")


def main() -> None:
    checks = {}
    contract = strict_json(CONTRACT)
    selfcheck = strict_json(SELFCHECK / "selfcheck.json")
    inventory = contract["source_inventory"]
    frozen = [REPO / rel for rel in inventory] + [CONTRACT, REQUEST, M2117_CONTRACT]
    frozen_before = {str(p): sha(p) for p in frozen}

    need(contract["schema"] == "m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_v1",
         "contract_schema_exact", checks)
    need(contract["status"] == "SOURCE_ONLY__M2126_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "contract_source_only_status", checks)
    budget = {"license_queries": 1, "vcs_compiles": 1, "simv_runs": 2,
              "saif_files": 2, "dc_runs": 0, "ptpx_runs": 0,
              "automatic_retry": False, "p1_serial": True,
              "reuse_old_artifacts": False}
    need(contract["execution_budget"] == budget, "future_budget_exact", checks)
    need(selfcheck["authorization"]["future_m2127_budget"] == budget,
         "selfcheck_future_budget_exact", checks)
    need(selfcheck["execution_performed"] == {"license_queries": 0, "vcs_compiles": 0,
         "simv_runs": 0, "saif_files": 0, "dc_runs": 0, "ptpx_runs": 0},
         "selfcheck_no_execution", checks)
    need(verify_seal(SELFCHECK) and (SELFCHECK / "RUN_COMPLETE.txt").read_text().strip().startswith("PASS_M2125"),
         "m2125_selfcheck_exhaustive_double_seal", checks)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract_inner_seal_exact", checks)
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract_outer_seal_exact", checks)
    need(len(inventory) == 15 and len(inventory) == len(set(inventory)),
         "source_inventory_15_unique", checks)
    need(all((REPO / rel).is_file() and not (REPO / rel).is_symlink()
             and sha(REPO / rel) == digest for rel, digest in inventory.items()),
         "source_inventory_15_of_15_exact_regular", checks)

    m2117 = strict_json(M2117_CONTRACT)
    need(len(m2117["source_inventory"]) == 21
         and all((REPO / rel).is_file() and not (REPO / rel).is_symlink()
                 and sha(REPO / rel) == digest
                 for rel, digest in m2117["source_inventory"].items()),
         "m2117_inventory_21_of_21_unchanged", checks)
    pred = contract["predecessor_fingerprint"]
    members = verify_seal(M2119, pred["manifest_sha256"], pred["outer_sha256"])
    need(set(members) == {"FAILED_DO_NOT_CITE.txt", "execution_counts.json",
         "license_preflight.log", "ordinary_lru4/rtl_execute.saif",
         "ordinary_lru4/rtl_sim.log", "ordinary_lru4/saif_parse.log",
         "vcs_compile.log"}, "m2119_exact_seven_members", checks)
    need(strict_json(M2119 / "execution_counts.json") == {
         "dc_runs": 0, "license_queries": 1, "ptpx_runs": 0,
         "saif_files": 0, "simv_runs": 1, "vcs_compiles": 1},
         "m2119_consumed_counts_exact", checks)
    need("status=FAILED_DO_NOT_CITE" in (M2119 / "FAILED_DO_NOT_CITE.txt").read_text()
         and "automatic_retry=false" in (M2119 / "FAILED_DO_NOT_CITE.txt").read_text(),
         "m2119_permanently_failed_no_retry", checks)
    saif_text = (M2119 / "ordinary_lru4/rtl_execute.saif").read_text(errors="replace")
    old_records = re.findall(r"\(T0\s+([0-9.eE+-]+)\)\s*\(T1\s+([0-9.eE+-]+)\)\s*"
                             r"\(TX\s+([0-9.eE+-]+)\)\s*\(TC\s+([0-9.eE+-]+)\)", saif_text)
    need(re.findall(r"\(DURATION\s+([0-9.eE+-]+)\)", saif_text) == ["60877.50"]
         and len(old_records) == 93971
         and sum(float(row[2]) != 0.0 for row in old_records) == 58277,
         "m2119_duration_record_tx_fingerprint_exact", checks)
    m2120_members = verify_seal(M2120, expected_outer=pred["m2120_review_outer_sha256"])
    need(sha(M2120 / "review.json") == pred["m2120_review_json_sha256"],
         "m2120_review_identity_exact", checks)
    old_review = strict_json(M2120 / "review.json")
    need(old_review["status"] == "PASS_M2120_M2119_FAILURE_HAMMER__M2119_CONSUMED_NO_POWER__M2125_SOURCE_AUTHORING_ONLY_ALLOWED",
         "m2120_failure_disposition_exact", checks)
    need(old_review["only_allowed_successor"]["source_identity"] == "M2125"
         and old_review["only_allowed_successor"]["direct_vcs_execution_authorized_now"] is False,
         "m2120_successor_boundary_exact", checks)
    need(len(m2120_members) > 0, "m2120_review_nonempty_exhaustive", checks)
    need(sha(DOC359) == DOC_SHA, "docs359_identity_preserved", checks)

    for label, path, expected in (
        ("vcs", VCS, contract["tool_identity"]["vcs"]),
        ("lmutil", LMUTIL, contract["tool_identity"]["lmutil"]),
    ):
        need(path.is_file() and not path.is_symlink() and os.access(path, os.X_OK),
             label + "_regular_nonsymlink_executable", checks)
        need(str(path) == expected["path"] and sha(path) == expected["sha256"],
             label + "_path_sha_exact", checks)

    runner = RUNNER.read_text()
    parser = PARSER.read_text()
    tb = TB.read_text()
    ucli = {axis: (HW / rel).read_text().splitlines() for axis, rel in {
        "ordinary_lru4": "dc_handoff/scripts/m2125_m2018_tsbg_ordinary_rtl_saif_window_diagnostic.ucli.tcl",
        "tsbg_b4": "dc_handoff/scripts/m2125_m2018_tsbg_tsbg_rtl_saif_window_diagnostic.ucli.tcl",
    }.items()}
    for path in (RUNNER, PARSER):
        proc = subprocess.run(["python3.12", "-m", "py_compile", str(path)], check=False)
        need(proc.returncode == 0, path.name + "_python_compile", checks)
    runner_static = subprocess.run(["python3.12", str(RUNNER), "--static"], cwd=REPO,
                                   text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, check=False, timeout=30)
    need(runner_static.returncode == 0, "runner_static_exit_zero", checks)
    runner_value = json.loads(runner_static.stdout)
    need(runner_value["status"] == "PASS_M2125_STATIC_RUNNER"
         and runner_value["source_count"] == 15
         and runner_value["execution_budget"] == budget,
         "runner_static_pass_exact", checks)
    parser_static = subprocess.run(["python3.12", str(PARSER), "static"], cwd=REPO,
                                   text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, check=False, timeout=30)
    parser_value = json.loads(parser_static.stdout)
    need(parser_static.returncode == 0 and parser_value["status"] == "PASS_M2125_STATIC_PARSER"
         and all(parser_value["checks"].values()), "parser_static_pass", checks)

    tree = ast.parse(runner)
    compile_lists = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "compile_command"
                             for target in node.targets)]
    sim_lists = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "sim_command"
                         for target in node.targets)]
    need(len(compile_lists) == 1 and isinstance(compile_lists[0], ast.List),
         "one_compile_command_literal", checks)
    compile_constants = [elt.value for elt in compile_lists[0].elts if isinstance(elt, ast.Constant)]
    need(compile_constants.count("+vcs+initreg+random") == 1
         and not any(isinstance(x, str) and x.startswith("+vcs+initreg+")
                     and x != "+vcs+initreg+random" for x in compile_constants),
         "compile_exact_one_initreg_random", checks)
    need(len(sim_lists) == 1 and isinstance(sim_lists[0], ast.List),
         "one_serial_sim_command_in_two_axis_loop", checks)
    sim_constants = [elt.value for elt in sim_lists[0].elts if isinstance(elt, ast.Constant)]
    need(sim_constants.count("+vcs+initreg+0") == 1
         and sim_constants.count("+WORKLOAD_SLOT=42") == 1,
         "runtime_exact_initreg_zero_and_slot42", checks)
    need(set(contract["axes"]) == {"ordinary_lru4", "tsbg_b4"}
         and runner.count('"plusarg": "+M2125_AXIS_') == 2
         and "for axis, cfg in AXES.items():" in runner,
         "two_fixed_axes_strictly_serial", checks)
    need(runner.count("run(compile_command") == 1 and runner.count("run(sim_command") == 1,
         "single_compile_and_serial_sim_launch_sites", checks)
    need("source_validation(require_review=True)" in runner
         and runner.index("source_validation(require_review=True)")
         < runner.index("LOCK.mkdir()") < runner.index('counts["license_queries"] += 1'),
         "review_collision_freshness_before_license_and_attempt", checks)
    need(not any(p.exists() for p in (RESULT, ATTEMPT, LOCK)),
         "m2127_result_attempt_lock_fresh", checks)
    blocked_match = re.search(r"blocked = \{(.*?)\}", runner, flags=re.S)
    blocked = set(re.findall(r'"([^"]+)"', blocked_match.group(1))) if blocked_match else set()
    need(blocked == {"vcs", "simv", "snps_shell", "dc_shell", "common_shell_exec",
                     "common_shell_exe", "pt_shell", "icc2_shell", "lmstat"},
         "same_uid_collision_set_exact", checks)

    forbidden_text = runner + tb + "\n".join(sum(ucli.values(), []))
    need(not re.search(r"\bforce\b|\brelease\b|assertoff|assertkill", forbidden_text, re.I),
         "no_force_release_assertion_suppression", checks)
    active_surface = tb + "\n".join(sum(ucli.values(), [])) + (
        HW / "dc_handoff/filelists/tcasii_m2125_m2018_tsbg_rtl_saif_window_diagnostic_vcs.f").read_text()
    need("UNIT_DELAY" not in active_surface
         and not re.search(r"(?:\$sdf|sdf_annotate|-sdf)", active_surface, re.I)
         and not any("UNIT_DELAY" in str(x) or "sdf" in str(x).lower()
                     for x in compile_constants + sim_constants),
         "no_unit_delay_or_sdf", checks)
    need("-xprop" not in runner and "+xprop" not in runner and "-2state" not in runner,
         "no_x_coercion_beyond_frozen_initreg", checks)
    need("measurement_window_active = 1'b1;\n        $display(\"M2125_RTL_SAIF_WINDOW_BEGIN" in tb
         and tb.count("@(negedge core.clk_core);") == 2
         and tb.count("#0.01;") == 3
         and "wait (core.full_execute_start_cycle >= 0);" in tb
         and "wait (core.base_done_cycle >= 0);" in tb
         and "wait (core.tsbg_done_cycle >= 0);" in tb,
         "settled_negedge_begin_and_selected_end_protocol", checks)
    need(tb.index("wait (core.full_execute_start_cycle >= 0);")
         < tb.index("@(negedge core.clk_core);")
         < tb.index("M2125_RTL_SAIF_WINDOW_BEGIN")
         < tb.index("wait (core.base_done_cycle >= 0);")
         < tb.rindex("@(negedge core.clk_core);")
         < tb.index("M2125_RTL_SAIF_WINDOW_END"),
         "window_boundary_order_exact", checks)

    expected_scopes = {
        "ordinary_lru4": "tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.core.dut_base.implementation",
        "tsbg_b4": "tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.core.dut_tsbg.implementation",
    }
    for axis, lines in ucli.items():
        scope = expected_scopes[axis]
        need(lines == ["power -gate_level all mda sv", "power " + scope, "run",
             "power -enable", "run", "power -disable",
             "power -report $::env(M2125_RTL_SAIF_FILE) 1e-9 " + scope, "quit"],
             "ucli_exact_selected_scope_and_two_stop_" + axis, checks)

    spec = importlib.util.spec_from_file_location("m2125_parser_hammer", PARSER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory(prefix="m2126.parser.") as temp_name:
        root = Path(temp_name)
        runtime = root / "runtime.log"
        write_runtime(runtime, "ordinary_lru4", 20292, 14304)
        need(mod.parse_runtime(runtime, "ordinary_lru4")["completion_ledger_exact"],
             "runtime_positive_exact_ledger", checks)
        clean_runtime = runtime.read_text()
        runtime.write_text(clean_runtime.replace("measurement_cycles=20292", "measurement_cycles=20291"))
        reject(lambda: mod.parse_runtime(runtime, "ordinary_lru4"),
               "runtime_cycle_mutation_rejected", checks)
        runtime.write_text(clean_runtime.replace("scalar_weight_reads=14304", "scalar_weight_reads=14303"))
        reject(lambda: mod.parse_runtime(runtime, "ordinary_lru4"),
               "runtime_read_mutation_rejected", checks)
        runtime.write_text(clean_runtime.replace("duration_ns=60876.00", "duration_ns=60877.50"))
        reject(lambda: mod.parse_runtime(runtime, "ordinary_lru4"),
               "runtime_duration_mutation_rejected", checks)
        runtime.write_text(clean_runtime + "Fatal: injected\n")
        reject(lambda: mod.parse_runtime(runtime, "ordinary_lru4"),
               "runtime_fatal_token_rejected", checks)

        saif = root / "axis.saif"
        write_saif(saif, 60876.0)
        positive = mod.parse_saif(saif, "ordinary_lru4")
        need(positive["record_count"] == 93971 and positive["tx_sum"] == 0.0
             and positive["conservation_failures"] == 0
             and all(value > 0 for value in positive["critical_nonzero_record_counts"].values()),
             "saif_positive_93971_tx0_conserved_active", checks)
        write_saif(saif, 60876.0, tx_bad=True)
        reject(lambda: mod.parse_saif(saif, "ordinary_lru4"), "saif_tx_nonzero_rejected", checks)
        write_saif(saif, 60876.0, conservation_bad=True)
        reject(lambda: mod.parse_saif(saif, "ordinary_lru4"), "saif_conservation_rejected", checks)
        write_saif(saif, 60876.0, record_count=93970)
        reject(lambda: mod.parse_saif(saif, "ordinary_lru4"), "saif_record_93970_rejected", checks)
        write_saif(saif, 60876.0, critical_zero="commit_accept")
        reject(lambda: mod.parse_saif(saif, "ordinary_lru4"), "saif_zero_critical_rejected", checks)
        write_saif(saif, 60877.5)
        reject(lambda: mod.parse_saif(saif, "ordinary_lru4"), "saif_old_duration_rejected", checks)

    # An absent/failed M2126 gate must fail before state or subprocess activity.
    spec = importlib.util.spec_from_file_location("m2125_runner_hammer", RUNNER)
    runner_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner_mod)
    with tempfile.TemporaryDirectory(prefix="m2126.prereview.") as temp_name:
        temp = Path(temp_name)
        runner_mod.REVIEW = temp / "missing_review"
        runner_mod.RESULT = temp / "result"
        runner_mod.ATTEMPT = temp / "attempt"
        runner_mod.LOCK = temp / "lock"
        reject(runner_mod.production, "missing_review_rejected", checks)
        need(not any((temp / name).exists() for name in ("result", "attempt", "lock")),
             "missing_review_creates_no_persistent_state", checks)

    need({str(p): sha(p) for p in frozen} == frozen_before,
         "all_m2125_and_predecessor_sources_unchanged", checks)
    output = {
        "schema": "m2126_m2125_mechanical_checks_r1_v1",
        "status": "PASS_M2126_MECHANICAL_CHECKS__NO_EDA_NO_LICENSE",
        "date_cst": "2026-09-04",
        "eda_invoked": False,
        "license_query_invoked": False,
        "checks": checks,
        "check_count": len(checks),
        "authorization_candidate": budget,
        "identity": {"runner_sha256": sha(RUNNER), "parser_sha256": sha(PARSER),
                     "tb_sha256": sha(TB), "contract_sha256": sha(CONTRACT),
                     "docs359_sha256": sha(DOC359)},
    }
    (HERE / "mechanical_checks.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(output["status"], "checks=" + str(len(checks)))


if __name__ == "__main__":
    main()
