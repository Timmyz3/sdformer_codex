#!/usr/bin/python3.12
"""Pure mutation tests for M2153; invokes no EDA, license query, or GPU."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


selfcheck = load("m2153_selfcheck", HERE / "selfcheck.py")
checker = load("m2153_checker", selfcheck.CHECKER)
inventory_module = load("m2153_inventory", selfcheck.INVENTORY)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_fixture(root: Path) -> Path:
    work = root.resolve()
    reports = work / "isolated_cwd/reports"
    frame = work / "isolated_cwd/frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm"
    design = work / "isolated_cwd/m2153_disposable_design.nlib"
    reports.mkdir(parents=True)
    frame.mkdir(parents=True)
    design.mkdir(parents=True)
    (frame / "native.db").write_bytes(b"\0M2153_FRAME_NATIVE_DB\n")
    (design / "native.db").write_bytes(b"\0M2153_DESIGN_NATIVE_DB\n")
    (reports / "reference_libraries.rpt").write_text("M2153 synthetic object report " + "R" * 96 + "\n")
    (reports / "design_library.rpt").write_text("M2153 synthetic design report " + "D" * 96 + "\n")
    isolated = work / "isolated_cwd"
    frame_stats = checker.tree_stats(frame)
    design_stats = checker.tree_stats(design)
    tech = "m2153_synthetic_technology"
    log_lines = [
        f"M2153_GATE1_OPTION_ROUND_TRIP_PASS cache={isolated / 'cache/library'}",
        f"M2153_GATE2_FRAME_CONVERSION_PASS status=1 frame={frame}",
        "M2153_GATE3_MASTER_COVERAGE_PASS count=94 views=4",
        f"M2153_GATE4_PHYSICAL_TECH_PASS site=core site_count=1 metals={checker.METALS} vias={checker.VIAS} tech={tech}",
        f"M2153_GATE5_RC_COMPATIBILITY_PASS name={checker.RC_NAME}",
        "M2153_GATE6_NONEMPTY_LIBRARY_OBJECTS_PASS "
        f"frame_files={frame_stats['regular_files']} frame_bytes={frame_stats['regular_bytes']} "
        f"design_files={design_stats['regular_files']} design_bytes={design_stats['regular_bytes']}",
        "RAW_PASS_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156_INDEPENDENT_RESULT_HAMMER",
    ]
    (work / "icc2_preflight.log").write_text("\n".join(log_lines) + "\n")
    (work / "icc2_preflight.rc").write_text("0\n")
    facts = {
        "status": "RAW_PASS_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156",
        "application_option_value": str(isolated / "cache/library"),
        "conversion_status": "1",
        "frame_ndm": str(frame),
        "frame_regular_files": str(frame_stats["regular_files"]),
        "frame_regular_bytes": str(frame_stats["regular_bytes"]),
        "design_lib": str(design),
        "design_regular_files": str(design_stats["regular_files"]),
        "design_regular_bytes": str(design_stats["regular_bytes"]),
        "current_library": "m2153_design",
        "tt_library": "tt",
        "ss_library": "ss",
        "ff_library": "ff",
        "physical_library": "physical",
        "mapped_master_union_count": "94",
        "tt_master_coverage": "94",
        "ss_master_coverage": "94",
        "ff_master_coverage": "94",
        "physical_master_coverage": "94",
        "core_site_name": "core",
        "core_site_count": "1",
        "routing_layers": checker.METALS,
        "via_layers": checker.VIAS,
        "current_technology": tech,
        "rc_technology_name": checker.RC_NAME,
        "rtl_imported": "false",
        "pnr_invoked": "false",
    }
    (reports / "machine_facts.txt").write_text("".join(f"{key}={value}\n" for key, value in facts.items()))
    masters = checker.MASTER_LIST.read_text().splitlines()
    (reports / "master_coverage.tsv").write_text(
        "master\ttt\tss\tff\tphysical\n" + "".join(f"{name}\t1\t1\t1\t1\n" for name in masters)
    )
    expected_env = {
        "HOME": str(isolated / "home"),
        "TMPDIR": str(isolated / "tmp"),
        "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
        "M2153_ISOLATED_CWD": str(isolated),
    }
    wrapper_observation = {
        "comm": "sh",
        "exe_path": "/usr/bin/dash",
        "cmdline": ["/bin/sh", str(checker.ICC2_WRAPPER), "-no_init", "-f", str(checker.TCL)],
        "selected_environment": expected_env,
    }
    actual_observation = {
        "comm": "icc2_exec",
        "exe_path": str(checker.ICC2_REAL),
        "cmdline": [str(checker.ICC2_REAL), "-root_path", "/opt/synopsys/icc2/V-2023.12-SP3", "-no_init", "-f", str(checker.TCL)],
        "selected_environment": expected_env,
    }
    identities = [
        {
            "pid": 100,
            "starttime_ticks": 1000,
            "first_ppid": 1,
            "parent_links": [{"ppid": 1, "parent_starttime_ticks": None}],
            "exec_observations": [wrapper_observation],
        },
        {
            "pid": 101,
            "starttime_ticks": 1001,
            "first_ppid": 100,
            "parent_links": [{"ppid": 100, "parent_starttime_ticks": 1000}],
            "exec_observations": [actual_observation],
        },
    ]
    process = {
        "schema": "m2153_icc2_process_tree_r1_v1",
        "root_pid": 100,
        "root_seen": True,
        "sample_count": 20,
        "unique_process_identity_count": 2,
        "exec_observation_count": 2,
        "icc2_wrapper_observation_count": 1,
        "icc2_wrapper_observations": [{"pid": 100, "starttime_ticks": 1000, **wrapper_observation}],
        "icc2_actual_exec_observation_count": 1,
        "icc2_actual_exec_observations": [{"pid": 101, "starttime_ticks": 1001, **actual_observation}],
        "tool_spawned_conversion_exec_observation_count": 0,
        "tool_spawned_conversion_exec_observations": [],
        "all_observed_processes": identities,
    }
    write_json(work / "process_tree.json", process)
    execution = {
        "schema": "m2155_m2153_execution_contract_r1_v1",
        "scope": "library_import_only",
        "license_queries": 1,
        "top_level_icc2_shell_runs": 1,
        "pnr_runs": 0,
        "automatic_retry": False,
        "icc2_invocation": [str(checker.ICC2_WRAPPER), "-no_init", "-f", str(checker.TCL)],
        "icc2_wrapper_sha256": checker.WRAPPER_SHA,
        "icc2_real_exec_path": str(checker.ICC2_REAL),
        "icc2_real_exec_sha256": checker.REAL_EXEC_SHA,
        "isolated_home": str(isolated / "home"),
        "isolated_tmpdir": str(isolated / "tmp"),
        "isolated_xdg_cache": str(isolated / "cache/xdg"),
        "isolated_library_cache": str(isolated / "cache/library"),
        "prior_m2135_collateral_action": "copied_byte_exact_original_preserved",
        "prior_m2135_collateral_sha256": checker.COLLATERAL_SHA,
    }
    write_json(work / "execution_contract.json", execution)
    root_inventory = inventory_module.inventory(checker.REPO)
    write_json(work / "repo_root_before.json", root_inventory)
    write_json(work / "repo_root_after.json", root_inventory)
    prior = work / "prior_m2135_collateral/icc2_output.txt"
    prior.parent.mkdir(parents=True)
    prior.write_bytes((checker.REPO / "icc2_output.txt").read_bytes())
    return work


def expect_reject(name: str, mutate) -> None:
    with tempfile.TemporaryDirectory(prefix=f"m2153_{name}_") as raw:
        work = make_fixture(Path(raw) / "work")
        mutate(work)
        try:
            checker.validate(work, work / "receipt.json")
        except Exception:
            print(f"PASS_MUTATION_REJECTED {name}")
            return
        raise AssertionError(f"mutation survived: {name}")


with tempfile.TemporaryDirectory(prefix="m2153_baseline_") as raw:
    work = make_fixture(Path(raw) / "work")
    payload = checker.validate(work, work / "receipt.json")
    assert payload["status"].startswith("RAW_PASS_M2155")
    print("PASS_SYNTHETIC_STRUCTURAL_BASELINE")


expect_reject("forged_gate", lambda work: (work / "icc2_preflight.log").write_text(
    (work / "icc2_preflight.log").read_text().replace(
        "M2153_GATE1_OPTION_ROUND_TRIP_PASS", "M2153_GATE1_FORGED_SEMANTICS", 1)))


def wrong_master(work: Path) -> None:
    path = work / "isolated_cwd/reports/master_coverage.tsv"
    lines = path.read_text().splitlines()
    lines[1] = "AAA_FAKE_MASTER\t1\t1\t1\t1"
    path.write_text("\n".join(lines) + "\n")


expect_reject("wrong_master", wrong_master)
expect_reject("empty_frame_ndm", lambda work: (work / "isolated_cwd/frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm/native.db").write_bytes(b""))


def pure_text_databases(work: Path) -> None:
    (work / "isolated_cwd/frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm/native.db").write_text("fake text only\n")
    (work / "isolated_cwd/m2153_disposable_design.nlib/native.db").write_text("fake text only\n")


expect_reject("pure_text_ndm_and_design_lib", pure_text_databases)


def contradictory_process(work: Path) -> None:
    path = work / "process_tree.json"
    payload = json.loads(path.read_text())
    payload["unique_process_identity_count"] = 99
    write_json(path, payload)


expect_reject("contradictory_process_count", contradictory_process)


def broken_parent(work: Path) -> None:
    path = work / "process_tree.json"
    payload = json.loads(path.read_text())
    payload["all_observed_processes"][1]["parent_links"] = [{"ppid": 999, "parent_starttime_ticks": 9999}]
    write_json(path, payload)


expect_reject("broken_process_parent", broken_parent)


def root_node_mutation(work: Path) -> None:
    path = work / "repo_root_after.json"
    payload = json.loads(path.read_text())
    payload["nodes"].append({"name": "zzz_forged_link", "node_type": "symlink", "mode_octal": "0777", "target": "/tmp"})
    payload["nodes"] = sorted(payload["nodes"], key=lambda item: item["name"])
    payload["node_count"] = len(payload["nodes"])
    write_json(path, payload)


expect_reject("repo_root_node_mutation", root_node_mutation)


def wrong_actual_exec(work: Path) -> None:
    path = work / "process_tree.json"
    payload = json.loads(path.read_text())
    payload["all_observed_processes"][1]["exec_observations"][0]["exe_path"] = "/tmp/fake_dgcom_exec"
    payload["icc2_actual_exec_observations"][0]["exe_path"] = "/tmp/fake_dgcom_exec"
    write_json(path, payload)


expect_reject("wrong_actual_exec", wrong_actual_exec)


def missing_no_init(work: Path) -> None:
    path = work / "process_tree.json"
    payload = json.loads(path.read_text())
    for section in (payload["all_observed_processes"][1]["exec_observations"], payload["icc2_actual_exec_observations"]):
        section[0]["cmdline"].remove("-no_init")
    write_json(path, payload)


expect_reject("actual_exec_missing_no_init", missing_no_init)


def omitted_conversion_classification(work: Path) -> None:
    path = work / "process_tree.json"
    payload = json.loads(path.read_text())
    payload["all_observed_processes"][1]["exec_observations"][0]["comm"] = "lm_shell_exec"
    payload["icc2_actual_exec_observations"][0]["comm"] = "lm_shell_exec"
    write_json(path, payload)


expect_reject("omitted_conversion_classification", omitted_conversion_classification)


def wrong_exec_sha(work: Path) -> None:
    path = work / "execution_contract.json"
    payload = json.loads(path.read_text())
    payload["icc2_real_exec_sha256"] = "0" * 64
    write_json(path, payload)


expect_reject("wrong_actual_exec_sha", wrong_exec_sha)


tcl = selfcheck.TCL.read_text()
runner = selfcheck.RUNNER.read_text()
monitor = selfcheck.MONITOR.read_text()
inventory_text = selfcheck.INVENTORY.read_text()
checker_text = selfcheck.CHECKER.read_text()
static_mutations = {
    "static_missing_no_init": (tcl, runner.replace('"${ICC2}" -no_init -f "${TCL}"', '"${ICC2}" -f "${TCL}"', 1), monitor, inventory_text, checker_text),
    "static_missing_isolated_home": (tcl, runner.replace('HOME="${ISOLATED}/home" ', "", 1), monitor, inventory_text, checker_text),
    "static_wildcard_core": (tcl.replace("get_site_defs -quiet -exact core", "get_site_defs -quiet *core*", 1), runner, monitor, inventory_text, checker_text),
    "static_second_icc2": (tcl, runner + '\n"${ICC2}" -no_init -f "${TCL}"\n', monitor, inventory_text, checker_text),
    "static_missing_actual_exec_pin": (tcl, runner.replace('sha_executable_exact 4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c "${ICC2_REAL}"', ": # removed", 1), monitor, inventory_text, checker_text),
    "static_missing_lm_shell_exec": (tcl, runner, monitor.replace('"lm_shell_exec",', "", 1), inventory_text, checker_text),
    "static_inventory_omits_symlink": (tcl, runner, monitor, inventory_text.replace("elif stat.S_ISLNK(mode):", "elif False:", 1), checker_text),
    "static_checker_omits_master_identity": (tcl, runner, monitor, inventory_text, checker_text.replace("observed_masters == frozen_masters", "len(observed_masters) == len(frozen_masters)", 1)),
}
for name, args in static_mutations.items():
    errors = selfcheck.source_errors(*args)
    assert errors, f"static mutation survived: {name}"
    print(f"PASS_STATIC_MUTATION_REJECTED {name} errors={len(errors)}")


with tempfile.TemporaryDirectory(prefix="m2153_inventory_types_") as raw:
    root = Path(raw) / "root"
    root.mkdir()
    (root / "regular").write_text("x")
    (root / "directory").mkdir()
    (root / "symlink").symlink_to("regular")
    os.mkfifo(root / "fifo")
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.bind(str(root / "socket"))
    try:
        payload = inventory_module.inventory(root)
    finally:
        endpoint.close()
    observed = {item["node_type"] for item in payload["nodes"]}
    assert {"regular", "directory", "symlink", "fifo", "socket"} <= observed
    print("PASS_ROOT_INVENTORY_RUNTIME_TYPES regular,directory,symlink,fifo,socket")


assert not list(HERE.rglob("__pycache__")), "selfcheck created forbidden pycache"
print("PASS_M2153_AUTHOR_MUTATION_TESTS total=20")
print("p0=0")
print("p1=0")
print("p2=0")
