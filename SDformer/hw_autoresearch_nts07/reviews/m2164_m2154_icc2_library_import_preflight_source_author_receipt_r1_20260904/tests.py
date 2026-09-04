#!/usr/bin/python3.12
"""Pure M2164 mutation tests; no EDA, license client, GPU, or git."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
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


selfcheck = load("m2164_selfcheck", HERE / "selfcheck.py")
checker = load("m2164_checker", selfcheck.CHECKER)
inventory_module = load("m2164_inventory", selfcheck.INVENTORY)
tests = 0


def expect_reject(label: str, fn) -> None:
    global tests
    try:
        fn()
    except Exception:
        tests += 1
        print(f"PASS_MUTATION_REJECTED {label}")
        return
    raise AssertionError(f"mutation survived: {label}")


def native_fixture(root: Path) -> tuple[Path, Path]:
    frame = root / "frame.ndm"
    design = root / "design.nlib"
    design.mkdir(parents=True)
    shutil.copyfile(checker.KNOWN_NDM, frame)
    shutil.copyfile(checker.KNOWN_NDM, design / "reflib.ndm")
    return frame, design


with tempfile.TemporaryDirectory(prefix="m2164_native_") as raw:
    frame, design = native_fixture(Path(raw))
    assert checker.native_ndm_member(frame, "frame")["size_bytes"] > 0
    assert checker.native_ndm_member(design / "reflib.ndm", "design")["size_bytes"] > 0
    tests += 2
    forged = Path(raw) / "forged.ndm"
    forged.write_bytes(b"\0FORGED_NOT_NATIVE")
    expect_reject("nul_prefixed_arbitrary_file", lambda: checker.native_ndm_member(forged, "forged"))
    wrong_magic = Path(raw) / "wrong_magic.ndm"
    wrong_magic.write_bytes(b"X" + checker.KNOWN_NDM.read_bytes()[1:])
    expect_reject("wrong_native_magic", lambda: checker.native_ndm_member(wrong_magic, "wrong magic"))
    wrong_release = Path(raw) / "wrong_release.ndm"
    blob = checker.KNOWN_NDM.read_bytes()
    wrong_release.write_bytes(blob[:31] + b"X" + blob[32:])
    expect_reject("wrong_library_manager_release", lambda: checker.native_ndm_member(wrong_release, "wrong release"))
    wrong_suffix = Path(raw) / "native.db"
    shutil.copyfile(checker.KNOWN_NDM, wrong_suffix)
    expect_reject("wrong_native_suffix", lambda: checker.native_ndm_member(wrong_suffix, "wrong suffix"))
    frame_dir = Path(raw) / "directory.ndm"
    frame_dir.mkdir()
    shutil.copyfile(checker.KNOWN_NDM, frame_dir / "native.db")
    expect_reject("directory_named_ndm", lambda: checker.native_ndm_member(frame_dir, "directory"))
    missing = Path(raw) / "missing_design.nlib/reflib.ndm"
    expect_reject("missing_design_reflib", lambda: checker.native_ndm_member(missing, "missing"))


def process_fixture() -> dict[str, object]:
    isolated = Path("/tmp/m2164_process_fixture/isolated_cwd")
    env = {
        "HOME": str(isolated / "home"),
        "TMPDIR": str(isolated / "tmp"),
        "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
        "M2153_ISOLATED_CWD": str(isolated),
    }
    wrapper = {
        "comm": "sh", "exe_path": "/usr/bin/dash",
        "cmdline": ["/bin/sh", str(checker.ICC2_WRAPPER), "-no_init", "-f", str(checker.TCL)],
        "selected_environment": env,
    }
    actual = {
        "comm": "icc2_exec", "exe_path": str(checker.ICC2_REAL),
        "cmdline": [str(checker.ICC2_REAL), "-no_init", "-f", str(checker.TCL)],
        "selected_environment": env,
    }
    identities = [
        {"pid": 100, "starttime_ticks": 1000, "first_ppid": 1,
         "parent_links": [{"ppid": 1, "parent_starttime_ticks": None}],
         "exec_observations": [wrapper]},
        {"pid": 101, "starttime_ticks": 1001, "first_ppid": 100,
         "parent_links": [{"ppid": 100, "parent_starttime_ticks": 1000}],
         "exec_observations": [actual]},
    ]
    flat_wrapper = {"pid": 100, "starttime_ticks": 1000, **wrapper}
    flat_actual = {"pid": 101, "starttime_ticks": 1001, **actual}
    return {
        "schema": "m2153_icc2_process_tree_r1_v1",
        "root_pid": 100, "root_seen": True, "sample_count": 10,
        "unique_process_identity_count": 2, "exec_observation_count": 2,
        "icc2_wrapper_observation_count": 1, "icc2_wrapper_observations": [flat_wrapper],
        "icc2_actual_exec_observation_count": 1, "icc2_actual_exec_observations": [flat_actual],
        "tool_spawned_conversion_exec_observation_count": 0,
        "tool_spawned_conversion_exec_observations": [],
        "all_observed_processes": identities,
    }


isolated = Path("/tmp/m2164_process_fixture/isolated_cwd")
baseline = process_fixture()
summary = checker.validate_process_tree(copy.deepcopy(baseline), isolated)
assert summary == {"identities": 2, "exec_observations": 2, "actual_exec_observations": 1}
tests += 1


def disconnected_cycle() -> None:
    p = process_fixture()
    actual = p["all_observed_processes"][1]["exec_observations"][0]
    p["all_observed_processes"][1] = {
        "pid": 101, "starttime_ticks": 1001, "first_ppid": 102,
        "parent_links": [{"ppid": 102, "parent_starttime_ticks": 1001}],
        "exec_observations": [actual],
    }
    p["all_observed_processes"].append({
        "pid": 102, "starttime_ticks": 1001, "first_ppid": 101,
        "parent_links": [{"ppid": 101, "parent_starttime_ticks": 1001}],
        "exec_observations": [{"comm": "helper", "exe_path": "/bin/true",
                               "cmdline": ["/bin/true"], "selected_environment": {}}],
    })
    p["unique_process_identity_count"] = 3
    p["exec_observation_count"] = 3
    checker.validate_process_tree(p, isolated)


expect_reject("disconnected_actual_parent_cycle", disconnected_cycle)


def reachable_extra_cycle() -> None:
    p = process_fixture()
    p["all_observed_processes"][0]["parent_links"].append(
        {"ppid": 101, "parent_starttime_ticks": 1001})
    checker.validate_process_tree(p, isolated)


expect_reject("reachable_cycle_extra_edge", reachable_extra_cycle)


def parent_after_child() -> None:
    p = process_fixture()
    p["all_observed_processes"][1]["parent_links"][0]["parent_starttime_ticks"] = 2000
    checker.validate_process_tree(p, isolated)


expect_reject("parent_start_after_child", parent_after_child)


def orphan() -> None:
    p = process_fixture()
    p["all_observed_processes"][1]["parent_links"] = [{"ppid": 999, "parent_starttime_ticks": 1}]
    checker.validate_process_tree(p, isolated)


expect_reject("orphan_nonroot", orphan)


def duplicate_root_pid() -> None:
    p = process_fixture()
    p["all_observed_processes"][1]["pid"] = 100
    checker.validate_process_tree(p, isolated)


expect_reject("duplicate_root_pid_identity", duplicate_root_pid)


def summary_count_forgery() -> None:
    p = process_fixture()
    p["unique_process_identity_count"] = 99
    checker.validate_process_tree(p, isolated)


expect_reject("summary_count_forgery", summary_count_forgery)


def actual_missing_no_init() -> None:
    p = process_fixture()
    p["all_observed_processes"][1]["exec_observations"][0]["cmdline"].remove("-no_init")
    p["icc2_actual_exec_observations"][0]["cmdline"].remove("-no_init")
    checker.validate_process_tree(p, isolated)


expect_reject("actual_missing_no_init", actual_missing_no_init)


def actual_wrong_tcl() -> None:
    p = process_fixture()
    for item in (p["all_observed_processes"][1]["exec_observations"][0],
                 p["icc2_actual_exec_observations"][0]):
        item["cmdline"][-1] = "/tmp/forged.tcl"
    checker.validate_process_tree(p, isolated)


expect_reject("actual_wrong_tcl", actual_wrong_tcl)


def actual_wrong_environment() -> None:
    p = process_fixture()
    p["all_observed_processes"][1]["exec_observations"][0]["selected_environment"]["HOME"] = "/root"
    p["icc2_actual_exec_observations"][0]["selected_environment"]["HOME"] = "/root"
    checker.validate_process_tree(p, isolated)


expect_reject("actual_wrong_environment", actual_wrong_environment)

def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def full_fixture(root: Path) -> Path:
    work = root.resolve()
    reports = work / "isolated_cwd/reports"
    frame = work / "isolated_cwd/frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm"
    design = work / "isolated_cwd/m2153_disposable_design.nlib"
    reports.mkdir(parents=True)
    frame.parent.mkdir(parents=True)
    design.mkdir(parents=True)
    shutil.copyfile(checker.KNOWN_NDM, frame)
    shutil.copyfile(checker.KNOWN_NDM, design / "reflib.ndm")
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
        "schema": "m2166_m2164_execution_contract_r1_v1",
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






def full_expect_reject(label: str, mutate) -> None:
    global tests
    with tempfile.TemporaryDirectory(prefix=f"m2164_full_{label}_") as raw:
        work = full_fixture(Path(raw) / "work")
        mutate(work)
        try:
            checker.validate(work, work / "receipt.json")
        except Exception:
            tests += 1
            print(f"PASS_FULL_MUTATION_REJECTED {label}")
            return
        raise AssertionError(f"full mutation survived: {label}")


with tempfile.TemporaryDirectory(prefix="m2164_full_baseline_") as raw:
    work = full_fixture(Path(raw) / "work")
    payload = checker.validate(work, work / "receipt.json")
    assert payload["status"] == "RAW_PASS_M2166_M2164_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2167_INDEPENDENT_RESULT_HAMMER"
    tests += 1


def forged_gate(work: Path) -> None:
    path = work / "icc2_preflight.log"
    path.write_text(path.read_text().replace("M2153_GATE1_OPTION_ROUND_TRIP_PASS", "M2153_GATE1_FORGED", 1))


full_expect_reject("forged_gate", forged_gate)


def wrong_master(work: Path) -> None:
    path = work / "isolated_cwd/reports/master_coverage.tsv"
    lines = path.read_text().splitlines()
    lines[1] = "AAA_FAKE_MASTER\t1\t1\t1\t1"
    path.write_text("\n".join(lines) + "\n")


full_expect_reject("wrong_master", wrong_master)


def nul_database_forgery(work: Path) -> None:
    frame = work / "isolated_cwd/frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm"
    design = work / "isolated_cwd/m2153_disposable_design.nlib/reflib.ndm"
    frame.write_bytes(b"\0FORGED_FRAME")
    design.write_bytes(b"\0FORGED_DESIGN")


full_expect_reject("nul_database_forgery", nul_database_forgery)


def root_inventory_drift(work: Path) -> None:
    path = work / "repo_root_after.json"
    payload = json.loads(path.read_text())
    payload["node_count"] += 1
    write_json(path, payload)


full_expect_reject("root_inventory_drift", root_inventory_drift)


def process_summary_forgery(work: Path) -> None:
    path = work / "process_tree.json"
    payload = json.loads(path.read_text())
    payload["exec_observation_count"] += 1
    write_json(path, payload)


full_expect_reject("process_summary_forgery", process_summary_forgery)

print(f"PASS_M2164_MUTATION_TESTS tests={tests} eda_runs=0 license_queries=0")
