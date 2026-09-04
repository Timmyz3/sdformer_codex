#!/usr/bin/python3.12
"""CPU-only source tests for M2189; never invokes LM, EDA, lmutil, or GPU."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
TCL = HW / "dc_handoff/scripts/run_lm_m2189_library_conversion_preflight.tcl"
OLD_TCL = HW / "dc_handoff/scripts/run_lm_m2180_library_conversion_preflight.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2189_m2181_lm_library_conversion_preflight_one_shot.sh"
MONITOR = HW / "dc_handoff/scripts/monitor_m2189_lm_conversion_process_tree.py"
CHECKER = HW / "system_simulator/scripts/check_m2189_lm_library_conversion_preflight.py"
LM = "/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell"
ACTUAL = "/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec"
MW = "/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checker():
    spec = importlib.util.spec_from_file_location("m2189_checker", CHECKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def obs(exe: str, cmdline: list[str], isolated: Path) -> dict:
    return {"comm": Path(exe).name, "exe_path": exe, "cmdline": cmdline,
            "selected_environment": {
                "HOME": str(isolated / "home"), "TMPDIR": str(isolated / "tmp"),
                "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
                "M2189_ISOLATED_CWD": str(isolated)}}


def good_tree(isolated: Path) -> dict:
    wrapper = obs("/usr/bin/bash", [LM, "-no_init", "-f", str(TCL)], isolated)
    actual = obs(ACTUAL, [ACTUAL, "-no_init", "-f", str(TCL)], isolated)
    milkyway = obs(MW, [MW, "-lm_mode"], isolated)
    identities = [
        {"pid": 100, "starttime_ticks": 1000, "first_ppid": 1,
         "parent_links": [{"ppid": 1, "parent_starttime_ticks": None}],
         "exec_observations": [wrapper]},
        {"pid": 101, "starttime_ticks": 1001, "first_ppid": 100,
         "parent_links": [{"ppid": 100, "parent_starttime_ticks": 1000}],
         "exec_observations": [actual]},
        {"pid": 102, "starttime_ticks": 1002, "first_ppid": 101,
         "parent_links": [{"ppid": 101, "parent_starttime_ticks": 1001}],
         "exec_observations": [milkyway]},
    ]
    return {"schema": "m2189_lm_conversion_process_tree_r1_v1", "root_pid": 100,
            "root_seen": True, "sample_count": 9,
            "unique_process_identity_count": 3, "exec_observation_count": 3,
            "lm_wrapper_observations": [{"pid": 100, "starttime_ticks": 1000, **wrapper}],
            "lm_actual_exec_observations": [{"pid": 101, "starttime_ticks": 1001, **actual}],
            "milkyway_child_observations": [{"pid": 102, "starttime_ticks": 1002, **milkyway}],
            "unexpected_process_identities": [], "unexpected_process_observations": [],
            "all_observed_processes": identities}


def rejected(module, payload: dict, isolated: Path) -> None:
    try:
        module.validate_process_tree(payload, isolated)
    except module.Failure:
        return
    raise AssertionError("mutation was accepted")


def add_identity(tree: dict, pid: int, parent: int, parent_ticks: int,
                 observation: dict) -> dict:
    item = copy.deepcopy(tree)
    row = {"pid": pid, "starttime_ticks": 1000 + pid - 99, "first_ppid": parent,
           "parent_links": [{"ppid": parent, "parent_starttime_ticks": parent_ticks}],
           "exec_observations": [observation]}
    item["all_observed_processes"].append(row)
    item["unique_process_identity_count"] += 1
    item["exec_observation_count"] += 1
    return item


def main() -> int:
    module = load_checker()
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    for path in (MONITOR, CHECKER):
        compile(path.read_text(), str(path), "exec")

    new = TCL.read_text()
    normalized = (new.replace("M2189", "M2180").replace("m2189", "m2180")
                  .replace("M2191", "M2182").replace("M2192", "M2183"))
    assert normalized == OLD_TCL.read_text(), "TCL semantic delta exceeds milestone renaming"
    ordered = [
        "set_app_options -name lib.setting.milkyway_exec -value $milkyway_exec",
        "get_app_option_value -name lib.setting.milkyway_exec",
        "M2189_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS",
        "generate_frame_from_mw $frame_name",
    ]
    positions = [new.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert new.count("generate_frame_from_mw $frame_name") == 1
    for command in ("create_lib", "read_verilog", "read_sverilog", "place_opt",
                    "clock_opt", "route_opt", "report_timing", "report_power"):
        assert not re.search(rf"(?m)^\s*{re.escape(command)}(?:\s|$)", new)
    runner = RUNNER.read_text()
    assert runner.count('"${LM_SHELL}" -no_init -f "${TCL}"') == 1
    assert runner.count('"${LMUTIL}" lmstat ') == 1
    assert "M2182_PERMANENTLY_UNAUTHORIZED" in runner

    with tempfile.TemporaryDirectory(prefix="m2189_source_test_") as td:
        base = Path(td)
        native = base / "good.ndm"
        native.write_bytes(module.NATIVE_HEADER + b"M2189_NATIVE_PAYLOAD")
        stats = module.validate_native_frame(native)
        assert stats["regular_files"] == 1 and stats["regular_bytes"] > len(module.NATIVE_HEADER)
        bad = base / "bad.ndm"
        bad.write_bytes(b"not-a-library-manager-frame")
        try:
            module.validate_native_frame(bad)
        except module.Failure:
            pass
        else:
            raise AssertionError("arbitrary NDM accepted")

        isolated = base / "isolated"
        tree = good_tree(isolated)
        assert module.validate_process_tree(tree, isolated) == {
            "identities": 3, "observations": 3, "actual_identities": 1,
            "milkyway_identities": 1, "unexpected_identities": 0,
            "unexpected_observations": 0}
        mutations: list[dict] = []
        # Required exhaustive descendant mutations.
        mutations.append(add_identity(tree, 103, 101, 1001,
            obs("/usr/bin/sleep", ["/usr/bin/sleep", "5"], isolated)))
        mutations.append(add_identity(tree, 103, 100, 1000,
            obs("/usr/bin/bash", [LM, "-no_init", "-f", str(TCL)], isolated)))
        mutations.append(add_identity(tree, 103, 100, 1000,
            obs(ACTUAL, [ACTUAL, "-no_init", "-f", str(TCL)], isolated)))
        mutations.append(add_identity(tree, 103, 101, 1001,
            obs(MW, [MW, "-lm_mode"], isolated)))
        item = copy.deepcopy(tree)
        item["all_observed_processes"][2]["parent_links"] = [
            {"ppid": 100, "parent_starttime_ticks": 1000}]
        mutations.append(item)
        for index in (1, 2):
            item = copy.deepcopy(tree)
            item["all_observed_processes"][index]["exec_observations"][0][
                "selected_environment"]["HOME"] = "/tmp/wrong"
            mutations.append(item)
        item = copy.deepcopy(tree)
        item["all_observed_processes"][0]["exec_observations"][0]["cmdline"][0] = "/tmp/fake_lm_shell"
        item["lm_wrapper_observations"] = []
        mutations.append(item)
        item = copy.deepcopy(tree)
        item["all_observed_processes"][1]["exec_observations"][0]["exe_path"] = "/tmp/fake_lm_shell_exec"
        item["lm_actual_exec_observations"] = []
        mutations.append(item)
        item = copy.deepcopy(tree)
        item["all_observed_processes"][2]["exec_observations"][0]["exe_path"] = "/tmp/fake_Milkyway"
        item["milkyway_child_observations"] = []
        mutations.append(item)
        item = copy.deepcopy(tree)
        item["all_observed_processes"][1]["exec_observations"].append(
            obs("/usr/bin/sleep", ["/usr/bin/sleep", "1"], isolated))
        item["exec_observation_count"] += 1
        mutations.append(item)
        item = copy.deepcopy(tree); item["schema"] = "wrong"; mutations.append(item)
        item = copy.deepcopy(tree); item["unique_process_identity_count"] = 2; mutations.append(item)
        item = copy.deepcopy(tree); item["exec_observation_count"] = 4; mutations.append(item)
        for item in mutations:
            rejected(module, item, isolated)

    print("PASS_M2189_SOURCE_TESTS tcl_semantic_delta=identity_only "
          f"native_controls=1 native_mutations=1 process_controls=1 "
          f"process_mutations={len(mutations)} eda_runs=0 license_queries=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
