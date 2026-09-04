#!/usr/bin/python3.12
"""Source-only tests for M2180. This file never invokes EDA or lmutil."""
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
TCL = HW / "dc_handoff/scripts/run_lm_m2180_library_conversion_preflight.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2180_m2171_lm_library_conversion_preflight_one_shot.sh"
MONITOR = HW / "dc_handoff/scripts/monitor_m2180_lm_conversion_process_tree.py"
CHECKER = HW / "system_simulator/scripts/check_m2180_lm_library_conversion_preflight.py"
M2171 = HW / "reviews/m2171_m2170_m2168_icc2_library_import_preflight_failure_hammer_r1_20260904"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
LM = "/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell"
ACTUAL = "/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec"
MW = "/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checker():
    spec = importlib.util.spec_from_file_location("m2180_checker", CHECKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def obs(exe: str, cmdline: list[str], isolated: Path) -> dict:
    return {"comm": Path(exe).name, "exe_path": exe, "cmdline": cmdline,
            "selected_environment": {
                "HOME": str(isolated / "home"), "TMPDIR": str(isolated / "tmp"),
                "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
                "M2180_ISOLATED_CWD": str(isolated)}}


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
    return {"schema": "m2180_lm_conversion_process_tree_r1_v1", "root_pid": 100,
            "root_seen": True, "sample_count": 9,
            "unique_process_identity_count": 3, "exec_observation_count": 3,
            "lm_wrapper_observations": [{"pid": 100, "starttime_ticks": 1000, **wrapper}],
            "lm_actual_exec_observations": [{"pid": 101, "starttime_ticks": 1001, **actual}],
            "milkyway_child_observations": [{"pid": 102, "starttime_ticks": 1002, **milkyway}],
            "unexpected_tool_observations": [], "all_observed_processes": identities}


def rejected(module, payload: dict, isolated: Path) -> None:
    try:
        module.validate_process_tree(payload, isolated)
    except module.Failure:
        return
    raise AssertionError("mutation was accepted")


def main() -> int:
    module = load_checker()
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    for path in (MONITOR, CHECKER):
        compile(path.read_text(), str(path), "exec")

    assert sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    assert sha(M2171 / "review.json") == "c42ffe2cea367f6a0bb43c73279ec1c340fd20f37fac990a5876c8193b52ccb9"
    assert sha(M2171 / "SHA256SUMS") == "296c041dc4018eb1d651910f384d2dad4fccb1678c9a439551ea147c53c40ec8"
    assert sha(M2171 / "SHA256SUMS.seal.sha256") == "c11c35e0596e622b6b9b8b9289eac2956ea8df01768944c94414df9bfd3e569b"

    source = TCL.read_text()
    ordered = [
        "set_app_options -name lib.setting.milkyway_exec -value $milkyway_exec",
        "get_app_option_value -name lib.setting.milkyway_exec",
        "M2180_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS",
        "generate_frame_from_mw $frame_name",
    ]
    positions = [source.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert source.count("generate_frame_from_mw $frame_name") == 1
    forbidden_commands = ("create_lib", "read_verilog", "read_sverilog", "place_opt",
                          "clock_opt", "route_opt", "report_timing", "report_power")
    for command in forbidden_commands:
        assert not re.search(rf"(?m)^\s*{re.escape(command)}(?:\s|$)", source)
    assert MW in RUNNER.read_text()
    assert RUNNER.read_text().count('"${LM_SHELL}" -no_init -f "${TCL}"') == 1
    assert RUNNER.read_text().count('"${LMUTIL}" lmstat ') == 1
    assert "pnr_runs=0" in RUNNER.read_text()
    assert "'automatic_retry':False" in RUNNER.read_text()

    with tempfile.TemporaryDirectory(prefix="m2180_source_test_") as td:
        base = Path(td)
        native = base / "good.ndm"
        native.write_bytes(module.NATIVE_HEADER + b"M2180_NATIVE_PAYLOAD")
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
            "milkyway_identities": 1}
        mutations: list[dict] = []
        for field in ("root_seen", "unexpected_tool_observations"):
            item = copy.deepcopy(tree)
            item[field] = False if field == "root_seen" else [{"exe_path": "/x/icc2_exec"}]
            mutations.append(item)
        item = copy.deepcopy(tree); item["lm_actual_exec_observations"] = []; mutations.append(item)
        item = copy.deepcopy(tree); item["milkyway_child_observations"] = []; mutations.append(item)
        item = copy.deepcopy(tree); item["all_observed_processes"][1]["exec_observations"][0]["exe_path"] = "/tmp/fake_lm_shell_exec"; mutations.append(item)
        item = copy.deepcopy(tree); item["all_observed_processes"][2]["exec_observations"][0]["exe_path"] = "/tmp/fake_Milkyway"; mutations.append(item)
        item = copy.deepcopy(tree); item["all_observed_processes"][2]["parent_links"] = [{"ppid": 100, "parent_starttime_ticks": 1000}]; mutations.append(item)
        item = copy.deepcopy(tree); item["all_observed_processes"][1]["exec_observations"][0]["selected_environment"]["HOME"] = "/tmp/wrong"; mutations.append(item)
        item = copy.deepcopy(tree); item["unique_process_identity_count"] = 2; mutations.append(item)
        item = copy.deepcopy(tree); item["exec_observation_count"] = 4; mutations.append(item)
        item = copy.deepcopy(tree); item["all_observed_processes"][1]["parent_links"].append({"ppid": 102, "parent_starttime_ticks": 1002}); mutations.append(item)
        item = copy.deepcopy(tree); item["schema"] = "wrong"; mutations.append(item)
        for item in mutations:
            rejected(module, item, isolated)

    print("PASS_M2180_SOURCE_TESTS native_controls=1 native_mutations=1 process_controls=1 process_mutations=12 eda_runs=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
