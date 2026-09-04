#!/usr/bin/python3.12
"""CPU-only M2205 source tests; never invokes LM, lmutil, EDA, or GPU."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
TCL = HW / "dc_handoff/scripts/run_lm_m2205_library_conversion_preflight.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2205_m2190_lm_library_conversion_preflight_one_shot.sh"
MONITOR = HW / "dc_handoff/scripts/monitor_m2205_lm_conversion_sampled_processes.py"
CENSUS = HW / "dc_handoff/scripts/census_m2205_same_uid_tools.py"
CHECKER = HW / "system_simulator/scripts/check_m2205_lm_library_conversion_preflight.py"
LM = "/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell"
ACTUAL = "/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec"
MW = "/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway"
GATE_TOKEN = "M2205_MONITOR_RELEASE_ACTUAL_STABLE\n"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checker():
    spec = importlib.util.spec_from_file_location("m2205_checker", CHECKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(isolated: Path) -> dict[str, str]:
    return {"HOME": str(isolated / "home"), "TMPDIR": str(isolated / "tmp"),
            "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
            "M2205_ISOLATED_CWD": str(isolated)}


def observation(exe: str, cmdline: list[str], isolated: Path, phase: str,
                env: bool = True) -> dict:
    return {"comm": Path(exe).name, "exe_path": exe, "cmdline": cmdline,
            "selected_environment": environment(isolated) if env else {}, "phase": phase}


def identity(pid: int, ticks: int, parent: int, parent_ticks: int | None,
             observations: list[dict]) -> dict:
    return {"pid": pid, "starttime_ticks": ticks, "first_ppid": parent,
            "parent_links": [{"ppid": parent, "parent_starttime_ticks": parent_ticks}],
            "exec_observations": observations}


def good_process(isolated: Path) -> dict:
    root = identity(100, 1000, 1, None, [observation(
        "/usr/bin/dash", [LM, "-no_init", "-f", str(TCL)], isolated,
        "bootstrap_pre_gate", env=False)])
    helper = identity(103, 1001, 100, 1000, [observation(
        "/usr/bin/dirname", ["dirname", LM], isolated, "bootstrap_pre_gate", env=False)])
    actual_pre = observation(ACTUAL, [ACTUAL, "-root_path", "/opt/synopsys/icc2/V-2023.12-SP3",
                                      "-no_init", "-f", str(TCL)], isolated,
                             "bootstrap_pre_gate")
    actual_post = copy.deepcopy(actual_pre)
    actual_post["phase"] = "post_gate"
    actual_all = identity(101, 1002, 100, 1000, [actual_pre, actual_post])
    actual_post_identity = identity(101, 1002, 100, 1000, [actual_post])
    mw_post = observation(MW, [MW, "-lm_mode"], isolated, "post_gate")
    mw_identity = identity(102, 1003, 101, 1002, [mw_post])
    return {
        "schema": "m2205_lm_conversion_sampled_process_contract_r1_v1",
        "status": "PASS_M2205_SAMPLED_POST_GATE_PROCESS_CONTRACT",
        "claim_scope": {
            "sampled_live_processes_only": True,
            "exhaustive_short_lived_processes": False,
            "sampling_interval_seconds": 0.005,
            "bootstrap_helpers_permitted_before_gate": True,
            "post_gate_actual_subtree_allowlist": [ACTUAL, MW],
        },
        "root_pid": 100, "root_starttime_ticks": 1000, "root_seen": True,
        "sample_count": 40,
        "gate": {"released": True, "created_by_monitor": True,
                 "token": GATE_TOKEN.rstrip("\n"), "tcl_wait_marker_seen": True,
                 "actual_stable_samples_required": 5,
                 "actual_stable_samples_observed": 7,
                 "frame_absent_before_release": True,
                 "release_monotonic_ns": 123456789},
        "actual_identity": {"pid": 101, "starttime_ticks": 1002, "exe_path": ACTUAL},
        "sampled_actual_identity_count": 1,
        "sampled_milkyway_identity_count": 1,
        "pre_gate_milkyway_observations": [],
        "post_gate_sample_count": 20,
        "unexpected_sampled_post_gate_descendants": [],
        "all_sampled_processes": [root, helper, actual_all, mw_identity],
        "post_gate_actual_subtree_processes": [actual_post_identity, mw_identity],
        "violation": "",
    }


def rejected(module, payload: dict, isolated: Path) -> None:
    try:
        module.validate_sampled_process(payload, isolated)
    except module.Failure:
        return
    raise AssertionError("process mutation accepted")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_full_work(module, base: Path) -> tuple[Path, Path]:
    work = base / "work"
    isolated = work / "isolated_cwd"
    for rel in ("home", "tmp", "cache/xdg", "cache/library", "frame_output",
                "frame_logs", "reports"):
        (isolated / rel).mkdir(parents=True, exist_ok=True)
    frame = isolated / "frame_output/m2205_tcbn28hpcplusbwp35p140_frame.ndm"
    frame.write_bytes(module.NATIVE_HEADER + b"M2205_NATIVE_PAYLOAD")
    frame_stats = module.validate_native_frame(frame)
    gate = work / "conversion.release.gate"
    gate.write_text(GATE_TOKEN)
    process = good_process(isolated)
    process["actual_identity"]["pid"] = 101
    write_json(work / "sampled_processes.json", process)
    log_lines = [
        f"M2205_GATE0_TCL_WAITING actual_pid=101 gate={gate}",
        f"M2205_GATE0_TCL_RELEASED actual_pid=101 gate={gate}",
        f"M2205_GATE1_LOCAL_OUTPUT_ROUND_TRIP_PASS cache={isolated / 'cache/library'}",
        f"M2205_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS exec={MW}",
        f"M2205_GATE3_FRAME_CONVERSION_PASS status=1 frame={frame}",
        f"M2205_GATE4_NONEMPTY_FRAME_PASS files=1 bytes={frame_stats['regular_bytes']}",
        "RAW_PASS_M2207_M2205_LM_LIBRARY_CONVERSION_PENDING_M2208_INDEPENDENT_RESULT_HAMMER",
    ]
    (work / "lm_preflight.log").write_text("\n".join(log_lines) + "\n")
    (work / "lm_preflight.rc").write_text("0\n")
    facts = [
        "status=RAW_PASS_M2207_M2205_LM_LIBRARY_CONVERSION_PENDING_M2208",
        "shell=lm_shell", "sampled_process_claim=true",
        "exhaustive_short_lived_process_claim=false", f"conversion_gate={gate}",
        f"local_output_dir={isolated / 'cache/library'}", f"milkyway_exec={MW}",
        "conversion_status=1", f"frame_ndm={frame}", "frame_regular_files=1",
        f"frame_regular_bytes={frame_stats['regular_bytes']}",
        "design_library_created=false", "rtl_imported=false", "pnr_invoked=false",
    ]
    (isolated / "reports/machine_facts.txt").write_text("\n".join(facts) + "\n")
    execution = {
        "schema": "m2207_m2205_lm_execution_contract_r1_v1",
        "scope": "lm_library_conversion_only", "license_queries": 1,
        "top_level_lm_shell_runs": 1, "generate_frame_commands": 1,
        "sampled_milkyway_identities": 1, "pnr_runs": 0, "automatic_retry": False,
        "process_claim": "sampled_live_processes_only__not_exhaustive_for_short_lived_helpers",
        "lm_invocation": [LM, "-no_init", "-f", str(TCL)],
        "lm_shell_sha256": "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
        "lm_shell_exec_path": ACTUAL,
        "lm_shell_exec_sha256": "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
        "milkyway_exec_path": MW,
        "milkyway_exec_sha256": "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
        "isolated_root": str(isolated), "runner_path": str(RUNNER),
    }
    write_json(work / "execution_contract.json", execution)
    census_common = {"schema": "m2205_same_uid_tool_census_r1_v1",
                     "uid": os.getuid(), "blocked_names": module.BLOCKED_NAMES,
                     "matching_processes": [], "matching_process_count": 0,
                     "status": "PASS_EMPTY"}
    for phase in ("before", "after"):
        write_json(work / f"same_uid_census_{phase}.json", {**census_common, "phase": phase})
    repo = {"schema": "synthetic_repo_inventory", "node_count": 0, "nodes": []}
    write_json(work / "repo_root_before.json", repo)
    write_json(work / "repo_root_after.json", repo)
    manifest = {
        "schema": "m2207_m2205_execution_output_manifest_r1_v1",
        "lm_return_code": 0, "top_level_lm_shell_runs": 1,
        "generate_frame_commands": 1, "automatic_retry": False, "pnr_runs": 0,
        "process_claim": "sampled_live_processes_only__not_exhaustive_for_short_lived_helpers",
        "execution_contract_sha256": sha(work / "execution_contract.json"),
        "lm_log_sha256": sha(work / "lm_preflight.log"),
        "sampled_processes_sha256": sha(work / "sampled_processes.json"),
        "same_uid_census_before_sha256": sha(work / "same_uid_census_before.json"),
        "same_uid_census_after_sha256": sha(work / "same_uid_census_after.json"),
        "frame": {"path": str(frame), **frame_stats},
        "exact_ordered_log_markers": log_lines,
    }
    write_json(work / "execution_output_manifest.json", manifest)
    return work, frame


def full_rejected(module, work: Path, output: Path) -> None:
    try:
        module.validate(work, output)
    except module.Failure:
        return
    raise AssertionError("full receipt mutation accepted")


def main() -> int:
    module = load_checker()
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    for path in (MONITOR, CENSUS, CHECKER):
        compile(path.read_text(), str(path), "exec")
    monitor_text = MONITOR.read_text()
    runner_text = RUNNER.read_text()
    assert "time.sleep(0.005)" in monitor_text
    assert not re.search(r"\b(subprocess|Popen|os\.system)\b", monitor_text)
    assert "/usr/bin/sleep" not in monitor_text and "/usr/bin/sleep" not in runner_text
    assert runner_text.count('"${LM_SHELL}" -no_init -f "${TCL}"') == 1
    assert runner_text.count('"${LMUTIL}" lmstat ') == 1
    assert runner_text.count('"${CENSUS}" --phase before') == 1
    assert runner_text.count('"${CENSUS}" --phase after') == 1
    assert "M2182_PERMANENTLY_UNAUTHORIZED=1" in runner_text
    assert "M2191_PERMANENTLY_UNAUTHORIZED=1" in runner_text

    tcl = TCL.read_text()
    wait_at = tcl.index("M2205_GATE0_TCL_WAITING")
    release_at = tcl.index("M2205_GATE0_TCL_RELEASED")
    proc_at = tcl.index("proc m2205_env")
    option_at = tcl.index("set_app_options -name lib.setting.milkyway_exec")
    generate_at = tcl.index("generate_frame_from_mw $frame_name")
    assert wait_at < release_at < proc_at < option_at < generate_at
    assert tcl.count("generate_frame_from_mw $frame_name") == 1
    assert GATE_TOKEN.strip() in tcl and "after 10" in tcl
    for command in ("create_lib", "read_verilog", "read_sverilog", "place_opt",
                    "clock_opt", "route_opt", "report_timing", "report_power"):
        assert not re.search(rf"(?m)^\s*{re.escape(command)}(?:\s|$)", tcl)

    with tempfile.TemporaryDirectory(prefix="m2205_source_test_") as td:
        base = Path(td)
        native = base / "good.ndm"
        native.write_bytes(module.NATIVE_HEADER + b"M2205_NATIVE_PAYLOAD")
        assert module.validate_native_frame(native)["regular_files"] == 1
        bad = base / "bad.ndm"
        bad.write_bytes(b"not-a-native-frame")
        try:
            module.validate_native_frame(bad)
        except module.Failure:
            pass
        else:
            raise AssertionError("native frame mutation accepted")

        isolated = base / "isolated"
        good = good_process(isolated)
        assert module.validate_sampled_process(good, isolated)["actual_identities"] == 1
        mutations: list[dict] = []
        for path, value in [
            (("schema",), "wrong"), (("status",), "FAIL"),
            (("claim_scope", "sampled_live_processes_only"), False),
            (("claim_scope", "exhaustive_short_lived_processes"), True),
            (("gate", "released"), False), (("gate", "created_by_monitor"), False),
            (("gate", "token"), "bad"), (("gate", "tcl_wait_marker_seen"), False),
            (("gate", "actual_stable_samples_observed"), 2),
            (("gate", "frame_absent_before_release"), False),
            (("gate", "release_monotonic_ns"), 0), (("violation",), "extra child"),
            (("post_gate_sample_count",), 0), (("sampled_actual_identity_count",), 2),
            (("sampled_milkyway_identity_count",), 2),
            (("actual_identity", "exe_path"), "/tmp/fake_lm"),
        ]:
            item = copy.deepcopy(good)
            target = item
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            mutations.append(item)
        item = copy.deepcopy(good)
        item["pre_gate_milkyway_observations"] = [{"exe_path": MW}]
        mutations.append(item)
        item = copy.deepcopy(good)
        item["unexpected_sampled_post_gate_descendants"] = [{"exe_path": "/usr/bin/sleep"}]
        mutations.append(item)
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][0]["exec_observations"][0]["exe_path"] = "/tmp/fake_lm"
        mutations.append(item)
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][1]["exec_observations"][0]["exe_path"] = "/tmp/fake_mw"
        mutations.append(item)
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][0]["exec_observations"][0]["selected_environment"]["HOME"] = "/tmp/wrong"
        mutations.append(item)
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][1]["exec_observations"][0]["selected_environment"]["HOME"] = "/tmp/wrong"
        mutations.append(item)
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][1]["parent_links"] = [
            {"ppid": 100, "parent_starttime_ticks": 1000}]
        mutations.append(item)
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][0]["exec_observations"][0]["cmdline"] = [ACTUAL, "-no_init"]
        mutations.append(item)
        item = copy.deepcopy(good)
        extra = identity(104, 1004, 101, 1002, [observation(
            "/usr/bin/sleep", ["/usr/bin/sleep", "5"], isolated, "post_gate", env=False)])
        item["post_gate_actual_subtree_processes"].append(extra)
        mutations.append(item)
        for mutation in mutations:
            rejected(module, mutation, isolated)

        work, _ = make_full_work(module, base / "full")
        receipt = work / "receipt.json"
        result = module.validate(work, receipt)
        assert result["status"].startswith("RAW_PASS_M2207")
        full_mutations = 0
        log = work / "lm_preflight.log"
        original = log.read_bytes(); log.write_bytes(original + b"unexpected\n")
        full_rejected(module, work, work / "reject_log.json"); log.write_bytes(original)
        full_mutations += 1
        before = work / "same_uid_census_before.json"
        original = before.read_bytes(); changed = json.loads(original)
        changed["matching_process_count"] = 1; write_json(before, changed)
        full_rejected(module, work, work / "reject_census.json"); before.write_bytes(original)
        full_mutations += 1
        frame = work / "isolated_cwd/frame_output/m2205_tcbn28hpcplusbwp35p140_frame.ndm"
        original = frame.read_bytes(); frame.write_bytes(original + b"drift")
        full_rejected(module, work, work / "reject_frame.json"); frame.write_bytes(original)
        full_mutations += 1
        process_path = work / "sampled_processes.json"
        original = process_path.read_bytes(); changed = json.loads(original)
        changed["claim_scope"]["exhaustive_short_lived_processes"] = True
        write_json(process_path, changed)
        full_rejected(module, work, work / "reject_process.json"); process_path.write_bytes(original)
        full_mutations += 1
        execution = work / "execution_contract.json"
        original = execution.read_bytes(); changed = json.loads(original)
        changed["top_level_lm_shell_runs"] = 2; write_json(execution, changed)
        full_rejected(module, work, work / "reject_execution.json"); execution.write_bytes(original)
        full_mutations += 1

    print("PASS_M2205_SOURCE_TESTS native_controls=1 native_mutations=1 "
          f"process_controls=1 process_mutations={len(mutations)} full_receipt_controls=1 "
          f"full_receipt_mutations={full_mutations} "
          "lm_runs=0 eda_runs=0 license_queries=0 gpu_runs=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
