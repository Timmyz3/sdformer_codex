#!/usr/bin/python3.12
"""CPU-only M2221 source tests; never invokes LM, lmutil, EDA, or GPU."""
from __future__ import annotations

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
TCL = HW / "dc_handoff/scripts/run_lm_m2221_command_option_discovery.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2223_m2222_m2221_lm_command_option_discovery_one_shot.sh"
CENSUS = HW / "dc_handoff/scripts/census_m2205_same_uid_tools.py"
INVENTORY = HW / "dc_handoff/scripts/inventory_m2153_repo_root.py"
CHECKER = HW / "system_simulator/scripts/check_m2223_lm_command_option_discovery.py"
CONTRACT = HW / "contracts/m2221_m2208_lm_command_option_discovery_source_contract_r1_20260904.json"
LM = "/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell"
ACTUAL = "/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec"
MW = "/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway"
RAW_PASS = "RAW_PASS_M2223_M2221_LM_COMMAND_OPTION_DISCOVERY_PENDING_M2224_RESULT_HAMMER"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hx(value: str) -> str:
    return value.encode().hex()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_checker():
    spec = importlib.util.spec_from_file_location("m2223_checker", CHECKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_full_work(module, base: Path) -> Path:
    work = base / "work"
    isolated = work / "isolated_cwd"
    for rel in ("home", "tmp", "cache/xdg", "cache/library", "frame_output",
                "frame_logs", "reports"):
        (isolated / rel).mkdir(parents=True, exist_ok=True)
    commands = {
        "generate_frame_from_mw": 1, "set_app_options": 1,
        "get_app_option_value": 1, "report_app_options": 1,
    }
    lines = [
        f"M2221_STARTUP mode=no_init setup_files=0 cwd_hex={hx(str(isolated))} "
        f"home_hex={hx(str(isolated / 'home'))}",
    ]
    lines.extend(f"M2221_COMMAND name={name} available={available}"
                 for name, available in commands.items())
    lines.extend([
        "M2221_OPTION name=lib.configuration.local_output_dir query_attempted=1 "
        f"query_rc=1 registered=0 value_hex= diagnostic_hex={hx('invalid option')}",
        "M2221_OPTION name=lib.setting.milkyway_exec query_attempted=1 query_rc=0 "
        f"registered=1 value_hex={hx('/old/Milkyway')} diagnostic_hex=",
        "M2221_MILKYWAY_SET attempted=1 set_rc=0 readback_attempted=1 readback_rc=0 "
        f"exact=1 value_hex={hx(MW)} set_diagnostic_hex= readback_diagnostic_hex=",
        "M2221_NO_SIDE_EFFECTS frame_files=0 ndm_files=0 nlib_files=0 "
        "generate_calls=0 create_lib_calls=0 pnr_calls=0",
        RAW_PASS,
    ])
    (work / "lm_discovery.log").write_text("\n".join(lines) + "\n")
    (work / "lm_discovery.rc").write_text("0\n")
    execution = {
        "schema": "m2223_m2221_lm_command_option_discovery_execution_contract_r1_v1",
        "scope": "lm_command_option_discovery_only", "startup_mode": "no_init",
        "license_queries": 1, "top_level_lm_shell_runs": 1,
        "generate_frame_commands": 0, "create_lib_commands": 0,
        "milkyway_process_runs": 0, "pnr_runs": 0, "automatic_retry": False,
        "lm_invocation": [LM, "-no_init", "-f", str(TCL)],
        "lm_shell_sha256": "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
        "lm_shell_exec_path": ACTUAL,
        "lm_shell_exec_sha256": "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
        "milkyway_exec_path": MW,
        "milkyway_exec_sha256": "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
        "isolated_root": str(isolated), "runner_path": str(RUNNER),
    }
    write_json(work / "execution_contract.json", execution)
    census = {"schema": "m2205_same_uid_tool_census_r1_v1", "uid": os.getuid(),
              "blocked_names": module.BLOCKED_NAMES, "matching_processes": [],
              "matching_process_count": 0, "status": "PASS_EMPTY"}
    for phase in ("before", "after"):
        write_json(work / f"same_uid_census_{phase}.json", {**census, "phase": phase})
    repo = {"schema": "synthetic_repo_inventory", "node_count": 0, "nodes": []}
    write_json(work / "repo_root_before.json", repo)
    write_json(work / "repo_root_after.json", repo)
    manifest = {
        "schema": "m2223_m2221_lm_command_option_discovery_output_manifest_r1_v1",
        "lm_return_code": 0, "license_queries": 1, "top_level_lm_shell_runs": 1,
        "generate_frame_commands": 0, "create_lib_commands": 0,
        "milkyway_process_runs": 0, "pnr_runs": 0, "automatic_retry": False,
        "execution_contract_sha256": sha(work / "execution_contract.json"),
        "lm_log_sha256": sha(work / "lm_discovery.log"),
        "same_uid_census_before_sha256": sha(work / "same_uid_census_before.json"),
        "same_uid_census_after_sha256": sha(work / "same_uid_census_after.json"),
        "repo_root_before_sha256": sha(work / "repo_root_before.json"),
        "repo_root_after_sha256": sha(work / "repo_root_after.json"),
    }
    write_json(work / "execution_output_manifest.json", manifest)
    return work


def rejected(module, work: Path, output: Path) -> None:
    try:
        module.validate(work, output)
    except module.Failure:
        return
    raise AssertionError("mutation accepted")


def main() -> int:
    module = load_checker()
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    for path in (CENSUS, INVENTORY, CHECKER):
        compile(path.read_text(), str(path), "exec")
    runner = RUNNER.read_text()
    tcl = TCL.read_text()
    contract = json.loads(CONTRACT.read_text())
    assert contract["status"] == "SOURCE_ONLY__M2222_REVIEW_REQUIRED__NO_LM_EDA_LICENSE_GPU"
    assert contract["execution_authority"]["direct_execution_authorized_now"] is False
    for relative, expected in contract["source_inventory"].items():
        path = ROOT / relative
        if path == Path(__file__).resolve():
            continue
        assert sha(path) == expected, relative
    assert runner.count('"${LM_SHELL}" -no_init -f "${TCL}"') == 1
    assert runner.count('"${LMUTIL}" lmstat ') == 1
    assert '"${CENSUS}" --phase before' in runner
    assert '"${CENSUS}" --phase after' in runner
    assert "monitor" not in runner.lower()
    assert "M2221_EXPECTED_SOURCE_REVIEW_SHA256" in runner
    assert not re.search(r"(?m)^\s*(generate_frame_from_mw|create_lib|open_lib|save_lib)\b", tcl)
    assert not re.search(r"(?m)^\s*(place_opt|clock_opt|route_opt|compile_fusion)\b", tcl)
    assert tcl.count("[list set_app_options -name $mw_name -value $milkyway_exec]") == 1
    assert tcl.count("[list get_app_option_value -name $mw_name]") == 1
    assert "exit 42" in tcl and "exit 0" in tcl
    assert "RAW_PASS_M2223_M2221_LM_COMMAND_OPTION_DISCOVERY_PENDING_M2224_RESULT_HAMMER" in tcl

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        work = make_full_work(module, base / "good")
        output = work / "receipt.json"
        result = module.validate(work, output)
        assert result["status"] == RAW_PASS
        assert result["commands"]["generate_frame_from_mw"] == 1
        assert result["options"]["lib.configuration.local_output_dir"]["registered"] == 0
        assert result["milkyway_exec_set_readback"]["exact"] == 1
        assert result["claim_boundary"]["library_conversion"] is False

        work = make_full_work(module, base / "frame")
        (work / "isolated_cwd/frame_output/forbidden.ndm").write_text("x")
        rejected(module, work, work / "receipt.json")

        work = make_full_work(module, base / "duplicate")
        log = work / "lm_discovery.log"
        log.write_text(log.read_text() + "M2221_COMMAND name=generate_frame_from_mw available=1\n")
        rejected(module, work, work / "receipt.json")

        work = make_full_work(module, base / "option")
        log = work / "lm_discovery.log"
        log.write_text(log.read_text().replace(
            "query_rc=1 registered=0 value_hex= diagnostic_hex=",
            "query_rc=0 registered=0 value_hex= diagnostic_hex=", 1))
        rejected(module, work, work / "receipt.json")

        work = make_full_work(module, base / "set")
        log = work / "lm_discovery.log"
        log.write_text(log.read_text().replace("exact=1 value_hex=", "exact=0 value_hex=", 1))
        rejected(module, work, work / "receipt.json")

        work = make_full_work(module, base / "execution")
        execution = json.loads((work / "execution_contract.json").read_text())
        execution["generate_frame_commands"] = 1
        write_json(work / "execution_contract.json", execution)
        rejected(module, work, work / "receipt.json")

        work = make_full_work(module, base / "inventory")
        repo = json.loads((work / "repo_root_after.json").read_text())
        repo["node_count"] = 1
        write_json(work / "repo_root_after.json", repo)
        rejected(module, work, work / "receipt.json")

        work = make_full_work(module, base / "manifest")
        manifest = json.loads((work / "execution_output_manifest.json").read_text())
        manifest["lm_return_code"] = 42
        write_json(work / "execution_output_manifest.json", manifest)
        rejected(module, work, work / "receipt.json")

    print("PASS_M2221_CPU_ONLY_SOURCE_AND_MUTATION_TESTS cases=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
