#!/usr/bin/python3.12
"""Fail-closed checker for the one raw M2223 no-conversion LM discovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
LM_SHELL = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell")
LM_EXEC = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec")
MILKYWAY = Path("/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway")
TCL = HW / "dc_handoff/scripts/run_lm_m2221_command_option_discovery.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2223_m2222_m2221_lm_command_option_discovery_one_shot.sh"
BLOCKED_NAMES = sorted({"vcs", "simv", "dc_shell", "pt_shell", "fm_shell", "icc2_shell",
                        "icc2_exec", "dgcom_exec", "lm_shell", "lm_shell_exec", "Milkyway",
                        "lmutil", "lmstat"})
COMMANDS = ("generate_frame_from_mw", "set_app_options", "get_app_option_value",
            "report_app_options")
OPTIONS = ("lib.configuration.local_output_dir", "lib.setting.milkyway_exec")
RAW_PASS = "RAW_PASS_M2223_M2221_LM_COMMAND_OPTION_DISCOVERY_PENDING_M2224_RESULT_HAMMER"


class Failure(RuntimeError):
    pass


def need(ok: bool, message: str) -> None:
    if not ok:
        raise Failure(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    need(path.is_file() and not path.is_symlink(), f"missing/symlink {path}")
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f"not object {path}")
    return value


def decode_hex(value: str, label: str) -> str:
    try:
        return bytes.fromhex(value).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise Failure(f"invalid {label} hex") from exc


def one_match(pattern: str, text: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    need(len(matches) == 1, f"{label} exact occurrence count {len(matches)}")
    return matches[0]


def validate_census(payload: dict, phase: str) -> None:
    need(payload == {"schema": "m2205_same_uid_tool_census_r1_v1", "phase": phase,
                     "uid": os.getuid(), "blocked_names": BLOCKED_NAMES,
                     "matching_processes": [], "matching_process_count": 0,
                     "status": "PASS_EMPTY"}, f"{phase} same-UID census")


def validate_option(record: dict[str, object], getter: int) -> None:
    attempted = record["query_attempted"]
    rc = record["query_rc"]
    registered = record["registered"]
    value = record["value"]
    diagnostic = record["diagnostic"]
    if getter == 0:
        need((attempted, rc, registered, value) == (0, -1, -1, "") and
             diagnostic == "get_app_option_value unavailable", "getter-absent option state")
    elif rc == 0:
        need((attempted, registered, diagnostic) == (1, 1, ""),
             "registered option state")
    else:
        need(attempted == 1 and rc > 0 and registered == 0 and value == "" and
             bool(diagnostic), "unregistered option state")


def validate(work: Path, output: Path) -> dict[str, object]:
    work = work.resolve(strict=True)
    need(work.is_dir() and not work.is_symlink() and not output.exists(), "work/output")
    isolated = work / "isolated_cwd"
    home = isolated / "home"
    frame = isolated / "frame_output"
    paths = {
        "log": work / "lm_discovery.log", "rc": work / "lm_discovery.rc",
        "execution": work / "execution_contract.json",
        "manifest": work / "execution_output_manifest.json",
        "before_census": work / "same_uid_census_before.json",
        "after_census": work / "same_uid_census_after.json",
        "repo_before": work / "repo_root_before.json",
        "repo_after": work / "repo_root_after.json",
    }
    for path in (isolated, home, frame, *paths.values()):
        need(path.exists() and not path.is_symlink(), f"missing/symlink {path}")
    need(paths["rc"].read_text().strip() == "0", "LM return code")
    need(not list(frame.iterdir()), "frame output directory not empty")
    need(not list(isolated.rglob("*.ndm")) and not list(isolated.rglob("*.nlib")),
         "forbidden NDM/NLIB side effect")

    text = paths["log"].read_text(errors="replace")
    need("M2221_FATAL_FAIL_CLOSED:" not in text, "Tcl fatal diagnostic")
    startup = one_match(
        r"^M2221_STARTUP mode=no_init setup_files=0 cwd_hex=([0-9a-f]*) home_hex=([0-9a-f]*)$",
        text, "startup")
    need(decode_hex(startup.group(1), "cwd") == str(isolated) and
         decode_hex(startup.group(2), "home") == str(home), "startup isolation identity")

    command_matches = list(re.finditer(r"^M2221_COMMAND name=([^ ]+) available=([01])$",
                                       text, re.MULTILINE))
    need(len(command_matches) == len(COMMANDS), "command marker count")
    commands: dict[str, int] = {}
    for match in command_matches:
        name, available = match.group(1), int(match.group(2))
        need(name in COMMANDS and name not in commands, "command identity/duplicate")
        commands[name] = available
    need(tuple(commands) == COMMANDS, "command marker order")

    option_matches = list(re.finditer(
        r"^M2221_OPTION name=([^ ]+) query_attempted=([01]) query_rc=(-?\d+) "
        r"registered=(-?\d+) value_hex=([0-9a-f]*) diagnostic_hex=([0-9a-f]*)$",
        text, re.MULTILINE))
    need(len(option_matches) == len(OPTIONS), "option marker count")
    options: dict[str, dict[str, object]] = {}
    for match in option_matches:
        name = match.group(1)
        need(name in OPTIONS and name not in options, "option identity/duplicate")
        options[name] = {
            "query_attempted": int(match.group(2)), "query_rc": int(match.group(3)),
            "registered": int(match.group(4)),
            "value": decode_hex(match.group(5), f"{name} value"),
            "diagnostic": decode_hex(match.group(6), f"{name} diagnostic"),
        }
    need(tuple(options) == OPTIONS, "option marker order")
    for record in options.values():
        validate_option(record, commands["get_app_option_value"])

    set_match = one_match(
        r"^M2221_MILKYWAY_SET attempted=([01]) set_rc=(-?\d+) "
        r"readback_attempted=([01]) readback_rc=(-?\d+) exact=(-?\d+) "
        r"value_hex=([0-9a-f]*) set_diagnostic_hex=([0-9a-f]*) "
        r"readback_diagnostic_hex=([0-9a-f]*)$", text, "Milkyway set/readback")
    set_record = {
        "attempted": int(set_match.group(1)), "set_rc": int(set_match.group(2)),
        "readback_attempted": int(set_match.group(3)),
        "readback_rc": int(set_match.group(4)), "exact": int(set_match.group(5)),
        "value": decode_hex(set_match.group(6), "set value"),
        "set_diagnostic": decode_hex(set_match.group(7), "set diagnostic"),
        "readback_diagnostic": decode_hex(set_match.group(8), "readback diagnostic"),
    }
    mw_registered = options[OPTIONS[1]]["registered"] == 1
    should_attempt = bool(commands["set_app_options"] and
                          commands["get_app_option_value"] and mw_registered)
    need(bool(set_record["attempted"]) == should_attempt, "set attempt gate")
    if not should_attempt:
        need((set_record["set_rc"], set_record["readback_attempted"],
              set_record["readback_rc"], set_record["exact"], set_record["value"],
              set_record["set_diagnostic"], set_record["readback_diagnostic"]) ==
             (-1, 0, -1, -1, "", "", ""), "set not-attempted state")
    elif set_record["set_rc"] != 0:
        need(set_record["set_rc"] > 0 and set_record["readback_attempted"] == 0 and
             set_record["readback_rc"] == -1 and set_record["exact"] == -1 and
             set_record["value"] == "" and bool(set_record["set_diagnostic"]) and
             set_record["readback_diagnostic"] == "", "set failure state")
    elif set_record["readback_rc"] == 0:
        need(set_record["readback_attempted"] == 1 and set_record["exact"] in (0, 1) and
             set_record["set_diagnostic"] == "" and
             set_record["readback_diagnostic"] == "", "readback success state")
        need((set_record["value"] == str(MILKYWAY)) == bool(set_record["exact"]),
             "readback exact flag")
    else:
        need(set_record["readback_attempted"] == 1 and set_record["readback_rc"] > 0 and
             set_record["exact"] == 0 and set_record["value"] == "" and
             set_record["set_diagnostic"] == "" and
             bool(set_record["readback_diagnostic"]), "readback failure state")

    no_side = one_match(
        r"^M2221_NO_SIDE_EFFECTS frame_files=0 ndm_files=0 nlib_files=0 "
        r"generate_calls=0 create_lib_calls=0 pnr_calls=0$", text, "no-side-effects")
    raw = one_match(rf"^{re.escape(RAW_PASS)}$", text, "raw pass")
    need(startup.start() < min(match.start() for match in command_matches) <
         min(match.start() for match in option_matches) < set_match.start() <
         no_side.start() < raw.start(), "discovery marker order")

    execution = read_json(paths["execution"])
    expected_execution = {
        "schema": "m2223_m2221_lm_command_option_discovery_execution_contract_r1_v1",
        "scope": "lm_command_option_discovery_only", "startup_mode": "no_init",
        "license_queries": 1, "top_level_lm_shell_runs": 1,
        "generate_frame_commands": 0, "create_lib_commands": 0,
        "milkyway_process_runs": 0, "pnr_runs": 0, "automatic_retry": False,
        "lm_invocation": [str(LM_SHELL), "-no_init", "-f", str(TCL)],
        "lm_shell_sha256": "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
        "lm_shell_exec_path": str(LM_EXEC),
        "lm_shell_exec_sha256": "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
        "milkyway_exec_path": str(MILKYWAY),
        "milkyway_exec_sha256": "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
        "isolated_root": str(isolated), "runner_path": str(RUNNER),
    }
    need(execution == expected_execution, "execution contract")
    validate_census(read_json(paths["before_census"]), "before")
    validate_census(read_json(paths["after_census"]), "after")
    need(read_json(paths["repo_before"]) == read_json(paths["repo_after"]),
         "repository root inventory drift")
    manifest = read_json(paths["manifest"])
    need(manifest == {
        "schema": "m2223_m2221_lm_command_option_discovery_output_manifest_r1_v1",
        "lm_return_code": 0, "license_queries": 1, "top_level_lm_shell_runs": 1,
        "generate_frame_commands": 0, "create_lib_commands": 0,
        "milkyway_process_runs": 0, "pnr_runs": 0, "automatic_retry": False,
        "execution_contract_sha256": sha(paths["execution"]),
        "lm_log_sha256": sha(paths["log"]),
        "same_uid_census_before_sha256": sha(paths["before_census"]),
        "same_uid_census_after_sha256": sha(paths["after_census"]),
        "repo_root_before_sha256": sha(paths["repo_before"]),
        "repo_root_after_sha256": sha(paths["repo_after"]),
    }, "execution/output manifest")

    result = {
        "schema": "m2223_m2221_lm_command_option_discovery_result_r1_v1",
        "status": RAW_PASS, "startup_mode": "no_init", "commands": commands,
        "options": options, "milkyway_exec_set_readback": set_record,
        "execution_manifest_sha256": sha(paths["manifest"]),
        "claim_boundary": {
            "command_option_discovery_only": True, "library_conversion": False,
            "design_library": False, "ndm_written": False, "pnr": False,
            "timing": False, "area": False, "power": False,
            "paper_ppa_ready": False, "execution_admitted": False,
            "pending_independent_m2224_result_hammer": True,
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = validate(args.work, args.output)
    print(value["status"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2223_CHECK_FAIL_CLOSED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
