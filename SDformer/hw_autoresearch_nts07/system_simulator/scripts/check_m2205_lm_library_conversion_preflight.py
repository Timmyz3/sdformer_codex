#!/usr/bin/python3.12
"""Fail-closed checker for the one raw M2207 LM-only conversion preflight."""
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
TCL = HW / "dc_handoff/scripts/run_lm_m2205_library_conversion_preflight.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2205_m2190_lm_library_conversion_preflight_one_shot.sh"
GATE_TOKEN = "M2205_MONITOR_RELEASE_ACTUAL_STABLE\n"
BLOCKED_NAMES = sorted({"vcs", "simv", "dc_shell", "pt_shell", "fm_shell", "icc2_shell",
                        "icc2_exec", "dgcom_exec", "lm_shell", "lm_shell_exec", "Milkyway",
                        "lmutil", "lmstat"})
NATIVE_HEADER = bytes.fromhex(
    "b2bdea03be02010000104c696272617279204d616e61676572002a562d323032332e31322d53503320666f72206c696e75783634202d2d204d61792030372c2032303234")


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


def parse_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        need(bool(sep) and key not in result, f"invalid/duplicate fact {line}")
        result[key] = value
    return result


def validate_native_frame(path: Path) -> dict[str, object]:
    need(path.is_file() and not path.is_symlink() and path.suffix == ".ndm",
         "frame is not one regular nonsymlink .ndm")
    blob = path.read_bytes()
    need(len(blob) > len(NATIVE_HEADER) and blob[:len(NATIVE_HEADER)] == NATIVE_HEADER,
         "frame native Library Manager header mismatch")
    return {"regular_files": 1, "regular_bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest()}


def validate_census(payload: dict, phase: str) -> None:
    need(payload == {"schema": "m2205_same_uid_tool_census_r1_v1", "phase": phase,
                     "uid": os.getuid(), "blocked_names": BLOCKED_NAMES,
                     "matching_processes": [], "matching_process_count": 0,
                     "status": "PASS_EMPTY"}, f"{phase} same-UID census")


def flatten(identities: list[dict]) -> tuple[dict[tuple[int, int], dict], list[dict]]:
    by_key: dict[tuple[int, int], dict] = {}
    flat: list[dict] = []
    for identity in identities:
        need(isinstance(identity, dict), "identity object")
        key = (identity.get("pid"), identity.get("starttime_ticks"))
        need(all(isinstance(item, int) and item > 0 for item in key) and key not in by_key,
             "identity key")
        need(isinstance(identity.get("parent_links"), list) and identity["parent_links"],
             "parent links")
        observations = identity.get("exec_observations")
        need(isinstance(observations, list) and observations, "exec observations")
        by_key[key] = identity
        for observation in observations:
            need(isinstance(observation, dict) and observation.get("phase") in
                 {"bootstrap_pre_gate", "post_gate"} and
                 isinstance(observation.get("exe_path"), str) and
                 isinstance(observation.get("cmdline"), list) and
                 all(isinstance(item, str) for item in observation["cmdline"]) and
                 isinstance(observation.get("selected_environment"), dict), "observation")
            flat.append({"pid": key[0], "starttime_ticks": key[1], **observation})
    return by_key, flat


def validate_sampled_process(payload: dict, isolated: Path) -> dict[str, int]:
    need(payload.get("schema") == "m2205_lm_conversion_sampled_process_contract_r1_v1",
         "sampled process schema")
    need(payload.get("status") == "PASS_M2205_SAMPLED_POST_GATE_PROCESS_CONTRACT",
         "sampled process status")
    expected_scope = {
        "sampled_live_processes_only": True,
        "exhaustive_short_lived_processes": False,
        "sampling_interval_seconds": 0.005,
        "bootstrap_helpers_permitted_before_gate": True,
        "post_gate_actual_subtree_allowlist": [str(LM_EXEC), str(MILKYWAY)],
    }
    need(payload.get("claim_scope") == expected_scope, "sampled process claim scope")
    need(payload.get("root_seen") is True and isinstance(payload.get("root_pid"), int) and
         isinstance(payload.get("root_starttime_ticks"), int), "root identity")
    gate = payload.get("gate")
    need(isinstance(gate, dict) and gate.get("released") is True and
         gate.get("created_by_monitor") is True and
         gate.get("token") == GATE_TOKEN.rstrip("\n") and
         gate.get("tcl_wait_marker_seen") is True and
         gate.get("actual_stable_samples_required") >= 3 and
         gate.get("actual_stable_samples_observed") >= gate["actual_stable_samples_required"] and
         gate.get("frame_absent_before_release") is True and
         isinstance(gate.get("release_monotonic_ns"), int) and gate["release_monotonic_ns"] > 0,
         "conversion gate evidence")
    need(payload.get("violation") == "" and
         payload.get("pre_gate_milkyway_observations") == [] and
         payload.get("unexpected_sampled_post_gate_descendants") == [] and
         isinstance(payload.get("post_gate_sample_count"), int) and
         payload["post_gate_sample_count"] > 0, "post-gate violation/census")
    bootstrap = payload.get("all_sampled_processes")
    post = payload.get("post_gate_actual_subtree_processes")
    need(isinstance(bootstrap, list) and bootstrap and isinstance(post, list) and post,
         "sampled identity lists")
    _, bootstrap_flat = flatten(bootstrap)
    post_by_key, post_flat = flatten(post)
    root_key = (payload["root_pid"], payload["root_starttime_ticks"])
    need(any((row["pid"], row["starttime_ticks"]) == root_key for row in bootstrap_flat),
         "root not sampled")
    actual = payload.get("actual_identity")
    need(isinstance(actual, dict) and actual.get("exe_path") == str(LM_EXEC), "actual identity")
    actual_key = (actual.get("pid"), actual.get("starttime_ticks"))
    need(actual_key in post_by_key and payload.get("sampled_actual_identity_count") == 1,
         "actual count")
    actual_rows = [row for row in post_flat if
                   (row["pid"], row["starttime_ticks"]) == actual_key]
    milkyway_rows = [row for row in post_flat if row["exe_path"] == str(MILKYWAY)]
    milkyway_keys = {(row["pid"], row["starttime_ticks"]) for row in milkyway_rows}
    need(actual_rows and all(row["exe_path"] == str(LM_EXEC) for row in actual_rows) and
         len(milkyway_keys) == 1 and payload.get("sampled_milkyway_identity_count") == 1,
         "actual/Milkyway sampled identity counts")
    mw_key = next(iter(milkyway_keys))
    need(set(post_by_key) == {actual_key, mw_key}, "post-gate sampled allowlist")
    expected_env = {"HOME": str(isolated / "home"), "TMPDIR": str(isolated / "tmp"),
                    "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
                    "M2205_ISOLATED_CWD": str(isolated)}
    for row in actual_rows + milkyway_rows:
        need(row["selected_environment"] == expected_env, "tool isolation environment")
    for row in actual_rows:
        need({"-no_init", "-f", str(TCL)}.issubset(set(row["cmdline"])),
             "actual command identity")
    parents: dict[tuple[int, int], set[tuple[int, int]]] = {key: set() for key in post_by_key}
    for key, identity in post_by_key.items():
        for link in identity["parent_links"]:
            parent = (link.get("ppid"), link.get("parent_starttime_ticks"))
            if parent in post_by_key:
                parents[key].add(parent)
    need(actual_key in parents[mw_key], "Milkyway not sampled directly below actual")
    return {"post_gate_sample_count": payload["post_gate_sample_count"],
            "actual_identities": 1, "milkyway_identities": 1,
            "unexpected_sampled_descendants": 0,
            "exhaustive_short_lived_process_claim": 0}


def validate(work: Path, output: Path) -> dict[str, object]:
    work = work.resolve(strict=True)
    need(work.is_dir() and not work.is_symlink() and not output.exists(), "work/output")
    isolated = work / "isolated_cwd"
    frame = isolated / "frame_output/m2205_tcbn28hpcplusbwp35p140_frame.ndm"
    facts_path = isolated / "reports/machine_facts.txt"
    gate_path = work / "conversion.release.gate"
    paths = {
        "log": work / "lm_preflight.log", "rc": work / "lm_preflight.rc",
        "process": work / "sampled_processes.json",
        "execution": work / "execution_contract.json",
        "manifest": work / "execution_output_manifest.json",
        "before_census": work / "same_uid_census_before.json",
        "after_census": work / "same_uid_census_after.json",
        "repo_before": work / "repo_root_before.json",
        "repo_after": work / "repo_root_after.json",
    }
    for path in (frame, facts_path, gate_path, *paths.values()):
        need(path.exists() and not path.is_symlink(), f"missing/symlink {path}")
    need(gate_path.read_text() == GATE_TOKEN, "conversion gate token")
    need(paths["rc"].read_text().strip() == "0", "LM return code")
    frame_stats = validate_native_frame(frame)
    need(not list(isolated.rglob("*.nlib")), "design .nlib unexpectedly created")
    need([path for path in isolated.rglob("*.ndm") if path != frame] == [], "extra NDM output")
    process_payload = read_json(paths["process"])
    process = validate_sampled_process(process_payload, isolated)
    actual_pid = process_payload["actual_identity"]["pid"]
    text = paths["log"].read_text(errors="replace")
    exact_lines = [
        f"M2205_GATE0_TCL_WAITING actual_pid={actual_pid} gate={gate_path}",
        f"M2205_GATE0_TCL_RELEASED actual_pid={actual_pid} gate={gate_path}",
        f"M2205_GATE1_LOCAL_OUTPUT_ROUND_TRIP_PASS cache={isolated / 'cache/library'}",
        f"M2205_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS exec={MILKYWAY}",
        f"M2205_GATE3_FRAME_CONVERSION_PASS status=1 frame={frame}",
        f"M2205_GATE4_NONEMPTY_FRAME_PASS files=1 bytes={frame_stats['regular_bytes']}",
        "RAW_PASS_M2207_M2205_LM_LIBRARY_CONVERSION_PENDING_M2208_INDEPENDENT_RESULT_HAMMER",
    ]
    positions = []
    for line in exact_lines:
        matches = list(re.finditer(rf"^{re.escape(line)}$", text, re.MULTILINE))
        need(len(matches) == 1, f"exact log line {line}")
        positions.append(matches[0].start())
    need(positions == sorted(positions), "log gate/conversion order")
    need("M2205_FATAL_FAIL_CLOSED:" not in text and "CMD-005" not in text,
         "LM fatal diagnostic")
    facts = parse_kv(facts_path)
    expected_facts = {
        "status": "RAW_PASS_M2207_M2205_LM_LIBRARY_CONVERSION_PENDING_M2208",
        "shell": "lm_shell", "sampled_process_claim": "true",
        "exhaustive_short_lived_process_claim": "false",
        "conversion_gate": str(gate_path),
        "local_output_dir": str(isolated / "cache/library"),
        "milkyway_exec": str(MILKYWAY), "conversion_status": "1",
        "frame_ndm": str(frame), "frame_regular_files": "1",
        "frame_regular_bytes": str(frame_stats["regular_bytes"]),
        "design_library_created": "false", "rtl_imported": "false", "pnr_invoked": "false",
    }
    need(facts == expected_facts, "machine facts")
    execution = read_json(paths["execution"])
    expected_execution = {
        "schema": "m2207_m2205_lm_execution_contract_r1_v1",
        "scope": "lm_library_conversion_only", "license_queries": 1,
        "top_level_lm_shell_runs": 1, "generate_frame_commands": 1,
        "sampled_milkyway_identities": 1, "pnr_runs": 0, "automatic_retry": False,
        "process_claim": "sampled_live_processes_only__not_exhaustive_for_short_lived_helpers",
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
    expected_manifest = {
        "schema": "m2207_m2205_execution_output_manifest_r1_v1",
        "lm_return_code": 0, "top_level_lm_shell_runs": 1,
        "generate_frame_commands": 1, "automatic_retry": False, "pnr_runs": 0,
        "process_claim": "sampled_live_processes_only__not_exhaustive_for_short_lived_helpers",
        "execution_contract_sha256": sha(paths["execution"]),
        "lm_log_sha256": sha(paths["log"]),
        "sampled_processes_sha256": sha(paths["process"]),
        "same_uid_census_before_sha256": sha(paths["before_census"]),
        "same_uid_census_after_sha256": sha(paths["after_census"]),
        "frame": {"path": str(frame), **frame_stats},
        "exact_ordered_log_markers": exact_lines,
    }
    need(manifest == expected_manifest, "execution/output manifest")
    result = {
        "schema": "m2207_m2205_lm_library_conversion_preflight_result_r1_v1",
        "status": "RAW_PASS_M2207_M2205_LM_LIBRARY_CONVERSION_PENDING_M2208_INDEPENDENT_RESULT_HAMMER",
        "frame": frame_stats, "sampled_processes": process,
        "execution_manifest_sha256": sha(paths["manifest"]),
        "claim_boundary": {"library_conversion_only": True,
                           "sampled_live_processes_only": True,
                           "exhaustive_short_lived_processes": False,
                           "design_library": False, "pnr": False, "timing": False,
                           "area": False, "power": False, "paper_ppa_ready": False},
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
        print(f"M2205_CHECK_FAIL_CLOSED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
