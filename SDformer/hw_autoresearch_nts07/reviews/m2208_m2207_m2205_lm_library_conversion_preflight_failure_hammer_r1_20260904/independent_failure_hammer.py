#!/usr/bin/python3.12
"""Independent read-only M2208 failure hammer; invokes no LM/EDA/license/GPU/Git."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
RUNS = HW / "dc_handoff/runs"
ATTEMPT = RUNS / ".m2207_m2205_lm_library_conversion_preflight_attempt_consumed"
QUARANTINE = RUNS / "m2207_m2205_lm_library_conversion_preflight_raw_r1_20260904.failed_or_incomplete.3526667.quarantine"
M2206 = HW / "reviews/m2206_m2205_m2190_lm_library_conversion_preflight_source_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
CONTRACT = HW / "contracts/m2205_m2190_lm_library_conversion_preflight_source_contract_r1_20260904.json"
TCL = HW / "dc_handoff/scripts/run_lm_m2205_library_conversion_preflight.tcl"
MONITOR = HW / "dc_handoff/scripts/monitor_m2205_lm_conversion_sampled_processes.py"
CENSUS = HW / "dc_handoff/scripts/census_m2205_same_uid_tools.py"
CHECKER = HW / "system_simulator/scripts/check_m2205_lm_library_conversion_preflight.py"
RUNNER = HW / "dc_handoff/scripts/run_m2205_m2190_lm_library_conversion_preflight_one_shot.sh"
TEST = HW / "tests/test_m2205_lm_library_conversion_preflight_source.py"
LM = "/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell"
LM_EXEC = "/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec"
MILKYWAY = "/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway"
EXPECTED_SOURCE = {
    CONTRACT: "65ae56329a89088f9f329cce3be51fbd4d98378fbdd8741585262ed7164d6deb",
    TCL: "c9ecf9eda32bd8d79f65e108d84c2851dc8a392f5ca8019fda3bf4a035dc6505",
    MONITOR: "4dd651bf0c55afe95d05c589ebf12f242144393f1d66468affc373920a576394",
    CENSUS: "ec452719e68c5caa88039ec7e37512647e2c737d54842eb2adf55e66639160bf",
    CHECKER: "74b7c82cf4c39ce7648ad0f35ada34a9e239019aae8e7100879350dca143564b",
    RUNNER: "ae4d01346c948bf8de6be37c135fca2ea79473b83151ff0e2a62870a880f8867",
    TEST: "624bdd3373203bde36300434543147bf31b99a555bc1ca6d89730f5812787cff",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
BLOCKED = {"vcs", "simv", "dc_shell", "pt_shell", "fm_shell", "icc2_shell",
           "icc2_exec", "dgcom_exec", "lm_shell", "lm_shell_exec", "Milkyway",
           "lmutil", "lmstat"}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verify_seal(directory: Path) -> dict[str, object]:
    need(directory.is_dir() and not directory.is_symlink(), f"invalid sealed dir {directory}")
    need(not any(path.is_symlink() for path in directory.rglob("*")), "symlink in seal")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe seal path")
        need(sha(directory / rel) == digest, f"member drift {rel}")
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive seal")
    return {"manifest_sha256": sha(manifest), "members": len(listed), "exhaustive": True,
            "symbolic_links": 0}


def current_tool_census() -> dict[str, object]:
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            exe = Path(os.readlink(proc / "exe")).name
            argv_names = {Path(item.decode(errors="replace")).name for item in
                          (proc / "cmdline").read_bytes().split(b"\0") if item}
        except (OSError, ValueError):
            continue
        if comm in BLOCKED or exe in BLOCKED or BLOCKED & argv_names:
            matches.append({"pid": int(proc.name), "comm": comm, "exe": exe,
                            "argv_names": sorted(argv_names)})
    need(not matches, f"live same-UID tool processes: {matches}")
    return {"matching_processes": matches, "matching_process_count": 0,
            "status": "PASS_EMPTY_AT_M2208_REVIEW"}


def exact_line_count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, re.MULTILINE))


def main() -> int:
    census_before = current_tool_census()
    for path, digest in EXPECTED_SOURCE.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             f"source/frozen identity drift {path}")
    m2206_seal = verify_seal(M2206)
    need(sha(M2206 / "review.json") ==
         "293b28289769f67a30c961ca579054520626845bfdfd7b98c54ca1b95570e2fc",
         "M2206 review identity")
    m2206 = read_json(M2206 / "review.json")
    need(m2206.get("status") ==
         "PASS_M2206_M2205_SOURCE_HAMMER__M2207_ONE_SHOT_AUTHORIZED" and
         m2206.get("authorization") == {"m2207": True, "license_queries": 1,
         "top_level_lm_shell_runs": 1, "pnr_runs": 0, "automatic_retry": False},
         "M2206 authority")

    matching = sorted(path.name for path in RUNS.iterdir() if "m2207" in path.name)
    need(matching == sorted([ATTEMPT.name, QUARANTINE.name]), f"M2207 multiplicity {matching}")
    attempt_seal = verify_seal(ATTEMPT)
    quarantine_seal = verify_seal(QUARANTINE)
    need(ATTEMPT.joinpath("ATTEMPT_CONSUMED.txt").read_text() ==
         "status=M2207_ATTEMPT_CONSUMED\nlicense_queries=1\ntop_level_lm_shell_runs=1\npnr_runs=0\nretry=false\n",
         "attempt marker")
    need(QUARANTINE.joinpath("RUN_FAILED_OR_INCOMPLETE.txt").read_text() ==
         "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=1\nretry=false\n",
         "quarantine marker")
    need(QUARANTINE.joinpath("lm_preflight.rc").read_text() == "42\n", "LM return code")
    need(QUARANTINE.joinpath("process_monitor.log").read_text() ==
         "MonitorFailure: exactly one sampled Milkyway identity required\n", "monitor log")

    process = read_json(QUARANTINE / "sampled_processes.json")
    need(process.get("status") == "FAIL_M2205_SAMPLED_POST_GATE_PROCESS_CONTRACT" and
         process.get("violation") == "MonitorFailure: exactly one sampled Milkyway identity required",
         "monitor failure identity")
    scope = process.get("claim_scope")
    need(scope == {"sampled_live_processes_only": True,
                   "exhaustive_short_lived_processes": False,
                   "sampling_interval_seconds": 0.005,
                   "bootstrap_helpers_permitted_before_gate": True,
                   "post_gate_actual_subtree_allowlist": [LM_EXEC, MILKYWAY]}, "claim scope")
    gate = process.get("gate")
    need(isinstance(gate, dict) and gate.get("released") is True and
         gate.get("created_by_monitor") is True and gate.get("tcl_wait_marker_seen") is True and
         gate.get("actual_stable_samples_required") == 5 and
         gate.get("actual_stable_samples_observed", 0) >= 5 and
         gate.get("frame_absent_before_release") is True, "release gate evidence")
    actual = process.get("actual_identity")
    need(isinstance(actual, dict) and actual.get("exe_path") == LM_EXEC and
         process.get("sampled_actual_identity_count") == 1, "actual identity")
    need(process.get("sampled_milkyway_identity_count") == 0 and
         process.get("pre_gate_milkyway_observations") == [] and
         process.get("unexpected_sampled_post_gate_descendants") == [] and
         process.get("post_gate_actual_subtree_processes") == [], "Milkyway absence")
    need(all(obs.get("phase") == "bootstrap_pre_gate"
             for identity in process.get("all_sampled_processes", [])
             for obs in identity.get("exec_observations", [])), "unexpected post-gate observation")

    log = QUARANTINE.joinpath("lm_preflight.log").read_text(errors="replace")
    pid = actual["pid"]
    work_gate = (Path(read_json(QUARANTINE / "execution_contract.json")["isolated_root"]).parent /
                 "conversion.release.gate")
    wait_line = rf"^M2205_GATE0_TCL_WAITING actual_pid={pid} gate={re.escape(str(work_gate))}$"
    release_line = rf"^M2205_GATE0_TCL_RELEASED actual_pid={pid} gate={re.escape(str(work_gate))}$"
    fatal_line = r"^M2205_FATAL_FAIL_CLOSED: Invalid option name 'lib\.configuration\.local_output_dir'$"
    need(exact_line_count(log, wait_line) == 1 and exact_line_count(log, release_line) == 1 and
         exact_line_count(log, fatal_line) == 1, "exact executed log markers")
    for pattern in (
        r"^M2205_GATE1_LOCAL_OUTPUT_ROUND_TRIP_PASS(?: |$).*$",
        r"^M2205_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS(?: |$).*$",
        r"^M2205_GATE3_FRAME_CONVERSION_PASS(?: |$).*$",
        r"^M2205_GATE4_NONEMPTY_FRAME_PASS(?: |$).*$",
        r"^RAW_PASS_M2207_M2205_LM_LIBRARY_CONVERSION_PENDING_M2208_INDEPENDENT_RESULT_HAMMER$",
    ):
        need(exact_line_count(log, pattern) == 0, f"unexpected executed marker {pattern}")
    need("Invalid option name 'lib.configuration.local_output_dir'\n    while executing\n"
         '"set_app_options -name lib.configuration.local_output_dir -value $cache"' in log,
         "exact native failure stack")
    need(log.index("M2205_GATE0_TCL_RELEASED actual_pid=") <
         log.index("M2205_FATAL_FAIL_CLOSED: Invalid option name"), "failure ordering")

    execution = read_json(QUARANTINE / "execution_contract.json")
    need(execution.get("scope") == "lm_library_conversion_only" and
         execution.get("lm_invocation") == [LM, "-no_init", "-f", str(TCL)] and
         execution.get("license_queries") == 1 and
         execution.get("top_level_lm_shell_runs") == 1 and
         execution.get("pnr_runs") == 0 and execution.get("automatic_retry") is False,
         "execution contract")
    census = read_json(QUARANTINE / "same_uid_census_before.json")
    need(census.get("phase") == "before" and census.get("status") == "PASS_EMPTY" and
         census.get("matching_process_count") == 0 and census.get("matching_processes") == [],
         "pre-run same-UID census")

    isolated = QUARANTINE / "isolated_cwd"
    empty_dirs = [isolated / name for name in
                  ("cache/library", "frame_output", "frame_logs", "reports")]
    need(all(path.is_dir() and not path.is_symlink() and not any(path.iterdir())
             for path in empty_dirs), "conversion output directory not empty")
    need(not list(isolated.rglob("*.ndm")) and not list(isolated.rglob("*.nlib")),
         "NDM/design library unexpectedly present")
    absent = ["execution_output_manifest.json", "receipt.json", "same_uid_census_after.json",
              "repo_root_after.json", "RUN_COMPLETE.txt"]
    need(all(not QUARANTINE.joinpath(name).exists() for name in absent), "success-only artifact exists")
    need(not (RUNS / "m2207_m2205_lm_library_conversion_preflight_raw_r1_20260904").exists() and
         not (RUNS / ".m2207_m2205_lm_library_conversion_preflight_launch_lock").exists() and
         not any(path.name.startswith(".m2207_m2205_lm_library_conversion_preflight_work")
                 for path in RUNS.iterdir()), "canonical/work/lock residue")

    census_after = current_tool_census()
    result = {
        "schema": "m2208_m2207_m2205_lm_library_conversion_preflight_failure_mechanical_checks_r1_v1",
        "status": "PASS_M2208_M2207_FAILURE_HAMMER__FAILURE_CONFIRMED__M2207_RETRY_FORBIDDEN",
        "identity": {"attempt_manifest_sha256": sha(ATTEMPT / "SHA256SUMS"),
                     "quarantine_manifest_sha256": sha(QUARANTINE / "SHA256SUMS"),
                     "lm_log_sha256": sha(QUARANTINE / "lm_preflight.log"),
                     "process_receipt_sha256": sha(QUARANTINE / "sampled_processes.json"),
                     "m2206_review_sha256": sha(M2206 / "review.json"),
                     "docs359_sha256": sha(DOC359)},
        "seals": {"attempt": attempt_seal, "quarantine": quarantine_seal,
                  "m2206": m2206_seal},
        "root_cause": {"tool": "lm_shell V-2023.12-SP3",
                       "failing_command": "set_app_options -name lib.configuration.local_output_dir -value $cache",
                       "diagnostic": "Invalid option name 'lib.configuration.local_output_dir'",
                       "lm_return_code": 42,
                       "before_generate_frame_from_mw": True,
                       "milkyway_started": False},
        "monitor": {"failed_closed": True, "reason": process["violation"],
                    "actual_identities": 1, "sampled_milkyway_identities": 0,
                    "unexpected_sampled_post_gate_descendants": 0,
                    "microprocess_exhaustive_claim": False},
        "containment": {"unique_attempt_marker": True, "unique_quarantine": True,
                        "canonical_result": False, "live_work_or_lock": False,
                        "frame_or_ndm_files": 0, "design_library_files": 0,
                        "pnr_runs": 0, "success_receipt_or_manifest": False,
                        "source_hashes_unchanged": True,
                        "quarantine_and_attempt_double_sealed": True,
                        "same_uid_before": census, "same_uid_at_review_start": census_before,
                        "same_uid_at_review_end": census_after,
                        "post_run_census_from_failed_runner_available": False,
                        "pollution_claim": "no observed source, canonical-result, live-process, frame, design-library, or P&R pollution; no exhaustive whole-repository after-inventory claim because the runner failed before writing repo_root_after.json"},
        "authorization": {"m2207_retry": False, "new_lm_discovery_run": False,
                          "license_queries": 0, "lm_runs": 0, "eda_runs": 0,
                          "pnr_runs": 0, "automatic_retry": False},
        "execution": {"review_lm_runs": 0, "review_eda_runs": 0,
                      "review_license_queries": 0, "review_gpu_runs": 0,
                      "review_git_mutation": False, "m2205_modified": False,
                      "m2207_modified": False, "docs359_modified": False},
    }
    output = HERE / "mechanical_checks.json"
    need(not output.exists(), "fresh output required")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
