#!/usr/bin/python3.12
"""Read-only M2223 forensics; emit only the new M2224 review artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
HW = REPO / "hw_autoresearch_nts07"
NAME = "m2223_m2221_lm_command_option_discovery_raw_r1_20260904"
RAW = HW / "dc_handoff/runs" / (NAME + ".failed_or_incomplete.3569314.quarantine")
STAGING = HW / "dc_handoff/runs/.m2223_m2221_lm_command_option_discovery_work.3569314"
ATTEMPT = HW / "dc_handoff/runs/.m2223_m2221_lm_command_option_discovery_attempt_consumed"
SOURCE_REVIEW = HW / "reviews/m2222_m2221_m2208_lm_command_option_discovery_source_hammer_r1_20260904"
CONTRACT = HW / "contracts/m2221_m2208_lm_command_option_discovery_source_contract_r1_20260904.json"
TCL = HW / "dc_handoff/scripts/run_lm_m2221_command_option_discovery.tcl"
CHECKER = HW / "system_simulator/scripts/check_m2223_lm_command_option_discovery.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PASS = "RAW_PASS_M2223_M2221_LM_COMMAND_OPTION_DISCOVERY_PENDING_M2224_RESULT_HAMMER"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, label: str) -> None:
    if not ok:
        raise RuntimeError(label)


def seal_check(directory: Path) -> int:
    need(directory.is_dir() and not directory.is_symlink(), "sealed directory")
    need(not any(p.is_symlink() for p in directory.rglob("*")), "sealed symlink")
    manifest = directory / "SHA256SUMS"
    outer = (directory / "SHA256SUMS.seal.sha256").read_text().split()
    need(outer == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        rel = rel.removeprefix("*")
        need(rel not in listed and not Path(rel).is_absolute() and ".." not in Path(rel).parts,
             "manifest duplicate/escape")
        listed[rel] = digest
        need(sha(directory / rel) == digest, "inner seal " + rel)
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
              and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(set(listed) == actual, "non-exhaustive seal")
    return len(actual)


def one(pattern: str, text: str, label: str) -> re.Match[str]:
    found = list(re.finditer(pattern, text, re.MULTILINE))
    need(len(found) == 1, label + " occurrence")
    return found[0]


def main() -> None:
    counts = {"raw_members": seal_check(RAW), "attempt_members": seal_check(ATTEMPT),
              "m2222_members": seal_check(SOURCE_REVIEW)}
    contract = json.loads(CONTRACT.read_text())
    review = json.loads((SOURCE_REVIEW / "review.json").read_text())
    need(review["status"] == "PASS_M2222_M2221_SOURCE_HAMMER__M2223_ONE_SHOT_AUTHORIZED",
         "source authorization")
    need(review["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}, "source severities")
    need(sha(CONTRACT) == review["identity"]["contract_sha256"], "source contract hash")
    for path, digest in contract["source_inventory"].items():
        need(sha(REPO / path) == digest, "source hash " + path)
    need(sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359 identity")
    need(not (HW / "dc_handoff/runs" / NAME).exists() and not STAGING.exists(),
         "canonical/staging must remain absent")
    need((RAW / "RUN_FAILED_OR_INCOMPLETE.txt").read_text() ==
         "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=2\nretry=false\n", "failure marker")
    need(not (RAW / "receipt.json").exists() and not (RAW / "RUN_COMPLETE.txt").exists(),
         "no admitted predecessor receipt")
    need((RAW / "lm_discovery.rc").read_text() == "0\n", "LM return code")
    log = (RAW / "lm_discovery.log").read_text()
    need(sha(RAW / "lm_discovery.log") ==
         "a94cf17932c44c98ce52eafda06a4402b2ce41e9e0845bf179d052cd301b5ce8", "raw log identity")
    startup = one(r"^M2221_STARTUP mode=no_init setup_files=0 cwd_hex=([0-9a-f]+) home_hex=([0-9a-f]+)$",
                  log, "startup")
    original_isolated = str(STAGING / "isolated_cwd")
    need(bytes.fromhex(startup[1]).decode() == original_isolated, "original startup cwd")
    need(bytes.fromhex(startup[2]).decode() == original_isolated + "/home", "original startup home")
    need(log.count("M2221_FATAL_FAIL_CLOSED:") == 1 and
         '    puts stderr "M2221_FATAL_FAIL_CLOSED: $message"' in log[:startup.start()],
         "fatal literal is exactly one echoed source line")
    need(not re.search(r"^\s*M2221_FATAL_FAIL_CLOSED:", log, re.MULTILINE), "runtime fatal")
    need(re.search(r"^\s*M2221_FATAL_FAIL_CLOSED:", log + "\nM2221_FATAL_FAIL_CLOSED: injected\n",
                   re.MULTILINE) is not None, "fatal mutation must reject")
    command_log = (RAW / "isolated_cwd/lm_command.log").read_text()
    need("source -echo -verbose " + str(TCL) in command_log, "source echo identity")
    commands = re.findall(r"^M2221_COMMAND name=([^ ]+) available=([01])$", log, re.MULTILINE)
    need(commands == [(name, "1") for name in
         ("generate_frame_from_mw", "set_app_options", "get_app_option_value", "report_app_options")],
         "four runtime command observations")
    local = one(r"^M2221_OPTION name=lib\.configuration\.local_output_dir query_attempted=1 "
                r"query_rc=1 registered=0 value_hex= diagnostic_hex=([0-9a-f]+)$", log, "local option")
    need(bytes.fromhex(local[1]).decode() == "Invalid option name", "local option diagnostic")
    one(r"^M2221_OPTION name=lib\.setting\.milkyway_exec query_attempted=1 query_rc=0 "
        r"registered=1 value_hex= diagnostic_hex=$", log, "milkyway option")
    setting = one(r"^M2221_MILKYWAY_SET attempted=1 set_rc=0 readback_attempted=1 readback_rc=0 "
                  r"exact=1 value_hex=([0-9a-f]+) set_diagnostic_hex= readback_diagnostic_hex=$",
                  log, "set readback")
    need(bytes.fromhex(setting[1]).decode() == str(
         Path("/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway")), "Milkyway readback")
    no_side = one(r"^M2221_NO_SIDE_EFFECTS frame_files=0 ndm_files=0 nlib_files=0 "
                  r"generate_calls=0 create_lib_calls=0 pnr_calls=0$", log, "no side effects")
    raw_pass = one(r"^" + PASS + r"$", log, "raw pass")
    need(startup.start() < local.start() < setting.start() < no_side.start() < raw_pass.start(),
         "runtime marker order")
    need(len(re.findall(r"^" + PASS + r"$", log + "\n" + PASS + "\n", re.MULTILINE)) == 2,
         "duplicate pass mutation")
    for suffix in ("ndm", "nlib"):
        need(not list((RAW / "isolated_cwd").rglob("*." + suffix)), "forbidden output")
    need(not list((RAW / "isolated_cwd/frame_output").iterdir()), "frame output")
    for phase in ("before", "after"):
        census = json.loads((RAW / ("same_uid_census_" + phase + ".json")).read_text())
        need(census["phase"] == phase and census["status"] == "PASS_EMPTY" and
             census["matching_process_count"] == 0 and census["matching_processes"] == [], "census")
    need((RAW / "repo_root_before.json").read_bytes() == (RAW / "repo_root_after.json").read_bytes(),
         "repository inventory drift")
    execution = json.loads((RAW / "execution_contract.json").read_text())
    need(execution["isolated_root"] == original_isolated and execution["startup_mode"] == "no_init",
         "execution path identity")
    manifest = json.loads((RAW / "execution_output_manifest.json").read_text())
    for key, path in {"execution_contract_sha256": "execution_contract.json",
                      "lm_log_sha256": "lm_discovery.log",
                      "same_uid_census_before_sha256": "same_uid_census_before.json",
                      "same_uid_census_after_sha256": "same_uid_census_after.json",
                      "repo_root_before_sha256": "repo_root_before.json",
                      "repo_root_after_sha256": "repo_root_after.json"}.items():
        need(manifest[key] == sha(RAW / path), "manifest link " + key)
    for key in ("generate_frame_commands", "create_lib_commands", "milkyway_process_runs", "pnr_runs"):
        need(execution[key] == manifest[key] == 0, "no conversion count")
    need(execution["top_level_lm_shell_runs"] == manifest["top_level_lm_shell_runs"] == 1,
         "one LM run")
    spec = importlib.util.spec_from_file_location("m2223_frozen_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forbidden_output = OUT / "must_not_be_created.json"
    need(not forbidden_output.exists(), "checker diagnostic output virgin")
    try:
        module.validate(RAW, forbidden_output)
    except module.Failure as exc:
        need(str(exc) == "Tcl fatal diagnostic", "original checker first failure")
    else:
        raise RuntimeError("original checker unexpectedly accepted")
    need(not forbidden_output.exists(), "checker failure wrote no output")

    measurements = {
        **counts, "source_inventory_members": len(contract["source_inventory"]),
        "lm_return_code": 0, "pipeline_return_code": 2,
        "echoed_fatal_literals": 1, "runtime_fatal_markers": 0,
        "original_checker_failure_reproduced": "Tcl fatal diagnostic",
        "all_four_commands_available": True,
        "local_output_dir_registered": False, "local_output_dir_query_rc": 1,
        "milkyway_exec_registered": True, "milkyway_set_readback_exact": True,
        "original_staging_isolated_root": original_isolated,
        "current_quarantine_isolated_root": str(RAW / "isolated_cwd"),
        "authenticated_rename_mapping_required": True,
        "no_frame_ndm_nlib_observed": True, "repository_inventory_unchanged": True,
        "before_after_census_empty": True,
        "review_lm_runs": 0, "review_license_queries": 0, "review_eda_runs": 0,
    }
    identity = {"raw_directory": str(RAW.relative_to(REPO)),
                "raw_manifest_sha256": sha(RAW / "SHA256SUMS"),
                "raw_log_sha256": sha(RAW / "lm_discovery.log"),
                "execution_contract_sha256": sha(RAW / "execution_contract.json"),
                "source_review_sha256": sha(SOURCE_REVIEW / "review.json"),
                "source_contract_sha256": sha(CONTRACT), "checker_sha256": sha(CHECKER),
                "tcl_sha256": sha(TCL), "runner_sha256": contract["source_inventory"][
                    "hw_autoresearch_nts07/dc_handoff/scripts/run_m2223_m2222_m2221_lm_command_option_discovery_one_shot.sh"],
                "attempt_manifest_sha256": sha(ATTEMPT / "SHA256SUMS"),
                "docs359_sha256": sha(DOCS359), "independent_review_script_sha256": sha(Path(__file__))}
    result = {
        "schema": "m2224_m2223_lm_discovery_failure_forensics_r1_v1", "milestone": "M2224",
        "date_cst": "2026-09-05",
        "status": "PASS_M2224_FAILURE_FORENSICS__NEW_IDENTITY_PARSE_ONLY_RECOVERY_AUTHORIZED",
        "score_over_100": 97, "severity_counts": {"p0": 0, "p1": 0, "p2": 1},
        "identity": identity, "mechanical_checks": measurements,
        "findings": [
            {"id": "F1", "severity": "predecessor_p1", "resolved_in_review": True,
             "text": "The checker searched the whole echoed LM transcript for a literal fatal token. The one occurrence is an echoed puts source line; no runtime fatal marker exists. Frozen checker failure is reproduced without LM."},
            {"id": "F2", "severity": "successor_requirement", "resolved_in_review": False,
             "text": "Quarantine relocation changes filesystem paths. A successor must pin the exact PID-3569314 staging-to-quarantine mapping using both the sealed execution contract and runtime startup marker; arbitrary path normalization is prohibited."},
            {"id": "P2-1", "severity": "p2", "resolved_in_review": False,
             "text": "Before/after empty census plus source inspection is not an exhaustive microprocess monitor. Discovery proves command/option behavior only, not library compatibility or successful conversion."}],
        "authorization": {
            "new_identity_parse_only_recovery": True,
            "inputs_exact_raw_manifest_sha256": identity["raw_manifest_sha256"],
            "new_parser_requires_independent_source_review": True,
            "new_parser_requires_runtime_fatal_and_duplicate_marker_mutations": True,
            "new_parser_requires_exact_authenticated_rename_mapping": True,
            "new_result_requires_independent_result_review": True,
            "modify_old_raw_or_source": False, "retry_m2223": False,
            "license_queries": 0, "lm_runs": 0, "milkyway_runs": 0,
            "conversion_runs": 0, "pnr_runs": 0},
        "claim_boundary": {
            "forensic_runtime_observations": True, "production_receipt_admitted": False,
            "command_option_discovery_only": True, "library_conversion": False,
            "library_compatibility": False, "ndm_written": False, "pnr": False,
            "timing": False, "area": False, "power": False, "paper_ppa_ready": False},
    }
    (OUT / "mechanical_checks.json").write_text(json.dumps(measurements, indent=2, sort_keys=True) + "\n")
    (OUT / "review.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "RUN_COMPLETE.txt").write_text(result["status"] + "\n")
    members = sorted(p for p in OUT.rglob("*") if p.is_file() and
                     p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    (OUT / "SHA256SUMS").write_text("".join(sha(p) + "  " + str(p.relative_to(OUT)) + "\n" for p in members))
    (OUT / "SHA256SUMS.seal.sha256").write_text(sha(OUT / "SHA256SUMS") + "  SHA256SUMS\n")
    need(seal_check(OUT) == len(members), "review exhaustive double seal")
    print(json.dumps({"status": result["status"], "score": 97, "counts": counts,
                      "review_sha256": sha(OUT / "review.json")}))


if __name__ == "__main__":
    main()
