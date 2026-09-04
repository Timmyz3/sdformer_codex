#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind static hammer for M1090/M1091. Never imports or runs runner."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HW = Path(__file__).resolve().parents[2]
WRAPPER = HW / "rtl_m1090/m1090_c2_k1_observation_wrapper.sv"
TB = HW / "dc_handoff/tb/tb_m1090_c2_k1_observation_mapped_case0_short.sv"
RUNNER = HW / "dc_handoff/scripts/run_m1091_m1090_c2_observation_dc_mapped_vcs_one_shot_r1.py"
CONTRACT = HW / "contracts/m1090_c2_k1_observation_dc_mapped_vcs_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1090_c2_k1_observation_dc_mapped_vcs_release_r1_20260830.json"
FILELIST = HW / "dc_handoff/filelists/date_m1090_c2_k1_observation_logic_only_dc.f"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = Path(__file__).with_name("mechanical_checks.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_double(path: Path) -> bool:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    return (
        side.read_text().split() == [sha(path), path.relative_to(HW).as_posix()]
        and outer.read_text().split() == [sha(side), side.relative_to(HW).as_posix()]
    )


wrapper = WRAPPER.read_text(encoding="utf-8")
tb = TB.read_text(encoding="utf-8")
runner = RUNNER.read_text(encoding="utf-8")
contract = load(CONTRACT)
release = load(RELEASE)

obs_decl = set(re.findall(r"output logic(?:\s+\[[^\]]+\])?\s+(obs_[A-Za-z0-9_]+)", wrapper))
obs_lhs = set(re.findall(r"\b(obs_[A-Za-z0-9_]+)\s*=", wrapper))
implementation_body = wrapper.split("m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24", 1)[1].split("always_comb", 1)[0]
obs_rhs_feedback = []
for line_no, line in enumerate(wrapper.splitlines(), 1):
    for name in obs_decl:
        if name in line and not re.search(rf"\b{name}\s*=", line) and "output logic" not in line:
            obs_rhs_feedback.append({"line": line_no, "name": name, "text": line.strip()})

source_hashes_match = {}
source_symlink_free = {}
for rel, expected in contract["source_sha256"].items():
    path = HW / rel
    source_hashes_match[rel] = path.is_file() and sha(path) == expected
    source_symlink_free[rel] = path.is_file() and not path.is_symlink()

# Model the runner's authority checks. Both attacks satisfy its current logic:
# the expected outer hashes and runner hash come from attacker-controlled env.
self_signed_release_attack = {
    "mutate_contract_and_update_source_hashes": True,
    "recompute_contract_sidecar_and_outer": True,
    "mutate_release_to_new_contract_outer_and_runner_hash": True,
    "recompute_release_sidecar_and_outer": True,
    "set_M1091_EXPECTED_RELEASE_OUTER_SHA256_to_forged_outer": True,
    "set_M1091_EXPECTED_RUNNER_SHA256_to_current_or_forged_runner": True,
    "accepted_by_current_static_gate_model": (
        'os.environ.get("M1091_EXPECTED_RELEASE_OUTER_SHA256", "")' in runner
        and 'os.environ.get("M1091_EXPECTED_RUNNER_SHA256")' in runner
        and "verify_double(RELEASE" in runner
    ),
}
self_signed_go_attack = {
    "forge_review_status_and_authorization": True,
    "recompute_review_manifest_and_outer": True,
    "set_M1091_EXPECTED_M1092_OUTER_SHA256_to_forged_outer": True,
    "accepted_by_current_static_gate_model": (
        'verify_flat(M1092,os.environ.get("M1091_EXPECTED_M1092_OUTER_SHA256","")' in runner
        and 'review["status"]!="PASS_M1092_M1090_OBSERVATION_SOURCE_HAMMER__GO_ONE_M1091_ATTEMPT"' in runner
        and 'review["authorization"]["one_m1091_attempt"] is not True' in runner
    ),
}

tool_and_library_paths = [
    "/opt/synopsys/syn/V-2023.12-SP3",
    "/opt/synopsys/vcs/V-2023.12-SP1",
    "/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a",
]
tool_path_literal = all(token in runner for token in tool_and_library_paths)
tool_or_library_hash_pins = any(token in runner for token in ("DC_SHELL_SHA256", "VCS_SHA256", "SLOW_SHA256", "FAST_SHA256", "CELL_SHA256"))
tool_or_library_symlink_rejection = any(token in runner for token in ("DC_SHELL.is_symlink", "VCS.is_symlink", "SLOW.is_symlink", "FAST.is_symlink", "CELL.is_symlink"))

checks = {
    "source_identity": {
        "contract_double_seal_valid": check_double(CONTRACT),
        "release_double_seal_valid": check_double(RELEASE),
        "release_pins_contract_sha": release["contract_sha256"] == sha(CONTRACT),
        "release_pins_runner_sha": release["runner_sha256"] == sha(RUNNER),
        "all_contract_source_hashes_match": all(source_hashes_match.values()),
        "all_contract_source_paths_direct_symlink_free": all(source_symlink_free.values()),
        "filelist_hash_matches": sha(FILELIST) == release["filelist_sha256"],
        "docs359_unchanged": sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    },
    "wrapper": {
        "obs_output_count": len(obs_decl),
        "obs_names": sorted(obs_decl),
        "all_obs_assigned": obs_decl == obs_lhs,
        "obs_connected_into_implementation": bool(re.search(r"\.obs_", implementation_body)),
        "obs_rhs_feedback_occurrences": obs_rhs_feedback,
        "m1058_source_modified_by_wrapper": False,
    },
    "tb": {
        "maximum_post_header_cycles_128": "if(window_cycle==128)" in tb,
        "absolute_watchdog_1000ns": '#1000 $fatal(1,"M1090 absolute watchdog")' in tb,
        "all_22_obs_have_isunknown_check": all(f"`M1090_FAIL_X({name})" in tb for name in obs_decl),
        "header_timeout": "header not accepted within 16 cycles" in tb,
        "raw_timeout": "raw not accepted within 32 cycles" in tb,
        "stage_counter_print": "M1090_STAGE cycle=" in tb,
        "first_x_fatal": "M1090_FIRST_X cycle=" in tb and "M1090 fail-closed on first unknown" in tb,
        "saif_or_toggle_dump": bool(re.search(r"(?i)saif|\$toggle", tb)),
        "initreg": "+vcs+" + "initreg" in tb,
    },
    "runner": {
        "m1080_do_not_retry_hardcoded": "PASS_M1088_M1080_FAILURE_AUDIT__M1080_DO_NOT_RETRY" in runner,
        "attempt_before_dc": runner.index('phase="ATTEMPT_CONSUME_BEFORE_EDA"') < runner.index('phase="FRESH_DC_M1090_OBSERVATION_TOP"'),
        "fresh_namespace_collision_gate": 'if any(p.exists() or p.is_symlink() for p in (RESULT,ATTEMPT,WORK))' in runner,
        "lock_nonblocking": "LOCK_EX|fcntl.LOCK_NB" in runner,
        "failure_quarantine": "FAILED_DIAGNOSTIC_DO_NOT_CITE" in runner and "os.rename(WORK,FAILURE)" in runner,
        "dc_run_call_count": len(re.findall(r"rc=run\(\[str\(DC_SHELL\)", runner)),
        "vcs_compile_call_count": len(re.findall(r"rc=run\(\[str\(VCS\)", runner)),
        "simv_run_call_count": len(re.findall(r"rc=run\(\[str\(simv\)", runner)),
        "mapped_case_count": 1 if '"mapped_cases":1' in runner else 0,
        "no_activity": not re.search(r"(?i)(write_saif|read_saif|saif_map|toggle_start|toggle_stop)", runner),
        "no_random_init": "+vcs+" + "initreg" not in runner,
        "diagnostic_not_paper": '"diagnostic_only":True,"paper_citable":False' in runner,
        "absolute_tool_and_library_paths": tool_path_literal,
        "tool_and_library_hash_pinned": tool_or_library_hash_pins,
        "tool_and_library_symlink_rejected": tool_or_library_symlink_rejection,
    },
    "attacks": {
        "self_signed_contract_release": self_signed_release_attack,
        "self_signed_go_review": self_signed_go_attack,
        "existing_result_attempt_or_work_rejects": 'if any(p.exists() or p.is_symlink() for p in (RESULT,ATTEMPT,WORK))' in runner,
        "preheld_lock_rejects": "except BlockingIOError:fail(\"lock busy\")" in runner,
        "direct_source_symlink_rejects": 'path.is_symlink()' in runner,
        "tool_or_library_symlink_attack_rejects": tool_or_library_symlink_rejection,
    },
    "namespace": {
        "m1091_result_absent": not (HW / "results/m1091_m1090_c2_observation_dc_mapped_vcs_r1_20260830").exists(),
        "m1091_attempt_absent": not (HW / "results/.m1091_m1090_c2_observation_dc_mapped_vcs_attempt_consumed").exists(),
        "m1091_work_absent": not any((HW / "results").glob(".m1091_m1090_c2_observation_dc_mapped_vcs_work.*")),
        "old_c2_m1089_sources_absent": not any(p for p in HW.rglob("*m1089*") if "m1089_final_checkpoint_rebind_readonly_audit" not in p.as_posix()),
        "checkpoint_review_untouched": (HW / "reviews/m1089_final_checkpoint_rebind_readonly_audit_r1_20260830").is_dir(),
    },
}

issues = [
    {
        "severity": "P0",
        "id": "M1092-P0-01",
        "title": "Caller-controlled expected hashes make the release, runner and M1092 GO self-signed",
        "evidence": {
            "release_attack_accepted": self_signed_release_attack["accepted_by_current_static_gate_model"],
            "go_attack_accepted": self_signed_go_attack["accepted_by_current_static_gate_model"],
        },
    },
    {
        "severity": "P1",
        "id": "M1092-P1-01",
        "title": "Tool, library and cell-model identities are absolute paths but not hash-pinned or symlink-rejected",
        "evidence": {
            "absolute_paths": tool_path_literal,
            "hash_pinned": tool_or_library_hash_pins,
            "symlink_rejected": tool_or_library_symlink_rejection,
        },
    },
]

result = {
    "schema": "m1092_m1090_c2_observation_source_independent_hammer_checks_v1",
    "receipt_blind": True,
    "runner_imported_or_executed": False,
    "eda_commands": 0,
    "checks": checks,
    "issues": issues,
    "verdict": "STOP",
}
OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("STOP_M1092_SELF_SIGNED_AUTHORITY")
