#!/usr/bin/env python3
"""Read-only fresh hammer for the consumed M560/M533 r8 launch failure.

The script never invokes the author runner, VCS, or simv.  It verifies the
sealed result, reconstructs its inventory, checks the control-flow boundary in
the frozen runner source, and uses isolated Bash micro-probes to demonstrate
the same-declaration `local` expansion bug and the two-declaration repair.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess


RESULT_REL = "hw_autoresearch_nts07/results/m560_m533_m528_dead_write_only_1rw_vcs_r6_20260828"
RUNNER_REL = "hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m560_m533_m528_dead_write_only_1rw_r8_exact_sha.sh"
RELEASE_REL = "hw_autoresearch_nts07/contracts/m560_m533_m528_dead_write_only_1rw_vcs_launch_release_r1_20260828.json"
DOCS359_REL = "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise RuntimeError("duplicate JSON key: " + key)
            out[key] = value
        return out
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=pairs)


def verify_result_seal(root):
    manifest_path = os.path.join(root, "SHA256SUMS")
    seal_path = os.path.join(root, "SHA256SUMS.seal.sha256")
    mismatches = []
    members = []
    with open(manifest_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            expected, rel = line.rstrip("\n").split("  ", 1)
            actual = sha256_file(os.path.join(root, rel))
            members.append(rel)
            if actual != expected:
                mismatches.append({"path": rel, "expected": expected, "actual": actual})
    seal_line = open(seal_path, "r").read().strip()
    seal_expected, seal_rel = seal_line.split("  ", 1)
    return {
        "member_count": len(members),
        "members": sorted(members),
        "member_mismatches": mismatches,
        "members_ok": not mismatches,
        "manifest_sha256": sha256_file(manifest_path),
        "seal_file_sha256": sha256_file(seal_path),
        "seal_ok": seal_rel == "SHA256SUMS" and seal_expected == sha256_file(manifest_path),
    }


def microprobe(script):
    proc = subprocess.run(["bash", "-uc", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = os.path.abspath(args.repo_root)
    result_root = os.path.join(repo, RESULT_REL)
    runner_path = os.path.join(repo, RUNNER_REL)
    release_path = os.path.join(repo, RELEASE_REL)
    docs359_path = os.path.join(repo, DOCS359_REL)

    seal = verify_result_seal(result_root)
    inventory = strict_json(os.path.join(result_root, "ARTIFACT_INVENTORY.json"))
    receipt = strict_json(os.path.join(result_root, "RUN_FAILED_OR_INCOMPLETE.json"))
    release = strict_json(release_path)
    marker = open(os.path.join(result_root, "FAILED_DO_NOT_CITE"), "r").read().strip()

    terminal_names = {
        "ARTIFACT_INVENTORY.json", "FAILED_DO_NOT_CITE", "RUN_FAILED_OR_INCOMPLETE.json",
        "SHA256SUMS", "SHA256SUMS.seal.sha256",
    }
    live_names = set(os.listdir(result_root))
    artifact_names = live_names - terminal_names
    inventory_items = {x["path"]: x for x in inventory["items"]}
    inventory_errors = []
    if artifact_names != set(inventory_items):
        inventory_errors.append({"path_set": {"live": sorted(artifact_names), "inventory": sorted(inventory_items)}})
    for rel, item in inventory_items.items():
        path = os.path.join(result_root, rel)
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or os.path.islink(path):
            inventory_errors.append({"path": rel, "problem": "not plain regular"})
            continue
        actual_bytes = os.path.getsize(path)
        actual_sha = sha256_file(path)
        if item.get("type") != "regular" or item.get("bytes") != actual_bytes or item.get("sha256") != actual_sha:
            inventory_errors.append({"path": rel, "problem": "metadata/content mismatch"})

    expected_artifacts = {
        "collision_initial.json", "collision_final.json", "collision_postmkdir.json", "resource_prelaunch.log"
    }
    collisions = {}
    for name in ("collision_initial.json", "collision_final.json", "collision_postmkdir.json"):
        value = strict_json(os.path.join(result_root, name))
        collisions[name] = {
            "schema": value.get("schema"),
            "runner_pid": value.get("runner_pid"),
            "scanner_pid": value.get("scanner_pid"),
            "matches": value.get("matches"),
            "verdict": value.get("verdict"),
        }
    collision_pass = (
        all(x["verdict"] == "PASS" and x["matches"] == [] for x in collisions.values())
        and len({x["runner_pid"] for x in collisions.values()}) == 1
        and len({x["scanner_pid"] for x in collisions.values()}) == 3
    )

    prelaunch_lines = [x for x in open(os.path.join(result_root, "resource_prelaunch.log"), "r") if x.strip()]
    prelaunch_pass = len(prelaunch_lines) == 3 and all(
        "session_failcnt=0" in x and "user_failcnt=0" in x
        and "session_under_oom=0" in x and "session_oom_kill=0" in x
        and "user_under_oom=0" in x and "user_oom_kill=0" in x
        for x in prelaunch_lines
    )

    runner = open(runner_path, "r").read()
    runner_sha = sha256_file(runner_path)
    faulty_decl = 'local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0 tmp="${heartbeat}.tmp.$$"'
    fixed_decl_1 = "local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0"
    fixed_decl_2 = 'local tmp="${heartbeat}.tmp.$$"'
    source_checks = {
        "set_euo_pipefail": "set -euo pipefail" in runner,
        "faulty_same_local_declaration_exact": faulty_decl in runner,
        "monitor_phase_precedes_background_launch": runner.index('CURRENT_PHASE="runtime_monitor_start"') < runner.index('resource_monitor "${RESULT_DIR}/resource_runtime.log"'),
        "heartbeat_required_before_vcs_phase": runner.index("require_monitor_live before_compile") < runner.index('CURRENT_PHASE="vcs_compile"'),
        "vcs_command_after_vcs_phase": runner.index('CURRENT_PHASE="vcs_compile"') < runner.index('"${VCS_BIN}" -full64'),
        "simv_after_vcs": runner.index('"${VCS_BIN}" -full64') < runner.index("./simv 2>&1"),
        "result_existence_gate_blocks_reuse": '[[ ! -e "${RESULT_DIR}" ]] || fail "result/attempt already exists: ${RESULT_DIR}"' in runner,
        "minimal_r9_fix_line_1": fixed_decl_1,
        "minimal_r9_fix_line_2": fixed_decl_2,
    }

    same_decl_probe = microprobe(
        'f(){ local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0 tmp="${heartbeat}.tmp.$$"; :; }; f o v h r a'
    )
    split_decl_probe = microprobe(
        'f(){ local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0; local tmp="${heartbeat}.tmp.$$"; printf "%s\\n" "$tmp"; }; f o v h r a'
    )
    bash_n = subprocess.run(["bash", "-n", runner_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    semantic_root_cause_pass = (
        same_decl_probe["returncode"] != 0
        and "heartbeat: unbound variable" in same_decl_probe["stderr"]
        and split_decl_probe["returncode"] == 0
        and split_decl_probe["stdout"].startswith("h.tmp.")
    )
    same_decl_report = dict(same_decl_probe)
    split_decl_report = dict(split_decl_probe)
    split_decl_report["stdout"] = re.sub(r"h\.tmp\.[0-9]+", "h.tmp.<pid>", split_decl_report["stdout"])

    forbidden_execution_artifacts = {
        "compile.log", "sim.log", "simv", "resource_runtime.log", "RESOURCE_HEARTBEAT",
        "RESOURCE_FINAL_REQUEST", "RESOURCE_FINAL_ACK", "RESOURCE_VIOLATION",
    }
    absent_execution_artifacts = sorted(forbidden_execution_artifacts - live_names)
    all_execution_artifacts_absent = len(absent_execution_artifacts) == len(forbidden_execution_artifacts)

    receipt_checks = {
        "status_failure": receipt.get("status") == "FAILED_DO_NOT_CITE",
        "kind_failure": receipt.get("kind") == "failure",
        "paper_citable_false": receipt.get("paper_citable") is False,
        "phase_runtime_monitor_start": receipt.get("phase") == "runtime_monitor_start",
        "runner_exit_rc_1": receipt.get("runner_exit_rc") == 1,
        "child_not_started": receipt.get("child_rc") == "not_started",
        "monitor_cleanup_wait_rc_1": receipt.get("monitor_status") == "cleanup_wait_rc_1",
        "failure_message_monitor_startup": receipt.get("failure_message") == "monitor startup",
        "preflight_cleanup_rc_0": receipt.get("preflight_cleanup_rc") == "0",
        "inventory_sha_bound": receipt.get("artifact_inventory", {}).get("sha256") == sha256_file(os.path.join(result_root, "ARTIFACT_INVENTORY.json")),
        "runner_sha_bound": receipt.get("exact_live_launch_bindings", {}).get("runner_r8", {}).get("sha256") == runner_sha,
        "functional_claim_false": receipt.get("claim_boundary", {}).get("functional_vcs_only") is False,
        "speedup_false": receipt.get("claim_boundary", {}).get("speedup") is False,
        "ppa_false": receipt.get("claim_boundary", {}).get("ppa") is False,
        "energy_false": receipt.get("claim_boundary", {}).get("energy") is False,
        "headline_false": receipt.get("claim_boundary", {}).get("system_or_paper_headline") is False,
        "marker_exact": marker == "FAILED_DO_NOT_CITE phase=runtime_monitor_start runner_rc=1 child_rc=not_started monitor_status=cleanup_wait_rc_1",
    }

    unique = release.get("unique_attempt", {})
    release_intent = release.get("release_intent", {})
    consumed_checks = {
        "release_result_path_exact": unique.get("result_path") == RESULT_REL.replace("hw_autoresearch_nts07/", ""),
        "atomic_mkdir_is_consumption_point": unique.get("attempt_consumed_only_by_runner_atomic_result_mkdir") is True,
        "max_attempts_one": release_intent.get("max_attempts") == 1,
        "result_now_exists": os.path.isdir(result_root),
        "terminal_failure_double_sealed": seal["members_ok"] and seal["seal_ok"],
        "runner_refuses_existing_result": source_checks["result_existence_gate_blocks_reuse"],
    }

    docs359_sha = sha256_file(docs359_path)
    all_checks = (
        seal["members_ok"] and seal["seal_ok"]
        and not inventory_errors
        and artifact_names == expected_artifacts
        and collision_pass and prelaunch_pass
        and all(v for k, v in source_checks.items() if not k.startswith("minimal_r9"))
        and semantic_root_cause_pass and bash_n.returncode == 0
        and all_execution_artifacts_absent
        and all(receipt_checks.values())
        and all(consumed_checks.values())
        and docs359_sha == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    )

    output = {
        "schema": "m717_m560_m533_r8_monitor_start_failure_fresh_hammer_recompute_v1",
        "review_mode": "read_only_receipt_blind_no_runner_vcs_or_simv",
        "seal": seal,
        "inventory": {
            "live_names": sorted(live_names),
            "artifact_names": sorted(artifact_names),
            "expected_artifact_names": sorted(expected_artifacts),
            "inventory_errors": inventory_errors,
            "all_match": not inventory_errors and artifact_names == expected_artifacts,
        },
        "collision": {"records": collisions, "pass": collision_pass},
        "prelaunch": {"samples": len(prelaunch_lines), "pass": prelaunch_pass},
        "receipt_checks": receipt_checks,
        "source_checks": source_checks,
        "bash_static_syntax": {"returncode": bash_n.returncode, "stderr": bash_n.stderr},
        "isolated_shell_semantics": {
            "same_declaration": same_decl_report,
            "split_declaration": split_decl_report,
            "root_cause_confirmed": semantic_root_cause_pass,
        },
        "execution_boundary": {
            "forbidden_execution_artifacts": sorted(forbidden_execution_artifacts),
            "all_absent": all_execution_artifacts_absent,
            "phase_before_vcs": True,
            "child_rc": receipt.get("child_rc"),
            "runner_invoked_by_reviewer": False,
            "vcs_invoked_by_reviewer": False,
            "simv_invoked_by_reviewer": False,
        },
        "consumed_identity": {
            "checks": consumed_checks,
            "r8_permanently_consumed": all(consumed_checks.values()),
            "r9_new_unique_identity_allowed": True,
            "requirements": [
                "new_runner_path_and_sha",
                "new_source_contract",
                "fresh_static_candidate_release_review_chain",
                "new_result_path",
                "same_frozen_functional_sources_unless_separately_reviewed",
            ],
        },
        "minimal_r9_fix": {
            "line_1": fixed_decl_1,
            "line_2": fixed_decl_2,
            "rationale": "heartbeat is assigned before tmp expands under set -u",
            "no_functional_rtl_change_required": True,
        },
        "docs359_sha256": docs359_sha,
        "all_checks_pass": all_checks,
        "decision": {
            "failure_receipt": "ADMIT",
            "functional_vcs": "NO_CONCLUSION",
            "rtl_functionality": "NO_CONCLUSION",
            "r8": "PERMANENTLY_CONSUMED",
            "r9": "MAY_OPEN_ONE_NEW_IDENTITY_AFTER_MINIMAL_REPAIR_AND_FRESH_CHAIN",
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "all_checks_pass": all_checks,
        "failure_receipt": output["decision"]["failure_receipt"],
        "functional_vcs": output["decision"]["functional_vcs"],
        "r8": output["decision"]["r8"],
        "root_cause_confirmed": semantic_root_cause_pass,
        "same_decl_rc": same_decl_probe["returncode"],
        "split_decl_rc": split_decl_probe["returncode"],
    }, sort_keys=True))
    return 0 if all_checks else 2


if __name__ == "__main__":
    raise SystemExit(main())
