#!/usr/bin/env python3
"""Static-only M719/r9 author self-test. Never invokes runner, VCS or simv."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import re
import subprocess


NEW_RUNNER_SHA = "27f2d7c0f6a2a8569b16f161fe5fcadc0722dfdb0735ee36130c3fb29b964604"
OLD_RUNNER_SHA = "176c14d35bf170f75b3097d832b2a39cd97ef7869263c1a0a019d99af0f8746e"
CONTRACT_SHA = "fca6edc169aaa4d932bdbe506b3452e49f156e20b9d9c9939a30b9665bf76185"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_manifest(root):
    manifest = os.path.join(root, "SHA256SUMS")
    seal = os.path.join(root, "SHA256SUMS.seal.sha256")
    mismatches = []
    for line in open(manifest, "r"):
        if not line.strip():
            continue
        expected, rel = line.rstrip("\n").split("  ", 1)
        actual = sha256_file(os.path.join(root, rel))
        if actual != expected:
            mismatches.append([rel, expected, actual])
    expected_manifest, rel = open(seal, "r").read().strip().split("  ", 1)
    return {
        "members_ok": not mismatches,
        "member_mismatches": mismatches,
        "manifest_sha256": sha256_file(manifest),
        "seal_file_sha256": sha256_file(seal),
        "seal_ok": rel == "SHA256SUMS" and expected_manifest == sha256_file(manifest),
    }


def probe(script):
    p = subprocess.run(["bash", "-uc", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = os.path.abspath(args.repo_root)
    hw = os.path.join(repo, "hw_autoresearch_nts07")
    old_runner = os.path.join(hw, "dc_handoff", "scripts", "run_vcs_m560_m533_m528_dead_write_only_1rw_r8_exact_sha.sh")
    new_runner = os.path.join(hw, "dc_handoff", "scripts", "run_vcs_m719_m533_m528_dead_write_only_1rw_r9_exact_sha.sh")
    contract = os.path.join(hw, "contracts", "m719_m533_m528_dead_write_only_1rw_source_only_contract_r1_20260828.json")
    r8_result = os.path.join(hw, "results", "m560_m533_m528_dead_write_only_1rw_vcs_r6_20260828")
    m717 = os.path.join(hw, "reviews", "m717_m560_m533_r8_monitor_start_failure_fresh_hammer_r1_20260828")
    new_result = os.path.join(hw, "results", "m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828")
    new_attempt = new_result + ".attempt"

    old_text = open(old_runner, "r").read()
    new_text = open(new_runner, "r").read()
    old_fault = 'local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0 tmp="${heartbeat}.tmp.$$"'
    new_first = "local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0"
    new_second = 'local tmp="${heartbeat}.tmp.$$"'
    old_probe = probe('f(){ local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0 tmp="${heartbeat}.tmp.$$"; :; }; f o v h r a')
    new_probe = probe('f(){ local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0; local tmp="${heartbeat}.tmp.$$"; printf "%s\\n" "$tmp"; }; f o v h r a')
    new_probe_report = dict(new_probe)
    new_probe_report["stdout"] = re.sub(r"h\.tmp\.[0-9]+", "h.tmp.<pid>", new_probe_report["stdout"])

    syntax = subprocess.run(["bash", "-n", new_runner], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    r8_seal = verify_manifest(r8_result)
    m717_seal = verify_manifest(m717)
    contract_member = subprocess.run(
        ["sha256sum", "-c", os.path.basename(contract) + ".sha256"],
        cwd=os.path.dirname(contract), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    contract_outer = subprocess.run(
        ["sha256sum", "-c", os.path.basename(contract) + ".sha256.seal.sha256"],
        cwd=os.path.dirname(contract), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    frozen = {
        "top_r2": ("rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv", "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1"),
        "sva_r2": ("verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv", "b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b"),
        "tb_r4": ("tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r4.sv", "72a6cef71b0014111c176e6baa751e6d0bfa1ea20e50b5c39b4064bbbe8345ff"),
        "macro_adapter": ("rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv", "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783"),
        "binding_plan": ("rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json", "db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983"),
    }
    frozen_checks = {
        name: {"path": rel, "expected": expected, "actual": sha256_file(os.path.join(hw, rel)), "pass": sha256_file(os.path.join(hw, rel)) == expected}
        for name, (rel, expected) in frozen.items()
    }

    compile_start = 'CURRENT_PHASE="vcs_compile"'
    functional_tail_exact = old_text[old_text.index(compile_start):] == new_text[new_text.index(compile_start):]
    old_monitor = old_text[old_text.index("resource_monitor() {"):old_text.index("require_monitor_live() {")]
    new_monitor = new_text[new_text.index("resource_monitor() {"):new_text.index("require_monitor_live() {")]
    normalized_new_monitor = new_monitor.replace(new_first + "\n  " + new_second, old_fault)
    monitor_only_split = normalized_new_monitor == old_monitor

    wrong_old_negative = {
        "expected_new_sha": NEW_RUNNER_SHA,
        "new_runner_sha": sha256_file(new_runner),
        "old_runner_sha": sha256_file(old_runner),
        "new_binding_accepts": sha256_file(new_runner) == NEW_RUNNER_SHA,
        "old_binding_rejected": sha256_file(old_runner) != NEW_RUNNER_SHA and sha256_file(old_runner) == OLD_RUNNER_SHA,
    }
    future_paths = [
        os.path.join(hw, "contracts", "m719_m533_m528_dead_write_only_1rw_vcs_launch_admission_candidate_r1_20260828.json"),
        os.path.join(hw, "contracts", "m719_m533_m528_dead_write_only_1rw_vcs_launch_release_r1_20260828.json"),
        os.path.join(hw, "reviews", "m719_m533_r9_source_static_hammer_r1_20260828"),
        os.path.join(hw, "reviews", "m719_m533_r9_vcs_launch_admission_candidate_hammer_r1_20260828"),
        os.path.join(hw, "reviews", "m719_m533_r9_vcs_final_launch_release_hammer_r1_20260828"),
    ]
    absent = {
        "new_result_path": not os.path.exists(new_result),
        "new_attempt_marker": not os.path.exists(new_attempt),
        "future_launch_chain": all(not os.path.exists(p) for p in future_paths),
    }
    prereq_source = {
        "r8_failure_paths_present": "R8_FAILED_RESULT_DIR=" in new_text and "R8_FAILED_RECEIPT=" in new_text,
        "m717_paths_present": "M717_DIR=" in new_text and "M717_REVIEW=" in new_text,
        "verifier_present": "verify_r8_failure_and_m717_prerequisite()" in new_text,
        "called_before_preflight": new_text.index("verify_r8_failure_and_m717_prerequisite\n") < new_text.index('[[ ! -e "${RESULT_DIR}" ]]'),
        "called_before_atomic_mkdir": 'CURRENT_PHASE="pre_mkdir_r8_m717_final_revalidation"; verify_r8_failure_and_m717_prerequisite' in new_text,
    }

    all_pass = (
        sha256_file(new_runner) == NEW_RUNNER_SHA
        and sha256_file(old_runner) == OLD_RUNNER_SHA
        and sha256_file(contract) == CONTRACT_SHA
        and syntax.returncode == 0
        and old_text.count(old_fault) == 1 and old_fault not in new_text
        and new_text.count(new_first) >= 1 and new_text.count(new_second) == 1
        and old_probe["returncode"] == 127 and "heartbeat: unbound variable" in old_probe["stderr"]
        and new_probe["returncode"] == 0 and new_probe["stdout"].startswith("h.tmp.")
        and wrong_old_negative["new_binding_accepts"] and wrong_old_negative["old_binding_rejected"]
        and functional_tail_exact and monitor_only_split
        and all(x["pass"] for x in frozen_checks.values())
        and r8_seal["members_ok"] and r8_seal["seal_ok"]
        and m717_seal["members_ok"] and m717_seal["seal_ok"]
        and contract_member.returncode == 0 and contract_outer.returncode == 0
        and all(absent.values()) and all(prereq_source.values())
        and sha256_file(os.path.join(hw, "docs", "359_DATE终局冻结_20260813.md")) == DOCS359_SHA
    )
    output = {
        "schema": "m719_m533_r9_source_only_static_selftest_v1",
        "status": "PASS" if all_pass else "FAIL",
        "runner_identity": {
            "new_path": "dc_handoff/scripts/run_vcs_m719_m533_m528_dead_write_only_1rw_r9_exact_sha.sh",
            "new_sha256": sha256_file(new_runner),
            "old_path": "dc_handoff/scripts/run_vcs_m560_m533_m528_dead_write_only_1rw_r8_exact_sha.sh",
            "old_sha256": sha256_file(old_runner),
            "contract_sha256": sha256_file(contract),
        },
        "wrong_old_runner_negative": wrong_old_negative,
        "wrong_old_runner_negative_pass": wrong_old_negative["new_binding_accepts"] and wrong_old_negative["old_binding_rejected"],
        "bash_n": {"returncode": syntax.returncode, "stderr": syntax.stderr},
        "isolated_reproducer": {
            "old_same_local": old_probe,
            "new_split_local": new_probe_report,
            "old_expected_rc": 127,
            "new_expected_rc": 0,
            "pass": old_probe["returncode"] == 127 and "heartbeat: unbound variable" in old_probe["stderr"] and new_probe["returncode"] == 0,
        },
        "semantic_scope": {
            "monitor_function_diff_normalizes_to_exact_r8": monitor_only_split,
            "vcs_compile_through_terminal_tail_byte_exact": functional_tail_exact,
            "frozen_functional_sources": frozen_checks,
            "hard_prerequisite_source_checks": prereq_source,
        },
        "sealed_prerequisites": {"r8_failure": r8_seal, "m717_review": m717_seal},
        "contract_double_seal": {"member_rc": contract_member.returncode, "outer_rc": contract_outer.returncode},
        "absence": absent,
        "new_result_path_absent": absent["new_result_path"],
        "old_same_local_rc": old_probe["returncode"],
        "new_split_local_rc": new_probe["returncode"],
        "execution": {"runner_runs": 0, "vcs_runs": 0, "simv_runs": 0, "eda_runs": 0},
        "docs359_sha256": sha256_file(os.path.join(hw, "docs", "359_DATE终局冻结_20260813.md")),
        "all_pass": all_pass,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "all_pass": all_pass,
        "wrong_old_runner_negative_pass": output["wrong_old_runner_negative_pass"],
        "old_same_local_rc": output["old_same_local_rc"],
        "new_split_local_rc": output["new_split_local_rc"],
        "new_result_path_absent": output["new_result_path_absent"],
        "runner_sha256": output["runner_identity"]["new_sha256"],
    }, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
