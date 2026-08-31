#!/usr/bin/env python3
"""Receipt-blind, read-only source hammer for M719/M533 r9.

This checker never invokes the author runner, VCS, simv, or any EDA tool.  It
checks immutable identities and seals, performs only Bash syntax checking and
two isolated `local`-declaration microprobes, and audits the runner text.
The authoritative review is emitted at the exact M719 path consumed by the
frozen runner; an optional M721 package may only hand off that sealed review.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess


HW = "hw_autoresearch_nts07"
RUNNER_R9 = "dc_handoff/scripts/run_vcs_m719_m533_m528_dead_write_only_1rw_r9_exact_sha.sh"
RUNNER_R8 = "dc_handoff/scripts/run_vcs_m560_m533_m528_dead_write_only_1rw_r8_exact_sha.sh"
CONTRACT_R9 = "contracts/m719_m533_m528_dead_write_only_1rw_source_only_contract_r1_20260828.json"
CONTRACT_R8 = "contracts/m560_m533_m528_dead_write_only_1rw_source_only_contract_r6_20260828.json"
REQUEST_DIR = "reviews/m719_m533_r9_source_static_hammer_r1_REQUEST_20260828"
HANDOFF_DIR = "reviews/m719_m533_r9_monitor_local_repair_author_handoff_r1_20260828"
R8_RESULT_DIR = "results/m560_m533_m528_dead_write_only_1rw_vcs_r6_20260828"
M717_DIR = "reviews/m717_m560_m533_r8_monitor_start_failure_fresh_hammer_r1_20260828"
DOCS359 = "docs/359_DATE终局冻结_20260813.md"
AUTHORITATIVE_REVIEW = "reviews/m719_m533_r9_source_static_hammer_r1_20260828/review.json"

EXPECTED = {
    "runner_r9": "27f2d7c0f6a2a8569b16f161fe5fcadc0722dfdb0735ee36130c3fb29b964604",
    "runner_r8": "176c14d35bf170f75b3097d832b2a39cd97ef7869263c1a0a019d99af0f8746e",
    "contract_r9": "fca6edc169aaa4d932bdbe506b3452e49f156e20b9d9c9939a30b9665bf76185",
    "contract_r9_sidecar": "a7cb56648348fd2e43f90b6990c0ed7e6f26c7787b7b88ea0f315b6f93d4c576",
    "contract_r9_outer": "39f3a2d7aa011baa050786211e29736df5063c7caf8bdf4329fd1758c5ef35b1",
    "contract_r8": "05b1de2db55ba12670ece983737ca92ff85221bde30de8ace4f13b2538b48825",
    "request_json": "677e6f2b10741678876bbdfa84b15c7d2fde262d0bd6f333e72b5f24acefe0d4",
    "request_manifest": "ca9b4fcc987cf1314c9b59dc8e555eb1dee812db944d08b000f4bfa4e2f217e7",
    "request_outer": "8bd10e249c351c864a4f7d53cba74427570a29795ebec8a28f38d5d249f7a608",
    "handoff_json": "cb7e7d02f0787b01d8ff5b88b4fd5af582c823a3b51ac6a28592dfde8085442a",
    "handoff_manifest": "04737ca707cf1517a5cf1b51b405d0e611eafc1bed19b275d13b7fd4d90f668e",
    "handoff_outer": "c5cf9b599cfbabad63933ff1d53814247c66fd2ebf5eecb582aa0920a2ecbcdb",
    "r8_failure": "d390ae62512d33e3f959de4dd1cf00546fde2253f9d8a52b0b6de5470568f393",
    "r8_manifest": "6061f952794dd8e30b734e123566a2b58aa6fd017f86821af6ca114c505e1d91",
    "r8_outer": "5a3f607edf6d0021b4e45ef8eb941465dd45ffe4b145549465b1888ed472eb4b",
    "m717_review": "ec0f06f3b1f4b112812a6bc101d52ba0078b4f83759928ec715971c7e2ea2bfc",
    "m717_manifest": "3f6b789d16aab8fc7b866f090619e0c7ec073b42bce42ff15010a0374c09a74b",
    "m717_outer": "ef1d93f2faef2f67313620b9d3215b550526ccf8fbdc84bead4d7a131fd27d8e",
    "top": "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1",
    "sva": "b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b",
    "tb": "72a6cef71b0014111c176e6baa751e6d0bfa1ea20e50b5c39b4064bbbe8345ff",
    "macro": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "binding": "db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
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

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("non-finite JSON number")
        if isinstance(value, dict):
            for key, member in value.items():
                finite(key)
                finite(member)
        elif isinstance(value, list):
            for member in value:
                finite(member)

    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=pairs, parse_constant=reject)
    finite(value)
    return value


def regular(path):
    mode = os.lstat(path).st_mode
    return stat.S_ISREG(mode) and not os.path.islink(path)


def verify_dir_seal(path):
    manifest = os.path.join(path, "SHA256SUMS")
    outer = os.path.join(path, "SHA256SUMS.seal.sha256")
    errors = []
    members = []
    if not regular(manifest) or not regular(outer):
        raise RuntimeError("missing/non-regular directory seal: " + path)
    with open(manifest, "r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            expected, rel = raw.rstrip("\n").split("  ", 1)
            member = os.path.join(path, rel)
            members.append(rel)
            if not regular(member):
                errors.append(rel + ":not_regular")
            elif sha256_file(member) != expected:
                errors.append(rel + ":sha")
    outer_expected, outer_rel = open(outer, encoding="utf-8").read().strip().split("  ", 1)
    if outer_rel != "SHA256SUMS" or outer_expected != sha256_file(manifest):
        errors.append("outer_seal")
    return {
        "member_count": len(members),
        "manifest_sha256": sha256_file(manifest),
        "outer_seal_file_sha256": sha256_file(outer),
        "errors": errors,
        "pass": not errors,
    }


def verify_json_sidecars(path):
    sidecar = path + ".sha256"
    outer = sidecar + ".seal.sha256"
    for member in (path, sidecar, outer):
        if not regular(member):
            raise RuntimeError("missing/non-regular JSON seal member: " + member)
    expected, rel = open(sidecar, encoding="utf-8").read().strip().split("  ", 1)
    outer_expected, outer_rel = open(outer, encoding="utf-8").read().strip().split("  ", 1)
    return {
        "member_pass": rel == os.path.basename(path) and expected == sha256_file(path),
        "outer_pass": outer_rel == os.path.basename(sidecar) and outer_expected == sha256_file(sidecar),
        "sidecar_file_sha256": sha256_file(sidecar),
        "outer_sidecar_file_sha256": sha256_file(outer),
    }


def extract_function(text, name):
    match = re.search(r"^" + re.escape(name) + r"\(\) \{\n.*?^\}\n", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise RuntimeError("function not found: " + name)
    return match.group(0)


def microprobe(source):
    proc = subprocess.run(
        ["bash", "-uc", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo = os.path.abspath(args.repo_root)
    hw = os.path.join(repo, HW)
    path = lambda rel: os.path.join(hw, rel)

    checks = {}
    details = {}

    request_seal = verify_dir_seal(path(REQUEST_DIR))
    handoff_seal = verify_dir_seal(path(HANDOFF_DIR))
    r8_seal = verify_dir_seal(path(R8_RESULT_DIR))
    m717_seal = verify_dir_seal(path(M717_DIR))
    r9_sidecars = verify_json_sidecars(path(CONTRACT_R9))
    r8_sidecars = verify_json_sidecars(path(CONTRACT_R8))
    details["seals"] = {
        "request": request_seal,
        "author_handoff": handoff_seal,
        "r8_failure": r8_seal,
        "m717": m717_seal,
        "r9_contract": r9_sidecars,
        "r8_contract": r8_sidecars,
    }
    checks["all_supplied_member_and_outer_seals"] = all(
        x["pass"] for x in (request_seal, handoff_seal, r8_seal, m717_seal)
    ) and all(
        x["member_pass"] and x["outer_pass"] for x in (r9_sidecars, r8_sidecars)
    )

    request = strict_json(path(REQUEST_DIR + "/request.json"))
    handoff = strict_json(path(HANDOFF_DIR + "/handoff.json"))
    contract = strict_json(path(CONTRACT_R9))
    old_contract = strict_json(path(CONTRACT_R8))
    failure = strict_json(path(R8_RESULT_DIR + "/RUN_FAILED_OR_INCOMPLETE.json"))
    m717 = strict_json(path(M717_DIR + "/review.json"))
    checks["strict_json_all_inputs"] = True

    observed_hashes = {
        "runner_r9": sha256_file(path(RUNNER_R9)),
        "runner_r8": sha256_file(path(RUNNER_R8)),
        "contract_r9": sha256_file(path(CONTRACT_R9)),
        "contract_r9_sidecar": sha256_file(path(CONTRACT_R9 + ".sha256")),
        "contract_r9_outer": sha256_file(path(CONTRACT_R9 + ".sha256.seal.sha256")),
        "contract_r8": sha256_file(path(CONTRACT_R8)),
        "request_json": sha256_file(path(REQUEST_DIR + "/request.json")),
        "request_manifest": sha256_file(path(REQUEST_DIR + "/SHA256SUMS")),
        "request_outer": sha256_file(path(REQUEST_DIR + "/SHA256SUMS.seal.sha256")),
        "handoff_json": sha256_file(path(HANDOFF_DIR + "/handoff.json")),
        "handoff_manifest": sha256_file(path(HANDOFF_DIR + "/SHA256SUMS")),
        "handoff_outer": sha256_file(path(HANDOFF_DIR + "/SHA256SUMS.seal.sha256")),
        "r8_failure": sha256_file(path(R8_RESULT_DIR + "/RUN_FAILED_OR_INCOMPLETE.json")),
        "r8_manifest": sha256_file(path(R8_RESULT_DIR + "/SHA256SUMS")),
        "r8_outer": sha256_file(path(R8_RESULT_DIR + "/SHA256SUMS.seal.sha256")),
        "m717_review": sha256_file(path(M717_DIR + "/review.json")),
        "m717_manifest": sha256_file(path(M717_DIR + "/SHA256SUMS")),
        "m717_outer": sha256_file(path(M717_DIR + "/SHA256SUMS.seal.sha256")),
        "docs359": sha256_file(path(DOCS359)),
    }
    checks["all_exact_external_identities"] = all(observed_hashes[key] == EXPECTED[key] for key in observed_hashes)
    details["observed_hashes"] = observed_hashes

    r9_text = open(path(RUNNER_R9), encoding="utf-8").read()
    r8_text = open(path(RUNNER_R8), encoding="utf-8").read()
    bash_n = subprocess.run(
        ["bash", "-n", path(RUNNER_R9)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    checks["runner_bash_n"] = bash_n.returncode == 0
    details["bash_n"] = {"rc": bash_n.returncode, "stdout": bash_n.stdout, "stderr": bash_n.stderr}

    old_decl = '  local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0 tmp="${heartbeat}.tmp.$$"\n'
    new_decl = (
        '  local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0\n'
        '  local tmp="${heartbeat}.tmp.$$"\n'
    )
    old_monitor = extract_function(r8_text, "resource_monitor")
    new_monitor = extract_function(r9_text, "resource_monitor")
    normalized_new_monitor = new_monitor.replace(new_decl, old_decl)
    checks["local_split_is_only_monitor_function_delta"] = (
        new_monitor.count(new_decl) == 1
        and old_monitor.count(old_decl) == 1
        and normalized_new_monitor == old_monitor
    )
    tail_marker = 'CURRENT_PHASE="vcs_compile"; CHILD_RC="running"'
    checks["vcs_compile_through_terminal_tail_byte_exact"] = (
        r9_text[r9_text.index(tail_marker):] == r8_text[r8_text.index(tail_marker):]
    )

    old_probe = microprobe(
        'f(){ local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0 tmp="${heartbeat}.tmp.$$"; :; }; f o v h r a'
    )
    new_probe = microprobe(
        'f(){ local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0; local tmp="${heartbeat}.tmp.$$"; printf "%s\\n" "$tmp"; }; f o v h r a'
    )
    checks["old_same_local_rc_127"] = old_probe["rc"] == 127 and "heartbeat: unbound variable" in old_probe["stderr"]
    checks["new_split_local_rc_0"] = new_probe["rc"] == 0 and new_probe["stdout"].startswith("h.tmp.")
    details["local_microprobes"] = {
        "old_same_local": old_probe,
        "new_split_local": {
            "rc": new_probe["rc"],
            "stdout": re.sub(r"h\.tmp\.[0-9]+", "h.tmp.<pid>", new_probe["stdout"]),
            "stderr": new_probe["stderr"],
        },
    }

    bound_r9 = contract.get("runner_r9", {}).get("sha256")
    checks["wrong_old_runner_negative"] = bound_r9 == EXPECTED["runner_r9"] and bound_r9 != EXPECTED["runner_r8"]
    checks["new_unique_result_identity"] = (
        contract.get("runner_r9", {}).get("result_path") == "results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828"
        and contract.get("runner_r9", {}).get("attempt_marker_path") == "results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828.attempt"
        and 'RESULT_DIR="${HW_ROOT}/results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828"' in r9_text
    )

    source_map = {"top_r2": "top", "sva_r2": "sva", "tb_r4": "tb", "macro_adapter": "macro", "macro_binding_plan": "binding"}
    frozen_actual = {}
    frozen_pass = True
    for field, expected_key in source_map.items():
        new_item = contract["frozen_functional_sources"][field]
        old_item = old_contract["frozen_functional_sources"][field]
        actual = sha256_file(path(new_item["path"]))
        frozen_actual[field] = actual
        frozen_pass = frozen_pass and (
            new_item["path"] == old_item["path"]
            and new_item["sha256"] == old_item["sha256"] == EXPECTED[expected_key] == actual
        )
    checks["rtl_tb_sva_macro_binding_byte_frozen"] = frozen_pass
    details["frozen_source_hashes"] = frozen_actual

    checks["r8_failure_exact_semantics"] = (
        failure.get("status") == "FAILED_DO_NOT_CITE"
        and failure.get("phase") == "runtime_monitor_start"
        and failure.get("child_rc") == "not_started"
        and failure.get("runner_exit_rc") == 1
        and failure.get("paper_citable") is False
    )
    checks["m717_exact_semantics"] = (
        m717.get("status") == "ADMIT_FAILURE_RECEIPT__R8_PERMANENTLY_CONSUMED__FUNCTIONAL_NO_CONCLUSION"
        and m717.get("decision", {}).get("r8") == "PERMANENTLY_CONSUMED"
        and m717.get("decision", {}).get("functional_vcs") == "NO_CONCLUSION"
        and m717.get("minimal_r9_fix", {}).get("new_unique_identity_allowed") is True
        and m717.get("minimal_r9_fix", {}).get("maximum_new_identities_now") == 1
    )

    lines = r9_text.splitlines()
    prereq_calls = [
        i + 1 for i, line in enumerate(lines)
        if "verify_r8_failure_and_m717_prerequisite" in line
        and "verify_r8_failure_and_m717_prerequisite()" not in line
    ]
    source_review_line = next(i + 1 for i, line in enumerate(lines) if line.strip() == 'verify_review_double_seal "${SOURCE_STATIC_DIR}"')
    resource_line = next(i + 1 for i, line in enumerate(lines) if 'CURRENT_PHASE="pre_mkdir_resource"' in line)
    mkdir_line = next(i + 1 for i, line in enumerate(lines) if line.strip() == 'if mkdir -- "${RESULT_DIR}"; then')
    checks["r8_m717_prerequisite_checked_twice"] = (
        len(prereq_calls) == 2
        and prereq_calls[0] < source_review_line
        and resource_line < prereq_calls[1] < mkdir_line
    )
    details["prerequisite_call_lines"] = prereq_calls

    collision_calls = [line for line in lines if "scan_same_uid_collisions" in line and "()" not in line]
    resource_tokens = [
        "CommitLimit:", "Committed_AS:", "MemAvailable:", "SwapFree:",
        "memory.failcnt", "under_oom", "oom_kill", "134217728", "33554432",
        "same_uid_synopsys_vcs_simv_collision_must_be_zero",
    ]
    checks["resource_and_collision_fail_closed"] = (
        len(collision_calls) == 3
        and all(token in r9_text for token in resource_tokens)
        and 'CURRENT_PHASE="pre_mkdir_collision_initial"' in r9_text
        and 'CURRENT_PHASE="pre_mkdir_collision_final"' in r9_text
        and 'CURRENT_PHASE="post_mkdir_collision"' in r9_text
    )
    atomic_segment = r9_text[r9_text.index('CURRENT_PHASE="pre_mkdir_atomic_attempt_publication"'):r9_text.index('CURRENT_PHASE="post_mkdir_evidence_copy"')]
    checks["atomic_attempt_publication_fail_closed"] = (
        "trap '' INT TERM HUP" in atomic_segment
        and 'if [[ -e "${RESULT_DIR}" ]]' in atomic_segment
        and atomic_segment.count('mkdir -- "${RESULT_DIR}"') == 1
        and 'rm -rf -- "${RESULT_DIR}"' not in r9_text
        and 'rmdir -- "${RESULT_DIR}"' not in r9_text
    )
    checks["terminal_success_and_failure_double_sealed"] = all(token in r9_text for token in (
        "trap cleanup EXIT", "build_artifact_inventory failure", "write_terminal_receipt failure",
        "seal_terminal_members", 'CURRENT_PHASE="success_terminal_seal"', "build_artifact_inventory success",
        "write_terminal_receipt success 0", "TERMINAL_SEALED=1", "trap - EXIT", "exit 0",
    ))

    future_paths = [
        "results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828",
        "results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828.attempt",
        "contracts/m719_m533_m528_dead_write_only_1rw_vcs_launch_admission_candidate_r1_20260828.json",
        "reviews/m719_m533_r9_vcs_launch_admission_candidate_hammer_r1_20260828",
        "contracts/m719_m533_m528_dead_write_only_1rw_vcs_launch_release_r1_20260828.json",
        "reviews/m719_m533_r9_vcs_final_launch_release_hammer_r1_20260828",
    ]
    absence = {rel: not os.path.lexists(path(rel)) for rel in future_paths}
    checks["new_result_attempt_and_future_chain_absent"] = all(absence.values())
    details["future_absence"] = absence

    canonical_review = request.get("required_pass_identity", {}).get("review_path")
    hardcoded_review = "reviews/m719_m533_r9_source_static_hammer_r1_20260828/review.json"
    checks["request_and_runner_agree_on_m719_canonical_review"] = (
        canonical_review == hardcoded_review
        and 'SOURCE_STATIC_DIR="${HW_ROOT}/reviews/m719_m533_r9_source_static_hammer_r1_20260828"' in r9_text
    )
    checks["authoritative_delivery_is_chain_consumable"] = AUTHORITATIVE_REVIEW == canonical_review
    details["canonical_path"] = {
        "request_required": canonical_review,
        "runner_hardcoded": hardcoded_review,
        "authoritative_delivery": AUTHORITATIVE_REVIEW,
        "mismatch": AUTHORITATIVE_REVIEW != canonical_review,
    }

    checks["request_and_handoff_no_execution_authorization"] = (
        all(value == 0 for value in request.get("forbidden_actions", {}).values() if isinstance(value, int))
        and handoff.get("claim_boundary", {}).get("launch_authorized") is False
        and contract.get("required_fresh_chain", {}).get("another_vcs_attempt_authorized_now") is False
    )
    checks["docs359_frozen"] = observed_hashes["docs359"] == EXPECTED["docs359"]

    source_mechanics_pass = all(checks.values())
    all_pass = all(checks.values())
    payload = {
        "schema": "m721_m719_m533_r9_source_fresh_static_hammer_recompute_v1",
        "date": "2026-08-28",
        "status": "PASS" if all_pass else "FAIL_STATIC_CHECKS",
        "source_mechanics_pass": source_mechanics_pass,
        "all_checks_pass": all_pass,
        "checks": checks,
        "details": details,
        "static_selftest": {
            "wrong_old_runner_negative_pass": checks["wrong_old_runner_negative"],
            "old_same_local_rc": old_probe["rc"],
            "new_split_local_rc": new_probe["rc"],
            "new_result_path_absent": absence["results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828"],
        },
        "execution_receipt": {
            "author_runner_executed_by_reviewer": False,
            "vcs_runs": 0,
            "simv_runs": 0,
            "hdl_eda_runs": 0,
            "result_or_attempt_identity_created": False,
            "launch_candidate_or_release_created": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if source_mechanics_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
