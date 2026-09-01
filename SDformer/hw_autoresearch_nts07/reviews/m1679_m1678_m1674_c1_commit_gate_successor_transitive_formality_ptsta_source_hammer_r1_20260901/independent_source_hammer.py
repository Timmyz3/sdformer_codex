#!/usr/bin/env python3
"""Different-author read-only hammer for the M1678 C1 source successor.

M1678 may change only the live Committed_AS headroom gate from 48 GiB to
24 GiB plus fresh M1678/M1679/M1680 namespaces, authority and provenance
labels.  This hammer never invokes the runner or any EDA tool.

Python syntax is compatible with CPython 3.6.
"""
from __future__ import print_function

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
NEW = HW / (
    "dc_handoff/scripts/run_m1678_m1674_c1_commit_gate_successor_"
    "transitive_formality_ptsta_exact_closed_one_shot.sh")
OLD = HW / (
    "dc_handoff/scripts/run_m1674_m1665_c1_transitive_formality_ptsta_"
    "exact_closed_one_shot.sh")
TEST = HW / (
    "system_simulator/tests/test_m1678_c1_commit_gate_successor_source.py")
CONTRACT = HW / (
    "contracts/m1678_m1674_c1_commit_gate_successor_transitive_formality_"
    "ptsta_source_contract_r1_20260901.json")
AUTHOR_DIR = HW / (
    "reviews/m1678_m1674_c1_commit_gate_successor_transitive_formality_"
    "ptsta_source_author_receipt_r1_20260901")
AUTHOR = AUTHOR_DIR / "author_receipt.json"
M1674_CONTRACT = HW / (
    "contracts/m1674_m1665_c1_transitive_formality_ptsta_source_"
    "contract_r1_20260901.json")
M1675 = HW / (
    "reviews/m1675_m1674_m1665_c1_transitive_formality_ptsta_source_"
    "hammer_r1_20260901")
M1676 = HW / (
    "contracts/m1676_m1675_m1674_m1665_c1_transitive_formality_ptsta_"
    "launch_release_r1_20260901.json")
M1665 = HW / (
    "dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_"
    "dc_recovered_canonical_r1_20260901")
M1667 = HW / (
    "reviews/m1667_m1665_c1_canonical_recovery_result_hammer_r1_20260901")
M993 = HW / (
    "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_"
    "canonical_r1_20260829")
RTL_FM = HW / "dc_handoff/scripts/run_formality_m1674_c1_rtl_to_m993_transitive.tcl"
GATE_FM = HW / "dc_handoff/scripts/run_formality_m1674_c1_m993_to_m1665_gate_to_gate.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1674_c1_m1665_slowmax_fastmin.tcl"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RELEASE = HW / (
    "contracts/m1680_m1679_m1678_m1674_c1_commit_gate_successor_"
    "transitive_formality_ptsta_launch_release_r1_20260901.json")
RESULT = HW / (
    "dc_handoff/runs/m1678_m1674_c1_commit_gate_successor_transitive_"
    "formality_ptsta_r1_20260901")
ATTEMPT = HW / (
    "dc_handoff/runs/.m1678_m1674_c1_commit_gate_successor_transitive_"
    "formality_ptsta_attempt_consumed")
LOCK = HW / (
    "dc_handoff/runs/.m1678_m1674_c1_commit_gate_successor_transitive_"
    "formality_ptsta_launch_lock")

EXPECTED = {
    "new": "07c4a80048fe6344901a07c2e6d0e9c053879dc0ee8a1a771a3832c4c42b7484",
    "old": "55409e053c7392de2e5962d7d8a9430cfc6429483ea3d774cd7ff4906305b944",
    "test": "94b44944c23013e5cc91d5ea2551320a9a9159ee3117b8edc248703a6ecbf46e",
    "contract": "0a42ecf5ecf9d22a1efdc36a636d7623d78a56912acc4b4c16787163d5b0c944",
    "contract_sidecar": "ba0870b1a195e5392855d987573a2b6f5a6429c1b01093a73f01df5b724e751b",
    "contract_outer": "5392ec5ad9ec88361bb2a76e3b55c4be4f6b5c9333a5c37f2fa41c3698065b13",
    "author": "01687c04c72676ea1c9d02c61260fe8d713d20d33318cdf043e10432aa5300c1",
    "author_manifest": "efc7d600658b53f1e468d65a7d5d2a33b319d832203139e860e12628b63db502",
    "author_outer": "293454b57cbecc8585dad02e301df3903bf2b4e8636231915d6a40c64e485094",
    "m1674_contract": "16424c8442febfccc22d3e0f920c96b4a8f6df7ae3b53dcbca072de9fc5e6bc9",
    "m1674_contract_outer": "afb2db429ce6bb706f33d90447dbf33b50685322ffd53712fbd6447169086955",
    "m1675_review": "644fba82b931b4bcc84287731ce6144a6fae94127fe8b8cf466e2512bf8b88e7",
    "m1675_outer": "73a01b08f7f21781512f0f0c2da38189d2a96875568f1848f17bc6a87cd0e07b",
    "m1676": "121e0843c69dccbb2039d9127e3732754d2d299bf5a818c1c3038b1d940be5a6",
    "m1676_outer": "5cc03cd4c50de76c5c801e59b9f8513115855beffa1b066d3b188bfa68b9be50",
    "m1665_manifest": "a16b9fb100bf7f1b3c6e7453035a5bf89a8f2ffbbeeca1d373038f6e899dba72",
    "m1665_outer": "12d87acb439b0cc171d3f42cd4f169fa6a531946c9c3c120cc9babc9c36fbc08",
    "m1667_review": "bcec72d13d08ddd38252eda93472a48ee1b9406563780b273544bf863f7b1db0",
    "m1667_outer": "c942b7b7461fdd4317a398f822d21b3f31be87f7e8f73f17c04bf11e965db5d9",
    "m993_manifest": "8aeda1372387692201badb90a7d81eb7d908f803c6cd652aab22dace5043d093",
    "m993_outer": "0cc3b953342d6f149183e5fdf55b97174f69f97701574b0a79f05a5068ff6689",
    "rtl_fm": "d3a72876d9b40f73c47834da123388fa40263cf017c61586f2113b352a7bc3de",
    "gate_fm": "6df82c2435ab312263fd133a8e52371ea3de1004bc493d9553879eafaf3d1e12",
    "pt_tcl": "e289faa0abb9f8e7136305158ef086e20bd7e77d2f960e436f51138a431241a1",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def must(value, message):
    if not value:
        raise HammerError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    must(stat.S_ISREG(mode) and not path.is_symlink(),
         label + " must be regular non-symlink")
    must(sha(path) == expected, label + " SHA drift")


def verify_file_seal(path, payload_sha, sidecar_sha, outer_sha, label):
    regular_exact(path, payload_sha, label)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular_exact(sidecar, sidecar_sha, label + " sidecar")
    regular_exact(outer, outer_sha, label + " outer")
    must(sidecar.read_text(encoding="ascii") ==
         payload_sha + "  " + path.name + "\n", label + " sidecar content")
    must(outer.read_text(encoding="ascii") ==
         sidecar_sha + "  " + sidecar.name + "\n", label + " outer content")


def verify_dir_seal(path, manifest_sha=None, outer_sha=None):
    path = Path(path)
    must(path.is_dir() and not path.is_symlink(), "directory root drift")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    if manifest_sha is not None:
        regular_exact(manifest, manifest_sha, "directory manifest")
    if outer_sha is not None:
        regular_exact(outer, outer_sha, "directory outer")
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=str(path),
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                          cwd=str(path), stdout=subprocess.DEVNULL)


def between(text, start, end):
    left = text.index(start)
    right = text.index(end, left)
    return text[left:right]


def assignment(text, name):
    match = re.search(r"^" + re.escape(name) + r"=(.*)$", text, re.M)
    must(match is not None, "missing assignment " + name)
    return match.group(1)


COMMON_ASSIGNMENTS = (
    "M1665_DIR", "M1665_ORIGINAL", "M1667_DIR", "M993_DIR",
    "M993_ORIGINAL", "TOP", "RTL_FILELIST", "RTL", "MACRO_RTL",
    "M993_NETLIST", "M993_SVF", "M1665_NETLIST", "M1665_DDC",
    "M1665_SVF", "M1665_SDC", "RTL_TO_M993_TCL", "GATE_TO_GATE_TCL",
    "PT_TCL", "DOC359", "FM_SHELL", "PT_SHELL", "LMUTIL",
    "LICENSE_FILE", "STD_SLOW", "STD_FAST", "MACRO_ROOT", "MACRO_SLOW",
    "MACRO_FAST", "MACRO_MANIFEST")


def validate_contract(row):
    must(type(row) is dict and row.get("schema") ==
         "m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_source_contract_r1_v1",
         "contract schema")
    must(row.get("status") ==
         "SOURCE_ONLY_M1678_C1_COMMIT_GATE_SUCCESSOR__NO_EDA_AUTHORIZED",
         "contract status")
    must(row.get("source_files") == {
        "dc_handoff/scripts/run_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_exact_closed_one_shot.sh": EXPECTED["new"],
        "system_simulator/tests/test_m1678_c1_commit_gate_successor_source.py": EXPECTED["test"]},
        "contract source identities")
    must(row.get("resource_gate_delta") == {
        "field": "commit_headroom_min_kib", "m1674": 50331648,
        "m1678": 25165824,
        "all_other_runtime_gates_byte_or_predicate_identical": True},
        "contract resource delta")
    must(row.get("authorization_now") == {"formality_runs": 0,
        "pt_runs": 0, "dc_runs": 0, "vcs_runs": 0, "ptpx_runs": 0,
        "gpu_runs": 0, "remote_runs": 0, "attempts_created": 0},
        "contract current authority")
    must(row.get("future_execution_budget") == {
        "formality_processes_exact": 2, "prime_time_processes_exact": 1,
        "all_other_eda_processes": 0, "max_attempts": 1, "retry": False},
        "contract future budget")
    live = row.get("unchanged_live_gates", {})
    must(live.get("mem_available_min_kib") == 16777216 and
         live.get("disk_available_min_kib") == 4194304 and
         live.get("swap_gate") == "none, identical to M1674" and
         live.get("same_uid_all_eda_zero_before_resource_and_before_attempt")
             is True and
         live.get("formality_and_prime_time_license_availability_before_attempt")
             is True and
         live.get("attempt_consumed_before_first_eda") is True,
         "contract unchanged live gates")
    failure = row.get("failure_policy", {})
    must(failure.get("retry_same_identity") is False and
         failure.get("tool_failure_publishes_sealed_quarantine") is True and
         failure.get("fail_on_any_formality_nonproof") is True and
         failure.get("fail_on_any_prime_time_gate") is True,
         "contract failure policy")
    claims = row.get("claim_boundary", {})
    must(claims.get("source_candidate") is True and
         all(value is False for key, value in claims.items()
             if key != "source_candidate"), "contract claim boundary")


def validate_runner(text, mode):
    must(mode == 0o755, "runner permission drift")
    must(text.count('"${commit_headroom}" -ge 25165824') == 1 and
         '"${commit_headroom}" -ge 50331648' not in text,
         "live commit headroom predicate drift")
    must(text.count('"${mem_available}" -ge 16777216') == 1 and
         text.count('"${disk_available}" -ge 4194304') == 1,
         "MemAvailable/disk gate drift")
    must("SwapFree" not in text and "swap_free" not in text,
         "new Swap gate is forbidden")
    must(text.count('[[ -z "$(same_uid_eda)" ]] || exit 4') == 2,
         "same-UID double gate drift")
    must(text.count("for feature in Formality PrimeTime") == 1 and
         text.count("lmstat -c 27030@ic.ismd-nemo") == 1,
         "license gate drift")
    fm1 = '"${FM_SHELL}" -f "${RTL_TO_M993_TCL}"'
    fm2 = '"${FM_SHELL}" -f "${GATE_TO_GATE_TCL}"'
    pt = '"${PT_SHELL}" -f "${PT_TCL}"'
    must(text.count(fm1) == text.count(fm2) == text.count(pt) == 1,
         "exact 2fm+1pt count drift")
    must(text.index(fm1) < text.index(fm2) < text.index(pt),
         "2fm+1pt serial order drift")
    attempt = 'mkdir "${ATTEMPT}"'
    must(text.index(attempt) < text.index(fm1), "attempt/EDA order drift")
    authority_counts = {
        "M1678_EXPECTED_RUNNER_SHA256": 2,
        "M1678_EXPECTED_RELEASE_SHA256": 2,
        "PASS_M1679_M1678_C1_COMMIT_GATE_SUCCESSOR_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT": 1,
        "AUTHORIZE_ONE_M1678_C1_COMMIT_GATE_SUCCESSOR_FORMALITY_PTSTA_ATTEMPT": 1,
        "future_m1678_attempts": 1}
    for token, count in authority_counts.items():
        must(text.count(token) == count and
             text.index(token) < text.index(attempt),
             "future authority drift " + token)
    result_gate_counts = {"Verification SUCCEEDED": 2,
        "No unmatched points": 2, "No failing compare points": 2,
        "No aborted compare points": 2, "No unverified compare points": 2,
        "setup<0 or hold<0 or setup_tns!=0 or hold_tns!=0": 1,
        "int(machine['macro_count'])!=9": 1,
        "forbidden timing exception": 1}
    for token, count in result_gate_counts.items():
        must(text.count(token) == count, "result gate drift " + token)
    must("FAILED_OR_INCOMPLETE_DO_NOT_CITE" in text and
         ".failed_or_incomplete.$$.quarantine" in text and
         text.count("retry=false") == 2 and "retry=true" not in text,
         "quarantine/no-retry drift")
    must("formality_runs_authorized=2" in text and
         "pt_runs_authorized=1" in text and
         "commit_headroom_min_kib=25165824" in text and
         "predecessor_commit_headroom_min_kib=50331648" in text,
         "attempt marker identity drift")
    must("dc_shell" not in between(text, fm1, pt + ")") and
         "ptpx" not in between(text, fm1, pt + ")") and
         "vcs" not in between(text, fm1, pt + ")"),
         "extra EDA call drift")


def exact_diff_audit(old, new):
    for name in COMMON_ASSIGNMENTS:
        must(assignment(old, name) == assignment(new, name),
             "common assignment drift " + name)
    function_start = "sha_file() {"
    same_uid_end = "on_exit() {"
    must(between(old, function_start, same_uid_end).replace(
             "ERROR: M1674", "ERROR: M1678") ==
         between(new, function_start, same_uid_end),
         "helper/seal/sameUID function drift")
    must(between(old, "on_exit() {", "trap on_exit EXIT INT TERM") ==
         between(new, "on_exit() {", "trap on_exit EXIT INT TERM"),
         "failure quarantine function drift")
    gate_start = "# No live tool, namespace, resource or license state"
    gate_end = 'mkdir "${LOCK}"'
    old_gate = between(old, gate_start, gate_end)
    new_gate = between(new, gate_start, gate_end)
    normalized = new_gate.replace("25165824", "50331648").replace(
        "/tmp/m1678_c1_license.", "/tmp/m1674_c1_license.")
    must(normalized == old_gate, "resource/license pre-attempt slice drift")
    eda_start = 'export M1674_SNAPSHOT_ROOT="${HW_ROOT}"'
    eda_end = '/usr/bin/python3 - "${WORK}" <<\'PY\''
    must(between(old, eda_start, eda_end) ==
         between(new, eda_start, eda_end),
         "EDA execution and shell result gates are not byte-identical")
    result_py_start = '/usr/bin/python3 - "${WORK}" <<\'PY\''
    receipt_start = "receipt={"
    must(between(old, result_py_start, receipt_start) ==
         between(new, result_py_start, receipt_start),
         "post-run PT/Formality reparse gates drift")
    return {"common_assignments_exact": len(COMMON_ASSIGNMENTS),
        "helper_same_uid_slice": "EXACT_AFTER_DIAGNOSTIC_PREFIX_NORMALIZATION",
        "failure_quarantine_slice": "BYTE_IDENTICAL",
        "resource_pre_attempt_slice":
            "ONLY_50331648_TO_25165824_AND_M1674_TO_M1678_TMP_PREFIX",
        "eda_execution_and_shell_result_gate_slice": "BYTE_IDENTICAL",
        "postrun_reparse_gate_slice": "BYTE_IDENTICAL"}


def runner_mutations(base, mode):
    cases = []

    def reject(name, text=None, permission=None):
        candidate = base if text is None else text
        candidate_mode = mode if permission is None else permission
        try:
            validate_runner(candidate, candidate_mode)
        except (HammerError, ValueError):
            cases.append(name)
            return
        raise HammerError("runner mutation accepted: " + name)

    reject("permission_world_writable", permission=0o777)
    reject("permission_non_executable", permission=0o644)
    reject("commit_gate_restored_48g", base.replace(
        '"${commit_headroom}" -ge 25165824',
        '"${commit_headroom}" -ge 50331648', 1))
    reject("commit_gate_lowered_8g", base.replace(
        '"${commit_headroom}" -ge 25165824',
        '"${commit_headroom}" -ge 8388608', 1))
    reject("mem_gate_lowered", base.replace(
        '"${mem_available}" -ge 16777216',
        '"${mem_available}" -ge 8388608', 1))
    reject("disk_gate_lowered", base.replace(
        '"${disk_available}" -ge 4194304',
        '"${disk_available}" -ge 2097152', 1))
    reject("swap_gate_added", base.replace(
        'disk_available="$(df -Pk', 'SwapFree=1\ndisk_available="$(df -Pk', 1))
    reject("same_uid_gate_removed", base.replace(
        '[[ -z "$(same_uid_eda)" ]] || exit 4', ':', 1))
    reject("license_feature_changed", base.replace(
        "for feature in Formality PrimeTime", "for feature in Formality DC", 1))
    reject("extra_pt_process", base.replace(
        'export M1674_FM_OUTPUT_DIR="${WORK}/rtl_to_m993"',
        '"${PT_SHELL}" -f "${PT_TCL}"\nexport M1674_FM_OUTPUT_DIR="${WORK}/rtl_to_m993"', 1))
    reject("tool_order_swapped", base.replace(
        '"${FM_SHELL}" -f "${RTL_TO_M993_TCL}"', "__FM1__", 1).replace(
        '"${FM_SHELL}" -f "${GATE_TO_GATE_TCL}"',
        '"${FM_SHELL}" -f "${RTL_TO_M993_TCL}"', 1).replace(
        "__FM1__", '"${FM_SHELL}" -f "${GATE_TO_GATE_TCL}"', 1))
    reject("setup_gate_relaxed", base.replace(
        "setup<0 or hold<0", "setup<-1 or hold<0", 1))
    reject("retry_enabled", base.replace("retry=false", "retry=true", 1))
    reject("quarantine_removed", base.replace(
        "FAILED_OR_INCOMPLETE_DO_NOT_CITE", "FAILED", 1))
    reject("runner_pin_renamed", base.replace(
        "M1678_EXPECTED_RUNNER_SHA256", "M1678_RUNNER_SHA256"))
    reject("release_pin_renamed", base.replace(
        "M1678_EXPECTED_RELEASE_SHA256", "M1678_RELEASE_SHA256"))
    reject("review_status_changed", base.replace(
        "PASS_M1679_M1678_C1_COMMIT_GATE_SUCCESSOR_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT",
        "PASS", 1))
    reject("release_status_changed", base.replace(
        "AUTHORIZE_ONE_M1678_C1_COMMIT_GATE_SUCCESSOR_FORMALITY_PTSTA_ATTEMPT",
        "AUTHORIZE_ALL", 1))
    reject("result_gate_deleted", base.replace("No unmatched points",
                                               "Unmatched ignored", 1))
    reject("macro_gate_relaxed", base.replace(
        "int(machine['macro_count'])!=9", "False", 1))
    must(len(cases) == 20, "runner mutation count")
    return cases


def contract_mutations(base):
    cases = []

    def reject(name, mutate):
        candidate = copy.deepcopy(base)
        mutate(candidate)
        try:
            validate_contract(candidate)
        except (HammerError, KeyError, TypeError):
            cases.append(name)
            return
        raise HammerError("contract mutation accepted: " + name)

    reject("contract_status", lambda d: d.update(status="PASS"))
    reject("contract_runner_sha", lambda d:
           d["source_files"].update({list(d["source_files"])[0]: "0" * 64}))
    reject("contract_threshold", lambda d:
           d["resource_gate_delta"].update(m1678=1))
    reject("contract_formality_now", lambda d:
           d["authorization_now"].update(formality_runs=1))
    reject("contract_two_attempts", lambda d:
           d["future_execution_budget"].update(max_attempts=2))
    reject("contract_retry", lambda d:
           d["future_execution_budget"].update(retry=True))
    reject("contract_swap_gate", lambda d:
           d["unchanged_live_gates"].update(swap_gate="required"))
    reject("contract_sameuid_off", lambda d:
           d["unchanged_live_gates"].update(
               same_uid_all_eda_zero_before_resource_and_before_attempt=False))
    reject("contract_quarantine_off", lambda d:
           d["failure_policy"].update(
               tool_failure_publishes_sealed_quarantine=False))
    reject("contract_paper_claim", lambda d:
           d["claim_boundary"].update(paper_citable=True))
    must(len(cases) == 10, "contract mutation count")
    return cases


def main():
    identities = {"new": NEW, "old": OLD, "test": TEST,
        "contract": CONTRACT, "author": AUTHOR,
        "m1674_contract": M1674_CONTRACT,
        "m1675_review": M1675 / "review.json", "m1676": M1676,
        "m1665_manifest": M1665 / "SHA256SUMS",
        "m1667_review": M1667 / "review.json",
        "m993_manifest": M993 / "SHA256SUMS", "rtl_fm": RTL_FM,
        "gate_fm": GATE_FM, "pt_tcl": PT_TCL, "docs359": DOC359}
    for name, path in identities.items():
        regular_exact(path, EXPECTED[name], name)
    verify_file_seal(CONTRACT, EXPECTED["contract"],
        EXPECTED["contract_sidecar"], EXPECTED["contract_outer"],
        "M1678 contract")
    verify_dir_seal(AUTHOR_DIR, EXPECTED["author_manifest"],
                    EXPECTED["author_outer"])
    verify_file_seal(M1674_CONTRACT, EXPECTED["m1674_contract"],
        sha(Path(str(M1674_CONTRACT) + ".sha256")),
        EXPECTED["m1674_contract_outer"], "M1674 contract")
    verify_dir_seal(M1675, None, EXPECTED["m1675_outer"])
    regular_exact(M1675 / "review.json", EXPECTED["m1675_review"],
                  "M1675 review")
    verify_file_seal(M1676, EXPECTED["m1676"],
        sha(Path(str(M1676) + ".sha256")), EXPECTED["m1676_outer"],
        "M1676 release")
    verify_dir_seal(M1665, EXPECTED["m1665_manifest"], EXPECTED["m1665_outer"])
    verify_dir_seal(M1667, None, EXPECTED["m1667_outer"])
    verify_dir_seal(M993, EXPECTED["m993_manifest"], EXPECTED["m993_outer"])
    must(not os.path.lexists(str(RELEASE)) and
         not os.path.lexists(str(RESULT)) and
         not os.path.lexists(str(ATTEMPT)) and
         not os.path.lexists(str(LOCK)), "future namespace not fresh")
    subprocess.check_call(["bash", "-n", str(NEW)])
    old_text = OLD.read_text(encoding="utf-8")
    new_text = NEW.read_text(encoding="utf-8")
    mode = stat.S_IMODE(NEW.lstat().st_mode)
    validate_runner(new_text, mode)
    diff_result = exact_diff_audit(old_text, new_text)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    validate_contract(contract)
    author = json.loads(AUTHOR.read_text(encoding="utf-8"))
    must(author.get("status") ==
         "PASS_SOURCE_AUTHORING_ONLY_M1678_C1_COMMIT_GATE_SUCCESSOR__NO_EDA_RUN" and
         author.get("authorization", {}).get("all_eda_now") is False and
         author.get("claim_boundary", {}).get("paper_citable") is False,
         "author receipt boundary")
    runner_attacks = runner_mutations(new_text, mode)
    contract_attacks = contract_mutations(contract)
    output = {"schema":
        "m1679_m1678_c1_commit_gate_successor_source_independent_hammer_r1_v1",
        "status": "PASS_M1679_M1678_C1_COMMIT_GATE_SUCCESSOR_SOURCE_HAMMER",
        "python": sys.version.split()[0], "bash_n": "PASS",
        "exact_identity_checks": len(identities),
        "exact_diff": diff_result,
        "runner_mutations_rejected": runner_attacks,
        "contract_mutations_rejected": contract_attacks,
        "total_mutations_rejected": len(runner_attacks) +
                                    len(contract_attacks),
        "future_release_present": False, "attempt_present": False,
        "result_present": False, "eda_invoked": False,
        "authorization": "M1680 release authoring only; no EDA now"}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
