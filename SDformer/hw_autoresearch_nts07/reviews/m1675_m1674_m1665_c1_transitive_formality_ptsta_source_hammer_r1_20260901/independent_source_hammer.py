#!/usr/bin/env python3
from __future__ import print_function

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1674_m1665_c1_transitive_formality_ptsta_exact_closed_one_shot.sh"
RTL_FM = HW / "dc_handoff/scripts/run_formality_m1674_c1_rtl_to_m993_transitive.tcl"
GATE_FM = HW / "dc_handoff/scripts/run_formality_m1674_c1_m993_to_m1665_gate_to_gate.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1674_c1_m1665_slowmax_fastmin.tcl"
TEST = HW / "system_simulator/tests/test_m1674_c1_transitive_formality_ptsta_source.py"
CONTRACT = HW / "contracts/m1674_m1665_c1_transitive_formality_ptsta_source_contract_r1_20260901.json"
AUTHOR_DIR = HW / "reviews/m1674_m1665_c1_transitive_formality_ptsta_source_author_receipt_r1_20260901"
AUTHOR = AUTHOR_DIR / "author_receipt.json"
M1665 = HW / "dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901"
M1667 = HW / "reviews/m1667_m1665_c1_canonical_recovery_result_hammer_r1_20260901"
M993 = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "dc_handoff/runs/m1674_m1665_c1_transitive_formality_ptsta_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1674_m1665_c1_transitive_formality_ptsta_attempt_consumed"
RELEASE = HW / "contracts/m1676_m1675_m1674_m1665_c1_transitive_formality_ptsta_launch_release_r1_20260901.json"


EXPECTED = {
    "runner": "55409e053c7392de2e5962d7d8a9430cfc6429483ea3d774cd7ff4906305b944",
    "rtl_fm": "d3a72876d9b40f73c47834da123388fa40263cf017c61586f2113b352a7bc3de",
    "gate_fm": "6df82c2435ab312263fd133a8e52371ea3de1004bc493d9553879eafaf3d1e12",
    "pt_tcl": "e289faa0abb9f8e7136305158ef086e20bd7e77d2f960e436f51138a431241a1",
    "test": "f8a8cd06a5a0d6a1975fce1a9ce5c4140d5f1adbf0217fcc46ee5999ca4843de",
    "contract": "16424c8442febfccc22d3e0f920c96b4a8f6df7ae3b53dcbca072de9fc5e6bc9",
    "contract_outer": "afb2db429ce6bb706f33d90447dbf33b50685322ffd53712fbd6447169086955",
    "author": "818095938b0c8a60b205f12d34d85535ed12057c5ce5a7d223e8022b7cb74f9f",
    "author_outer": "3a0c3f156568f59edec22f61a9f1b06a499cdc50b3827ddc5d5305b0d217ad3d",
    "m1665_manifest": "a16b9fb100bf7f1b3c6e7453035a5bf89a8f2ffbbeeca1d373038f6e899dba72",
    "m1665_outer": "12d87acb439b0cc171d3f42cd4f169fa6a531946c9c3c120cc9babc9c36fbc08",
    "m1665_receipt": "07601960b22b5f1d23226d5a60ce25c92b9652bc9700d058d6a4aea38e08b4e6",
    "m1667_review": "bcec72d13d08ddd38252eda93472a48ee1b9406563780b273544bf863f7b1db0",
    "m1667_outer": "c942b7b7461fdd4317a398f822d21b3f31be87f7e8f73f17c04bf11e965db5d9",
    "m993_manifest": "8aeda1372387692201badb90a7d81eb7d908f803c6cd652aab22dace5043d093",
    "m993_outer": "0cc3b953342d6f149183e5fdf55b97174f69f97701574b0a79f05a5068ff6689",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def must(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_dir_seal(path):
    for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
        must((path / name).is_file() and not (path / name).is_symlink(), "seal absent " + str(path / name))
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=str(path), stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=str(path), stdout=subprocess.DEVNULL)


def verify_file_seal(path):
    must(path.is_file() and not path.is_symlink(), "payload absent")
    subprocess.check_call(["sha256sum", "-c", path.name + ".sha256"], cwd=str(path.parent), stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", path.name + ".sha256.seal.sha256"], cwd=str(path.parent), stdout=subprocess.DEVNULL)


def canonical_release():
    return {
        "schema": "m1676_m1675_m1674_m1665_c1_transitive_formality_ptsta_launch_release_r1_v1",
        "date": "2026-09-01", "milestone": "M1676",
        "status": "AUTHORIZE_ONE_M1674_C1_TRANSITIVE_FORMALITY_PTSTA_ATTEMPT",
        "identity": {
            "runner_path": "dc_handoff/scripts/run_m1674_m1665_c1_transitive_formality_ptsta_exact_closed_one_shot.sh",
            "runner_sha256": EXPECTED["runner"],
            "source_contract_path": "contracts/m1674_m1665_c1_transitive_formality_ptsta_source_contract_r1_20260901.json",
            "source_contract_sha256": EXPECTED["contract"],
            "author_receipt_path": "reviews/m1674_m1665_c1_transitive_formality_ptsta_source_author_receipt_r1_20260901/author_receipt.json",
            "author_receipt_sha256": EXPECTED["author"],
            "source_hammer_path": "reviews/m1675_m1674_m1665_c1_transitive_formality_ptsta_source_hammer_r1_20260901/review.json",
            "source_hammer_sha256": "HAMMER_SHA_PLACEHOLDER",
            "m1665_manifest_sha256": EXPECTED["m1665_manifest"],
            "m1667_review_sha256": EXPECTED["m1667_review"],
            "future_result": "dc_handoff/runs/m1674_m1665_c1_transitive_formality_ptsta_r1_20260901",
            "future_attempt": "dc_handoff/runs/.m1674_m1665_c1_transitive_formality_ptsta_attempt_consumed",
        },
        "authorization": {"launch_now": True, "max_attempts": 1, "formality_runs": 2,
                          "pt_runs": 1, "dc_runs": 0, "vcs_runs": 0, "ptpx_runs": 0,
                          "gpu_runs": 0, "remote_runs": 0, "retry": False},
        "execution_order": ["RTL-to-M993 Formality", "different-process M993-to-M1665 gate-to-gate Formality",
                            "independent PrimeTime slow-max/fast-min", "different-author result hammer before any claim"],
        "result_policy": {"all_three_processes_must_exit_zero": True,
                          "both_formality_proofs_must_succeed": True,
                          "prime_time_setup_hold_must_be_nonnegative": True,
                          "macro_count_exact": 9, "fresh_result_hammer_required": True},
        "claim_boundary": {"launch_release": True, "formality": False, "prime_time": False,
                           "power": False, "energy": False, "cycle_speedup": False,
                           "system_speedup": False, "paper_ppa_ready": False,
                           "paper_citable": False, "headline": False},
    }


def validate_release(d):
    def exact(o, keys, name):
        must(type(o) is dict and set(o) == set(keys), name + " keyset")
    exact(d, ("schema", "date", "milestone", "status", "identity", "authorization",
              "execution_order", "result_policy", "claim_boundary"), "top")
    must(d["schema"] == canonical_release()["schema"], "schema")
    must(d["status"] == canonical_release()["status"], "status")
    expected = canonical_release()
    for key in ("identity", "authorization", "execution_order", "result_policy", "claim_boundary"):
        must(d[key] == expected[key], key)


def mutation_results():
    cases = []
    base = canonical_release()
    mutations = [
        ("top_extra_key", lambda d: d.update({"bypass": True})),
        ("wrong_schema", lambda d: d.__setitem__("schema", "wrong")),
        ("wrong_status", lambda d: d.__setitem__("status", "wrong")),
        ("runner_path", lambda d: d["identity"].__setitem__("runner_path", "/tmp/runner")),
        ("runner_sha", lambda d: d["identity"].__setitem__("runner_sha256", "0" * 64)),
        ("contract_path", lambda d: d["identity"].__setitem__("source_contract_path", "/tmp/contract")),
        ("contract_sha", lambda d: d["identity"].__setitem__("source_contract_sha256", "0" * 64)),
        ("author_sha", lambda d: d["identity"].__setitem__("author_receipt_sha256", "0" * 64)),
        ("hammer_path", lambda d: d["identity"].__setitem__("source_hammer_path", "/tmp/hammer")),
        ("hammer_sha", lambda d: d["identity"].__setitem__("source_hammer_sha256", "0" * 64)),
        ("m1665_manifest", lambda d: d["identity"].__setitem__("m1665_manifest_sha256", "0" * 64)),
        ("m1667_review", lambda d: d["identity"].__setitem__("m1667_review_sha256", "0" * 64)),
        ("future_result", lambda d: d["identity"].__setitem__("future_result", "/tmp/result")),
        ("future_attempt", lambda d: d["identity"].__setitem__("future_attempt", "/tmp/attempt")),
        ("identity_extra_key", lambda d: d["identity"].update({"bypass": True})),
        ("launch_false", lambda d: d["authorization"].__setitem__("launch_now", False)),
        ("two_attempts", lambda d: d["authorization"].__setitem__("max_attempts", 2)),
        ("one_formality", lambda d: d["authorization"].__setitem__("formality_runs", 1)),
        ("two_pt", lambda d: d["authorization"].__setitem__("pt_runs", 2)),
        ("dc_authorized", lambda d: d["authorization"].__setitem__("dc_runs", 1)),
        ("vcs_authorized", lambda d: d["authorization"].__setitem__("vcs_runs", 1)),
        ("ptpx_authorized", lambda d: d["authorization"].__setitem__("ptpx_runs", 1)),
        ("gpu_authorized", lambda d: d["authorization"].__setitem__("gpu_runs", 1)),
        ("remote_authorized", lambda d: d["authorization"].__setitem__("remote_runs", 1)),
        ("retry_authorized", lambda d: d["authorization"].__setitem__("retry", True)),
        ("order_swapped", lambda d: d["execution_order"].reverse()),
        ("formality_policy_off", lambda d: d["result_policy"].__setitem__("both_formality_proofs_must_succeed", False)),
        ("negative_slack_allowed", lambda d: d["result_policy"].__setitem__("prime_time_setup_hold_must_be_nonnegative", False)),
        ("macro_count_eight", lambda d: d["result_policy"].__setitem__("macro_count_exact", 8)),
        ("result_hammer_off", lambda d: d["result_policy"].__setitem__("fresh_result_hammer_required", False)),
        ("claim_formality_true", lambda d: d["claim_boundary"].__setitem__("formality", True)),
        ("claim_paper_true", lambda d: d["claim_boundary"].__setitem__("paper_citable", True)),
        ("claim_headline_true", lambda d: d["claim_boundary"].__setitem__("headline", True)),
    ]
    validate_release(base)
    for name, mutate in mutations:
        candidate = copy.deepcopy(base)
        mutate(candidate)
        try:
            validate_release(candidate)
        except AssertionError:
            cases.append({"name": name, "rejected": True})
        else:
            cases.append({"name": name, "rejected": False})
    must(all(x["rejected"] for x in cases), "mutation accepted")
    return cases


def main():
    paths = {"runner": RUNNER, "rtl_fm": RTL_FM, "gate_fm": GATE_FM,
             "pt_tcl": PT_TCL, "test": TEST, "contract": CONTRACT,
             "contract_outer": Path(str(CONTRACT) + ".sha256.seal.sha256"),
             "author": AUTHOR, "author_outer": AUTHOR_DIR / "SHA256SUMS.seal.sha256",
             "m1665_manifest": M1665 / "SHA256SUMS", "m1665_outer": M1665 / "SHA256SUMS.seal.sha256",
             "m1665_receipt": M1665 / "m1665_recovered_c1_dc_receipt.json",
             "m1667_review": M1667 / "review.json", "m1667_outer": M1667 / "SHA256SUMS.seal.sha256",
             "m993_manifest": M993 / "SHA256SUMS", "m993_outer": M993 / "SHA256SUMS.seal.sha256",
             "docs359": DOC359}
    for key, path in paths.items():
        must(path.is_file() and not path.is_symlink(), "missing/nonregular " + key)
        must(sha(path) == EXPECTED[key], "identity " + key)
    verify_file_seal(CONTRACT)
    for path in (AUTHOR_DIR, M1665, M1665 / "original_quarantine", M1667, M993, M993 / "original_quarantine"):
        verify_dir_seal(path)
    must(not RELEASE.exists() and not ATTEMPT.exists() and not RESULT.exists(), "source-only namespace violated")

    contract = json.loads(CONTRACT.read_text())
    author = json.loads(AUTHOR.read_text())
    runner = RUNNER.read_text()
    rtl = RTL_FM.read_text()
    gate = GATE_FM.read_text()
    pt = PT_TCL.read_text()
    must(contract["status"] == "SOURCE_ONLY_M1674_C1_TRANSITIVE_FORMALITY_PTSTA__NO_EDA_AUTHORIZED", "contract status")
    must(all(v == 0 for v in contract["authorization_now"].values()), "current authority")
    must(author["authorization"]["all_eda_now"] is False, "author EDA authority")
    must(author["claim_boundary"]["paper_citable"] is False, "author paper boundary")

    fm1 = '"${FM_SHELL}" -f "${RTL_TO_M993_TCL}"'
    fm2 = '"${FM_SHELL}" -f "${GATE_TO_GATE_TCL}"'
    ptcall = '"${PT_SHELL}" -f "${PT_TCL}"'
    must(runner.count(fm1) == 1 and runner.count(fm2) == 1 and runner.count(ptcall) == 1, "tool counts")
    must(runner.index(fm1) < runner.index(fm2) < runner.index(ptcall), "tool order")
    attempt = 'mkdir "${ATTEMPT}"'
    must(runner.index(attempt) < runner.index(fm1), "attempt must precede EDA")
    for token in ('verify_dir_seal "${HAMMER_DIR}"', 'verify_file_seal "${RELEASE}"',
                  'M1674_EXPECTED_RUNNER_SHA256', 'M1674_EXPECTED_RELEASE_SHA256',
                  '[[ -z "$(same_uid_eda)" ]] || exit 4', 'for feature in Formality PrimeTime',
                  'commit_headroom', 'mem_available', 'disk_available'):
        must(token in runner and runner.index(token) < runner.index(attempt), "preflight " + token)
    must(runner.count('[[ -z "$(same_uid_eda)" ]] || exit 4') == 2, "same UID recheck")
    must('FAILED_OR_INCOMPLETE_DO_NOT_CITE' in runner and '.quarantine' in runner, "failure quarantine")
    must('retry=false' in runner, "no retry")

    must('set_svf $svf_file' in rtl and 'M1674_M993_SVF' in rtl, "M993 SVF bridge")
    must('M1674_M1665' not in rtl, "direct M1665 proof leakage")
    must('M1674_M993_MAPPED_NETLIST' in gate and 'M1674_M1665_MAPPED_NETLIST' in gate, "gate bridge")
    must('set_svf' not in gate, "incremental SVF misuse")
    must('M1674_M1665_SVF' not in rtl + gate, "incremental SVF direct use")
    for proof in (rtl, gate):
        for token in ('report_unmatched_points', 'report_failing_points', 'report_aborted_points',
                      'report_unverified_points', 'read_db -technology_library $macro_slow_db'):
            must(token in proof, "Formality completeness " + token)

    for token in ('read_verilog $mapped_netlist', 'read_sdc $mapped_sdc',
                  'set_min_library $std_slow_db -min_version $std_fast_db',
                  'set_min_library $macro_slow_db -min_version $macro_fast_db',
                  '-max ssg0p9v125c', '-min ffg1p05vm40c', 'macro_count != 9',
                  'setup_slack < 0.0 || $hold_slack < 0.0',
                  'setup_tns_ns=0.0', 'hold_tns_ns=0.0',
                  'setup_violating_paths=0', 'hold_violating_paths=0',
                  'parasitics=none_no_read_parasitics_command', 'pt_eco=false'):
        must(token in pt, "PT completeness " + token)
    forbidden = re.compile(r'(^|\n)\s*(set_false_path|set_multicycle_path|set_min_delay|set_max_delay|set_disable_timing|set_case_analysis|fix_eco_timing|write_changes|read_parasitics)\b')
    must(forbidden.search(pt) is None, "PT exception/ECO")
    must(contract["frozen_timing_point"] == {"technology_nm": 28, "clock_period_ns": 3.0,
        "setup_uncertainty_ns": 0.2, "hold_uncertainty_ns": 0.05, "ideal_clock": True,
        "wireload": "ZeroWireload", "macro_cell": "TS1N28HPCPHVTB128X128M4S", "macro_count": 9,
        "timing_exception_counts": {"false_path": 0, "multicycle_path": 0, "min_delay": 0,
            "max_delay": 0, "disabled_timing_arc": 0, "case_analysis": 0}}, "timing point")

    mutations = mutation_results()
    out = {
        "status": "PASS", "python": sys.version.split()[0],
        "identity_checks": len(paths), "directory_seals": 6,
        "source_unit_tests_expected": 7, "negative_mutations_total": len(mutations),
        "negative_mutations_rejected": sum(1 for x in mutations if x["rejected"]),
        "eda_invoked": False, "attempt_created": False, "result_created": False,
        "transitive_chain": "RTL->M993(original SVF) AND M993->M1665(no SVF)",
        "prime_time": "independent mapped-Verilog+SDC slow/max fast/min; nine macros; no ECO/exceptions",
        "claim_boundary": "source review only; no Formality/PT result or paper claim",
        "mutations": mutations,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
