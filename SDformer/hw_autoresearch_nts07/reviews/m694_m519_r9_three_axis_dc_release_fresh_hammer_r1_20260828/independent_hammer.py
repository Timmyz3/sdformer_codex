#!/usr/bin/env python3
"""Independent static M694 hammer.  This script never invokes an EDA tool."""

from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_dc_m519_r9_setup_area_three_axis_exact_sha.sh"
CONTRACT = HW / "contracts/m519_r9_setup_area_three_axis_recovery_contract_r2_20260828.json"
ADMISSION = HW / "contracts/m519_r9_setup_area_three_axis_dc_launch_admission_r2_20260828.json"
M576 = HW / "reviews/m576_m519_r8_dc_launch_admission_candidate_hammer_r1_20260828"
M580 = HW / "reviews/m580_m519_r8_dc_final_launch_release_hammer_r1_20260828"
M693 = HW / "reviews/m693_m519_r9_three_axis_dc_release_author_handoff_r1_20260828"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def safe_member(name):
    member = PurePosixPath(name)
    require(not member.is_absolute() and member.parts and
            ".." not in member.parts, "unsafe seal member")
    return member


def verify_dir_seal(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "unsafe seal dir")
    sums = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="utf-8").strip().split() ==
            [sha256(sums), "SHA256SUMS"], "outer seal mismatch")
    for raw in sums.read_text(encoding="utf-8").splitlines():
        expected, name = raw.split(None, 1)
        member = directory / safe_member(name.strip()).as_posix()
        observed = os.lstat(str(member))
        require(stat.S_ISREG(observed.st_mode) and
                not stat.S_ISLNK(observed.st_mode) and
                sha256(member) == expected, "sealed member mismatch")
    return {"manifest_sha256": sha256(sums),
            "outer_seal_file_sha256": sha256(outer)}


def verify_file_seal(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sidecar.read_text(encoding="utf-8").strip().split() ==
            [sha256(path), path.name], "member sidecar mismatch")
    require(outer.read_text(encoding="utf-8").strip().split() ==
            [sha256(sidecar), sidecar.name], "file outer seal mismatch")


def main():
    contract = strict_json(CONTRACT)
    admission = strict_json(ADMISSION)
    m576 = strict_json(M576 / "review.json")
    m580 = strict_json(M580 / "review.json")
    handoff = strict_json(M693 / "handoff.json")
    verify_file_seal(CONTRACT)
    verify_file_seal(ADMISSION)
    seals = {"m576": verify_dir_seal(M576), "m580": verify_dir_seal(M580),
             "m693": verify_dir_seal(M693)}

    require(sha256(RUNNER) ==
            "0a1e1b0d2b391e45c43e0ec337a0b1114a407fc94be0d3d0ce37e103986e909c",
            "runner identity drift")
    require(sha256(CONTRACT) ==
            "74b13288e9bd13aa07feb68abc9f1f95b5255962bc80a6a1f759103f2608bf41",
            "contract identity drift")
    require(sha256(ADMISSION) ==
            "608a4afb0fe5a706a0f90700b2967231a5aeb3ef3ee6714a99f43d260d6242d3",
            "admission identity drift")
    require(subprocess.call(["/usr/bin/bash", "-n", str(RUNNER)]) == 0,
            "runner syntax failure")

    auth = admission["authorization"]
    require(auth == {"max_attempts": 1, "run_dc": True,
                     "run_formality": False, "run_pt": False,
                     "run_ptpx": False, "run_remote": False,
                     "run_vcs": False}, "authorization is not DC-only one-shot")
    identity = admission["identity"]
    require(identity["dc_runner_sha256"] == sha256(RUNNER) and
            identity["recovery_contract_sha256"] == sha256(CONTRACT) and
            identity["dc_runner_path"] ==
            "dc_handoff/scripts/run_dc_m519_r9_setup_area_three_axis_exact_sha.sh" and
            identity["recovery_contract_path"] ==
            "contracts/m519_r9_setup_area_three_axis_recovery_contract_r2_20260828.json",
            "runner/contract/admission SHA closure failure")
    require(contract["setup_area_flow"]["runner_sha256"] == sha256(RUNNER) and
            contract["setup_area_flow"]["clock_period_ns"] == 3.0 and
            contract["setup_area_flow"]["point_order"] == [
                {"id": "k1", "arch_mode": 0},
                {"id": "k8", "arch_mode": 1},
                {"id": "k1x8", "arch_mode": 2}],
            "three-axis flow drift")

    exact = contract["exact_files"]
    require(len(exact) == 17 and set(exact) == {
        "dc_handoff/scripts/run_dc_m519_r9_setup_area_three_axis_exact_sha.sh",
        "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl",
        "dc_handoff/filelists/date_m519_r5_channel_local_fault_three_axis_logic_only_dc.f",
        "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc",
        "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv",
        "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
        "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
        "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
        "rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv",
        "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
        "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
        "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k8_registered_release_8bank_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
        "rtl_m519/m519_fc2_registered_release_matched_8bank_raw4_acc24.sv",
        "docs/359_DATE终局冻结_20260813.md"}, "exact-file set drift")
    for name, expected in exact.items():
        path = HW / name
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == expected, "exact-file identity drift: " + name)

    provenance = admission["fresh_successor_provenance"]
    require(sha256(M576 / "review.json") == provenance["candidate_hammer_sha256"] and
            sha256(M576 / "SHA256SUMS") ==
            provenance["candidate_hammer_manifest_file_sha256"] and
            sha256(M576 / "SHA256SUMS.seal.sha256") ==
            provenance["candidate_hammer_outer_seal_file_sha256"] and
            provenance["candidate_hammer_status"] == m576["status"] ==
            "PASS_M553_M519_R8_DC_LAUNCH_ADMISSION_CANDIDATE_HAMMER" and
            m576["verdict"] == "PASS" and m576["score_out_of_100"] == 100 and
            m576["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0},
            "M576 exact successor provenance mismatch")
    require(sha256(HW / "contracts/m519_r8_setup_area_three_axis_dc_launch_admission_r1_20260827.json") ==
            admission["repair_provenance"]["immutable_r8_release_sha256"] and
            sha256(M580 / "review.json") ==
            admission["repair_provenance"]["immutable_m580_review_sha256"] and
            m580["status"] ==
            "FAIL_FINAL_RELEASE_HAMMER__M576_STATUS_PROVENANCE_MISMATCH__NO_DC_AUTHORIZED",
            "immutable R8/M580 provenance drift")

    # Existing VCS admission is read-only evidence; no VCS is invoked here.
    vcs_result = HW / "results/m519_r5_channel_local_fault_vcs_r1_20260827"
    vcs_review = HW / "reviews/m519_r5_channel_local_fault_vcs_receipt_blind_hammer_r1_20260827"
    verify_dir_seal(vcs_result)
    verify_dir_seal(vcs_review)
    vcs_receipt = strict_json(vcs_result /
        "m519_r5_channel_local_fault_vcs_receipt_r1.json")
    vcs_verdict = strict_json(vcs_review /
        "m519_r5_channel_local_fault_vcs_receipt_blind_hammer_verdict_r1.json")
    require(vcs_receipt["status"] ==
            "PASS_M519_R5_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_REVIEW" and
            vcs_verdict["status"] ==
            "PASS_VCS_RECEIPT_BLIND__DC_RUNNER_REPAIR_AND_NEW_STATIC_ADMISSION_ALLOWED__DC_NOT_AUTHORIZED" and
            vcs_verdict["severity_counts"]["p0"] == 0 and
            vcs_verdict["severity_counts"]["p1"] == 0,
            "existing VCS admission drift")

    tools = contract["tool_identity"]
    actual = Path(tools["dc_shell_actual_executable"])
    require(actual.read_bytes()[:4] == b"\x7fELF" and
            sha256(actual) == tools["dc_shell_actual_executable_sha256"],
            "actual DC executable is not the frozen ELF")
    require(Path(tools["dc_shell"]).resolve() ==
            Path(tools["dc_shell_wrapper"]).resolve() and
            sha256(tools["dc_shell"]) == tools["dc_shell_sha256"] and
            sha256(tools["dc_shell_wrapper"]) ==
            tools["dc_shell_wrapper_sha256"], "entry/wrapper identity drift")
    for key in ("slow", "fast"):
        path = Path(tools[key + "_library"])
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == tools[key + "_library_sha256"],
                key + " TSMC DB drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs359 drift")

    runner = RUNNER.read_text(encoding="utf-8")
    require("candidate_hammer_status' \"${m519_r9_admission}\")\" ==" in runner and
            "'.status' \"${m519_r9_m576}/review.json\")\"" in runner and
            "m519_r9_run_point k1 0" in runner and
            "m519_r9_run_point k8 1" in runner and
            "m519_r9_run_point k1x8 2" in runner and
            runner.count('"${m519_r9_dc}" -f') == 1,
            "runner static gate/axis/launch structure drift")
    canonical = HW / admission["unique_attempt"]["canonical_result_path"]
    attempt = HW / admission["unique_attempt"]["attempt_sentinel_path"]
    require(not canonical.exists() and not canonical.is_symlink() and
            not attempt.exists() and not attempt.is_symlink(),
            "R9 attempt identity already consumed")
    require(handoff["status"] == "READY_FOR_FRESH_M694_HAMMER__NO_EDA_EXECUTED",
            "author handoff status drift")

    result = {
        "schema": "m694_m519_r9_three_axis_dc_release_independent_hammer_v1",
        "status": "PASS_STATIC_RELEASE_CHAIN__LIVE_MACHINE_GATES_REPORTED_SEPARATELY",
        "runner_sha256": sha256(RUNNER),
        "contract_sha256": sha256(CONTRACT),
        "admission_sha256": sha256(ADMISSION),
        "m576_exact_status": m576["status"],
        "exact_files_rehashed": len(exact),
        "vcs_evidence": "PASS_EXISTING_SEALED_NO_VCS_RUN",
        "tool_identity": "PASS_ENTRY_WRAPPER_ACTUAL_ELF_AND_TWO_TSMC_DB",
        "axes": ["k1:ARCH_MODE0", "k8:ARCH_MODE1", "k1x8:ARCH_MODE2"],
        "clock_period_ns": 3.0,
        "authorization": auth,
        "canonical_absent": True,
        "attempt_absent": True,
        "seals": seals,
        "eda_executed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
