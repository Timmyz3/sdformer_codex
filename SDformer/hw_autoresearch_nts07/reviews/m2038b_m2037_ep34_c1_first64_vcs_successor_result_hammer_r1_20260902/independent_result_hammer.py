#!/usr/bin/env python3
"""Read-only independent result hammer for the sealed M2037 VCS successor."""

import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / (
    "results/m2037_m2031_ep34_c1_first64_model_rtl_calibration_"
    "vcs_successor_r1_20260902"
)
ATTEMPT = HW / (
    "results/.m2037_m2031_ep34_c1_first64_model_rtl_calibration_"
    "vcs_successor_attempt_consumed"
)
M2032 = HW / "reviews/m2032_m2031_ep34_c1_first64_model_rtl_calibration_source_hammer_r1_20260902"
M2034 = HW / "reviews/m2034_m2033_ep34_c1_first64_model_rtl_calibration_runner_source_hammer_r1_20260902"
M2035 = HW / "reviews/m2035_m2033_ep34_c1_first64_vcs_seal_failure_hammer_r1_20260902"
M2036 = HW / "reviews/m2036_m2037_ep34_c1_first64_model_rtl_calibration_successor_runner_source_hammer_r1_20260902"

EXPECTED_PASS = (
    "PASS_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION rows=64 active=64 "
    "input_nnz=565 residual_nnz=192 exact_parent_rows=4 issue=196 "
    "parent_edges=58 dead_elisions=31 macro_reads=54 macro_writes=33 "
    "forwards=4 deadline_holds=6 stalls=14 psum_commits=64 "
    "row_completions=64 numeric_commits=64 rtl_cycle_speedup=false "
    "full_network=false system_speedup=false"
)

PINS = {
    RESULT / "SHA256SUMS": "1f9f43bbadf503e6e874a803490e437f5e522e7d630dbb3b21926766dacef27e",
    RESULT / "SHA256SUMS.seal.sha256": "895c143c473169403216eb4426e9d64c2fca29f4c6b8a22e7a068c9a0a9c1dca",
    RESULT / "receipt.json": "46ee304d62004ffeb1719f7cbb450ff7e44d65461a859358c530a68f4481e162",
    RESULT / "generated_symlink_removal.json": "92bd21875017a32fc0216388f429dcbdf065b30346f3ee37d60f604ca92c69da",
    RESULT / "compile.log": "eddbe0c653d511628c3f03a409e53a1f13b15f6201ba2a9ede9530c3dd7e40bf",
    RESULT / "sim.log": "635b7d6227419d0952d9bf9a3bf08729bea559cfddde7616ee0d55c674b68f6e",
    M2032 / "review.json": "f0b6ce291ec25b52815db25c0bc8e76d87162c9b3821fa9d3b7eb3577bfa238a",
    M2034 / "review.json": "3eb091f8385e73745deea40e82cb4a04711b22f3b91e619692c5d0156b027544",
    M2035 / "review.json": "e3b8bffe5b9c0d33d326b5431ba79c9bcacec67527c4f996786cb5dd5f634654",
    M2036 / "review.json": "738e79dfd5f9880f2fa9983d895a69a085c460868f276ca8d36280a52d5890b1",
    M2036 / "launch_release.json": "c51bb1e520d61bf558410c5725fced81c17488aef6f8406366d6cc39642b9e1a",
    HW / "dc_handoff/scripts/run_vcs_m2033_m2031_ep34_c1_first64_model_rtl_calibration_one_shot.sh":
        "9ecfea0331368385421c2b7bfbf84d00fe9bf6f4d793f8fc07bfa2b25fc047b3",
    HW / "docs/359_DATE终局冻结_20260813.md":
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, label):
    if not value:
        raise AssertionError(label)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(token):
        raise ValueError("non-standard JSON token: " + token)

    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root")
    return value


def verify_tree(directory, expected_members=None):
    require(directory.is_dir() and not directory.is_symlink(), "real sealed directory")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest identity")
    require(outer.is_file() and not outer.is_symlink(), "outer identity")
    require(outer.read_text(encoding="utf-8").split() == [sha256(manifest), "SHA256SUMS"],
            "outer seal")
    listed = set()
    for row in manifest.read_text(encoding="utf-8").splitlines():
        fields = row.split(None, 1)
        require(len(fields) == 2, "manifest row")
        digest, relative = fields
        relative = relative.lstrip("*")
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts and relative not in listed,
                "manifest member path")
        target = directory / path
        require(target.is_file() and not target.is_symlink(), "manifest member identity")
        require(sha256(target) == digest, "manifest member SHA")
        listed.add(relative)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    require(not any(path.is_symlink() for path in directory.rglob("*")), "symlink in tree")
    require(actual == listed, "complete sealed topology")
    if expected_members is not None:
        require(len(listed) == expected_members, "sealed member count")
    return listed


def main():
    for path, expected in PINS.items():
        require(path.is_file() and not path.is_symlink(), "pinned file: " + str(path))
        require(sha256(path) == expected, "pinned SHA: " + str(path))

    result_members = verify_tree(RESULT, 96)
    require("receipt.json" in result_members and
            "generated_symlink_removal.json" in result_members and
            "simv.daidir/_2545240_archive_1.so" in result_members,
            "critical result members")
    verify_tree(ATTEMPT, 1)
    for upstream in (M2032, M2034, M2035, M2036):
        verify_tree(upstream)

    require((RESULT / "compile.rc").read_text(encoding="utf-8") == "0\n", "compile rc")
    require((RESULT / "sim.rc").read_text(encoding="utf-8") == "0\n", "sim rc")
    require((RESULT / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M2037_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_SUCCESSOR_PENDING_INDEPENDENT_REVIEW\n",
            "run complete token")
    sim = (RESULT / "sim.log").read_text(encoding="utf-8", errors="replace")
    compile_log = (RESULT / "compile.log").read_text(encoding="utf-8", errors="replace")
    require(sim.splitlines().count(EXPECTED_PASS) == 1, "exact terminal cardinality")
    forbidden = re.compile(
        r"(^|[^A-Za-z])(Error|Fatal|Assertion.*failed)|\$fatal|global watchdog expired|"
        r"counter mismatch|numeric mismatch|protocol_error", re.MULTILINE)
    require(forbidden.search(compile_log + "\n" + sim) is None, "functional error token")

    attempt_text = (ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text(encoding="utf-8")
    require(attempt_text == (
        "status=M2037_SUCCESSOR_ATTEMPT_CONSUMED\n"
        "vcs_compile_runs=1\nsimv_runs=1\nretry=false\n"), "attempt marker")
    siblings = HW / "results"
    require(len(list(siblings.glob(".m2037*attempt_consumed*"))) == 1, "unique attempt")
    require(not list(siblings.glob(".m2037*stage*")), "no private stage")
    require(not list(siblings.glob("m2037*quarantine*")), "no M2037 quarantine")

    removal = strict_json(RESULT / "generated_symlink_removal.json")
    require(removal == {
        "schema": "m2037_expected_vcs_archive_symlink_removal_r1_v1",
        "status": "RECORDED_AND_UNLINKED_EXPECTED_VCS_ARCHIVE_SYMLINK",
        "link_path": "csrc/_2545240_archive_1.so",
        "raw_target": ".//../simv.daidir//_2545240_archive_1.so",
        "resolved_target_path": "simv.daidir/_2545240_archive_1.so",
        "target_size_bytes": 573992,
        "target_sha256": "83632f8b4f001e977ce3ed4b263a672e7834caa02e9910ca48fb0324da64a144",
        "remaining_symlinks_after_unlink": 0,
    }, "exact symlink-removal record")
    target = RESULT / removal["resolved_target_path"]
    require(target.is_file() and not target.is_symlink(), "retained archive target")
    require(target.stat().st_size == removal["target_size_bytes"] and
            sha256(target) == removal["target_sha256"], "archive target size/SHA")
    require(not os.path.lexists(RESULT / removal["link_path"]), "removed link remains")

    receipt = strict_json(RESULT / "receipt.json")
    require(receipt.get("schema") ==
            "m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_receipt_r1_v1",
            "receipt schema")
    require(receipt.get("status") ==
            "PASS_M2037_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_SUCCESSOR_PENDING_INDEPENDENT_REVIEW",
            "receipt status")
    require(receipt.get("identity", {}).get("generated_symlink_removal_sha256") ==
            sha256(RESULT / "generated_symlink_removal.json"), "receipt removal binding")
    require(receipt.get("identity", {}).get("compile_log_sha256") == sha256(RESULT / "compile.log") and
            receipt.get("identity", {}).get("sim_log_sha256") == sha256(RESULT / "sim.log"),
            "receipt log bindings")
    require(receipt.get("execution") == {
        "automatic_retry": False, "macro_model": "foundry UNIT_DELAY functional",
        "simv_runs": 1, "vcs_compile_runs": 1}, "execution population")
    require(receipt.get("model_to_rtl_counts") == {
        "issue_accepts": 196, "parent_edges": 58, "dead_write_elisions": 31,
        "macro_reads": 54, "macro_writes": 33, "forwards": 4,
        "deadline_holds": 6, "issue_stalls": 14, "psum_commits": 64,
        "row_completions": 64, "numeric_commits": 64}, "receipt counts")
    require(receipt.get("payload_boundary") == {
        "masks": "real ep34 sealed-ledger prefix",
        "signed12_values": "synthetic deterministic function of source index and lane",
        "psum_prior": "all zero",
        "real_weight_or_real_psum_numeric_calibration": False}, "payload boundary")
    boundary = receipt.get("claim_boundary", {})
    require(boundary.get("single_real_tile_event_and_synthetic_numeric_calibration") is True and
            boundary.get("functional_vcs") is True and
            all(boundary.get(key) is False for key in (
                "cpu_model_1p694510x_upgraded_to_rtl", "rtl_cycle_speedup", "same_area",
                "timing", "power", "energy", "full_network", "system_speedup", "headline")),
            "claim boundary")

    expected_status = {
        M2032 / "review.json": "PASS_M2032_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_SOURCE_HAMMER",
        M2034 / "review.json": "PASS_M2034_M2033_RUNNER_SOURCE_HAMMER",
        M2035 / "review.json": "PASS_M2035_M2033_CANONICAL_SEAL_FAILURE_HAMMER",
        M2036 / "review.json": "PASS_M2036_M2037_SUCCESSOR_RUNNER_SOURCE_HAMMER",
    }
    for path, status in expected_status.items():
        require(strict_json(path).get("status") == status, "upstream status")
    release = strict_json(M2036 / "launch_release.json")
    require(release.get("status") ==
            "AUTHORIZED_EXACTLY_ONE_M2037_SUCCESSOR_VCS_COMPILE_AND_SIM" and
            release.get("execution_budget") == {
                "vcs_compile_runs": 1, "simv_runs": 1, "automatic_retry": False},
            "M2036 release")
    require(receipt["identity"]["successor_runner_review_sha256"] == sha256(M2036 / "review.json") and
            receipt["identity"]["launch_release_sha256"] == sha256(M2036 / "launch_release.json"),
            "receipt M2036 bindings")

    old = strict_json(M2035 / "review.json")
    old_canonical = siblings / "m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902"
    old_quarantine = list(siblings.glob(
        "m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902."
        "failed_or_incomplete.*.quarantine"))
    require(not os.path.lexists(old_canonical) and len(old_quarantine) == 1,
            "old M2033 permanent state")
    require(old["claim_boundary"]["old_functional_vcs_result_citable"] is False and
            old["failure_classification"]["old_attempt_reusable"] is False,
            "old M2033 cannot be cited")

    print(json.dumps({
        "status": "PASS_M2038B_M2037_EP34_C1_FIRST64_VCS_SUCCESSOR_RESULT_HAMMER",
        "score": 99,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 1},
        "result_members": len(result_members),
        "result_symlinks": 0,
        "exact_pass_lines": 1,
        "compile_rc": 0,
        "sim_rc": 0,
        "eda_launched": False,
        "gpu_launched": False,
        "license_query_launched": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
