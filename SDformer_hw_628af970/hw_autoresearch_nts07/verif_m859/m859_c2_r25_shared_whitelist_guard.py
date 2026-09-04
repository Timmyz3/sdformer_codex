#!/usr/bin/env python3
"""M859/C2 R25 single-whitelist result publication guard.

R25 centralizes the exact 15 payload keys and the receipt filename/schema/
status.  The real runner asks this guard to write the pending receipt, staging
uses the same whitelist, and recursive verification/publication uses the same
derived member set.  Hardware, VCS commands and R24 recursive safety remain
unchanged.  Python 3.6 compatible; this module never invokes EDA.
"""

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
R851_DIR = HERE.parent / "verif_m851"
sys.path.insert(0, str(R851_DIR))
import m851_c2_r24_recursive_seal_guard as r851  # noqa: E402

base = r851.base
r848 = r851.r848

M856_DIR = (
    "reviews/m856_m851_c2_r24_recursive_seal_source_fresh_hammer_r1_20260829"
)
M856_STATUS = (
    "FAIL_M851_R24_SOURCE_GATE__NO_RELEASE_AUTHORIZED__"
    "RUNNER_RECEIPT_NOT_IN_WHITELIST"
)
M856_REVIEW_SHA256 = (
    "829aa823f431dc64727e3f16efcd1a200fc9e797846b8b5030957400e9362f1b"
)
M856_MANIFEST_SHA256 = (
    "1115ae859642950870b6370acf61f50cd2780c66b0f4d098d9f598bd4c6b5903"
)
M856_OUTER_SHA256 = (
    "96fd220ea2061390dccb0563ce3e0592a5d6ea0d7f0b067146032e32eeccda67"
)

RECEIPT_FILENAME = "m859_c2_r25_shared_whitelist_vcs_receipt_r1.json"
RECEIPT_SCHEMA = "m859_c2_r25_shared_whitelist_vcs_receipt_v1"
RECEIPT_STATUS = (
    "PASS_M859_R25_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_HAMMER"
)
RUN_COMPLETE_STATUS = RECEIPT_STATUS

# The only payload-key authority for receipt writing checks, staging,
# recursive verification and recursive publication.
WHITELIST = (
    "RUN_COMPLETE.txt",
    "launch_identity.txt",
    RECEIPT_FILENAME,
    "attack/compile.log",
    "attack/compile.rc",
    "attack/sim.log",
    "attack/sim.rc",
    "attack/assert.report",
    "attack/assert.report.disablelog",
    "equalbw/compile.log",
    "equalbw/compile.rc",
    "equalbw/sim.log",
    "equalbw/sim.rc",
    "equalbw/assert.report",
    "equalbw/assert.report.disablelog",
)
RESULT_MEMBERS = WHITELIST + ("SHA256SUMS", "SHA256SUMS.seal.sha256")

SOURCE_HAMMER_STATUS = (
    "PASS100_M859_R25_SHARED_WHITELIST_SOURCE__"
    "AUTHORIZE_ONE_FRESH_RELEASE_ONLY"
)
RELEASE_STATUS = (
    "AUTHORIZED_ONE_M859_R25_SHARED_WHITELIST_CHANNEL_SPLIT_VCS_ATTEMPT"
)
FINAL_HAMMER_STATUS = (
    "PASS100_M859_R25_SHARED_WHITELIST_FINAL_LAUNCH__"
    "ONE_VCS_ATTEMPT_AUTHORIZED"
)
FINAL_HAMMER_AUTHORIZATION = dict(base.FINAL_HAMMER_AUTHORIZATION)


def require_exact_mapping(actual, expected, label):
    base.require_exact_typed_mapping(actual, expected, label)


def receipt_value(runner_sha256, contract_sha256, candidate_sha256,
                  release_sha256, final_outer_sha256):
    return {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "runner_sha256": runner_sha256,
        "contract_sha256": contract_sha256,
        "candidate_sha256": candidate_sha256,
        "release_sha256": release_sha256,
        "final_hammer_outer_seal_sha256": final_outer_sha256,
        "tool": "Synopsys VCS V-2023.12-SP1",
        "publication": {
            "source": "PRIVATE_VCS_WORK_WITH_TOOL_SYMLINKS_ALLOWED",
            "canonical": "EXACT_15_REGULAR_FILE_SHARED_WHITELIST_DOUBLE_SEALED",
            "source_dev_inode_size_sha_pre_post_stable": True,
            "shared_whitelist_authority":
                "verif_m859.m859_c2_r25_shared_whitelist_guard.WHITELIST",
        },
        "attack_contract": {
            "same_cycle_slot_reuse": 1,
            "ledger_conservation": True,
            "illegal_response_closes_both": True,
            "legal_response_survives_request_fault": True,
        },
        "exact_cycles": {
            "k8": [51, 131, 486, 1231, 14],
            "k1x8": [53, 133, 499, 1246, 14],
        },
        "frozen_k1_vs_k1x8": "SOURCE_SHA_BOUND_ONLY__NOT_RERUN_OR_CHANGED",
        "claim_boundary": {
            "vcs_validated": True,
            "dc": False,
            "ppa": False,
            "system_speedup": False,
            "headline": False,
            "paper_citable": False,
        },
    }


def write_pending_receipt(work, runner_sha256, contract_sha256,
                          candidate_sha256, release_sha256,
                          final_outer_sha256):
    work = Path(work)
    base.require(work.is_dir() and not work.is_symlink(),
                 "receipt work must be nonsymlink directory")
    for value in (runner_sha256, contract_sha256, candidate_sha256,
                  release_sha256, final_outer_sha256):
        base.require(len(value) == 64 and
                     all(ch in "0123456789abcdef" for ch in value),
                     "receipt SHA identity invalid")
    path = work / RECEIPT_FILENAME
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW, 0o600)
    try:
        payload = (json.dumps(receipt_value(
            runner_sha256, contract_sha256, candidate_sha256,
            release_sha256, final_outer_sha256), indent=2,
            sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            base.require(count > 0, "receipt short write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    value = base.strict_json(path)
    base.require(value.get("schema") == RECEIPT_SCHEMA and
                 value.get("status") == RECEIPT_STATUS,
                 "written receipt identity drift")
    return {
        "status": "PASS_M859_R25_PENDING_RECEIPT_WRITTEN",
        "filename": RECEIPT_FILENAME,
        "schema": RECEIPT_SCHEMA,
        "receipt_status": RECEIPT_STATUS,
        "sha256": base.sha256(path),
    }


def _validate_receipt_in_work(work):
    work = Path(work)
    path = work / RECEIPT_FILENAME
    value = base.strict_json(path)
    base.require(value.get("schema") == RECEIPT_SCHEMA and
                 value.get("status") == RECEIPT_STATUS,
                 "staged receipt filename/schema/status drift")


def stage_result_whitelist(work, stage):
    work = Path(work)
    stage = Path(stage)
    base.require(work.is_dir() and not work.is_symlink(),
                 "work must be nonsymlink directory")
    base.require(not os.path.lexists(str(stage)),
                 "private result stage already exists")
    _validate_receipt_in_work(work)
    stage.mkdir(mode=0o700)
    work_fd = os.open(str(work), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    records = []
    try:
        for relative in WHITELIST:
            records.append(r848._copy_one(work_fd, stage, relative))
    finally:
        os.close(work_fd)
    actual_files = set()
    actual_directories = set()
    for member in stage.rglob("*"):
        base.require(not member.is_symlink(), "symlink in R25 result stage")
        relative = member.relative_to(stage).as_posix()
        if member.is_file():
            actual_files.add(relative)
        elif member.is_dir():
            actual_directories.add(relative)
        else:
            raise base.Failure("nonregular R25 stage member: " + relative)
    base.require(actual_files == set(WHITELIST),
                 "R25 shared whitelist file mismatch")
    base.require(actual_directories == {"attack", "equalbw"},
                 "R25 shared whitelist directory mismatch")
    staged_receipt = base.strict_json(stage / RECEIPT_FILENAME)
    base.require(staged_receipt.get("schema") == RECEIPT_SCHEMA and
                 staged_receipt.get("status") == RECEIPT_STATUS,
                 "copied receipt identity drift")
    return {
        "status": "PASS_M859_R25_SHARED_WHITELIST_STAGED",
        "member_count": len(records),
        "receipt_filename": RECEIPT_FILENAME,
        "receipt_schema": RECEIPT_SCHEMA,
        "receipt_status": RECEIPT_STATUS,
        "symlinks": 0,
        "extras": 0,
    }


def verify_recursive_sealed_directory(path):
    value = r851.verify_recursive_sealed_directory(path, RESULT_MEMBERS)
    receipt = base.strict_json(Path(path) / RECEIPT_FILENAME)
    base.require(receipt.get("schema") == RECEIPT_SCHEMA and
                 receipt.get("status") == RECEIPT_STATUS,
                 "verified receipt filename/schema/status drift")
    return value


def publish_recursive_noreplace(source, destination):
    before = verify_recursive_sealed_directory(source)
    base._rename_noreplace(Path(source), Path(destination))
    base.require(not os.path.lexists(str(source)),
                 "R25 stage remained after publication")
    after = verify_recursive_sealed_directory(destination)
    base.require(before == after, "R25 published identity changed")
    return after


def verify_m856_authority(hw_root, contract):
    hw_root = Path(hw_root).resolve()
    directory = hw_root / M856_DIR
    identity = base.verify_sealed_directory(directory)
    base.regular_exact(directory / "review.json", M856_REVIEW_SHA256,
                       "M856 R24 negative source hammer")
    base.require(identity["manifest_sha256"] == M856_MANIFEST_SHA256 and
                 identity["outer_seal_file_sha256"] == M856_OUTER_SHA256,
                 "M856 negative hammer seal drift")
    review = base.strict_json(directory / "review.json")
    base.require(review.get("status") == M856_STATUS and
                 review.get("score_out_of_100") == 88 and
                 (review.get("p0_count"), review.get("p1_count"),
                  review.get("p2_count")) == (1, 0, 0) and
                 review.get("claim_boundary", {}).get(
                     "source_gate_passed") is False and
                 review.get("claim_boundary", {}).get(
                     "release_authorized") is False and
                 review.get("claim_boundary", {}).get(
                     "vcs_run_by_reviewer") is False,
                 "M856 negative semantics drift")
    expected = {
        "directory": M856_DIR,
        "review_sha256": M856_REVIEW_SHA256,
        "manifest_sha256": M856_MANIFEST_SHA256,
        "outer_seal_file_sha256": M856_OUTER_SHA256,
        "status": M856_STATUS,
        "score_out_of_100": 88,
        "p0_count": 1,
        "m851_release_authorized": False,
        "m851_launch_authorized": False,
        "required_successor": "M859_R25_SHARED_15_KEY_WHITELIST",
    }
    require_exact_mapping(contract.get("m856_negative_source_authority"),
                          expected, "contract M856 authority drift")
    return identity


def validate_source(hw_root, contract_path, candidate_path, runner_path):
    source = r851.validate_source(hw_root, contract_path, candidate_path,
                                  runner_path)
    contract = base.strict_json(contract_path)
    authority = verify_m856_authority(hw_root, contract)
    source["m856_outer_seal_file_sha256"] = authority[
        "outer_seal_file_sha256"]
    source["status"] = "PASS_M859_R25_SHARED_WHITELIST_SOURCE__NO_VCS_OR_EDA"
    return source


def expected_source_target(source):
    return {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "m856_outer_seal_file_sha256":
            source["m856_outer_seal_file_sha256"],
    }


def validate_launch_chain(hw_root, contract_path, candidate_path, runner_path,
                          source_hammer_dir, release_path, final_hammer_dir,
                          expected_final_outer):
    source = validate_source(hw_root, contract_path, candidate_path,
                             runner_path)
    source_hammer_dir = Path(source_hammer_dir).resolve()
    release_path = Path(release_path).resolve()
    final_hammer_dir = Path(final_hammer_dir).resolve()
    source_identity = base.verify_sealed_directory(source_hammer_dir)
    review = base.strict_json(source_hammer_dir / "review.json")
    base.require(review.get("status") == SOURCE_HAMMER_STATUS and
                 review.get("score_out_of_100") == 100 and
                 (review.get("p0_count"), review.get("p1_count"),
                  review.get("p2_count")) == (0, 0, 0),
                 "R25 source hammer semantics drift")
    require_exact_mapping(review.get("review_target"),
                          expected_source_target(source),
                          "R25 source hammer target drift")
    base.verify_double_sealed_file(release_path)
    release = base.strict_json(release_path)
    base.require(release.get("schema") ==
                 "m859_c2_r25_shared_whitelist_vcs_launch_admission_v1" and
                 release.get("status") == RELEASE_STATUS,
                 "R25 release schema/status drift")
    require_exact_mapping(release.get("authorization"), {
        "launch_now": True, "run_vcs": True, "run_simv": True,
        "query_license": True, "run_eda": False, "max_attempts": 1,
    }, "R25 release authorization drift")
    require_exact_mapping(release.get("source_binding"), {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "source_hammer_outer_seal_file_sha256":
            source_identity["outer_seal_file_sha256"],
        "m856_outer_seal_file_sha256":
            source["m856_outer_seal_file_sha256"],
    }, "R25 release source binding drift")
    final_identity = base.verify_sealed_directory(final_hammer_dir)
    base.require(final_identity["outer_seal_file_sha256"] ==
                 expected_final_outer, "R25 final outer pin drift")
    final_review = base.strict_json(final_hammer_dir / "review.json")
    base.require(final_review.get("status") == FINAL_HAMMER_STATUS and
                 final_review.get("score_out_of_100") == 100 and
                 (final_review.get("p0_count"), final_review.get("p1_count"),
                  final_review.get("p2_count")) == (0, 0, 0),
                 "R25 final hammer semantics drift")
    require_exact_mapping(final_review.get("authorization"),
                          FINAL_HAMMER_AUTHORIZATION,
                          "R25 final authorization drift")
    require_exact_mapping(final_review.get("review_target"), {
        "release_sha256": base.sha256(release_path),
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
    }, "R25 final hammer target drift")
    return {
        "status": "PASS_M859_R25_EXACT_LAUNCH_CHAIN",
        "release_sha256": base.sha256(release_path),
        "final_hammer_outer_seal_file_sha256": expected_final_outer,
        "source": source,
    }


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    local = (
        "write-pending-receipt", "stage-result-whitelist",
        "verify-recursive-sealed-directory", "publish-recursive-no-replace",
        "validate-source", "validate-launch-chain",
    )
    if command not in local:
        return r851.main()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    receipt = sub.add_parser("write-pending-receipt")
    receipt.add_argument("--work", required=True)
    for name in ("runner-sha256", "contract-sha256", "candidate-sha256",
                 "release-sha256", "final-hammer-outer-seal-sha256"):
        receipt.add_argument("--" + name, required=True)
    stage = sub.add_parser("stage-result-whitelist")
    stage.add_argument("--work", required=True)
    stage.add_argument("--stage", required=True)
    verify = sub.add_parser("verify-recursive-sealed-directory")
    verify.add_argument("--path", required=True)
    publish = sub.add_parser("publish-recursive-no-replace")
    publish.add_argument("--source", required=True)
    publish.add_argument("--destination", required=True)
    source = sub.add_parser("validate-source")
    source.add_argument("--hw-root", required=True)
    source.add_argument("--contract", required=True)
    source.add_argument("--candidate", required=True)
    source.add_argument("--runner", required=True)
    launch = sub.add_parser("validate-launch-chain")
    launch.add_argument("--hw-root", required=True)
    launch.add_argument("--contract", required=True)
    launch.add_argument("--candidate", required=True)
    launch.add_argument("--runner", required=True)
    launch.add_argument("--source-hammer", required=True)
    launch.add_argument("--release", required=True)
    launch.add_argument("--final-hammer", required=True)
    launch.add_argument("--expected-final-outer", required=True)
    args = parser.parse_args()
    if args.command == "write-pending-receipt":
        value = write_pending_receipt(
            args.work, args.runner_sha256, args.contract_sha256,
            args.candidate_sha256, args.release_sha256,
            args.final_hammer_outer_seal_sha256)
    elif args.command == "stage-result-whitelist":
        value = stage_result_whitelist(args.work, args.stage)
    elif args.command == "verify-recursive-sealed-directory":
        value = verify_recursive_sealed_directory(args.path)
    elif args.command == "publish-recursive-no-replace":
        value = publish_recursive_noreplace(args.source, args.destination)
    elif args.command == "validate-source":
        value = validate_source(args.hw_root, args.contract, args.candidate,
                                args.runner)
    else:
        value = validate_launch_chain(
            args.hw_root, args.contract, args.candidate, args.runner,
            args.source_hammer, args.release, args.final_hammer,
            args.expected_final_outer)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
