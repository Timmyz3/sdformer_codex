#!/usr/bin/env python3
"""M848/C2 R23 whitelist-publication and launch-identity guard.

Python 3.6 compatible.  VCS may create symlinks in its private work tree, but
only an exact set of regular control/evidence files may enter the canonical
result.  Each source is opened with O_NOFOLLOW and is stable in device,
inode, size and SHA-256 across the copy.  This module never invokes EDA.
"""

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
R837_DIR = HERE.parent / "verif_m837"
sys.path.insert(0, str(R837_DIR))
import m837_c2_r22_identity_compat_guard as r837  # noqa: E402

base = r837.base

M846_DIR = (
    "reviews/m846_m837_c2_r22_result_stage_seal_failure_hammer_r1_20260829"
)
M846_STATUS = (
    "PASS100_M837_R22_RESULT_STAGE_SEAL_FAILURE_HAMMER__"
    "ATTEMPT_CONSUMED__NO_CANONICAL_RESULT__NO_PERFORMANCE_CITATION"
)
M846_REVIEW_SHA256 = (
    "db20afd5659beb4016669371b73af19e299897fe45ae22214fd92a77440fef17"
)
M846_MANIFEST_SHA256 = (
    "9ef6666f9557f0d13fa43149e5c78245763540969c044f3b216a61e53ec9a532"
)
M846_OUTER_SHA256 = (
    "4357180e1c431639a90f91aacb42b8f967c010cb014f24e364c992fdc901acb5"
)
M837_ATTEMPT = "results/.m837_c2_r22_unicode_channel_split_vcs_attempt_consumed"
M837_ATTEMPT_JSON_SHA256 = (
    "87ff6c7e802f85ec9afbcab5649851d94c3dd90929f0ddf983be61f87e81c30a"
)
M837_ATTEMPT_MANIFEST_SHA256 = (
    "0c7fadf5555e6f7195d0af0f987ed7fcae778ad9a983ea0b983b4abf470a1b23"
)
M837_ATTEMPT_OUTER_SHA256 = (
    "94086ab67cf581701b3c2048610a453087454f298bf27145055e39bd93bff5f3"
)

SOURCE_HAMMER_STATUS = (
    "PASS100_M848_R23_WHITELIST_SOURCE__AUTHORIZE_ONE_FRESH_RELEASE_ONLY"
)
RELEASE_STATUS = "AUTHORIZED_ONE_M848_R23_WHITELIST_CHANNEL_SPLIT_VCS_ATTEMPT"
FINAL_HAMMER_STATUS = (
    "PASS100_M848_R23_WHITELIST_FINAL_LAUNCH__ONE_VCS_ATTEMPT_AUTHORIZED"
)
FINAL_HAMMER_AUTHORIZATION = dict(base.FINAL_HAMMER_AUTHORIZATION)

WHITELIST = (
    "RUN_COMPLETE.txt",
    "launch_identity.txt",
    "m848_c2_r23_whitelist_vcs_receipt_r1.json",
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


def require_exact_mapping(actual, expected, label):
    base.require_exact_typed_mapping(actual, expected, label)


def _stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_size)


def _hash_open_file(handle):
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def _open_regular_nofollow(root_fd, relative):
    parts = relative.split("/")
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                              os.O_NOFOLLOW, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW,
                     dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    value = os.fstat(fd)
    base.require(stat.S_ISREG(value.st_mode),
                 "whitelist source is not regular: " + relative)
    return fd


def _copy_one(work_fd, stage, relative):
    source_fd = _open_regular_nofollow(work_fd, relative)
    destination = stage / relative
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    destination_fd = None
    post_fd = None
    try:
        before = os.fstat(source_fd)
        with os.fdopen(os.dup(source_fd), "rb") as source:
            before_sha = _hash_open_file(source)
        destination_fd = os.open(str(destination), flags, 0o600)
        digest = hashlib.sha256()
        os.lseek(source_fd, 0, os.SEEK_SET)
        while True:
            block = os.read(source_fd, 1 << 20)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                base.require(written > 0, "short whitelist destination write")
                view = view[written:]
        os.fsync(destination_fd)
        copied_sha = digest.hexdigest()
        after = os.fstat(source_fd)
        with os.fdopen(os.dup(source_fd), "rb") as source:
            after_sha = _hash_open_file(source)
        post_fd = _open_regular_nofollow(work_fd, relative)
        path_after = os.fstat(post_fd)
        with os.fdopen(os.dup(post_fd), "rb") as source:
            path_after_sha = _hash_open_file(source)
        destination_stat = os.fstat(destination_fd)
        base.require(_stat_identity(before) == _stat_identity(after),
                     "source dev/inode/size changed during copy: " + relative)
        base.require(_stat_identity(before) == _stat_identity(path_after),
                     "source path identity changed during copy: " + relative)
        base.require(before_sha == copied_sha == after_sha == path_after_sha,
                     "source SHA changed during copy: " + relative)
        base.require(stat.S_ISREG(destination_stat.st_mode) and
                     destination_stat.st_size == before.st_size,
                     "destination regularity/size drift: " + relative)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if post_fd is not None:
            os.close(post_fd)
        os.close(source_fd)
    base.require(not destination.is_symlink() and
                 base.sha256(destination) == before_sha,
                 "destination SHA drift: " + relative)
    return {
        "path": relative,
        "source_dev": before.st_dev,
        "source_inode": before.st_ino,
        "size": before.st_size,
        "sha256": before_sha,
        "source_pre_post_stable": True,
        "destination_regular_nonsymlink": True,
    }


def stage_result_whitelist(work, stage):
    work = Path(work)
    stage = Path(stage)
    base.require(work.is_dir() and not work.is_symlink(),
                 "work must be a nonsymlink directory")
    base.require(not os.path.lexists(str(stage)),
                 "private result stage already exists")
    stage.mkdir(mode=0o700)
    work_fd = os.open(str(work), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    records = []
    try:
        for relative in WHITELIST:
            records.append(_copy_one(work_fd, stage, relative))
    finally:
        os.close(work_fd)
    actual = set()
    for member in stage.rglob("*"):
        base.require(not member.is_symlink(), "symlink in result stage")
        if member.is_file():
            actual.add(member.relative_to(stage).as_posix())
    base.require(actual == set(WHITELIST),
                 "private result stage whitelist mismatch")
    return {
        "status": "PASS_M848_R23_EXACT_REGULAR_WHITELIST_STAGED",
        "member_count": len(records),
        "members": records,
        "symlinks": 0,
        "extras": 0,
    }


def verify_m846_authority(hw_root, contract):
    hw_root = Path(hw_root).resolve()
    directory = hw_root / M846_DIR
    identity = base.verify_sealed_directory(directory)
    base.regular_exact(directory / "review.json", M846_REVIEW_SHA256,
                       "M846 R22 failure hammer")
    base.require(identity["manifest_sha256"] == M846_MANIFEST_SHA256 and
                 identity["outer_seal_file_sha256"] == M846_OUTER_SHA256,
                 "M846 failure hammer seal drift")
    review = base.strict_json(directory / "review.json")
    base.require(review.get("status") == M846_STATUS and
                 review.get("score_out_of_100") == 100 and
                 review.get("p0") == [] and review.get("p1") == [] and
                 review.get("p2") == [],
                 "M846 failure hammer semantics drift")
    boundary = review.get("failure_classification", {})
    base.require(boundary.get("attempt_consumed") is True and
                 boundary.get("formal_attempt_reusable") is False and
                 boundary.get("canonical_result_absent") is True and
                 boundary.get("failure_phase") == "RESULT_STAGE_SEAL",
                 "M846 spent-attempt boundary drift")
    expected = {
        "directory": M846_DIR,
        "review_sha256": M846_REVIEW_SHA256,
        "manifest_sha256": M846_MANIFEST_SHA256,
        "outer_seal_file_sha256": M846_OUTER_SHA256,
        "status": M846_STATUS,
        "m837_attempt_consumed": True,
        "m837_attempt_reusable": False,
        "m837_canonical_result_exists": False,
    }
    require_exact_mapping(contract.get("m846_spent_attempt_authority"),
                          expected, "contract M846 authority drift")
    attempt = hw_root / M837_ATTEMPT
    attempt_identity = base.verify_sealed_directory(attempt, {
        "attempt.json", "SHA256SUMS", "SHA256SUMS.seal.sha256",
    })
    base.regular_exact(attempt / "attempt.json", M837_ATTEMPT_JSON_SHA256,
                       "spent M837 attempt")
    base.require(attempt_identity["manifest_sha256"] ==
                 M837_ATTEMPT_MANIFEST_SHA256 and
                 attempt_identity["outer_seal_file_sha256"] ==
                 M837_ATTEMPT_OUTER_SHA256,
                 "spent M837 attempt seal drift")
    return identity


def validate_source(hw_root, contract_path, candidate_path, runner_path):
    source = r837.validate_source(hw_root, contract_path, candidate_path,
                                  runner_path)
    contract = base.strict_json(contract_path)
    authority = verify_m846_authority(hw_root, contract)
    source["m846_outer_seal_file_sha256"] = authority[
        "outer_seal_file_sha256"]
    source["status"] = "PASS_M848_R23_WHITELIST_SOURCE__NO_VCS_OR_EDA"
    return source


def expected_source_target(source):
    return {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "m846_outer_seal_file_sha256":
            source["m846_outer_seal_file_sha256"],
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
                 "R23 source hammer PASS100 semantics drift")
    require_exact_mapping(review.get("review_target"),
                          expected_source_target(source),
                          "R23 source hammer target drift")
    base.verify_double_sealed_file(release_path)
    release = base.strict_json(release_path)
    base.require(release.get("schema") ==
                 "m848_c2_r23_whitelist_vcs_launch_admission_v1" and
                 release.get("status") == RELEASE_STATUS,
                 "R23 release schema/status drift")
    require_exact_mapping(release.get("authorization"), {
        "launch_now": True, "run_vcs": True, "run_simv": True,
        "query_license": True, "run_eda": False, "max_attempts": 1,
    }, "R23 release authorization drift")
    require_exact_mapping(release.get("source_binding"), {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "source_hammer_outer_seal_file_sha256":
            source_identity["outer_seal_file_sha256"],
        "m846_outer_seal_file_sha256":
            source["m846_outer_seal_file_sha256"],
    }, "R23 release source binding drift")
    final_identity = base.verify_sealed_directory(final_hammer_dir)
    base.require(final_identity["outer_seal_file_sha256"] ==
                 expected_final_outer, "R23 final outer pin drift")
    final_review = base.strict_json(final_hammer_dir / "review.json")
    base.require(final_review.get("status") == FINAL_HAMMER_STATUS and
                 final_review.get("score_out_of_100") == 100 and
                 (final_review.get("p0_count"), final_review.get("p1_count"),
                  final_review.get("p2_count")) == (0, 0, 0),
                 "R23 final hammer PASS100 semantics drift")
    require_exact_mapping(final_review.get("authorization"),
                          FINAL_HAMMER_AUTHORIZATION,
                          "R23 final authorization drift")
    require_exact_mapping(final_review.get("review_target"), {
        "release_sha256": base.sha256(release_path),
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
    }, "R23 final hammer target drift")
    return {
        "status": "PASS_M848_R23_EXACT_LAUNCH_CHAIN",
        "release_sha256": base.sha256(release_path),
        "final_hammer_outer_seal_file_sha256": expected_final_outer,
        "source": source,
    }


def self_test():
    result = base.self_test()
    result["status"] = "PASS_M848_R23_WHITELIST_GUARD_SELF_TEST"
    return result


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command not in ("validate-source", "validate-launch-chain",
                       "stage-result-whitelist", "self-test"):
        return base.main()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
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
    stage = sub.add_parser("stage-result-whitelist")
    stage.add_argument("--work", required=True)
    stage.add_argument("--stage", required=True)
    sub.add_parser("self-test")
    args = parser.parse_args()
    base.require(args.command is not None, "missing command")
    if args.command == "validate-source":
        value = validate_source(args.hw_root, args.contract, args.candidate,
                                args.runner)
    elif args.command == "validate-launch-chain":
        value = validate_launch_chain(
            args.hw_root, args.contract, args.candidate, args.runner,
            args.source_hammer, args.release, args.final_hammer,
            args.expected_final_outer)
    elif args.command == "stage-result-whitelist":
        value = stage_result_whitelist(args.work, args.stage)
    else:
        value = self_test()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
