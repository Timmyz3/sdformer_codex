#!/usr/bin/env python3
"""M851/C2 R24 recursive exact-result seal guard.

The inherited base verifier intentionally supports either recursive sealing
without an exact set or an exact *flat* root set.  R24 adds a separate,
strict recursive verifier/publisher for the fixed 15-file C2 result shape;
it does not weaken or modify the inherited verifier.
"""

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
R848_DIR = HERE.parent / "verif_m848"
sys.path.insert(0, str(R848_DIR))
import m848_c2_r23_whitelist_guard as r848  # noqa: E402

base = r848.base

M850_DIR = "reviews/m850_m848_c2_r23_whitelist_source_fresh_hammer_r1_20260829"
M850_STATUS = (
    "FAIL_M848_R23_SOURCE_GATE__NO_RELEASE_AUTHORIZED__"
    "NESTED_EXACT_MEMBER_VERIFIER_REQUIRED"
)
M850_REVIEW_SHA256 = (
    "eef474a5aff01776c96eaa58a4f88d9020aae3bcd6ea193be90ca01ee01fda64"
)
M850_MANIFEST_SHA256 = (
    "2f99495ac7df3dd91b1c7e0c153212a9b605deec6d13a7fc64607582fd2ddb70"
)
M850_OUTER_SHA256 = (
    "75f0c2daaf7afbe66cea9d149dd0978d08f9094e1cda849eb1e9e9788b826ced"
)

SOURCE_HAMMER_STATUS = (
    "PASS100_M851_R24_RECURSIVE_SEAL_SOURCE__"
    "AUTHORIZE_ONE_FRESH_RELEASE_ONLY"
)
RELEASE_STATUS = (
    "AUTHORIZED_ONE_M851_R24_RECURSIVE_SEAL_CHANNEL_SPLIT_VCS_ATTEMPT"
)
FINAL_HAMMER_STATUS = (
    "PASS100_M851_R24_RECURSIVE_SEAL_FINAL_LAUNCH__"
    "ONE_VCS_ATTEMPT_AUTHORIZED"
)
FINAL_HAMMER_AUTHORIZATION = dict(base.FINAL_HAMMER_AUTHORIZATION)
RESULT_MEMBERS = tuple(r848.WHITELIST) + (
    "SHA256SUMS", "SHA256SUMS.seal.sha256",
)


def require_exact_mapping(actual, expected, label):
    base.require_exact_typed_mapping(actual, expected, label)


def _hash_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    value = hashlib.sha256()
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            break
        value.update(block)
    return value.hexdigest()


def _walk_fd(directory_fd, prefix, files, directories):
    for name in sorted(os.listdir(directory_fd)):
        base.require(name not in ("", ".", "..") and "/" not in name,
                     "unsafe recursive member name")
        relative = name if not prefix else prefix + "/" + name
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(value.st_mode):
            child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY |
                               os.O_NOFOLLOW, dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                base.require((opened.st_dev, opened.st_ino) ==
                             (value.st_dev, value.st_ino),
                             "recursive directory identity changed: " +
                             relative)
                directories.add(relative)
                _walk_fd(child_fd, relative, files, directories)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(value.st_mode):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW,
                         dir_fd=directory_fd)
            try:
                opened = os.fstat(fd)
                digest_first = _hash_fd(fd)
                after = os.fstat(fd)
                digest_second = _hash_fd(fd)
                base.require(
                    (value.st_dev, value.st_ino, value.st_size) ==
                    (opened.st_dev, opened.st_ino, opened.st_size) ==
                    (after.st_dev, after.st_ino, after.st_size),
                    "recursive file identity changed: " + relative)
                base.require(digest_first == digest_second,
                             "recursive file SHA changed: " + relative)
                files[relative] = {
                    "sha256": digest_first,
                    "dev": opened.st_dev,
                    "inode": opened.st_ino,
                    "size": opened.st_size,
                }
            finally:
                os.close(fd)
        else:
            raise base.Failure("nonregular recursive member: " + relative)


def _manifest_from_bytes(content):
    entries = {}
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise base.Failure("manifest is not UTF-8")
    for number, raw in enumerate(lines, 1):
        base.require(len(raw) >= 67 and raw[64:66] == "  ",
                     "malformed manifest line {}".format(number))
        digest = raw[:64]
        relative = raw[66:]
        base.require(all(ch in "0123456789abcdef" for ch in digest),
                     "noncanonical manifest digest")
        base._safe_relative(relative)
        base.require(relative not in entries,
                     "duplicate manifest member: " + relative)
        entries[relative] = digest
    base.require(entries, "empty recursive manifest")
    return entries


def _read_member_nofollow(directory, relative, expected_record):
    root_fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY |
                      os.O_NOFOLLOW)
    fd = None
    try:
        fd = r848._open_regular_nofollow(root_fd, relative)
        value = os.fstat(fd)
        chunks = []
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        content = b"".join(chunks)
        after = os.fstat(fd)
        actual = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "dev": value.st_dev,
            "inode": value.st_ino,
            "size": value.st_size,
        }
        base.require(actual == expected_record and
                     (value.st_dev, value.st_ino, value.st_size) ==
                     (after.st_dev, after.st_ino, after.st_size),
                     "recursive seal member path identity changed: " +
                     relative)
        return content
    finally:
        if fd is not None:
            os.close(fd)
        os.close(root_fd)


def verify_recursive_sealed_directory(directory, exact_members):
    directory = Path(directory)
    base.require(directory.is_dir() and not directory.is_symlink(),
                 "recursive sealed path must be nonsymlink directory")
    root_fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY |
                      os.O_NOFOLLOW)
    files = {}
    directories = set()
    try:
        root_identity_before = os.fstat(root_fd)
        _walk_fd(root_fd, "", files, directories)
        root_identity_after = os.fstat(root_fd)
        base.require((root_identity_before.st_dev, root_identity_before.st_ino) ==
                     (root_identity_after.st_dev, root_identity_after.st_ino),
                     "recursive root identity changed")
    finally:
        os.close(root_fd)
    expected_files = set(exact_members)
    expected_directories = set()
    for relative in expected_files:
        parts = relative.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            expected_directories.add("/".join(parts[:index]))
    base.require(set(files) == expected_files,
                 "recursive exact file population mismatch")
    base.require(directories == expected_directories,
                 "recursive exact directory population mismatch")
    manifest_bytes = _read_member_nofollow(
        directory, "SHA256SUMS", files["SHA256SUMS"])
    manifest = _manifest_from_bytes(manifest_bytes)
    expected_payloads = expected_files - {
        "SHA256SUMS", "SHA256SUMS.seal.sha256",
    }
    base.require(set(manifest) == expected_payloads,
                 "recursive manifest population mismatch")
    for relative, expected in manifest.items():
        base.require(files[relative]["sha256"] == expected,
                     "recursive manifest SHA drift: " + relative)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    outer_expected = manifest_sha + "  SHA256SUMS\n"
    outer_bytes = _read_member_nofollow(
        directory, "SHA256SUMS.seal.sha256",
        files["SHA256SUMS.seal.sha256"])
    base.require(outer_bytes == outer_expected.encode("utf-8"),
                 "recursive outer seal drift")
    base.require(files["SHA256SUMS"]["sha256"] == manifest_sha and
                 files["SHA256SUMS.seal.sha256"]["sha256"] ==
                 hashlib.sha256(outer_bytes).hexdigest(),
                 "recursive seal file walk identity drift")
    return {
        "manifest_sha256": manifest_sha,
        "outer_seal_file_sha256": hashlib.sha256(outer_bytes).hexdigest(),
        "member_count": len(manifest),
        "file_count_including_seals": len(files),
        "directory_count": len(directories),
    }


def publish_recursive_noreplace(source, destination, exact_members):
    source = Path(source)
    destination = Path(destination)
    before = verify_recursive_sealed_directory(source, exact_members)
    base._rename_noreplace(source, destination)
    base.require(not os.path.lexists(str(source)),
                 "recursive source remained after publication")
    after = verify_recursive_sealed_directory(destination, exact_members)
    base.require(before == after, "recursive published identity changed")
    return after


def verify_m850_authority(hw_root, contract):
    hw_root = Path(hw_root).resolve()
    directory = hw_root / M850_DIR
    identity = base.verify_sealed_directory(directory)
    base.regular_exact(directory / "review.json", M850_REVIEW_SHA256,
                       "M850 R23 negative source hammer")
    base.require(identity["manifest_sha256"] == M850_MANIFEST_SHA256 and
                 identity["outer_seal_file_sha256"] == M850_OUTER_SHA256,
                 "M850 negative hammer seal drift")
    review = base.strict_json(directory / "review.json")
    base.require(review.get("status") == M850_STATUS and
                 review.get("score_out_of_100") == 88 and
                 (review.get("p0_count"), review.get("p1_count"),
                  review.get("p2_count")) == (1, 0, 0) and
                 review.get("claim_boundary", {}).get(
                     "source_gate_passed") is False and
                 review.get("claim_boundary", {}).get(
                     "release_authorized") is False and
                 review.get("claim_boundary", {}).get(
                     "vcs_run_by_reviewer") is False,
                 "M850 negative status drift")
    expected = {
        "directory": M850_DIR,
        "review_sha256": M850_REVIEW_SHA256,
        "manifest_sha256": M850_MANIFEST_SHA256,
        "outer_seal_file_sha256": M850_OUTER_SHA256,
        "status": M850_STATUS,
        "score_out_of_100": 88,
        "p0_count": 1,
        "m848_release_authorized": False,
        "m848_launch_authorized": False,
        "required_successor": "M851_R24_RECURSIVE_EXACT_SEAL_VERIFIER",
    }
    require_exact_mapping(contract.get("m850_negative_source_authority"),
                          expected, "contract M850 authority drift")
    return identity


def validate_source(hw_root, contract_path, candidate_path, runner_path):
    source = r848.validate_source(hw_root, contract_path, candidate_path,
                                  runner_path)
    contract = base.strict_json(contract_path)
    authority = verify_m850_authority(hw_root, contract)
    source["m850_outer_seal_file_sha256"] = authority[
        "outer_seal_file_sha256"]
    source["status"] = "PASS_M851_R24_RECURSIVE_SEAL_SOURCE__NO_VCS_OR_EDA"
    return source


def expected_source_target(source):
    return {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "m850_outer_seal_file_sha256":
            source["m850_outer_seal_file_sha256"],
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
                 "R24 source hammer PASS100 semantics drift")
    require_exact_mapping(review.get("review_target"),
                          expected_source_target(source),
                          "R24 source hammer target drift")
    base.verify_double_sealed_file(release_path)
    release = base.strict_json(release_path)
    base.require(release.get("schema") ==
                 "m851_c2_r24_recursive_seal_vcs_launch_admission_v1" and
                 release.get("status") == RELEASE_STATUS,
                 "R24 release schema/status drift")
    require_exact_mapping(release.get("authorization"), {
        "launch_now": True, "run_vcs": True, "run_simv": True,
        "query_license": True, "run_eda": False, "max_attempts": 1,
    }, "R24 release authorization drift")
    require_exact_mapping(release.get("source_binding"), {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "source_hammer_outer_seal_file_sha256":
            source_identity["outer_seal_file_sha256"],
        "m850_outer_seal_file_sha256":
            source["m850_outer_seal_file_sha256"],
    }, "R24 release source binding drift")
    final_identity = base.verify_sealed_directory(final_hammer_dir)
    base.require(final_identity["outer_seal_file_sha256"] ==
                 expected_final_outer, "R24 final outer pin drift")
    final_review = base.strict_json(final_hammer_dir / "review.json")
    base.require(final_review.get("status") == FINAL_HAMMER_STATUS and
                 final_review.get("score_out_of_100") == 100 and
                 (final_review.get("p0_count"), final_review.get("p1_count"),
                  final_review.get("p2_count")) == (0, 0, 0),
                 "R24 final hammer PASS100 semantics drift")
    require_exact_mapping(final_review.get("authorization"),
                          FINAL_HAMMER_AUTHORIZATION,
                          "R24 final authorization drift")
    require_exact_mapping(final_review.get("review_target"), {
        "release_sha256": base.sha256(release_path),
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
    }, "R24 final hammer target drift")
    return {
        "status": "PASS_M851_R24_EXACT_LAUNCH_CHAIN",
        "release_sha256": base.sha256(release_path),
        "final_hammer_outer_seal_file_sha256": expected_final_outer,
        "source": source,
    }


def self_test():
    value = base.self_test()
    value["status"] = "PASS_M851_R24_RECURSIVE_SEAL_GUARD_SELF_TEST"
    return value


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    local = ("validate-source", "validate-launch-chain",
             "verify-recursive-sealed-directory",
             "publish-recursive-no-replace", "self-test")
    if command not in local:
        return r848.main()
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
    verify = sub.add_parser("verify-recursive-sealed-directory")
    verify.add_argument("--path", required=True)
    publish = sub.add_parser("publish-recursive-no-replace")
    publish.add_argument("--source", required=True)
    publish.add_argument("--destination", required=True)
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
    elif args.command == "verify-recursive-sealed-directory":
        value = verify_recursive_sealed_directory(args.path, RESULT_MEMBERS)
    elif args.command == "publish-recursive-no-replace":
        value = publish_recursive_noreplace(args.source, args.destination,
                                            RESULT_MEMBERS)
    else:
        value = self_test()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
