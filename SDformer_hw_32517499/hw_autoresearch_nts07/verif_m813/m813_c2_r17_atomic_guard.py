#!/usr/bin/env python3
"""M813/C2 R17 source and atomic-publication guard.

Python 3.6 compatible.  This module never invokes VCS, simv, lmutil, or EDA.
It provides strict JSON parsing, double-seal verification, Linux
renameat2(RENAME_NOREPLACE), flat attempt creation, and sealed failure receipts.
"""

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile


AT_FDCWD = -100
RENAME_NOREPLACE = 1
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M811_REVIEW_DIR = "reviews/m803_c2_r16_channel_split_source_fresh_hammer_r1_20260828"
M811_REVIEW_SHA256 = "6c844bf92c0bd66753dc56e03f1fb4e840e156aff914311938a56e366bb14d0e"
M811_MANIFEST_SHA256 = "1eccd30053f3ab386e1a38adc8324c3f2c8399c35a298f856279c4fe4cb9a245"
M811_OUTER_SEAL_FILE_SHA256 = "809b089f8df6ba30f44f4057ddff7617b8ecddc3b5add7a2dccf22cb265a3cba"
SOURCE_HAMMER_STATUS = "PASS100_M813_R17_ATOMIC_SOURCE__AUTHORIZE_ONE_RELEASE_ONLY"
RELEASE_STATUS = "AUTHORIZED_ONE_M813_R17_ATOMIC_CHANNEL_SPLIT_VCS_ATTEMPT"
FINAL_HAMMER_STATUS = "PASS100_M813_R17_FINAL_LAUNCH__ONE_VCS_ATTEMPT_AUTHORIZED"


class Failure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Failure("duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def reject_constant(value):
    raise Failure("non-finite JSON constant: " + value)


def strict_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),
            "JSON must be a regular nonsymlink file: " + str(path))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate_pairs,
                         parse_constant=reject_constant)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n",
                          encoding="utf-8")


def regular_exact(path, expected, label):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),
            label + " must be a regular nonsymlink file")
    require(len(expected) == 64 and sha256(path) == expected,
            label + " SHA drift")


def _safe_relative(name):
    rel = Path(name)
    require(name and not rel.is_absolute() and "\x00" not in name,
            "unsafe manifest member: " + repr(name))
    require(all(part not in ("", ".", "..") for part in rel.parts),
            "unsafe manifest member: " + repr(name))
    require(name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
            "manifest recursively names seal file")
    return rel


def _manifest_entries(directory):
    manifest = Path(directory) / "SHA256SUMS"
    require(manifest.is_file() and not manifest.is_symlink(),
            "missing regular SHA256SUMS")
    entries = {}
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        require(len(raw) >= 67 and raw[64:66] == "  ",
                "malformed manifest line {}".format(number))
        digest = raw[:64]
        require(all(ch in "0123456789abcdef" for ch in digest),
                "noncanonical digest on manifest line {}".format(number))
        name = raw[66:]
        rel = _safe_relative(name)
        require(name not in entries, "duplicate manifest member: " + name)
        entries[name] = digest
        member = Path(directory) / rel
        require(member.is_file() and not member.is_symlink(),
                "manifest member is missing/nonregular: " + name)
        require(sha256(member) == digest, "manifest member SHA drift: " + name)
    require(entries, "empty sealed manifest")
    return entries


def verify_sealed_directory(directory, exact_root_members=None):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed path must be a nonsymlink directory")
    entries = _manifest_entries(directory)
    outer = directory / "SHA256SUMS.seal.sha256"
    require(outer.is_file() and not outer.is_symlink(),
            "missing regular outer seal")
    expected_outer = sha256(directory / "SHA256SUMS") + "  SHA256SUMS\n"
    require(outer.read_text(encoding="utf-8") == expected_outer,
            "outer seal drift")
    actual = set()
    for member in directory.rglob("*"):
        require(not member.is_symlink(), "symlink in sealed directory")
        if member.is_file():
            actual.add(member.relative_to(directory).as_posix())
    expected = set(entries) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    require(actual == expected,
            "sealed population mismatch actual={} expected={}".format(
                sorted(actual), sorted(expected)))
    if exact_root_members is not None:
        require(actual == set(exact_root_members),
                "flat root member mismatch")
        require(all("/" not in name for name in actual),
                "flat root contains nested member")
    return {
        "manifest_sha256": sha256(directory / "SHA256SUMS"),
        "outer_seal_file_sha256": sha256(outer),
        "member_count": len(entries),
    }


def verify_double_sealed_file(path):
    path = Path(path)
    regular_exact(path, sha256(path), "sealed payload")
    inner = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(inner.is_file() and not inner.is_symlink(), "missing file inner seal")
    require(outer.is_file() and not outer.is_symlink(), "missing file outer seal")
    require(inner.read_text(encoding="utf-8") ==
            sha256(path) + "  " + path.name + "\n", "file inner seal drift")
    require(outer.read_text(encoding="utf-8") ==
            sha256(inner) + "  " + inner.name + "\n", "file outer seal drift")
    return {
        "payload_sha256": sha256(path),
        "outer_seal_file_sha256": sha256(outer),
    }


def seal_directory(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "seal target must be a nonsymlink directory")
    require(not (directory / "SHA256SUMS").exists() and
            not (directory / "SHA256SUMS.seal.sha256").exists(),
            "seal files already exist")
    members = []
    for member in directory.rglob("*"):
        require(not member.is_symlink(), "symlink in seal target")
        if member.is_file():
            rel = member.relative_to(directory).as_posix()
            _safe_relative(rel)
            members.append((rel, sha256(member)))
    require(members, "cannot seal empty directory")
    members.sort()
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(digest, rel)
                                for rel, digest in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        sha256(manifest) + "  SHA256SUMS\n", encoding="utf-8")
    return verify_sealed_directory(directory)


def _rename_noreplace(source, destination):
    source = Path(source)
    destination = Path(destination)
    require(source.parent.resolve() == destination.parent.resolve(),
            "atomic publication requires sibling source/destination")
    require(source.exists() and not source.is_symlink(),
            "publication source missing or symlink")
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "renameat2"),
            "Linux renameat2(RENAME_NOREPLACE) unavailable")
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(AT_FDCWD, os.fsencode(str(source)), AT_FDCWD,
                       os.fsencode(str(destination)), RENAME_NOREPLACE)
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise Failure("atomic publication destination collision: " +
                          str(destination))
        raise Failure("renameat2(RENAME_NOREPLACE) failed errno={}: {}".format(
            code, os.strerror(code)))


def publish_noreplace(source, destination, exact_root_members=None):
    source = Path(source)
    destination = Path(destination)
    source_identity = verify_sealed_directory(source, exact_root_members)
    _rename_noreplace(source, destination)
    require(not source.exists() and not source.is_symlink(),
            "source remained after atomic publication")
    destination_identity = verify_sealed_directory(destination,
                                                   exact_root_members)
    require(source_identity == destination_identity,
            "published identity changed")
    return destination_identity


def create_attempt_stage(stage, identity):
    stage = Path(stage)
    require(not os.path.lexists(str(stage)), "attempt stage already exists")
    stage.mkdir(mode=0o700)
    require(set(identity) == {
        "schema", "status", "runner_sha256", "contract_sha256",
        "candidate_sha256", "release_sha256", "final_hammer_outer_seal_sha256",
        "claim_boundary",
    }, "attempt identity key set drift")
    require(identity["schema"] == "m813_c2_r17_atomic_vcs_attempt_v1" and
            identity["status"] == "ONE_M813_R17_VCS_ATTEMPT_CONSUMED" and
            identity["claim_boundary"] == {
                "vcs_complete": False, "paper_citable": False,
                "system_speedup": False,
            }, "attempt identity semantics drift")
    for key in ("runner_sha256", "contract_sha256", "candidate_sha256",
                "release_sha256", "final_hammer_outer_seal_sha256"):
        require(len(identity[key]) == 64 and
                all(ch in "0123456789abcdef" for ch in identity[key]),
                "attempt SHA field invalid: " + key)
    write_json(stage / "attempt.json", identity)
    seal_directory(stage)
    return verify_sealed_directory(stage, {
        "attempt.json", "SHA256SUMS", "SHA256SUMS.seal.sha256",
    })


def attempt_identity(runner_sha256, contract_sha256, candidate_sha256,
                     release_sha256, final_hammer_outer_seal_sha256):
    return {
        "schema": "m813_c2_r17_atomic_vcs_attempt_v1",
        "status": "ONE_M813_R17_VCS_ATTEMPT_CONSUMED",
        "runner_sha256": runner_sha256,
        "contract_sha256": contract_sha256,
        "candidate_sha256": candidate_sha256,
        "release_sha256": release_sha256,
        "final_hammer_outer_seal_sha256": final_hammer_outer_seal_sha256,
        "claim_boundary": {
            "vcs_complete": False,
            "paper_citable": False,
            "system_speedup": False,
        },
    }


def verify_attempt(path, expected):
    identity = verify_sealed_directory(path, {
        "attempt.json", "SHA256SUMS", "SHA256SUMS.seal.sha256",
    })
    value = strict_json(Path(path) / "attempt.json")
    require(value == expected, "attempt identity content drift")
    return identity


def _read_log(path):
    if not path:
        return "no runner log was available\n"
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        return "runner log was absent or nonregular: {}\n".format(path)
    return path.read_text(encoding="utf-8", errors="replace")


def write_failure_quarantine(parent, primary_name, metadata, runner_log=None):
    parent = Path(parent)
    require(parent.is_dir() and not parent.is_symlink(),
            "failure quarantine parent is invalid")
    require("/" not in primary_name and primary_name not in ("", ".", ".."),
            "failure quarantine primary name is unsafe")
    require(metadata.get("schema") == "m813_c2_r17_failure_receipt_v1" and
            metadata.get("status") ==
            "FAILED_OR_INCOMPLETE_DO_NOT_CITE_PERFORMANCE" and
            metadata.get("claim_boundary") == {
                "failure_boundary_citable": True,
                "paper_performance_citable": False,
                "vcs_complete": False,
                "system_speedup": False,
            }, "failure metadata semantics drift")
    stage = Path(tempfile.mkdtemp(prefix=".m813_failure_stage.", dir=str(parent)))
    log_text = _read_log(runner_log)
    (stage / "driver.log").write_text(log_text, encoding="utf-8")
    value = dict(metadata)
    value["driver_log_sha256"] = sha256(stage / "driver.log")
    write_json(stage / "failure.json", value)
    seal_directory(stage)
    exact = {"failure.json", "driver.log", "SHA256SUMS",
             "SHA256SUMS.seal.sha256"}
    verify_sealed_directory(stage, exact)
    names = [primary_name]
    names.extend(primary_name + ".collision_fallback.{}.{}".format(os.getpid(), index)
                 for index in range(64))
    collisions = []
    for name in names:
        destination = parent / name
        try:
            publish_noreplace(stage, destination, exact)
            return {
                "status": "PASS_SEALED_FAILURE_QUARANTINE_PUBLISHED",
                "path": str(destination),
                "collision_count": len(collisions),
                "identity": verify_sealed_directory(destination, exact),
            }
        except Failure as error:
            if "destination collision" not in str(error):
                raise
            collisions.append(str(destination))
    raise Failure("all failure quarantine destinations collided; sealed stage remains: " +
                  str(stage))


def _contained(root, rel):
    root = Path(root).resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise Failure("source path escapes root: " + rel)
    return path


def validate_source(hw_root, contract_path, candidate_path, runner_path):
    hw_root = Path(hw_root).resolve()
    contract_path = Path(contract_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    runner_path = Path(runner_path).resolve()
    verify_double_sealed_file(contract_path)
    verify_double_sealed_file(candidate_path)
    contract = strict_json(contract_path)
    candidate = strict_json(candidate_path)
    require(contract.get("schema") == "m813_c2_r17_atomic_source_only_contract_v1" and
            contract.get("status") == "SOURCE_ONLY_M813_R17__NO_VCS_AUTHORIZATION",
            "source contract status/schema drift")
    require(contract.get("authorization") == {
        "launch_now": False, "run_vcs": False, "run_simv": False,
        "query_license": False, "run_eda": False,
        "create_attempt": False, "create_result": False,
        "author_true_release": False,
    }, "source contract authorization drift")
    regular_exact(runner_path, contract["runner_sha256"], "M813 runner")
    regular_exact(candidate_path, contract["candidate_sha256"], "M813 candidate")
    require(candidate.get("schema") ==
            "m813_c2_r17_atomic_vcs_launch_candidate_source_only_v1" and
            candidate.get("status") ==
            "SOURCE_ONLY_M813_R17_VCS_CANDIDATE__FRESH_HAMMER_REQUIRED__NO_LAUNCH",
            "source candidate status/schema drift")
    require(candidate.get("authorization") == {
        "source_validation": True, "atomic_self_tests": True,
        "fresh_source_hammer": True, "author_true_release": False,
        "launch_now": False, "run_vcs": False, "run_eda": False,
        "create_attempt": False, "create_result": False,
    }, "source candidate authorization drift")
    require(candidate.get("authoritative_contract_path") ==
            contract_path.relative_to(hw_root).as_posix(),
            "candidate contract path drift")
    require(candidate.get("runner_path") ==
            runner_path.relative_to(hw_root).as_posix(),
            "candidate runner path drift")
    require(candidate.get("frozen_exact_cycles") == {
        "k8": [51, 131, 486, 1231, 14],
        "k1x8": [53, 133, 499, 1246, 14],
    }, "five exact cycle gates drift")
    require(candidate.get("claim_boundary") == {
        "source_only": True, "rtl_modified": False,
        "vcs_validated": False, "dc_validated": False,
        "ppa_ready": False, "system_speedup": False,
        "paper_citable": False,
    }, "candidate claim boundary drift")
    source_map = contract.get("source_sha256")
    require(isinstance(source_map, dict) and len(source_map) >= 20,
            "source SHA map is incomplete")
    for rel, expected in source_map.items():
        path = _contained(hw_root, rel)
        regular_exact(path, expected, "source " + rel)
    require(sha256(hw_root / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    m811_dir = hw_root / M811_REVIEW_DIR
    m811_identity = verify_sealed_directory(m811_dir)
    regular_exact(m811_dir / "review.json", M811_REVIEW_SHA256, "M811 review")
    require(m811_identity["manifest_sha256"] == M811_MANIFEST_SHA256 and
            m811_identity["outer_seal_file_sha256"] ==
            M811_OUTER_SEAL_FILE_SHA256, "M811 double seal drift")
    m811 = strict_json(m811_dir / "review.json")
    require(m811.get("status") ==
            "FAIL_M803_R16_SOURCE_GATE__NO_VCS_CANDIDATE_AUTHORIZED__ADDITIVE_RUNNER_REPAIR_REQUIRED" and
            m811.get("score_out_of_100") == 92 and
            (m811.get("p0_count"), m811.get("p1_count"), m811.get("p2_count")) ==
            (1, 2, 0), "M811 repair authority drift")
    require(contract.get("m811_repair_authority") == {
        "directory": M811_REVIEW_DIR,
        "review_sha256": M811_REVIEW_SHA256,
        "manifest_sha256": M811_MANIFEST_SHA256,
        "outer_seal_file_sha256": M811_OUTER_SEAL_FILE_SHA256,
    }, "contract M811 authority binding drift")
    return {
        "status": "PASS_M813_R17_SOURCE_IDENTITY__NO_VCS_OR_EDA",
        "runner_sha256": sha256(runner_path),
        "contract_sha256": sha256(contract_path),
        "candidate_sha256": sha256(candidate_path),
        "source_count": len(source_map),
        "docs359_sha256": DOCS359_SHA256,
    }


def validate_launch_chain(hw_root, contract_path, candidate_path, runner_path,
                          source_hammer_dir, release_path, final_hammer_dir,
                          expected_final_outer):
    source = validate_source(hw_root, contract_path, candidate_path, runner_path)
    hw_root = Path(hw_root).resolve()
    source_hammer_dir = Path(source_hammer_dir).resolve()
    release_path = Path(release_path).resolve()
    final_hammer_dir = Path(final_hammer_dir).resolve()
    source_hammer_identity = verify_sealed_directory(source_hammer_dir)
    source_review = strict_json(source_hammer_dir / "review.json")
    require(source_review.get("status") == SOURCE_HAMMER_STATUS and
            source_review.get("score_out_of_100") == 100 and
            (source_review.get("p0_count"), source_review.get("p1_count"),
             source_review.get("p2_count")) == (0, 0, 0),
            "source hammer PASS100 semantics drift")
    require(source_review.get("review_target") == {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
    }, "source hammer target binding drift")
    verify_double_sealed_file(release_path)
    release = strict_json(release_path)
    require(release.get("schema") == "m813_c2_r17_atomic_vcs_launch_admission_v1" and
            release.get("status") == RELEASE_STATUS and
            release.get("authorization") == {
                "launch_now": True, "run_vcs": True, "run_simv": True,
                "query_license": True, "run_eda": False,
                "max_attempts": 1,
            }, "true release semantics drift")
    require(release.get("source_binding") == {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "source_hammer_outer_seal_file_sha256":
            source_hammer_identity["outer_seal_file_sha256"],
    }, "true release source binding drift")
    final_identity = verify_sealed_directory(final_hammer_dir)
    require(final_identity["outer_seal_file_sha256"] == expected_final_outer,
            "caller final-hammer outer seal pin drift")
    final_review = strict_json(final_hammer_dir / "review.json")
    require(final_review.get("status") == FINAL_HAMMER_STATUS and
            final_review.get("score_out_of_100") == 100 and
            (final_review.get("p0_count"), final_review.get("p1_count"),
             final_review.get("p2_count")) == (0, 0, 0) and
            final_review.get("authorization", {}).get("launch_now") is True,
            "final hammer PASS100 semantics drift")
    require(final_review.get("review_target") == {
        "release_sha256": sha256(release_path),
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
    }, "final hammer target binding drift")
    return {
        "status": "PASS_M813_R17_EXACT_LAUNCH_CHAIN",
        "release_sha256": sha256(release_path),
        "final_hammer_outer_seal_file_sha256": expected_final_outer,
        "source": source,
    }


def self_test():
    with tempfile.TemporaryDirectory(prefix="m813_guard_selftest.") as raw:
        root = Path(raw)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"status":"A","status":"B"}\n', encoding="utf-8")
        duplicate_rejected = False
        try:
            strict_json(duplicate)
        except Failure:
            duplicate_rejected = True
        stage = root / "stage"
        stage.mkdir()
        (stage / "sentinel").write_text("sealed\n", encoding="utf-8")
        seal_directory(stage)
        target = root / "target"
        target.mkdir()
        (target / "attacker").write_text("preserve\n", encoding="utf-8")
        collision_rejected = False
        try:
            publish_noreplace(stage, target)
        except Failure:
            collision_rejected = True
        require((target / "attacker").read_text(encoding="utf-8") ==
                "preserve\n" and not (target / "stage").exists(),
                "collision polluted canonical destination")
        return {
            "status": "PASS_M813_ATOMIC_GUARD_SELF_TEST",
            "duplicate_json_rejected": duplicate_rejected,
            "renameat2_collision_rejected": collision_rejected,
            "canonical_unpolluted": True,
            "vcs_executed": False,
            "eda_executed": False,
        }


def main():
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

    attempt = sub.add_parser("create-attempt-stage")
    attempt.add_argument("--stage", required=True)
    for name in ("runner-sha256", "contract-sha256", "candidate-sha256",
                 "release-sha256", "final-hammer-outer-seal-sha256"):
        attempt.add_argument("--" + name, required=True)

    verify_attempt_parser = sub.add_parser("verify-attempt")
    verify_attempt_parser.add_argument("--path", required=True)
    for name in ("runner-sha256", "contract-sha256", "candidate-sha256",
                 "release-sha256", "final-hammer-outer-seal-sha256"):
        verify_attempt_parser.add_argument("--" + name, required=True)

    seal = sub.add_parser("seal-directory")
    seal.add_argument("--path", required=True)

    verify = sub.add_parser("verify-sealed-directory")
    verify.add_argument("--path", required=True)
    verify.add_argument("--exact-root-members", default="")

    publish = sub.add_parser("publish-no-replace")
    publish.add_argument("--source", required=True)
    publish.add_argument("--destination", required=True)
    publish.add_argument("--exact-root-members", default="")

    failure = sub.add_parser("write-failure-quarantine")
    failure.add_argument("--parent", required=True)
    failure.add_argument("--primary-name", required=True)
    failure.add_argument("--phase", required=True)
    failure.add_argument("--return-code", required=True, type=int)
    failure.add_argument("--attempt-consumed", required=True,
                         choices=("true", "false"))
    failure.add_argument("--runner-sha256", required=True)
    failure.add_argument("--contract-sha256", required=True)
    failure.add_argument("--candidate-sha256", required=True)
    failure.add_argument("--release-sha256", required=True)
    failure.add_argument("--final-hammer-outer-seal-sha256", required=True)
    failure.add_argument("--runner-log", default="")

    sub.add_parser("self-test")
    args = parser.parse_args()
    require(args.command is not None, "missing command")

    if args.command == "validate-source":
        result = validate_source(args.hw_root, args.contract, args.candidate,
                                 args.runner)
    elif args.command == "validate-launch-chain":
        result = validate_launch_chain(
            args.hw_root, args.contract, args.candidate, args.runner,
            args.source_hammer, args.release, args.final_hammer,
            args.expected_final_outer)
    elif args.command == "create-attempt-stage":
        identity = attempt_identity(
            args.runner_sha256, args.contract_sha256, args.candidate_sha256,
            args.release_sha256, args.final_hammer_outer_seal_sha256)
        result = create_attempt_stage(args.stage, identity)
    elif args.command == "verify-attempt":
        identity = attempt_identity(
            args.runner_sha256, args.contract_sha256, args.candidate_sha256,
            args.release_sha256, args.final_hammer_outer_seal_sha256)
        result = verify_attempt(args.path, identity)
    elif args.command == "seal-directory":
        result = seal_directory(args.path)
    elif args.command == "verify-sealed-directory":
        exact = set(args.exact_root_members.split(",")) \
            if args.exact_root_members else None
        result = verify_sealed_directory(args.path, exact)
    elif args.command == "publish-no-replace":
        exact = set(args.exact_root_members.split(",")) \
            if args.exact_root_members else None
        result = publish_noreplace(args.source, args.destination, exact)
    elif args.command == "write-failure-quarantine":
        metadata = {
            "schema": "m813_c2_r17_failure_receipt_v1",
            "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE_PERFORMANCE",
            "phase": args.phase,
            "return_code": args.return_code,
            "attempt_consumed": args.attempt_consumed == "true",
            "runner_sha256": args.runner_sha256,
            "contract_sha256": args.contract_sha256,
            "candidate_sha256": args.candidate_sha256,
            "release_sha256": args.release_sha256,
            "final_hammer_outer_seal_sha256":
                args.final_hammer_outer_seal_sha256,
            "claim_boundary": {
                "failure_boundary_citable": True,
                "paper_performance_citable": False,
                "vcs_complete": False,
                "system_speedup": False,
            },
        }
        result = write_failure_quarantine(args.parent, args.primary_name,
                                          metadata, args.runner_log or None)
    else:
        result = self_test()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as error:
        print("M813 guard failure: " + str(error), file=os.sys.stderr)
        raise SystemExit(3)
