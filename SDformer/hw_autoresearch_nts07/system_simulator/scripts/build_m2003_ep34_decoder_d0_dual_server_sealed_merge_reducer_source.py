#!/usr/bin/env python3
"""Source-only fail-closed merge/reducer for the split M1704 D0 campaign.

The local campaign is allowed to complete ordinals 0..4499 and the remote
campaign 4500..8699.  A future reviewed one-shot release may open exactly one
remote tar archive, verify every remote attempt/result, retain any already
completed local overlap only after a deterministic-core equality check,
quarantine (never delete) an interrupted local work directory in the remote
range, fill only missing result namespaces, and invoke the unchanged M1688
strong 8,700-shard reducer through M1704.

This source revision exposes only ``--describe`` and ``--preflight``.  It does
not open an archive, stop a process, merge a namespace, or run the reducer.
CPython 3.6 safe.
"""
from __future__ import print_function

import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/"
    "test_m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_"
    "source_contract_r1_20260902.json")
M1704_SOURCE = HERE / (
    "build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py")
M1704_TEST = HW / (
    "system_simulator/tests/"
    "test_m1704_ep34_decoder_d0_execution_authority_adapter_source.py")
M1704_CONTRACT = HW / (
    "contracts/m1704_ep34_decoder_d0_execution_authority_adapter_"
    "source_contract_r1_20260901.json")
M1705_REVIEW = HW / (
    "reviews/m1705_m1704_ep34_decoder_d0_execution_authority_adapter_"
    "source_independent_review_r1_20260901")
M1706_RELEASE = HW / (
    "contracts/m1706_m1705_m1704_ep34_decoder_d0_8700_shard_"
    "campaign_release_r1_20260901.json")
FUTURE_REVIEW = HW / (
    "reviews/m2004_m2003_ep34_decoder_d0_dual_server_sealed_merge_"
    "reducer_source_hammer_r1_20260902")
FUTURE_RELEASE = HW / (
    "contracts/m2005_m2004_m2003_ep34_decoder_d0_dual_server_sealed_"
    "merge_reducer_release_r1_20260902.json")
RESULT = HW / (
    "results/m2003_ep34_decoder_d0_8700_shard_dual_server_reducer_"
    "r1_20260902")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__LOCAL_0_4499_REMOTE_4500_8699__M2004_REVIEW_REQUIRED__"
    "NO_ARCHIVE_OPEN_NO_MERGE_NO_REDUCER")
RESULT_SCHEMA = (
    "m2003_ep34_decoder_d0_8700_shard_dual_server_reducer_result_r1_v1")
REVIEW_STATUS = (
    "PASS_M2004_M2003_DECODER_D0_DUAL_SERVER_SEALED_MERGE_REDUCER_SOURCE__"
    "AUTHORIZE_M2005_RELEASE_AUTHORING_ONLY")
RELEASE_SCHEMA = (
    "m2005_m2004_m2003_ep34_decoder_d0_dual_server_sealed_merge_"
    "reducer_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_REMOTE_ARCHIVE_VERIFY_MERGE_AND_D0_REDUCER")
M1704_SOURCE_SHA256 = (
    "abc052025a5ed6975fa9d200581b182fa723a7b4b0330bc341b4bc2712204820")
M1704_TEST_SHA256 = (
    "9d804d08d69621b66cd687c9a4a272f344c62eb7dff97204ebd25efe57eb25f4")
M1704_CONTRACT_SHA256 = (
    "1a133259f01addec024530944bec739a9a783fa381fc08f736721219b15bd554")
M1705_REVIEW_SHA256 = (
    "c10f9b2c2eef2784fccd22a01c320a9c2ea45df17dc63a9c2cc3fafae9b28f1b")
M1706_RELEASE_SHA256 = (
    "43c7096fe90263abf7593d41c3222675bc9153ca4529436b3a57405c550fe7e0")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
REMOTE_START = 4500
REMOTE_STOP = 8700
LOCAL_STOP = 4500
HEX = frozenset("0123456789abcdef")


class M2003Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M2003Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M2003Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_m1704():
    regular_exact(M1704_SOURCE, M1704_SOURCE_SHA256, "exact M1704 source")
    spec = importlib.util.spec_from_file_location("m2003_exact_m1704",
                                                  str(M1704_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1704")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m1704_ep34_decoder_d0_execution_authority_adapter_source_r1_v1"
            and module.B.G.TOTAL_SHARDS == 8700,
            "M1704 grid/source identity drift")
    return module


M1704 = load_m1704()
B = M1704.B


def _absent_with_sidecars(path, label):
    rows = (Path(path), Path(str(path) + ".sha256"),
            Path(str(path) + ".sha256.seal.sha256"))
    require(all(not os.path.lexists(str(row)) for row in rows),
            label + " or sidecar exists")


def identity():
    return {
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m1704_source_sha256": M1704_SOURCE_SHA256,
        "m1704_test_sha256": M1704_TEST_SHA256,
        "m1704_contract_sha256": M1704_CONTRACT_SHA256,
        "m1705_review_sha256": M1705_REVIEW_SHA256,
        "m1706_release_sha256": M1706_RELEASE_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def validate_source_stage():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M1704_SOURCE, M1704_SOURCE_SHA256, "exact M1704 source")
    regular_exact(M1704_TEST, M1704_TEST_SHA256, "exact M1704 test")
    regular_exact(M1704_CONTRACT, M1704_CONTRACT_SHA256,
                  "exact M1704 contract")
    regular_exact(M1705_REVIEW / "review.json", M1705_REVIEW_SHA256,
                  "exact M1705 review")
    regular_exact(M1706_RELEASE, M1706_RELEASE_SHA256,
                  "exact M1706 release")
    require(M1704.validate_future_review_and_release() == M1706_RELEASE_SHA256,
            "M1704 execution authority chain drift")
    B.verify_double_sealed_file(SOURCE_CONTRACT, "M2003 source contract")
    _absent_with_sidecars(FUTURE_RELEASE, "future M2005 release")
    require(not FUTURE_REVIEW.exists(), "future M2004 review exists")
    require(not RESULT.exists(), "future M2003 result exists")
    return {"identity": identity(), "total_shards": 8700,
            "local_range": [0, LOCAL_STOP],
            "remote_range": [REMOTE_START, REMOTE_STOP],
            "archive_opened": False, "merge_executed": False,
            "reducer_executed": False}


def _result_token(ordinal):
    return "m1681_ep34_decoder_d0_shard_{:04d}_r1_20260901".format(
        ordinal)


def _archive_names(ordinal):
    token = _result_token(ordinal)
    base = "hw_autoresearch_nts07/results/" + token
    attempt = ("hw_autoresearch_nts07/results/." + token +
               ".attempt_consumed")
    return {"directory": base, "attempt": attempt,
            "result_json": base + "/result.json",
            "manifest": base + "/SHA256SUMS",
            "outer": base + "/SHA256SUMS.seal.sha256"}


def expected_archive_population(start=REMOTE_START, stop=REMOTE_STOP):
    require(type(start) is int and type(stop) is int and
            0 <= start < stop <= 8700, "invalid archive range")
    directories = set()
    files = set()
    for ordinal in range(start, stop):
        names = _archive_names(ordinal)
        directories.add(names["directory"])
        files.update((names["attempt"], names["result_json"],
                      names["manifest"], names["outer"]))
    return {"directories": directories, "files": files}


def _safe_member_name(name):
    require(type(name) is str and name and "\\" not in name and
            not name.startswith("/") and not name.startswith("./"),
            "unsafe archive member name")
    clean = name[:-1] if name.endswith("/") else name
    path = Path(clean)
    require(path.as_posix() == clean and ".." not in path.parts,
            "unsafe archive member path")
    return clean


def inspect_archive(archive, expected_sha256):
    archive = Path(archive)
    require(len(expected_sha256) == 64 and
            all(item in HEX for item in expected_sha256),
            "invalid expected archive SHA")
    regular_exact(archive, expected_sha256, "remote archive")
    expected = expected_archive_population()
    seen_dirs = set()
    seen_files = set()
    with tarfile.open(str(archive), "r:") as stream:
        for member in stream:
            name = _safe_member_name(member.name)
            require(name not in seen_dirs and name not in seen_files,
                    "duplicate archive member")
            require(not member.issym() and not member.islnk() and
                    not member.ischr() and not member.isblk() and
                    not member.isfifo(), "non-regular archive member")
            if member.isdir():
                require(name in expected["directories"],
                        "unexpected archive directory")
                seen_dirs.add(name)
            else:
                require(member.isreg() and name in expected["files"],
                        "unexpected archive file")
                if name.endswith(".attempt_consumed"):
                    require(stat.S_IMODE(member.mode) == 0o400,
                            "remote attempt mode is not 0400")
                require(0 <= member.size <= (4 << 20),
                        "archive member exceeds size bound")
                seen_files.add(name)
    require(seen_dirs == expected["directories"] and
            seen_files == expected["files"],
            "archive population is incomplete or excessive")
    return {"archive_sha256": expected_sha256,
            "directories": len(seen_dirs), "files": len(seen_files),
            "ordinals": REMOTE_STOP - REMOTE_START}


def extract_archive_once(archive, expected_sha256, staging_parent):
    inspection = inspect_archive(archive, expected_sha256)
    staging_parent = Path(staging_parent)
    staging_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="m2003_remote_pack_",
                                 dir=str(staging_parent)))
    with tarfile.open(str(archive), "r:") as stream:
        for member in stream:
            name = _safe_member_name(member.name)
            target = root / name
            require(str(target.resolve()).startswith(str(root.resolve()) +
                    os.sep), "archive extraction escaped staging")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=False)
                os.chmod(str(target), 0o700)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            mode = 0o400 if name.endswith(".attempt_consumed") else 0o600
            descriptor = os.open(str(target), flags, mode)
            source = stream.extractfile(member)
            require(source is not None, "cannot read archive member")
            try:
                with os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(source, output, 1 << 20)
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                source.close()
            os.chmod(str(target), mode)
    return root, inspection


def _staged_paths(root, ordinal):
    names = _archive_names(ordinal)
    return {key: Path(root) / value for key, value in names.items()}


def verify_staged_shard(root, ordinal):
    require(REMOTE_START <= ordinal < REMOTE_STOP,
            "staged ordinal outside remote range")
    paths = _staged_paths(root, ordinal)
    attempt = paths["attempt"]
    mode = attempt.lstat().st_mode
    require(stat.S_ISREG(mode) and not attempt.is_symlink() and
            stat.S_IMODE(mode) == 0o400, "staged attempt topology/mode drift")
    attempt_sha = sha256(attempt)
    seal = B.verify_sealed_tree(paths["directory"],
        allow_ignored_pycache=False, label="staged remote shard")
    row = B.strict_json(paths["result_json"])
    B.validate_shard_receipt(row, ordinal, attempt_sha,
                             M1706_RELEASE_SHA256)
    require(row.get("release_sha256") == M1706_RELEASE_SHA256 and
            row.get("checkpoint_sha256") == B.G.CHECKPOINT_SHA256 and
            row.get("resource_manifest_sha256") == B.G.RESOURCE_SHA256,
            "staged shard authority identity drift")
    return {"row": row, "attempt_sha256": attempt_sha, "seal": seal}


def deterministic_core(row):
    keys = ("schema", "status", "source_sha256", "release_sha256",
            "attempt_sha256", "checkpoint_sha256",
            "resource_manifest_sha256", "shard_ordinal", "shard",
            "configuration_order", "metrics", "integer_ratio_inputs",
            "payload_fd_sha256", "payload_fd_size", "automatic_retry",
            "shard_isolated", "monolithic_full_call", "full_decoder",
            "system_speedup", "paper_result")
    require(all(key in row for key in keys),
            "result lacks deterministic core field")
    return dict((key, row[key]) for key in keys)


def _pid_alive(pid):
    require(type(pid) is int and pid > 1, "unsafe controller PID")
    try:
        os.kill(pid, 0)
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        raise
    return True


def validate_runtime_authority(archive_sha256, stopped_pids):
    review_seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M2004")
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and review.get("identity") == identity() and
            review.get("authorization") == {
                "m2005_release_authoring": True,
                "archive_open": False, "merge": False,
                "reducer": False, "shard_runs": 0,
                "payload_opens": 0, "eda_runs": 0, "gpu_runs": 0},
            "M2004 review authority drift")
    release_sha = B.verify_double_sealed_file(FUTURE_RELEASE, "M2005 release")
    release = B.strict_json(FUTURE_RELEASE)
    expected_identity = dict(identity(),
        m2004_review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        m2004_manifest_sha256=review_seal["manifest_sha256"],
        m2004_outer_file_sha256=review_seal["outer_file_sha256"])
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == expected_identity and
            release.get("archive_sha256") == archive_sha256 and
            release.get("remote_range") == [REMOTE_START, REMOTE_STOP] and
            release.get("local_required_range") == [0, LOCAL_STOP] and
            release.get("stopped_pids") == stopped_pids and
            release.get("authorization") == {
                "archive_open": 1, "archive_extract": 1, "merge": 1,
                "reducer": 1, "result_publish": 1, "shard_runs": 0,
                "payload_opens": 0, "deletes": 0, "overwrites": 0,
                "eda_runs": 0, "gpu_runs": 0},
            "M2005 runtime release drift")
    require(all(not _pid_alive(pid) for pid in stopped_pids),
            "declared local campaign PID is still alive")
    return release_sha


def _copy_result_tree(source, target):
    require(not os.path.lexists(str(target)), "result target already exists")
    work = Path(str(target) + ".m2003_import_work")
    require(not os.path.lexists(str(work)), "import work already exists")
    shutil.copytree(str(source), str(work), symlinks=False)
    B.verify_sealed_tree(work, allow_ignored_pycache=False,
                         label="copied remote shard")
    work.rename(target)


def _quarantine_work(work, quarantine_root, ordinal):
    quarantine_root = Path(quarantine_root)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / ("ordinal_{:04d}.interrupted_work".format(
        ordinal))
    require(not os.path.lexists(str(target)), "recovery quarantine collision")
    work.rename(target)
    members = []
    for path in sorted(target.rglob("*")):
        require(not path.is_symlink(), "interrupted work contains symlink")
        if path.is_file():
            members.append({"name": path.relative_to(target).as_posix(),
                            "sha256": sha256(path),
                            "size": path.stat().st_size})
    return {"ordinal": ordinal, "path": str(target), "members": members}


def merge_and_reduce(archive, expected_sha256, stopped_pids,
                     staging_parent, quarantine_root):
    release_sha = validate_runtime_authority(expected_sha256, stopped_pids)
    require(not RESULT.exists(), "M2003 result already exists")
    for ordinal in range(LOCAL_STOP):
        M1704.M1688.verify_sealed_shard(ordinal)
    stage, inspection = extract_archive_once(archive, expected_sha256,
                                              staging_parent)
    counts = {"installed": 0, "local_overlap_retained": 0,
              "interrupted_work_quarantined": 0}
    quarantined = []
    for ordinal in range(REMOTE_START, REMOTE_STOP):
        remote = verify_staged_shard(stage, ordinal)
        paths = B.namespace_paths(ordinal)
        present = dict((key, os.path.lexists(str(value)))
                       for key, value in paths.items())
        if present["result"]:
            require(present == {"result": True, "attempt": True,
                                "work": False, "failure": False},
                    "local completed overlap topology drift")
            local = M1704.M1688.verify_sealed_shard(ordinal)
            require(deterministic_core(local["row"]) ==
                    deterministic_core(remote["row"]),
                    "local/remote deterministic core mismatch")
            counts["local_overlap_retained"] += 1
            continue
        require(not present["failure"], "local failed-no-retry overlap exists")
        if present["work"]:
            require(present["attempt"], "work exists without attempt")
            quarantined.append(_quarantine_work(paths["work"],
                                                quarantine_root, ordinal))
            counts["interrupted_work_quarantined"] += 1
        if present["attempt"]:
            require(sha256(paths["attempt"]) == remote["attempt_sha256"],
                    "local/remote attempt identity mismatch")
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(str(paths["attempt"]), flags, 0o400)
            try:
                payload = _staged_paths(stage, ordinal)["attempt"].read_bytes()
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(str(paths["attempt"]), 0o400)
        _copy_result_tree(_staged_paths(stage, ordinal)["directory"],
                          paths["result"])
        M1704.M1688.verify_sealed_shard(ordinal)
        counts["installed"] += 1
    aggregate = M1704.reduce_complete_sealed_shards()
    require(aggregate.get("complete_shards") == 8700 and
            aggregate.get("full_decoder") is False and
            aggregate.get("system_speedup") is False,
            "strong D0 reducer boundary drift")
    work = Path(str(RESULT) + ".work")
    require(not os.path.lexists(str(work)), "M2003 result work exists")
    work.mkdir(parents=True, mode=0o700)
    receipt = {"schema": RESULT_SCHEMA,
        "status": "COMPLETE_8700_D0_DUAL_SERVER_MERGE__HAMMER_REQUIRED",
        "source_sha256": sha256(SOURCE), "release_sha256": release_sha,
        "archive": inspection, "merge_counts": counts,
        "quarantined_interrupted_work": quarantined,
        "staging_root": str(stage), "aggregate": aggregate,
        "shard_runs": 0, "payload_opens": 0, "deletes": 0,
        "overwrites": 0, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}
    (work / "result.json").write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    B.seal_work_tree(work)
    B.verify_sealed_tree(work, allow_ignored_pycache=False,
                         label="M2003 reducer result")
    work.rename(RESULT)
    return receipt


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "ranges": {"local_required": [0, LOCAL_STOP],
                   "remote_archive": [REMOTE_START, REMOTE_STOP],
                   "total_shards": 8700},
        "merge_policy": {"local_completed_overlap_preferred": True,
            "deterministic_core_equality_required": True,
            "rss_excluded_from_overlap_equality": True,
            "interrupted_work_quarantined_not_deleted": True,
            "missing_only_install": True, "overwrite": False,
            "delete": False, "all_campaign_pids_must_be_stopped": True},
        "reducer": "unchanged M1688 strong exact-sibling reducer via M1704",
        "claim_boundary": {"source_only": True, "archive_open": False,
            "merge": False, "reducer": False, "shard_runs": 0,
            "payload_opens": 0, "gpu_runs": 0, "eda_runs": 0,
            "full_d0_result": False, "full_decoder": False,
            "system_speedup": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    output = describe()
    if args.preflight:
        output = {"schema": SCHEMA,
                  "status": "PASS_M2003_SOURCE_PREFLIGHT__NO_ARCHIVE_NO_MERGE",
                  "authorities": validate_source_stage(),
                  "claim_boundary": describe()["claim_boundary"]}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
