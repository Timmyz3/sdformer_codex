#!/usr/bin/env python3
"""Additive repair of all M2004 findings for the split D0 campaign.

M2003 is immutable and M2005 is forbidden.  This successor adds a sealed live
process-identity receipt, a non-vacuous stopped-process gate, one immutable tar
FD/open, all-remote verification plus an immutable plan before mutation,
explicit M1706 binding for every local row, and overlap equality over the exact
validated receipt key set minus only machine-specific RSS.

The CLI remains source-only.  A future M2007 review may authorize one process
identity capture; a later archive-bound M2008 release is required for merge and
reduction.  CPython 3.6 safe.
"""
from __future__ import print_function

import argparse
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
    "test_m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_"
    "source_contract_r1_20260902.json")
M2003_SOURCE = HERE / (
    "build_m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_source.py")
M2003_TEST = HW / (
    "system_simulator/tests/"
    "test_m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_source.py")
M2003_CONTRACT = HW / (
    "contracts/m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_"
    "source_contract_r1_20260902.json")
M2004_REVIEW = HW / (
    "reviews/m2004_m2003_ep34_decoder_d0_dual_server_sealed_merge_"
    "reducer_source_hammer_r1_20260902")
FORBIDDEN_M2005 = HW / (
    "contracts/m2005_m2004_m2003_ep34_decoder_d0_dual_server_sealed_"
    "merge_reducer_release_r1_20260902.json")
FUTURE_REVIEW = HW / (
    "reviews/m2007_m2006_ep34_decoder_d0_dual_server_atomic_merge_"
    "successor_source_hammer_r1_20260902")
PRESTOP = HW / (
    "results/m2006_ep34_decoder_d0_local_campaign_process_identity_"
    "r1_20260902")
FUTURE_RELEASE = HW / (
    "contracts/m2008_m2007_m2006_ep34_decoder_d0_dual_server_atomic_"
    "merge_release_r1_20260902.json")
ATTEMPT = HW / (
    "results/.m2006_ep34_decoder_d0_dual_server_atomic_merge_"
    "attempt_consumed")
PLAN = HW / (
    "results/m2006_ep34_decoder_d0_remote_4500_8699_verified_plan_"
    "r1_20260902")
RESULT = HW / (
    "results/m2006_ep34_decoder_d0_8700_shard_atomic_merge_reducer_"
    "r1_20260902")
FAILURE = HW / (
    "results/m2006_ep34_decoder_d0_atomic_merge_failed_no_retry_"
    "r1_20260902")
STAGING_PARENT = HW / "staging/m2006_decoder_d0_remote_pack"
QUARANTINE_ROOT = HW / (
    "recovery_quarantine/m2006_decoder_d0_interrupted_remote_overlap")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = (
    "m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_source_r1_v1")
STATUS = (
    "SOURCE_ONLY__M2004_FIVE_P1_REPAIRED__M2005_FORBIDDEN__"
    "M2007_REVIEW_REQUIRED")
REVIEW_STATUS = (
    "PASS_M2007_M2006_DECODER_D0_DUAL_SERVER_ATOMIC_MERGE_SOURCE__"
    "AUTHORIZE_ONE_PROCESS_IDENTITY_CAPTURE_AND_M2008_AUTHORING")
RELEASE_SCHEMA = (
    "m2008_m2007_m2006_ep34_decoder_d0_dual_server_atomic_merge_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_M2006_SINGLE_FD_ARCHIVE_VERIFY_PLAN_MERGE_REDUCER")
M2003_SOURCE_SHA256 = (
    "49d1467dc49d711b9b70ca0e10b0bebb849aa601fdbcfced24a5ecbcef944c85")
M2003_TEST_SHA256 = (
    "4b617c3ee6c80aa7ce386ef51e30a676ed663ba13ab540658e49172665b3ac13")
M2003_CONTRACT_SHA256 = (
    "6c02aa25457e8855d2513688235d39934df6d1102474a8f31e062b13d7d0036d")
M2004_REVIEW_SHA256 = (
    "7d70e2c5d4d9c83f2fed65d27db5979b396a19195727229e5a76c999f69a2a2d")
M2004_MANIFEST_SHA256 = (
    "a5c56179aa077d72e031acf461f59cea1674e4cb4802d66e32ca892c4652b674")
M2004_OUTER_SHA256 = (
    "0ed49541d5d908b77ac60697d369670b529cdaa47c3a4b8923ce06ff04aae81f")
M1706_RELEASE_SHA256 = (
    "43c7096fe90263abf7593d41c3222675bc9153ca4529436b3a57405c550fe7e0")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
EXPECTED_ROLES = (
    "launcher_bash", "pool_controller", "worker_0", "worker_1", "worker_2")
REMOTE_START = 4500
REMOTE_STOP = 8700
LOCAL_STOP = 4500


class M2006Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M2006Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M2006Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_m2003():
    regular_exact(M2003_SOURCE, M2003_SOURCE_SHA256, "exact M2003 source")
    spec = importlib.util.spec_from_file_location("m2006_exact_m2003",
                                                  str(M2003_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M2003")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_source_r1_v1"
            and module.B.G.TOTAL_SHARDS == 8700,
            "M2003 source/grid identity drift")
    return module


M2003 = load_m2003()
B = M2003.B
M1704 = M2003.M1704


def _absent_with_sidecars(path, label):
    rows = (Path(path), Path(str(path) + ".sha256"),
            Path(str(path) + ".sha256.seal.sha256"))
    require(all(not os.path.lexists(str(row)) for row in rows),
            label + " or sidecar exists")


def validate_m2004_failure():
    seal = B.verify_sealed_tree(M2004_REVIEW, M2004_REVIEW_SHA256,
        M2004_MANIFEST_SHA256, M2004_OUTER_SHA256, False, "M2004")
    row = B.strict_json(M2004_REVIEW / "review.json")
    require(row.get("status") ==
            "FAIL_M2004_M2003_DECODER_D0_DUAL_SERVER_SEALED_MERGE_REDUCER_SOURCE__NO_M2005_RELEASE__SUCCESSOR_REQUIRED"
            and row.get("verdict") == "FAIL_CLOSED_NO_M2005_RELEASE" and
            row.get("score_over_100") == 63 and
            row.get("severity_counts") == {"p0": 0, "p1": 5, "p2": 1}
            and [item.get("id") for item in row.get("p1", [])] == [
                "P1_STOPPED_PID_GATE_IS_VACUOUS",
                "P1_REMOTE_SHARDS_ARE_MUTATED_BEFORE_FULL_REMOTE_VERIFICATION",
                "P1_ARCHIVE_OPEN_AUTHORITY_AND_IDENTITY_ARE_NOT_SINGLE_USE",
                "P1_LOCAL_PREFIX_NOT_BOUND_TO_M1706_RELEASE",
                "P1_OVERLAP_CORE_EXCLUDES_MORE_THAN_RSS"] and
            row.get("authorization") == {
                "successor_source_authoring": True,
                "m2005_release_authoring": False,
                "archive_open": False, "archive_extract": False,
                "merge": False, "reducer": False, "shard_runs": 0,
                "payload_opens": 0, "gpu_runs": 0, "eda_runs": 0},
            "M2004 disposition drift")
    return seal


def identity():
    return {"source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m2003_source_sha256": M2003_SOURCE_SHA256,
        "m2003_test_sha256": M2003_TEST_SHA256,
        "m2003_contract_sha256": M2003_CONTRACT_SHA256,
        "m2004_review_sha256": M2004_REVIEW_SHA256,
        "m2004_manifest_sha256": M2004_MANIFEST_SHA256,
        "m2004_outer_file_sha256": M2004_OUTER_SHA256,
        "m1706_release_sha256": M1706_RELEASE_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def validate_source_stage():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M2003_SOURCE, M2003_SOURCE_SHA256, "exact M2003 source")
    regular_exact(M2003_TEST, M2003_TEST_SHA256, "exact M2003 test")
    regular_exact(M2003_CONTRACT, M2003_CONTRACT_SHA256,
                  "exact M2003 contract")
    validate_m2004_failure()
    B.verify_double_sealed_file(SOURCE_CONTRACT, "M2006 source contract")
    _absent_with_sidecars(FORBIDDEN_M2005, "forbidden M2005 release")
    require(not FUTURE_REVIEW.exists(), "future M2007 review exists")
    _absent_with_sidecars(FUTURE_RELEASE, "future M2008 release")
    require(not PRESTOP.exists() and not PLAN.exists() and
            not RESULT.exists() and not FAILURE.exists() and
            not ATTEMPT.exists(), "future runtime artifact exists")
    return {"identity": identity(), "m2004": "five_p1_bound",
            "archive_opened": False, "process_receipt": False,
            "merge": False, "reducer": False}


def _proc_record(pid):
    require(type(pid) is int and pid > 1, "unsafe process PID")
    root = Path("/proc") / str(pid)
    stat_text = (root / "stat").read_text(encoding="ascii")
    close = stat_text.rfind(")")
    require(close > 0, "malformed proc stat")
    fields = stat_text[close + 2:].split()
    require(len(fields) > 19, "truncated proc stat")
    cmdline = (root / "cmdline").read_bytes()
    cwd = os.readlink(str(root / "cwd"))
    return {"pid": pid, "ppid": int(fields[1]),
            "starttime_ticks": int(fields[19]),
            "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
            "cmdline_text": cmdline.replace(b"\0", b" ").decode(
                "utf-8", "surrogateescape").strip(),
            "cwd": cwd}


def classify_process_records(records):
    require(type(records) is list and len(records) == 5,
            "exactly five live campaign processes required")
    require(len(set(row.get("pid") for row in records)) == 5,
            "campaign process PIDs must be unique")
    launchers = [row for row in records if
        "build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py"
        in row.get("cmdline_text", "") and "range(1,8700)" in
        row.get("cmdline_text", "")]
    require(len(launchers) == 1, "exact M1704 launcher identity missing")
    launcher = launchers[0]
    controllers = [row for row in records if row.get("ppid") ==
        launcher["pid"] and row.get("cmdline_text") == "python3 -"]
    require(len(controllers) == 1, "exact pool controller identity missing")
    controller = controllers[0]
    workers = sorted([row for row in records if row.get("ppid") ==
        controller["pid"] and row.get("cmdline_text") == "python3 -"],
        key=lambda row: row["pid"])
    require(len(workers) == 3, "exact three-worker identity missing")
    ordered = [("launcher_bash", launcher),
               ("pool_controller", controller)] + [
               ("worker_{}".format(index), row)
               for index, row in enumerate(workers)]
    require(tuple(role for role, _row in ordered) == EXPECTED_ROLES,
            "campaign role topology drift")
    cwd = str(HW.parent)
    require(all(row.get("cwd") == cwd for _role, row in ordered),
            "campaign cwd identity drift")
    return [dict(row, role=role) for role, row in ordered]


def capture_process_identity(pids):
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("authorization", {}).get(
                "process_identity_capture") == 1 and
            review.get("authorization", {}).get("archive_open") == 0 and
            review.get("identity") == identity(),
            "M2007 process-capture authority drift")
    require(not PRESTOP.exists(), "process identity receipt exists")
    rows = classify_process_records([_proc_record(pid) for pid in pids])
    work = Path(str(PRESTOP) + ".work")
    require(not work.exists(), "process identity work exists")
    work.mkdir(parents=True, mode=0o700)
    receipt = {"schema": SCHEMA, "status":
        "SEALED_LIVE_M1704_LOCAL_CAMPAIGN_PROCESS_IDENTITY__STOP_PENDING",
        "source_sha256": sha256(SOURCE), "processes": rows,
        "captured_while_all_five_alive": True,
        "archive_opened": False, "merge": False, "reducer": False}
    (work / "result.json").write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    B.seal_work_tree(work)
    B.verify_sealed_tree(work, allow_ignored_pycache=False,
                         label="M2006 process identity")
    work.rename(PRESTOP)
    return receipt


def validate_process_receipt():
    seal = B.verify_sealed_tree(PRESTOP, allow_ignored_pycache=False,
                                label="M2006 process identity")
    row = B.strict_json(PRESTOP / "result.json")
    require(row.get("schema") == SCHEMA and row.get("status") ==
            "SEALED_LIVE_M1704_LOCAL_CAMPAIGN_PROCESS_IDENTITY__STOP_PENDING"
            and row.get("source_sha256") == sha256(SOURCE) and
            row.get("captured_while_all_five_alive") is True and
            row.get("archive_opened") is False and
            row.get("merge") is False and row.get("reducer") is False,
            "process identity receipt drift")
    classified = classify_process_records(row.get("processes"))
    require(classified == row["processes"], "process receipt role drift")
    return row, seal


def _same_process_alive(row):
    try:
        current = _proc_record(row["pid"])
    except (OSError, M2006Error):
        return False
    return current["starttime_ticks"] == row["starttime_ticks"]


def exact_receipt_core(row):
    keys = {"schema", "status", "source_sha256", "release_sha256",
        "attempt_sha256", "checkpoint_sha256", "resource_manifest_sha256",
        "shard_ordinal", "shard", "configuration_order", "metrics",
        "integer_ratio_inputs", "payload_fd_sha256", "payload_fd_size",
        "rss", "automatic_retry", "shard_isolated", "monolithic_full_call",
        "full_decoder", "system_speedup", "paper_result",
        "independent_result_hammer_pending"}
    require(set(row) == keys, "unexpected or missing shard receipt key")
    return dict((key, row[key]) for key in sorted(keys - {"rss"}))


def verify_local_shard(ordinal):
    verified = M1704.M1688.verify_sealed_shard(ordinal)
    row = verified["row"]
    require(row.get("release_sha256") == M1706_RELEASE_SHA256 and
            row.get("checkpoint_sha256") == B.G.CHECKPOINT_SHA256 and
            row.get("resource_manifest_sha256") == B.G.RESOURCE_SHA256,
            "local shard is not exact M1706 identity")
    exact_receipt_core(row)
    return verified


def _validate_member_population(members, start=REMOTE_START,
                                stop=REMOTE_STOP):
    expected = M2003.expected_archive_population(start, stop)
    directories = set()
    files = set()
    for member in members:
        name = M2003._safe_member_name(member.name)
        require(name not in directories and name not in files,
                "duplicate archive member")
        require(not member.issym() and not member.islnk() and
                not member.ischr() and not member.isblk() and
                not member.isfifo(), "special archive member")
        if member.isdir():
            require(name in expected["directories"],
                    "unexpected archive directory")
            directories.add(name)
        else:
            require(member.isreg() and name in expected["files"],
                    "unexpected archive file")
            if name.endswith(".attempt_consumed"):
                require(stat.S_IMODE(member.mode) == 0o400,
                        "remote attempt mode is not 0400")
            require(0 <= member.size <= (4 << 20),
                    "archive member exceeds size bound")
            files.add(name)
    require(directories == expected["directories"] and
            files == expected["files"], "archive population mismatch")
    return {"directories": len(directories), "files": len(files),
            "ordinals": stop - start}


def single_fd_verify_and_extract(archive, expected_sha256,
                                 start=REMOTE_START, stop=REMOTE_STOP,
                                 staging_parent=STAGING_PARENT):
    archive = Path(archive)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(archive), flags)
    root = None
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "archive FD is not regular")
        with os.fdopen(descriptor, "rb", closefd=False) as opened:
            digest = hashlib.sha256()
            for block in iter(lambda: opened.read(1 << 20), b""):
                digest.update(block)
            require(digest.hexdigest() == expected_sha256,
                    "archive FD SHA drift")
            opened.seek(0)
            with tarfile.open(fileobj=opened, mode="r:") as stream:
                members = stream.getmembers()
                inspection = _validate_member_population(members,
                                                         start, stop)
                staging_parent = Path(staging_parent)
                staging_parent.mkdir(parents=True, exist_ok=True)
                root = Path(tempfile.mkdtemp(prefix="single_fd_",
                                             dir=str(staging_parent)))
                for member in members:
                    name = M2003._safe_member_name(member.name)
                    target = root / name
                    require(str(target.resolve()).startswith(
                        str(root.resolve()) + os.sep),
                        "archive extraction escaped staging")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=False)
                        os.chmod(str(target), 0o700)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    if hasattr(os, "O_NOFOLLOW"):
                        output_flags |= os.O_NOFOLLOW
                    mode = (0o400 if name.endswith(".attempt_consumed")
                            else 0o600)
                    output_fd = os.open(str(target), output_flags, mode)
                    source = stream.extractfile(member)
                    require(source is not None, "cannot read tar member")
                    try:
                        with os.fdopen(output_fd, "wb") as output:
                            shutil.copyfileobj(source, output, 1 << 20)
                            output.flush()
                            os.fsync(output.fileno())
                    finally:
                        source.close()
                    os.chmod(str(target), mode)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size,
                 after.st_mtime_ns), "archive FD identity changed")
    finally:
        os.close(descriptor)
    inspection["archive_sha256"] = expected_sha256
    return root, inspection


def verify_all_remote_before_mutation(stage, start=REMOTE_START,
                                      stop=REMOTE_STOP,
                                      verifier=None, artifact_projector=None):
    verifier = verifier or M2003.verify_staged_shard
    rows = []
    for ordinal in range(start, stop):
        verified = verifier(stage, ordinal)
        core = exact_receipt_core(verified["row"])
        if artifact_projector is None:
            paths = M2003._staged_paths(stage, ordinal)
            artifact = {"result_json_sha256": sha256(paths["result_json"]),
                        "manifest_sha256":
                            verified["seal"]["manifest_sha256"]}
        else:
            artifact = artifact_projector(stage, ordinal, verified)
        require(set(artifact) == {"result_json_sha256", "manifest_sha256"},
                "verified plan artifact projection drift")
        rows.append(dict({"ordinal": ordinal,
            "attempt_sha256": verified["attempt_sha256"],
            "deterministic_core_sha256": canonical_sha(core)}, **artifact))
    require(len(rows) == stop - start, "remote verified plan incomplete")
    return rows


def _publish_verified_plan(rows, archive_sha256, staging_root):
    require(not PLAN.exists(), "verified plan exists")
    work = Path(str(PLAN) + ".work")
    require(not work.exists(), "verified plan work exists")
    work.mkdir(parents=True, mode=0o700)
    payload = {"schema": SCHEMA,
        "status": "ALL_4200_REMOTE_SHARDS_VERIFIED_BEFORE_MUTATION",
        "archive_sha256": archive_sha256, "staging_root": str(staging_root),
        "rows": rows, "row_count": len(rows),
        "canonical_namespace_mutated": False}
    (work / "result.json").write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    B.seal_work_tree(work)
    seal = B.verify_sealed_tree(work, allow_ignored_pycache=False,
                                label="M2006 verified import plan")
    work.rename(PLAN)
    return payload, seal


def validate_runtime_release():
    review_seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M2007")
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and review.get("identity") == identity(),
            "M2007 runtime authority drift")
    processes, process_seal = validate_process_receipt()
    release_sha = B.verify_double_sealed_file(FUTURE_RELEASE, "M2008 release")
    release = B.strict_json(FUTURE_RELEASE)
    expected_identity = dict(identity(),
        m2007_review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        m2007_manifest_sha256=review_seal["manifest_sha256"],
        m2007_outer_file_sha256=review_seal["outer_file_sha256"],
        prestop_result_sha256=sha256(PRESTOP / "result.json"),
        prestop_manifest_sha256=process_seal["manifest_sha256"],
        prestop_outer_file_sha256=process_seal["outer_file_sha256"])
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == expected_identity and
            release.get("archive_path") ==
                "/tmp/m1704_remote_sealed_shards_4500_8699_20260902.tar" and
            len(release.get("archive_sha256", "")) == 64 and
            release.get("remote_range") == [REMOTE_START, REMOTE_STOP] and
            release.get("local_required_range") == [0, LOCAL_STOP] and
            release.get("processes") == processes["processes"] and
            release.get("authorization") == {
                "overall_attempt": 1, "archive_open": 1,
                "archive_extract": 1, "verified_plan_publish": 1,
                "merge": 1, "manual_resume_from_plan": 1,
                "reducer": 1, "result_publish": 1, "shard_runs": 0,
                "payload_opens": 0, "deletes": 0, "overwrites": 0,
                "gpu_runs": 0, "eda_runs": 0},
            "M2008 release drift")
    require(all(not _same_process_alive(row) for row in processes["processes"]),
            "captured local campaign process is still alive")
    return release, release_sha


def _consume_attempt(release, release_sha):
    require(not ATTEMPT.exists(), "overall merge attempt already consumed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(ATTEMPT), flags, 0o400)
    payload = {"schema": SCHEMA, "release_sha256": release_sha,
        "archive_sha256": release["archive_sha256"],
        "prestop_result_sha256": sha256(PRESTOP / "result.json"),
        "attempt_before_archive_open": True, "automatic_retry": False}
    with os.fdopen(descriptor, "wb") as output:
        output.write(json.dumps(payload, sort_keys=True,
                                allow_nan=False).encode("utf-8") + b"\n")
        output.flush()
        os.fsync(output.fileno())
    os.chmod(str(ATTEMPT), 0o400)
    return sha256(ATTEMPT)


def _validate_overall_attempt(release, release_sha):
    mode = ATTEMPT.lstat().st_mode
    require(stat.S_ISREG(mode) and not ATTEMPT.is_symlink() and
            stat.S_IMODE(mode) == 0o400,
            "overall attempt topology/mode drift")
    row = B.strict_json(ATTEMPT)
    require(row == {"schema": SCHEMA, "release_sha256": release_sha,
        "archive_sha256": release["archive_sha256"],
        "prestop_result_sha256": sha256(PRESTOP / "result.json"),
        "attempt_before_archive_open": True, "automatic_retry": False},
        "overall attempt identity drift")
    return sha256(ATTEMPT)


def _load_verified_plan(release):
    seal = B.verify_sealed_tree(PLAN, allow_ignored_pycache=False,
                                label="M2006 verified import plan")
    row = B.strict_json(PLAN / "result.json")
    require(row.get("schema") == SCHEMA and row.get("status") ==
            "ALL_4200_REMOTE_SHARDS_VERIFIED_BEFORE_MUTATION" and
            row.get("archive_sha256") == release["archive_sha256"] and
            row.get("row_count") == REMOTE_STOP - REMOTE_START and
            row.get("canonical_namespace_mutated") is False and
            type(row.get("rows")) is list and
            [item.get("ordinal") for item in row["rows"]] ==
                list(range(REMOTE_START, REMOTE_STOP)),
            "verified import plan identity/population drift")
    index = dict((item["ordinal"], item) for item in row["rows"])
    require(len(index) == REMOTE_STOP - REMOTE_START,
            "verified import plan duplicates")
    return row, seal, index


def _verify_staged_against_plan(stage, ordinal, expected):
    verified = M2003.verify_staged_shard(stage, ordinal)
    paths = M2003._staged_paths(stage, ordinal)
    observed = {"ordinal": ordinal,
        "attempt_sha256": verified["attempt_sha256"],
        "result_json_sha256": sha256(paths["result_json"]),
        "manifest_sha256": verified["seal"]["manifest_sha256"],
        "deterministic_core_sha256": canonical_sha(
            exact_receipt_core(verified["row"]))}
    require(observed == expected, "staged shard changed after verified plan")
    return verified


def _copy_attempt(source, target):
    require(not os.path.lexists(str(target)), "attempt target already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(target), flags, 0o400)
    with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as out:
        shutil.copyfileobj(input_stream, out, 1 << 20)
        out.flush()
        os.fsync(out.fileno())
    os.chmod(str(target), 0o400)


def _quarantine_interrupted_work(work, ordinal):
    QUARANTINE_ROOT.mkdir(parents=True, exist_ok=True)
    target = QUARANTINE_ROOT / (
        "ordinal_{:04d}.interrupted_work".format(ordinal))
    if target.exists() and not work.exists():
        return {"ordinal": ordinal, "path": str(target),
                "already_quarantined": True}
    require(work.is_dir() and not work.is_symlink() and not target.exists(),
            "interrupted work quarantine topology drift")
    work.rename(target)
    members = []
    for path in sorted(target.rglob("*")):
        require(not path.is_symlink(), "interrupted work contains symlink")
        if path.is_file():
            members.append({"name": path.relative_to(target).as_posix(),
                            "sha256": sha256(path),
                            "size": path.stat().st_size})
    return {"ordinal": ordinal, "path": str(target),
            "already_quarantined": False, "members": members}


def _install_or_compare_remote(stage, ordinal, plan_row):
    remote = _verify_staged_against_plan(stage, ordinal, plan_row)
    paths = B.namespace_paths(ordinal)
    present = dict((key, os.path.lexists(str(value)))
                   for key, value in paths.items())
    if present["result"]:
        require(present == {"result": True, "attempt": True,
                            "work": False, "failure": False},
                "completed overlap topology drift")
        local = verify_local_shard(ordinal)
        require(canonical_sha(exact_receipt_core(local["row"])) ==
                plan_row["deterministic_core_sha256"],
                "local/remote overlap core mismatch")
        return {"disposition": "local_overlap_retained", "quarantine": None}
    require(not present["failure"], "failed-no-retry overlap exists")
    quarantine = None
    if present["work"]:
        require(present["attempt"], "work exists without attempt")
        quarantine = _quarantine_interrupted_work(paths["work"], ordinal)
    if present["attempt"]:
        require(sha256(paths["attempt"]) == remote["attempt_sha256"],
                "local/remote attempt identity mismatch")
    else:
        _copy_attempt(M2003._staged_paths(stage, ordinal)["attempt"],
                      paths["attempt"])
    require(not paths["result"].exists(), "result appeared during install")
    M2003._copy_result_tree(
        M2003._staged_paths(stage, ordinal)["directory"], paths["result"])
    verify_local_shard(ordinal)
    return {"disposition": "remote_installed", "quarantine": quarantine}


def _merge_from_verified_plan(stage, plan_index):
    counts = {"remote_installed": 0, "local_overlap_retained": 0,
              "interrupted_work_quarantined": 0,
              "already_quarantined_on_resume": 0}
    quarantined = []
    for ordinal in range(REMOTE_START, REMOTE_STOP):
        disposition = _install_or_compare_remote(
            stage, ordinal, plan_index[ordinal])
        counts[disposition["disposition"]] += 1
        if disposition["quarantine"] is not None:
            row = disposition["quarantine"]
            key = ("already_quarantined_on_resume" if
                   row.get("already_quarantined") else
                   "interrupted_work_quarantined")
            counts[key] += 1
            quarantined.append(row)
    require(counts["remote_installed"] +
            counts["local_overlap_retained"] ==
            REMOTE_STOP - REMOTE_START,
            "merge population conservation failed")
    return counts, quarantined


def _publish_result(release_sha, attempt_sha, plan_seal, counts,
                    quarantined, aggregate, resumed):
    require(not RESULT.exists(), "M2006 result exists")
    work = Path(str(RESULT) + ".work")
    require(not work.exists(), "M2006 result work exists")
    work.mkdir(parents=True, mode=0o700)
    row = {"schema": SCHEMA, "status":
        "COMPLETE_8700_D0_ATOMIC_DUAL_SERVER_MERGE__HAMMER_REQUIRED",
        "source_sha256": sha256(SOURCE), "release_sha256": release_sha,
        "overall_attempt_sha256": attempt_sha,
        "verified_plan_manifest_sha256": plan_seal["manifest_sha256"],
        "merge_counts": counts, "quarantined_work": quarantined,
        "aggregate": aggregate, "manual_resume_used": resumed,
        "archive_open_count": 0 if resumed else 1,
        "shard_runs": 0, "payload_opens": 0, "deletes": 0,
        "overwrites": 0, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}
    (work / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    B.seal_work_tree(work)
    B.verify_sealed_tree(work, allow_ignored_pycache=False,
                         label="M2006 merged D0 result")
    work.rename(RESULT)
    return row


def _finish_merge(release, release_sha, attempt_sha, stage, resumed):
    _plan, plan_seal, plan_index = _load_verified_plan(release)
    counts, quarantined = _merge_from_verified_plan(stage, plan_index)
    for ordinal in range(8700):
        verify_local_shard(ordinal)
    aggregate = M1704.reduce_complete_sealed_shards()
    require(aggregate.get("complete_shards") == 8700 and
            aggregate.get("full_decoder") is False and
            aggregate.get("system_speedup") is False,
            "strong D0 reducer boundary drift")
    return _publish_result(release_sha, attempt_sha, plan_seal, counts,
                           quarantined, aggregate, resumed)


def merge_and_reduce():
    """Initial one-shot: all verification and plan publication precede merge."""
    release, release_sha = validate_runtime_release()
    attempt_sha = _consume_attempt(release, release_sha)
    for ordinal in range(LOCAL_STOP):
        verify_local_shard(ordinal)
    stage = None
    try:
        stage, _inspection = single_fd_verify_and_extract(
            Path(release["archive_path"]), release["archive_sha256"])
        rows = verify_all_remote_before_mutation(stage)
        _publish_verified_plan(rows, release["archive_sha256"], stage)
        return _finish_merge(release, release_sha, attempt_sha, stage, False)
    except BaseException as error:
        if not FAILURE.exists():
            work = Path(str(FAILURE) + ".work")
            if not work.exists():
                work.mkdir(parents=True, mode=0o700)
                (work / "result.json").write_text(json.dumps({
                    "schema": SCHEMA, "status":
                    "FAILED_NO_AUTOMATIC_RETRY__MANUAL_PLAN_RESUME_ONLY",
                    "source_sha256": sha256(SOURCE),
                    "release_sha256": release_sha,
                    "attempt_sha256": attempt_sha,
                    "stage": str(stage) if stage is not None else None,
                    "verified_plan_exists": PLAN.exists(),
                    "error_type": type(error).__name__,
                    "error": str(error), "automatic_retry": False
                    }, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
                B.seal_work_tree(work)
                work.rename(FAILURE)
        raise


def manual_resume_from_verified_plan():
    """No archive reopen; valid only after a sealed plan and failed initial run."""
    release, release_sha = validate_runtime_release()
    attempt_sha = _validate_overall_attempt(release, release_sha)
    require(FAILURE.exists() and PLAN.exists() and not RESULT.exists(),
            "manual resume topology is not authorized")
    plan, _seal, _index = _load_verified_plan(release)
    stage = Path(plan["staging_root"])
    require(stage.is_dir() and not stage.is_symlink(),
            "verified staging root unavailable")
    return _finish_merge(release, release_sha, attempt_sha, stage, True)


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "repairs": {"sealed_live_process_identity": True,
            "exact_five_role_stop_gate": True,
            "overall_attempt_before_archive_open": True,
            "single_immutable_archive_fd": True,
            "all_remote_verified_before_mutation": True,
            "immutable_verified_plan": True,
            "local_rows_explicit_m1706": True,
            "overlap_excludes_only_rss": True,
            "manual_resume_from_plan_defined": True},
        "claim_boundary": {"source_only": True,
            "process_identity_capture": False, "archive_open": False,
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
            "status": "PASS_M2006_SOURCE_PREFLIGHT__NO_RUNTIME_ACTION",
            "authorities": validate_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
