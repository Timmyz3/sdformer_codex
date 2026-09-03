#!/usr/bin/env python3
"""Narrow successor closing M2007 publication/resume findings.

M2006 remains immutable and M2008 is forbidden.  This source retains its
single-FD, all-before-mutate, explicit-M1706 and exact-minus-RSS mechanisms,
but replaces canonical publication with Linux renameat2(RENAME_NOREPLACE),
validates/promotes an orphan import-work tree during manual resume, records
campaign and resume-leg archive-open counts separately, and binds exact raw
cmdline bytes to every live process record.

CLI is source-only; no production process/archive/namespace is touched.
"""
from __future__ import print_function

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/"
    "test_m2009_ep34_decoder_d0_noreplace_resume_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m2009_ep34_decoder_d0_noreplace_resume_successor_"
    "source_contract_r1_20260902.json")
M2006_SOURCE = HERE / (
    "build_m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_source.py")
M2006_TEST = HW / (
    "system_simulator/tests/"
    "test_m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_source.py")
M2006_CONTRACT = HW / (
    "contracts/m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_"
    "source_contract_r1_20260902.json")
M2007_REVIEW = HW / (
    "reviews/m2007_m2006_ep34_decoder_d0_dual_server_atomic_merge_"
    "successor_source_hammer_r1_20260902")
FORBIDDEN_M2008 = HW / (
    "contracts/m2008_m2007_m2006_ep34_decoder_d0_dual_server_atomic_"
    "merge_release_r1_20260902.json")
FUTURE_REVIEW = HW / (
    "reviews/m2010_m2009_ep34_decoder_d0_noreplace_resume_successor_"
    "source_hammer_r1_20260902")
PRESTOP = HW / (
    "results/m2009_ep34_decoder_d0_local_campaign_process_identity_"
    "r1_20260902")
FUTURE_RELEASE = HW / (
    "contracts/m2011_m2010_m2009_ep34_decoder_d0_noreplace_resume_"
    "release_r1_20260902.json")
ATTEMPT = HW / (
    "results/.m2009_ep34_decoder_d0_noreplace_resume_attempt_consumed")
PLAN = HW / (
    "results/m2009_ep34_decoder_d0_remote_4500_8699_verified_plan_"
    "r1_20260902")
RESULT = HW / (
    "results/m2009_ep34_decoder_d0_8700_shard_noreplace_reducer_"
    "r1_20260902")
FAILURE = HW / (
    "results/m2009_ep34_decoder_d0_merge_failed_manual_resume_allowed_"
    "r1_20260902")
STAGING_PARENT = HW / "staging/m2009_decoder_d0_remote_pack"
QUARANTINE_ROOT = HW / (
    "recovery_quarantine/m2009_decoder_d0_interrupted_remote_overlap")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m2009_ep34_decoder_d0_noreplace_resume_successor_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__M2007_TWO_P1_TWO_P2_REPAIRED__M2008_FORBIDDEN__"
    "M2010_REVIEW_REQUIRED")
REVIEW_STATUS = (
    "PASS_M2010_M2009_DECODER_D0_NOREPLACE_RESUME_SOURCE__"
    "AUTHORIZE_PROCESS_CAPTURE_AND_M2011_AUTHORING")
RELEASE_SCHEMA = (
    "m2011_m2010_m2009_ep34_decoder_d0_noreplace_resume_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_M2009_ONE_FD_PLAN_NOREPLACE_MERGE_REDUCER")
M2006_SOURCE_SHA256 = (
    "230538001861e117f8b2adb64d8f008e651eef592e3c334863dada6670497530")
M2006_TEST_SHA256 = (
    "055321de907fe845e45f5bf902f874ebeabefe70b3407094d9739a3154808adb")
M2006_CONTRACT_SHA256 = (
    "49d56e9975649b8ff595e3264d453a36606a240bc5efa5cdbba7b12f1f515480")
M2007_REVIEW_SHA256 = (
    "45209abdd8184267a74a0ef404d35aa4f0b00023c2f5cfe859c6124d88becd3d")
M2007_MANIFEST_SHA256 = (
    "9f12bd60abe47ebe7c54a7c1c13c61543d5d4597c810cbccf8b54c5af1581bbb")
M2007_OUTER_SHA256 = (
    "9f17803e7247d319ed42796271e20a5ed1961c57f22a4636980227aecaa0852e")
M1706_RELEASE_SHA256 = (
    "43c7096fe90263abf7593d41c3222675bc9153ca4529436b3a57405c550fe7e0")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
EXPECTED_ROLES = (
    "launcher_bash", "pool_controller", "worker_0", "worker_1", "worker_2")
REMOTE_START = 4500
REMOTE_STOP = 8700
LOCAL_STOP = 4500
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class M2009Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M2009Error(message)


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
        raise M2009Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_m2006():
    regular_exact(M2006_SOURCE, M2006_SOURCE_SHA256, "exact M2006 source")
    spec = importlib.util.spec_from_file_location("m2009_exact_m2006",
                                                  str(M2006_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M2006")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_source_r1_v1"
            and module.B.G.TOTAL_SHARDS == 8700,
            "M2006 source/grid identity drift")
    return module


M2006 = load_m2006()
B = M2006.B
M1704 = M2006.M1704
M2003 = M2006.M2003


def _absent_with_sidecars(path, label):
    rows = (Path(path), Path(str(path) + ".sha256"),
            Path(str(path) + ".sha256.seal.sha256"))
    require(all(not os.path.lexists(str(row)) for row in rows),
            label + " or sidecar exists")


def validate_m2007_failure():
    B.verify_sealed_tree(M2007_REVIEW, M2007_REVIEW_SHA256,
        M2007_MANIFEST_SHA256, M2007_OUTER_SHA256, False, "M2007")
    row = B.strict_json(M2007_REVIEW / "review.json")
    require(row.get("status", "").startswith("FAIL_M2007") and
            row.get("score_over_100") == 76 and
            row.get("severity_counts") == {"p0": 0, "p1": 2, "p2": 2}
            and [item.get("id") for item in row.get("p1", [])] == [
                "P1_INTERRUPTED_IMPORT_WORK_STRANDS_MANUAL_RESUME",
                "P1_NO_OVERWRITE_PUBLICATION_IS_CHECK_THEN_RENAME"] and
            row.get("authorization", {}).get(
                "successor_source_authoring") is True and
            row.get("authorization", {}).get(
                "process_identity_capture") == 0 and
            row.get("authorization", {}).get("m2008_release_authoring")
                is False,
            "M2007 disposition drift")


def identity():
    return {"source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m2006_source_sha256": M2006_SOURCE_SHA256,
        "m2006_test_sha256": M2006_TEST_SHA256,
        "m2006_contract_sha256": M2006_CONTRACT_SHA256,
        "m2007_review_sha256": M2007_REVIEW_SHA256,
        "m2007_manifest_sha256": M2007_MANIFEST_SHA256,
        "m2007_outer_file_sha256": M2007_OUTER_SHA256,
        "m1706_release_sha256": M1706_RELEASE_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def rename_noreplace(source, target):
    source = Path(source)
    target = Path(target)
    require(os.path.lexists(str(source)), "noreplace source missing")
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "renameat2"), "renameat2 unavailable")
    result = libc.renameat2(
        ctypes.c_int(AT_FDCWD), ctypes.c_char_p(os.fsencode(str(source))),
        ctypes.c_int(AT_FDCWD), ctypes.c_char_p(os.fsencode(str(target))),
        ctypes.c_uint(RENAME_NOREPLACE))
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(target))
    return True


def _proc_record(pid):
    require(type(pid) is int and pid > 1, "unsafe process PID")
    root = Path("/proc") / str(pid)
    stat_text = (root / "stat").read_text(encoding="ascii")
    close = stat_text.rfind(")")
    require(close > 0, "malformed proc stat")
    fields = stat_text[close + 2:].split()
    require(len(fields) > 19, "truncated proc stat")
    raw = (root / "cmdline").read_bytes()
    return {"pid": pid, "ppid": int(fields[1]),
        "starttime_ticks": int(fields[19]),
        "cmdline_raw_hex": raw.hex(),
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
        "cmdline_text": raw.replace(b"\0", b" ").decode(
            "utf-8", "surrogateescape").strip(),
        "cwd": os.readlink(str(root / "cwd"))}


def classify_process_records(records):
    raw_keys = {"pid", "ppid", "starttime_ticks", "cmdline_raw_hex",
                "cmdline_sha256", "cmdline_text", "cwd"}
    require(type(records) is list and len(records) == 5,
            "exactly five live campaign processes required")
    for row in records:
        require(set(row) in (raw_keys, raw_keys | {"role"}),
                "process receipt exact key set drift")
        try:
            raw = bytes.fromhex(row["cmdline_raw_hex"])
        except (TypeError, ValueError):
            raise M2009Error("invalid raw cmdline encoding")
        require(hashlib.sha256(raw).hexdigest() == row["cmdline_sha256"] and
                raw.replace(b"\0", b" ").decode(
                    "utf-8", "surrogateescape").strip() ==
                row["cmdline_text"], "cmdline text/hash mismatch")
    require(len(set(row["pid"] for row in records)) == 5,
            "process PIDs are not unique")
    launchers = [row for row in records if
        "build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py"
        in row["cmdline_text"] and "range(1,8700)" in row["cmdline_text"]]
    require(len(launchers) == 1, "exact M1704 launcher missing")
    launcher = launchers[0]
    controllers = [row for row in records if row["ppid"] ==
        launcher["pid"] and row["cmdline_text"] == "python3 -"]
    require(len(controllers) == 1, "exact pool controller missing")
    controller = controllers[0]
    workers = sorted([row for row in records if row["ppid"] ==
        controller["pid"] and row["cmdline_text"] == "python3 -"],
        key=lambda row: row["pid"])
    require(len(workers) == 3, "exact three-worker population missing")
    ordered = [("launcher_bash", launcher),
               ("pool_controller", controller)] + [
               ("worker_{}".format(index), row)
               for index, row in enumerate(workers)]
    require(all(row["cwd"] == str(HW.parent) for _role, row in ordered),
            "campaign cwd drift")
    return [dict(row, role=role) for role, row in ordered]


def _same_process_alive(row):
    try:
        current = _proc_record(row["pid"])
    except FileNotFoundError:
        return False
    return current["starttime_ticks"] == row["starttime_ticks"]


def _publish_sealed_work(work, target, label):
    B.seal_work_tree(work)
    seal = B.verify_sealed_tree(work, allow_ignored_pycache=False,
                                label=label)
    rename_noreplace(work, target)
    return seal


def validate_source_stage():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M2006_SOURCE, M2006_SOURCE_SHA256, "exact M2006 source")
    regular_exact(M2006_TEST, M2006_TEST_SHA256, "exact M2006 test")
    regular_exact(M2006_CONTRACT, M2006_CONTRACT_SHA256,
                  "exact M2006 contract")
    validate_m2007_failure()
    B.verify_double_sealed_file(SOURCE_CONTRACT, "M2009 source contract")
    _absent_with_sidecars(FORBIDDEN_M2008, "forbidden M2008 release")
    require(not FUTURE_REVIEW.exists(), "future M2010 review exists")
    _absent_with_sidecars(FUTURE_RELEASE, "future M2011 release")
    require(all(not os.path.lexists(str(path)) for path in
        (PRESTOP, Path(str(PRESTOP) + ".work"), ATTEMPT, PLAN,
         Path(str(PLAN) + ".work"), RESULT, Path(str(RESULT) + ".work"),
         FAILURE, Path(str(FAILURE) + ".work"))),
        "future M2009 runtime artifact exists")
    return {"identity": identity(), "m2007": "two_p1_two_p2_bound",
            "process_capture": False, "archive_open": False,
            "merge": False, "reducer": False}


def capture_process_identity(pids):
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("identity") == identity() and
            review.get("authorization") == {
                "process_identity_capture": 1,
                "m2011_release_authoring": 1, "archive_open": 0,
                "merge": 0, "reducer": 0, "payload_opens": 0,
                "gpu_runs": 0, "eda_runs": 0},
            "M2010 process-capture authority drift")
    rows = classify_process_records([_proc_record(pid) for pid in pids])
    work = Path(str(PRESTOP) + ".work")
    work.mkdir(parents=True, mode=0o700)
    receipt = {"schema": SCHEMA, "status":
        "SEALED_LIVE_M1704_PROCESS_IDENTITY__STOP_PENDING",
        "source_sha256": sha256(SOURCE), "processes": rows,
        "captured_all_five_live": True, "archive_open": 0,
        "merge": False, "reducer": False}
    (work / "result.json").write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    _publish_sealed_work(work, PRESTOP, "M2009 process identity")
    return receipt


def validate_process_receipt():
    seal = B.verify_sealed_tree(PRESTOP, allow_ignored_pycache=False,
                                label="M2009 process identity")
    row = B.strict_json(PRESTOP / "result.json")
    require(row.get("schema") == SCHEMA and row.get("status") ==
            "SEALED_LIVE_M1704_PROCESS_IDENTITY__STOP_PENDING" and
            row.get("source_sha256") == sha256(SOURCE) and
            row.get("captured_all_five_live") is True and
            row.get("archive_open") == 0 and row.get("merge") is False and
            row.get("reducer") is False,
            "process receipt boundary drift")
    require(classify_process_records(row.get("processes")) == row["processes"],
            "process receipt role/hash drift")
    return row, seal


def validate_runtime_release():
    review_seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M2010")
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and review.get("identity") == identity(),
            "M2010 runtime authority drift")
    processes, process_seal = validate_process_receipt()
    release_sha = B.verify_double_sealed_file(FUTURE_RELEASE, "M2011 release")
    release = B.strict_json(FUTURE_RELEASE)
    expected_identity = dict(identity(),
        m2010_review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        m2010_manifest_sha256=review_seal["manifest_sha256"],
        m2010_outer_file_sha256=review_seal["outer_file_sha256"],
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
            "M2011 release drift")
    require(all(not _same_process_alive(row) for row in processes["processes"]),
            "captured campaign process is still alive")
    return release, release_sha


def _consume_attempt(release, release_sha):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(ATTEMPT), flags, 0o400)
    row = {"schema": SCHEMA, "release_sha256": release_sha,
        "archive_sha256": release["archive_sha256"],
        "prestop_result_sha256": sha256(PRESTOP / "result.json"),
        "attempt_before_archive_open": True, "automatic_retry": False}
    with os.fdopen(descriptor, "wb") as output:
        output.write(json.dumps(row, sort_keys=True,
                                allow_nan=False).encode("utf-8") + b"\n")
        output.flush()
        os.fsync(output.fileno())
    os.chmod(str(ATTEMPT), 0o400)
    return sha256(ATTEMPT)


def _validate_attempt(release, release_sha):
    mode = ATTEMPT.lstat().st_mode
    require(stat.S_ISREG(mode) and not ATTEMPT.is_symlink() and
            stat.S_IMODE(mode) == 0o400, "overall attempt mode drift")
    require(B.strict_json(ATTEMPT) == {
        "schema": SCHEMA, "release_sha256": release_sha,
        "archive_sha256": release["archive_sha256"],
        "prestop_result_sha256": sha256(PRESTOP / "result.json"),
        "attempt_before_archive_open": True, "automatic_retry": False},
        "overall attempt identity drift")
    return sha256(ATTEMPT)


def _publish_plan(rows, archive_sha, stage):
    work = Path(str(PLAN) + ".work")
    work.mkdir(parents=True, mode=0o700)
    row = {"schema": SCHEMA, "status":
        "ALL_4200_REMOTE_SHARDS_VERIFIED_BEFORE_MUTATION",
        "archive_sha256": archive_sha, "staging_root": str(stage),
        "rows": rows, "row_count": len(rows),
        "canonical_namespace_mutated": False}
    (work / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return row, _publish_sealed_work(work, PLAN, "M2009 verified plan")


def _load_plan(release):
    seal = B.verify_sealed_tree(PLAN, allow_ignored_pycache=False,
                                label="M2009 verified plan")
    row = B.strict_json(PLAN / "result.json")
    require(row.get("schema") == SCHEMA and row.get("status") ==
            "ALL_4200_REMOTE_SHARDS_VERIFIED_BEFORE_MUTATION" and
            row.get("archive_sha256") == release["archive_sha256"] and
            row.get("row_count") == REMOTE_STOP - REMOTE_START and
            row.get("canonical_namespace_mutated") is False and
            [item.get("ordinal") for item in row.get("rows", [])] ==
                list(range(REMOTE_START, REMOTE_STOP)), "plan identity drift")
    return row, seal, dict((item["ordinal"], item) for item in row["rows"])


def _staged_verified(stage, ordinal, plan_row):
    verified = M2003.verify_staged_shard(stage, ordinal)
    paths = M2003._staged_paths(stage, ordinal)
    observed = {"ordinal": ordinal,
        "attempt_sha256": verified["attempt_sha256"],
        "result_json_sha256": sha256(paths["result_json"]),
        "manifest_sha256": verified["seal"]["manifest_sha256"],
        "deterministic_core_sha256": M2006.canonical_sha(
            M2006.exact_receipt_core(verified["row"]))}
    require(observed == plan_row, "staged shard changed after plan")
    return verified


def _copy_attempt(source, target):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(target), flags, 0o400)
    with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as out:
        shutil.copyfileobj(incoming, out, 1 << 20)
        out.flush()
        os.fsync(out.fileno())
    os.chmod(str(target), 0o400)


def _promote_result_resumable(source, target, plan_row):
    import_work = Path(str(target) + ".m2009_import_work")
    if not import_work.exists():
        require(not os.path.lexists(str(import_work)),
                "import-work special entry exists")
        shutil.copytree(str(source), str(import_work), symlinks=False)
    seal = B.verify_sealed_tree(import_work, allow_ignored_pycache=False,
                                label="M2009 import work")
    row = B.strict_json(import_work / "result.json")
    require(sha256(import_work / "result.json") ==
            plan_row["result_json_sha256"] and
            seal["manifest_sha256"] == plan_row["manifest_sha256"] and
            M2006.canonical_sha(M2006.exact_receipt_core(row)) ==
            plan_row["deterministic_core_sha256"],
            "orphan import-work does not match verified plan")
    rename_noreplace(import_work, target)


def _quarantine_work(work, ordinal):
    QUARANTINE_ROOT.mkdir(parents=True, exist_ok=True)
    target = QUARANTINE_ROOT / (
        "ordinal_{:04d}.interrupted_work".format(ordinal))
    if target.exists() and not work.exists():
        return {"ordinal": ordinal, "path": str(target), "resumed": True}
    require(work.is_dir() and not work.is_symlink(),
            "interrupted work topology drift")
    rename_noreplace(work, target)
    return {"ordinal": ordinal, "path": str(target), "resumed": False}


def _install_one(stage, ordinal, plan_row):
    remote = _staged_verified(stage, ordinal, plan_row)
    paths = B.namespace_paths(ordinal)
    present = dict((key, os.path.lexists(str(path)))
                   for key, path in paths.items())
    if present["result"]:
        require(present == {"result": True, "attempt": True,
                            "work": False, "failure": False},
                "completed overlap topology drift")
        local = M2006.verify_local_shard(ordinal)
        require(M2006.canonical_sha(M2006.exact_receipt_core(local["row"])) ==
                plan_row["deterministic_core_sha256"],
                "local/remote deterministic core mismatch")
        return "local_overlap_retained", None
    require(not present["failure"], "failed overlap exists")
    quarantine = None
    if present["work"]:
        require(present["attempt"], "work exists without attempt")
        quarantine = _quarantine_work(paths["work"], ordinal)
    if present["attempt"]:
        require(sha256(paths["attempt"]) == remote["attempt_sha256"],
                "attempt identity mismatch")
    else:
        _copy_attempt(M2003._staged_paths(stage, ordinal)["attempt"],
                      paths["attempt"])
    _promote_result_resumable(
        M2003._staged_paths(stage, ordinal)["directory"],
        paths["result"], plan_row)
    M2006.verify_local_shard(ordinal)
    return "remote_installed", quarantine


def _merge(stage, index):
    counts = {"remote_installed": 0, "local_overlap_retained": 0,
              "quarantined": 0, "quarantine_resumed": 0}
    quarantine = []
    for ordinal in range(REMOTE_START, REMOTE_STOP):
        disposition, item = _install_one(stage, ordinal, index[ordinal])
        counts[disposition] += 1
        if item is not None:
            counts["quarantine_resumed" if item["resumed"] else
                   "quarantined"] += 1
            quarantine.append(item)
    require(counts["remote_installed"] + counts["local_overlap_retained"] ==
            REMOTE_STOP - REMOTE_START, "merge conservation failed")
    return counts, quarantine


def _publish_result(release_sha, attempt_sha, plan_seal, counts,
                    quarantine, aggregate, resumed):
    work = Path(str(RESULT) + ".work")
    work.mkdir(parents=True, mode=0o700)
    row = {"schema": SCHEMA, "status":
        "COMPLETE_8700_D0_NOREPLACE_MERGE__HAMMER_REQUIRED",
        "source_sha256": sha256(SOURCE), "release_sha256": release_sha,
        "attempt_sha256": attempt_sha,
        "plan_manifest_sha256": plan_seal["manifest_sha256"],
        "counts": counts, "quarantine": quarantine,
        "aggregate": aggregate, "manual_resume_used": resumed,
        "campaign_archive_open_count": 1,
        "resume_leg_archive_open_count": 0 if resumed else None,
        "shard_runs": 0, "payload_opens": 0, "deletes": 0,
        "overwrites": 0, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}
    (work / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    _publish_sealed_work(work, RESULT, "M2009 reducer result")
    return row


def _finish(release, release_sha, attempt_sha, stage, resumed):
    _plan, plan_seal, index = _load_plan(release)
    counts, quarantine = _merge(stage, index)
    for ordinal in range(8700):
        M2006.verify_local_shard(ordinal)
    aggregate = M1704.reduce_complete_sealed_shards()
    require(aggregate.get("complete_shards") == 8700 and
            aggregate.get("full_decoder") is False and
            aggregate.get("system_speedup") is False,
            "strong reducer boundary drift")
    return _publish_result(release_sha, attempt_sha, plan_seal, counts,
                           quarantine, aggregate, resumed)


def merge_and_reduce():
    release, release_sha = validate_runtime_release()
    attempt_sha = _consume_attempt(release, release_sha)
    for ordinal in range(LOCAL_STOP):
        M2006.verify_local_shard(ordinal)
    stage = None
    try:
        stage, _inspection = M2006.single_fd_verify_and_extract(
            Path(release["archive_path"]), release["archive_sha256"],
            REMOTE_START, REMOTE_STOP, STAGING_PARENT)
        rows = M2006.verify_all_remote_before_mutation(stage)
        _publish_plan(rows, release["archive_sha256"], stage)
        return _finish(release, release_sha, attempt_sha, stage, False)
    except BaseException as error:
        if not os.path.lexists(str(FAILURE)):
            work = Path(str(FAILURE) + ".work")
            if not work.exists():
                work.mkdir(parents=True, mode=0o700)
                (work / "result.json").write_text(json.dumps({
                    "schema": SCHEMA, "status":
                    "FAILED_MANUAL_PLAN_RESUME_ONLY",
                    "release_sha256": release_sha,
                    "attempt_sha256": attempt_sha,
                    "stage": str(stage) if stage is not None else None,
                    "plan_exists": PLAN.exists(),
                    "error_type": type(error).__name__,
                    "error": str(error), "automatic_retry": False
                    }, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
                _publish_sealed_work(work, FAILURE, "M2009 failure")
        raise


def manual_resume_from_plan():
    release, release_sha = validate_runtime_release()
    attempt_sha = _validate_attempt(release, release_sha)
    require(FAILURE.exists() and PLAN.exists() and not RESULT.exists(),
            "manual resume topology invalid")
    plan, _seal, _index = _load_plan(release)
    stage = Path(plan["staging_root"])
    require(stage.is_dir() and not stage.is_symlink(),
            "verified staging root unavailable")
    return _finish(release, release_sha, attempt_sha, stage, True)


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "repairs": {"renameat2_noreplace_all_canonical_publish": True,
            "orphan_import_work_validated_and_promoted": True,
            "campaign_archive_open_count": 1,
            "resume_leg_archive_open_count": 0,
            "process_exact_keys": True,
            "raw_cmdline_hash_self_consistent": True},
        "inherited_closed_m2004_findings": {
            "single_fd": True, "all_before_mutate": True,
            "explicit_m1706": True, "exact_minus_rss": True,
            "five_process_stop_receipt": True},
        "claim_boundary": {"source_only": True,
            "process_identity_capture": False, "archive_open": False,
            "merge": False, "reducer": False, "payload_opens": 0,
            "gpu_runs": 0, "eda_runs": 0, "full_d0_result": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    output = describe()
    if args.preflight:
        output = {"schema": SCHEMA,
            "status": "PASS_M2009_SOURCE_PREFLIGHT__NO_RUNTIME_ACTION",
            "authorities": validate_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
