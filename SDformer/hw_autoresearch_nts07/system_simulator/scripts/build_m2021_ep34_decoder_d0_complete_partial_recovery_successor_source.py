#!/usr/bin/env python3
"""M2021 source-only decoder-D0 recovery successor.

M2015 remains immutable and M2017 is forbidden.  This narrow successor fixes
the last independently reproduced recovery seam: an interrupted import may
already contain all three allowed filenames while its final seal is truncated.
Such topology-safe but unauthenticated evidence is preserved in a numbered
RENAME_NOREPLACE quarantine before the immutable staged tree is reverified and
copied afresh.  Authenticated plan-mismatching evidence remains preserved and
rejected.  Every inherited runtime namespace in M2015, M2012 and M2009 is also
bound to the M2021 paths.

The CLI exposes describe/preflight only.  It cannot capture production process
identity, open the remote archive, merge shards, run the reducer, GPU, or EDA.
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


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/"
    "test_m2021_ep34_decoder_d0_complete_partial_recovery_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m2021_ep34_decoder_d0_complete_partial_recovery_successor_"
    "source_contract_r1_20260902.json")
M2015_SOURCE = HERE / (
    "build_m2015_ep34_decoder_d0_authenticated_recovery_successor_source.py")
M2015_TEST = HW / (
    "system_simulator/tests/"
    "test_m2015_ep34_decoder_d0_authenticated_recovery_successor_source.py")
M2015_CONTRACT = HW / (
    "contracts/m2015_ep34_decoder_d0_authenticated_recovery_successor_"
    "source_contract_r1_20260902.json")
M2016_REVIEW = HW / (
    "reviews/m2016_m2015_ep34_decoder_d0_authenticated_recovery_"
    "successor_source_hammer_r1_20260902")
FORBIDDEN_M2017 = HW / (
    "contracts/m2017_m2016_m2015_ep34_decoder_d0_authenticated_recovery_"
    "release_r1_20260902.json")
FUTURE_REVIEW = HW / (
    "reviews/m2022_m2021_ep34_decoder_d0_complete_partial_recovery_"
    "successor_source_hammer_r1_20260902")
FUTURE_RELEASE = HW / (
    "contracts/m2023_m2022_m2021_ep34_decoder_d0_complete_partial_recovery_"
    "release_r1_20260902.json")
PRESTOP = HW / (
    "results/m2021_ep34_decoder_d0_local_campaign_process_identity_"
    "r1_20260902")
ATTEMPT = HW / (
    "results/.m2021_ep34_decoder_d0_complete_partial_recovery_attempt_consumed")
PLAN = HW / (
    "results/m2021_ep34_decoder_d0_remote_4500_8699_verified_plan_"
    "r1_20260902")
RESULT = HW / (
    "results/m2021_ep34_decoder_d0_8700_shard_complete_partial_recovery_"
    "reducer_r1_20260902")
FAILURE = HW / (
    "results/m2021_ep34_decoder_d0_merge_failed_manual_resume_allowed_"
    "r1_20260902")
STAGING_PARENT = HW / "staging/m2021_decoder_d0_remote_pack"
QUARANTINE_ROOT = HW / (
    "recovery_quarantine/m2021_decoder_d0_partial_import_work")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = (
    "m2021_ep34_decoder_d0_complete_partial_recovery_successor_source_r1_v1")
STATUS = (
    "SOURCE_ONLY__M2016_ONE_P1_ONE_P2_REPAIRED__M2017_FORBIDDEN__"
    "M2022_REVIEW_REQUIRED")
REVIEW_STATUS = (
    "PASS_M2022_M2021_DECODER_D0_COMPLETE_PARTIAL_RECOVERY_SOURCE__"
    "AUTHORIZE_PROCESS_CAPTURE_AND_M2023_AUTHORING")
RELEASE_SCHEMA = (
    "m2023_m2022_m2021_ep34_decoder_d0_complete_partial_recovery_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_M2021_ONE_FD_PLAN_COMPLETE_PARTIAL_RECOVERY_MERGE_REDUCER")
M2015_SOURCE_SHA256 = (
    "a60da7c35121ab15d069bd9006837f386c3b795179afc553b24d6290886ed5fb")
M2015_TEST_SHA256 = (
    "f4f2f5e87bfc1b3eacb493cb03cfb2d08ae83a471a2d9ffe185409b8540e501f")
M2015_CONTRACT_SHA256 = (
    "dcc1c1edc482283af1025f013f4e657a8a478f4eebb1861ebea529dc48de47e0")
M2016_REVIEW_SHA256 = (
    "9cf0cec26195e6317fd7185b964d4945ef8ad0eb567ed5b92db3438da202f848")
M2016_MANIFEST_SHA256 = (
    "7c03beb4722704cf6e3818ab8e935890986eb3c6ef3259bdf16c292481e8d57b")
M2016_OUTER_SHA256 = (
    "1d80db6372af9f03540c84703e2968875b1323a5e519d71abe7f5894f9a34645")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
RUNTIME_NAMES = (
    "PRESTOP", "ATTEMPT", "PLAN", "RESULT", "FAILURE", "STAGING_PARENT",
    "QUARANTINE_ROOT")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exact_m2015():
    if sha256(M2015_SOURCE) != M2015_SOURCE_SHA256:
        raise RuntimeError("exact M2015 source SHA drift")
    spec = importlib.util.spec_from_file_location("m2021_exact_m2015",
                                                  str(M2015_SOURCE))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import exact M2015")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCHEMA != (
            "m2015_ep34_decoder_d0_authenticated_recovery_successor_source_r1_v1"):
        raise RuntimeError("M2015 schema drift")
    return module


R = _load_exact_m2015()
Q = R.Q
P = R.P
B = R.B
M2006 = R.M2006
M1704 = R.M1704
M2003 = R.M2003
M2021Error = R.M2015Error
require = R.require
rename_noreplace = R.rename_noreplace
classify_process_records = R.classify_process_records
_proc_record = R._proc_record
LEGACY_RUNTIME = tuple(set(
    getattr(module, name) for module in (R, Q, P) for name in RUNTIME_NAMES))


def identity():
    return {"source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m2015_source_sha256": M2015_SOURCE_SHA256,
        "m2015_test_sha256": M2015_TEST_SHA256,
        "m2015_contract_sha256": M2015_CONTRACT_SHA256,
        "m2016_review_sha256": M2016_REVIEW_SHA256,
        "m2016_manifest_sha256": M2016_MANIFEST_SHA256,
        "m2016_outer_file_sha256": M2016_OUTER_SHA256,
        "m1706_release_sha256": P.M1706_RELEASE_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def validate_m2016_failure():
    B.verify_sealed_tree(M2016_REVIEW, M2016_REVIEW_SHA256,
        M2016_MANIFEST_SHA256, M2016_OUTER_SHA256, False, "M2016")
    row = B.strict_json(M2016_REVIEW / "review.json")
    require(row.get("score_over_100") == 86 and
            row.get("severity_counts") == {"p0": 0, "p1": 1, "p2": 1}
            and [item.get("id") for item in row.get("p1", [])] == [
                "P1_THREE_ALLOWED_NAMES_TRUNCATED_SEAL_STILL_STRANDS_RESUME"]
            and [item.get("id") for item in row.get("p2", [])] == [
                "P2_Q_MODULE_RETAINS_SIX_M2012_RUNTIME_PATH_CONSTANTS"]
            and row.get("authorization", {}).get(
                "successor_source_authoring") is True
            and row.get("authorization", {}).get(
                "process_identity_capture") == 0
            and row.get("authorization", {}).get(
                "m2017_release_authoring") is False,
            "M2016 disposition drift")


def _absent(path, label):
    paths = (Path(path), Path(str(path) + ".sha256"),
             Path(str(path) + ".sha256.seal.sha256"))
    require(all(not os.path.lexists(str(item)) for item in paths),
            label + " exists")


def validate_source_stage():
    P.regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    P.regular_exact(M2015_SOURCE, M2015_SOURCE_SHA256, "exact M2015 source")
    P.regular_exact(M2015_TEST, M2015_TEST_SHA256, "exact M2015 test")
    P.regular_exact(M2015_CONTRACT, M2015_CONTRACT_SHA256,
                    "exact M2015 contract")
    validate_m2016_failure()
    B.verify_double_sealed_file(SOURCE_CONTRACT, "M2021 source contract")
    _absent(FORBIDDEN_M2017, "forbidden M2017 release")
    require(not FUTURE_REVIEW.exists(), "future M2022 review exists")
    _absent(FUTURE_RELEASE, "future M2023 release")
    runtime = (PRESTOP, Path(str(PRESTOP) + ".work"), ATTEMPT, PLAN,
               Path(str(PLAN) + ".work"), RESULT,
               Path(str(RESULT) + ".work"), FAILURE,
               Path(str(FAILURE) + ".work"), STAGING_PARENT)
    require(all(not os.path.lexists(str(path)) for path in runtime),
            "future M2021 runtime artifact exists")
    require(all(not os.path.lexists(str(path)) for path in LEGACY_RUNTIME),
            "legacy decoder runtime artifact exists")
    return {"identity": identity(), "m2016": "one_p1_one_p2_bound",
            "process_capture": False, "archive_open": False,
            "merge": False, "reducer": False}


def _review_authorization():
    return {"process_identity_capture": 1,
        "m2023_release_authoring": 1, "archive_open": 0,
        "merge": 0, "reducer": 0, "payload_opens": 0,
        "gpu_runs": 0, "eda_runs": 0}


def validate_capture_review():
    seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M2022 capture authority")
    row = B.strict_json(FUTURE_REVIEW / "review.json")
    require(row.get("status") == REVIEW_STATUS and
            row.get("score_over_100", 0) >= 95 and
            row.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and row.get("identity") == identity() and
            row.get("authorization") == _review_authorization(),
            "M2022 process-capture authority drift")
    return row, seal


def capture_process_identity(pids):
    validate_capture_review()
    rows = classify_process_records([_proc_record(pid) for pid in pids])
    work = Path(str(PRESTOP) + ".work")
    work.mkdir(parents=True, mode=0o700)
    receipt = {"schema": SCHEMA, "status":
        "SEALED_LIVE_M1704_PROCESS_IDENTITY__STOP_PENDING",
        "source_sha256": sha256(SOURCE), "processes": rows,
        "captured_all_five_live": True, "reread_before_publish": True,
        "review_tree_authenticated_before_first_pid_read": True,
        "archive_open": 0, "merge": False, "reducer": False}
    (work / "result.json").write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    B.seal_work_tree(work)
    B.verify_sealed_tree(work, allow_ignored_pycache=False,
                         label="M2021 process identity work")
    Q.reread_process_records(rows, reader=_proc_record)
    rename_noreplace(work, PRESTOP)
    return receipt


def classify_import_orphan(work):
    """Authenticate complete names; all safe invalid seals remain partial."""
    topology = Q.inspect_import_work_topology(work)
    if topology == "incomplete":
        return "partial_unsealed"
    try:
        B.verify_sealed_tree(work, allow_ignored_pycache=False,
                             label="M2021 import orphan authentication")
    except Exception:
        return "partial_unsealed"
    return "authenticated_sealed"


def _ensure_quarantine_root():
    if not os.path.lexists(str(QUARANTINE_ROOT)):
        QUARANTINE_ROOT.mkdir(parents=True, mode=0o700)
    mode = QUARANTINE_ROOT.lstat().st_mode
    require(stat.S_ISDIR(mode) and not QUARANTINE_ROOT.is_symlink(),
            "quarantine root topology drift")


def quarantine_partial_import_work(work, target):
    """Preserve every topology-safe unauthenticated copy without overwrite."""
    work = Path(work)
    require(classify_import_orphan(work) == "partial_unsealed",
            "only topology-safe partial import-work may be quarantined")
    _ensure_quarantine_root()
    stem = Path(target).name + ".partial_import_work"
    for index in range(10000):
        candidate = QUARANTINE_ROOT / (stem + ".{:04d}".format(index))
        try:
            rename_noreplace(work, candidate)
            return candidate
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
    raise M2021Error("quarantine namespace exhausted")


def _promote_result_resumable(source, target, plan_row):
    source, target = Path(source), Path(target)
    import_work = Path(str(target) + ".m2021_import_work")
    if os.path.lexists(str(target)):
        require(not os.path.lexists(str(import_work)),
                "published target coexists with import-work")
        Q._verified_tree_matches_plan(target, plan_row,
                                      "M2021 already-published target")
        return "already_published"
    if os.path.lexists(str(import_work)):
        kind = classify_import_orphan(import_work)
        if kind == "authenticated_sealed":
            Q._verified_tree_matches_plan(
                import_work, plan_row, "M2021 authenticated import orphan")
        else:
            quarantine_partial_import_work(import_work, target)
            require(not os.path.lexists(str(import_work)),
                    "partial import-work quarantine failed")
    if not os.path.lexists(str(import_work)):
        Q._verified_tree_matches_plan(
            source, plan_row, "M2021 staged source before fresh copy")
        shutil.copytree(str(source), str(import_work), symlinks=False)
    Q._verified_tree_matches_plan(import_work, plan_row, "M2021 import work")
    rename_noreplace(import_work, target)
    return "published"


def _release_authorization():
    return {"overall_attempt": 1, "archive_open": 1,
        "archive_extract": 1, "verified_plan_publish": 1,
        "merge": 1, "manual_resume_from_plan": 1,
        "reducer": 1, "result_publish": 1, "shard_runs": 0,
        "payload_opens": 0, "deletes": 0, "overwrites": 0,
        "gpu_runs": 0, "eda_runs": 0}


def validate_runtime_release():
    review, review_seal = validate_capture_review()
    processes, process_seal = P.validate_process_receipt()
    release_sha = B.verify_double_sealed_file(FUTURE_RELEASE, "M2023 release")
    release = B.strict_json(FUTURE_RELEASE)
    expected_identity = dict(identity(),
        m2022_review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        m2022_manifest_sha256=review_seal["manifest_sha256"],
        m2022_outer_file_sha256=review_seal["outer_file_sha256"],
        prestop_result_sha256=sha256(PRESTOP / "result.json"),
        prestop_manifest_sha256=process_seal["manifest_sha256"],
        prestop_outer_file_sha256=process_seal["outer_file_sha256"])
    require(review.get("authorization") == _review_authorization() and
            release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == expected_identity and
            release.get("archive_path") ==
                "/tmp/m1704_remote_sealed_shards_4500_8699_20260902.tar" and
            len(release.get("archive_sha256", "")) == 64 and
            release.get("remote_range") == [P.REMOTE_START, P.REMOTE_STOP] and
            release.get("local_required_range") == [0, P.LOCAL_STOP] and
            release.get("processes") == processes["processes"] and
            release.get("authorization") == _release_authorization(),
            "M2023 release drift")
    require(all(not P._same_process_alive(row)
                for row in processes["processes"]),
            "captured campaign process is still alive")
    return release, release_sha


def _activate_successor_runtime():
    bindings = {
        "SOURCE": SOURCE, "TEST": TEST, "SOURCE_CONTRACT": SOURCE_CONTRACT,
        "FUTURE_REVIEW": FUTURE_REVIEW, "FUTURE_RELEASE": FUTURE_RELEASE,
        "PRESTOP": PRESTOP, "ATTEMPT": ATTEMPT, "PLAN": PLAN,
        "RESULT": RESULT, "FAILURE": FAILURE,
        "STAGING_PARENT": STAGING_PARENT, "QUARANTINE_ROOT": QUARANTINE_ROOT,
        "SCHEMA": SCHEMA, "STATUS": STATUS, "REVIEW_STATUS": REVIEW_STATUS,
        "RELEASE_SCHEMA": RELEASE_SCHEMA, "RELEASE_STATUS": RELEASE_STATUS,
        "identity": identity, "validate_source_stage": validate_source_stage,
        "capture_process_identity": capture_process_identity,
        "validate_runtime_release": validate_runtime_release,
        "_promote_result_resumable": _promote_result_resumable}
    for module in (R, Q, P):
        for name, value in bindings.items():
            setattr(module, name, value)


_activate_successor_runtime()
merge_and_reduce = P.merge_and_reduce
manual_resume_from_plan = P.manual_resume_from_plan


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "repairs": {
            "three_names_invalid_seal_local_noreplace_quarantine": True,
            "full_promote_recovery_and_idempotent_resume_tested": True,
            "authenticated_plan_mismatch_preserved_rejected": True,
            "staged_source_reverified_before_fresh_copy": True,
            "all_nested_runtime_paths_bound": True,
            "future_review_authenticated_before_first_pid_read": True,
            "five_process_reread_before_no_replace_publish": True},
        "inherited": {"single_archive_fd": True,
            "all_before_mutate": True, "sealed_plan": True,
            "all_rows_explicit_m1706": True, "exact_minus_rss": True,
            "campaign_archive_open_count": 1,
            "resume_leg_archive_open_count": 0},
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
            "status": "PASS_M2021_SOURCE_PREFLIGHT__NO_RUNTIME_ACTION",
            "authorities": validate_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
