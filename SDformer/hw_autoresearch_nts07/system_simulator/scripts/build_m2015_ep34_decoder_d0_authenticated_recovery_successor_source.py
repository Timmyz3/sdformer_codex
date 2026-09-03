#!/usr/bin/env python3
"""M2015 source-only successor closing the two M2013 findings.

M2012 remains immutable and M2014 is forbidden.  This source distinguishes an
authenticated, sealed import orphan from a normal interrupted copy whose three
pathnames may already exist but whose seal is incomplete.  It also authenticates
the complete future M2016 review tree, score, severities, identity and authority
before the first production PID read or receipt work-directory creation.

The CLI exposes describe/preflight only and performs no production action.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/"
    "test_m2015_ep34_decoder_d0_authenticated_recovery_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m2015_ep34_decoder_d0_authenticated_recovery_successor_"
    "source_contract_r1_20260902.json")
M2012_SOURCE = HERE / (
    "build_m2012_ep34_decoder_d0_recoverable_noreplace_successor_source.py")
M2012_TEST = HW / (
    "system_simulator/tests/"
    "test_m2012_ep34_decoder_d0_recoverable_noreplace_successor_source.py")
M2012_CONTRACT = HW / (
    "contracts/m2012_ep34_decoder_d0_recoverable_noreplace_successor_"
    "source_contract_r1_20260902.json")
M2013_REVIEW = HW / (
    "reviews/m2013_m2012_ep34_decoder_d0_recoverable_noreplace_"
    "successor_source_hammer_r1_20260902")
FORBIDDEN_M2014 = HW / (
    "contracts/m2014_m2013_m2012_ep34_decoder_d0_recoverable_noreplace_"
    "release_r1_20260902.json")
FUTURE_REVIEW = HW / (
    "reviews/m2016_m2015_ep34_decoder_d0_authenticated_recovery_"
    "successor_source_hammer_r1_20260902")
FUTURE_RELEASE = HW / (
    "contracts/m2017_m2016_m2015_ep34_decoder_d0_authenticated_recovery_"
    "release_r1_20260902.json")
PRESTOP = HW / (
    "results/m2015_ep34_decoder_d0_local_campaign_process_identity_"
    "r1_20260902")
ATTEMPT = HW / (
    "results/.m2015_ep34_decoder_d0_authenticated_recovery_attempt_consumed")
PLAN = HW / (
    "results/m2015_ep34_decoder_d0_remote_4500_8699_verified_plan_"
    "r1_20260902")
RESULT = HW / (
    "results/m2015_ep34_decoder_d0_8700_shard_authenticated_recovery_"
    "reducer_r1_20260902")
FAILURE = HW / (
    "results/m2015_ep34_decoder_d0_merge_failed_manual_resume_allowed_"
    "r1_20260902")
STAGING_PARENT = HW / "staging/m2015_decoder_d0_remote_pack"
QUARANTINE_ROOT = HW / (
    "recovery_quarantine/m2015_decoder_d0_partial_import_work")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m2015_ep34_decoder_d0_authenticated_recovery_successor_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__M2013_TWO_P1_REPAIRED__M2014_FORBIDDEN__"
    "M2016_REVIEW_REQUIRED")
REVIEW_STATUS = (
    "PASS_M2016_M2015_DECODER_D0_AUTHENTICATED_RECOVERY_SOURCE__"
    "AUTHORIZE_PROCESS_CAPTURE_AND_M2017_AUTHORING")
RELEASE_SCHEMA = (
    "m2017_m2016_m2015_ep34_decoder_d0_authenticated_recovery_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_M2015_ONE_FD_PLAN_AUTHENTICATED_RECOVERY_MERGE_REDUCER")
M2012_SOURCE_SHA256 = (
    "437faf4278acd9701ab6495fabfb08eaafba22065fb8aa63cc4194589d1de872")
M2012_TEST_SHA256 = (
    "71e2e281305f9bc154b74e5170975c5d0769b5a5b926f46a9d21931483ff0065")
M2012_CONTRACT_SHA256 = (
    "35e4be35e8bff708db3eceb33e484ef4bad2fbaa88eb6cc7970ec5f49d763667")
M2013_REVIEW_SHA256 = (
    "d43d29211934a78aa23c4a90bdf3774830e4ad257832e152b9cf2069d8476506")
M2013_MANIFEST_SHA256 = (
    "b52b73240a783678a12110721268742fcde32027020c839702a219bc6ae13814")
M2013_OUTER_SHA256 = (
    "b5280614b57fc89f246546d08f2275f254cc2b49826ac9e984b7ac9da14505b0")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exact_m2012():
    if sha256(M2012_SOURCE) != M2012_SOURCE_SHA256:
        raise RuntimeError("exact M2012 source SHA drift")
    spec = importlib.util.spec_from_file_location("m2015_exact_m2012",
                                                  str(M2012_SOURCE))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import exact M2012")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCHEMA != (
            "m2012_ep34_decoder_d0_recoverable_noreplace_successor_source_r1_v1"):
        raise RuntimeError("M2012 schema drift")
    return module


Q = _load_exact_m2012()
P = Q.P
B = Q.B
M2006 = Q.M2006
M1704 = Q.M1704
M2003 = Q.M2003
M2015Error = Q.M2012Error
require = Q.require
rename_noreplace = Q.rename_noreplace
classify_process_records = Q.classify_process_records
_proc_record = Q._proc_record
LEGACY_M2012_RUNTIME = (
    Q.PRESTOP, Q.ATTEMPT, Q.PLAN, Q.RESULT, Q.FAILURE, Q.STAGING_PARENT)


def identity():
    return {"source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m2012_source_sha256": M2012_SOURCE_SHA256,
        "m2012_test_sha256": M2012_TEST_SHA256,
        "m2012_contract_sha256": M2012_CONTRACT_SHA256,
        "m2013_review_sha256": M2013_REVIEW_SHA256,
        "m2013_manifest_sha256": M2013_MANIFEST_SHA256,
        "m2013_outer_file_sha256": M2013_OUTER_SHA256,
        "m1706_release_sha256": P.M1706_RELEASE_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def validate_m2013_failure():
    B.verify_sealed_tree(M2013_REVIEW, M2013_REVIEW_SHA256,
        M2013_MANIFEST_SHA256, M2013_OUTER_SHA256, False, "M2013")
    row = B.strict_json(M2013_REVIEW / "review.json")
    require(row.get("score_over_100") == 80 and
            row.get("severity_counts") == {"p0": 0, "p1": 2, "p2": 0}
            and [item.get("id") for item in row.get("p1", [])] == [
                "P1_THREE_ALLOWED_NAMES_WITH_TRUNCATED_SEAL_STRAND_RESUME",
                "P1_UNSEALED_REVIEW_CAN_AUTHORIZE_PROCESS_CAPTURE"]
            and row.get("authorization", {}).get(
                "m2015_successor_source_authoring") is True
            and row.get("authorization", {}).get(
                "process_identity_capture") == 0
            and row.get("authorization", {}).get(
                "m2014_release_authoring") is False,
            "M2013 disposition drift")


def _absent(path, label):
    paths = (Path(path), Path(str(path) + ".sha256"),
             Path(str(path) + ".sha256.seal.sha256"))
    require(all(not os.path.lexists(str(item)) for item in paths),
            label + " exists")


def validate_source_stage():
    P.regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    P.regular_exact(M2012_SOURCE, M2012_SOURCE_SHA256, "exact M2012 source")
    P.regular_exact(M2012_TEST, M2012_TEST_SHA256, "exact M2012 test")
    P.regular_exact(M2012_CONTRACT, M2012_CONTRACT_SHA256,
                    "exact M2012 contract")
    validate_m2013_failure()
    B.verify_double_sealed_file(SOURCE_CONTRACT, "M2015 source contract")
    _absent(FORBIDDEN_M2014, "forbidden M2014 release")
    require(not FUTURE_REVIEW.exists(), "future M2016 review exists")
    _absent(FUTURE_RELEASE, "future M2017 release")
    runtime = (PRESTOP, Path(str(PRESTOP) + ".work"), ATTEMPT, PLAN,
               Path(str(PLAN) + ".work"), RESULT,
               Path(str(RESULT) + ".work"), FAILURE,
               Path(str(FAILURE) + ".work"), STAGING_PARENT)
    require(all(not os.path.lexists(str(path)) for path in runtime),
            "future M2015 runtime artifact exists")
    require(all(not os.path.lexists(str(path))
                for path in LEGACY_M2012_RUNTIME),
            "legacy M2012 runtime artifact exists")
    return {"identity": identity(), "m2013": "two_p1_bound",
            "process_capture": False, "archive_open": False,
            "merge": False, "reducer": False}


def validate_capture_review():
    seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M2016 capture authority")
    row = B.strict_json(FUTURE_REVIEW / "review.json")
    require(row.get("status") == REVIEW_STATUS and
            row.get("score_over_100", 0) >= 95 and
            row.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and row.get("identity") == identity() and
            row.get("authorization") == {
                "process_identity_capture": 1,
                "m2017_release_authoring": 1, "archive_open": 0,
                "merge": 0, "reducer": 0, "payload_opens": 0,
                "gpu_runs": 0, "eda_runs": 0},
            "M2016 process-capture authority drift")
    return row, seal


def capture_process_identity(pids):
    # Authentication is deliberately before the first PID read and before the
    # no-replace receipt work namespace is created.
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
                         label="M2015 process identity work")
    Q.reread_process_records(rows, reader=_proc_record)
    rename_noreplace(work, PRESTOP)
    return receipt


def classify_import_orphan(work):
    """Authenticate a complete name set; invalid seal is a partial copy."""
    topology = Q.inspect_import_work_topology(work)
    if topology == "incomplete":
        return "partial_unsealed"
    try:
        B.verify_sealed_tree(work, allow_ignored_pycache=False,
                             label="M2015 import orphan authentication")
    except Exception:
        # Preserve it by quarantine, but do not let a normal final-file
        # truncation strand the only manual-resume path.
        return "partial_unsealed"
    return "authenticated_sealed"


def _promote_result_resumable(source, target, plan_row):
    import_work = Path(str(target) + ".m2015_import_work")
    if os.path.lexists(str(import_work)):
        kind = classify_import_orphan(import_work)
        if kind == "authenticated_sealed":
            # Authenticated but plan-mismatched evidence is never reclassified
            # as a normal crash and is preserved at the fixed name on failure.
            Q._verified_tree_matches_plan(
                import_work, plan_row, "M2015 authenticated import orphan")
        else:
            Q.quarantine_partial_import_work(import_work, target)
            require(not os.path.lexists(str(import_work)),
                    "partial import-work quarantine failed")
    if not os.path.lexists(str(import_work)):
        Q._verified_tree_matches_plan(
            source, plan_row, "M2015 staged source before fresh copy")
        shutil.copytree(str(source), str(import_work), symlinks=False)
    Q._verified_tree_matches_plan(import_work, plan_row, "M2015 import work")
    rename_noreplace(import_work, target)


def _activate_successor_runtime():
    # Q helpers resolve Q module globals; the inherited P runtime resolves P
    # globals.  Bind both explicitly so the nested exact-source inheritance is
    # testable and cannot silently publish into an old namespace.
    q_bindings = {"QUARANTINE_ROOT": QUARANTINE_ROOT}
    for name, value in q_bindings.items():
        setattr(Q, name, value)
    p_bindings = {
        "SOURCE": SOURCE, "TEST": TEST, "SOURCE_CONTRACT": SOURCE_CONTRACT,
        "FUTURE_REVIEW": FUTURE_REVIEW, "FUTURE_RELEASE": FUTURE_RELEASE,
        "PRESTOP": PRESTOP, "ATTEMPT": ATTEMPT, "PLAN": PLAN,
        "RESULT": RESULT, "FAILURE": FAILURE,
        "STAGING_PARENT": STAGING_PARENT, "QUARANTINE_ROOT": QUARANTINE_ROOT,
        "SCHEMA": SCHEMA, "STATUS": STATUS, "REVIEW_STATUS": REVIEW_STATUS,
        "RELEASE_SCHEMA": RELEASE_SCHEMA, "RELEASE_STATUS": RELEASE_STATUS,
        "identity": identity, "validate_source_stage": validate_source_stage,
        "capture_process_identity": capture_process_identity,
        "_promote_result_resumable": _promote_result_resumable}
    for name, value in p_bindings.items():
        setattr(P, name, value)


_activate_successor_runtime()
validate_runtime_release = P.validate_runtime_release
merge_and_reduce = P.merge_and_reduce
manual_resume_from_plan = P.manual_resume_from_plan


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "repairs": {
            "three_names_invalid_seal_is_quarantined_partial": True,
            "valid_sealed_plan_mismatch_is_preserved_rejected": True,
            "staged_source_reverified_before_fresh_copy": True,
            "future_review_tree_authenticated_before_first_pid_read": True,
            "review_score_at_least_95_and_zero_severity": True,
            "five_process_reread_before_no_replace_publish": True},
        "inherited": {"m2012_exact_source": True,
            "renameat2_noreplace": True, "single_archive_fd": True,
            "all_before_mutate": True, "sealed_plan": True,
            "explicit_m1706": True, "exact_minus_rss": True,
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
            "status": "PASS_M2015_SOURCE_PREFLIGHT__NO_RUNTIME_ACTION",
            "authorities": validate_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
