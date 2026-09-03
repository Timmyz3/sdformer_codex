#!/usr/bin/env python3
"""M2012 source-only successor for safe dual-server D0 closure.

M2009 is immutable and M2011 is forbidden.  This narrow successor preserves
M2009's single-archive-FD, all-before-mutate, sealed-plan and Linux
RENAME_NOREPLACE transaction.  It closes the two M2010 findings only:

* a normal partial import-work directory is moved, without overwrite, into an
  evidence quarantine before a fresh copy from the reverified staging tree;
* the exact five live process identities are re-read after the receipt is
  sealed and immediately before its no-replace publication.

The CLI remains source-only.  It cannot capture a process, open an archive,
merge a shard, run the reducer, or invoke GPU/EDA work.
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
    "test_m2012_ep34_decoder_d0_recoverable_noreplace_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m2012_ep34_decoder_d0_recoverable_noreplace_successor_"
    "source_contract_r1_20260902.json")
M2009_SOURCE = HERE / (
    "build_m2009_ep34_decoder_d0_noreplace_resume_successor_source.py")
M2009_TEST = HW / (
    "system_simulator/tests/"
    "test_m2009_ep34_decoder_d0_noreplace_resume_successor_source.py")
M2009_CONTRACT = HW / (
    "contracts/m2009_ep34_decoder_d0_noreplace_resume_successor_"
    "source_contract_r1_20260902.json")
M2010_REVIEW = HW / (
    "reviews/m2010_m2009_ep34_decoder_d0_noreplace_resume_successor_"
    "source_hammer_r1_20260902")
FORBIDDEN_M2011 = HW / (
    "contracts/m2011_m2010_m2009_ep34_decoder_d0_noreplace_resume_"
    "release_r1_20260902.json")
FUTURE_REVIEW = HW / (
    "reviews/m2013_m2012_ep34_decoder_d0_recoverable_noreplace_"
    "successor_source_hammer_r1_20260902")
FUTURE_RELEASE = HW / (
    "contracts/m2014_m2013_m2012_ep34_decoder_d0_recoverable_noreplace_"
    "release_r1_20260902.json")
PRESTOP = HW / (
    "results/m2012_ep34_decoder_d0_local_campaign_process_identity_"
    "r1_20260902")
ATTEMPT = HW / (
    "results/.m2012_ep34_decoder_d0_recoverable_noreplace_attempt_consumed")
PLAN = HW / (
    "results/m2012_ep34_decoder_d0_remote_4500_8699_verified_plan_"
    "r1_20260902")
RESULT = HW / (
    "results/m2012_ep34_decoder_d0_8700_shard_recoverable_noreplace_"
    "reducer_r1_20260902")
FAILURE = HW / (
    "results/m2012_ep34_decoder_d0_merge_failed_manual_resume_allowed_"
    "r1_20260902")
STAGING_PARENT = HW / "staging/m2012_decoder_d0_remote_pack"
QUARANTINE_ROOT = HW / (
    "recovery_quarantine/m2012_decoder_d0_partial_import_work")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m2012_ep34_decoder_d0_recoverable_noreplace_successor_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__M2010_ONE_P1_ONE_P2_REPAIRED__M2011_FORBIDDEN__"
    "M2013_REVIEW_REQUIRED")
REVIEW_STATUS = (
    "PASS_M2013_M2012_DECODER_D0_RECOVERABLE_NOREPLACE_SOURCE__"
    "AUTHORIZE_PROCESS_CAPTURE_AND_M2014_AUTHORING")
RELEASE_SCHEMA = (
    "m2014_m2013_m2012_ep34_decoder_d0_recoverable_noreplace_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_M2012_ONE_FD_PLAN_RECOVERABLE_NOREPLACE_MERGE_REDUCER")
M2009_SOURCE_SHA256 = (
    "188619aaeabb381bbc1581e02392c40e6c61b2ddf3faf313186c6f750b94d8d9")
M2009_TEST_SHA256 = (
    "1e4cdfaf87bb0a28b96963269ac62fd576b7177990264898ebe3a1350c610f81")
M2009_CONTRACT_SHA256 = (
    "6dadbec2875007d3d742a2f0f4ea2f62e523ba349452056a30a389713995fe3d")
M2010_REVIEW_SHA256 = (
    "c6294e092fc2c8dfe89ea71c15b7254024e5e80e6b0b00999c8f27e976a8617a")
M2010_MANIFEST_SHA256 = (
    "e1f7d71b5ca39251a74ddb8387331b82df86737fd7dabe962c29d0d2fcd20a78")
M2010_OUTER_SHA256 = (
    "a50983e92eed4e5330f01b718b3e92d97c02c64be566955f928dffb180c475c9")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
ALLOWED_IMPORT_FILES = frozenset(
    ("result.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exact_m2009():
    if _sha256(M2009_SOURCE) != M2009_SOURCE_SHA256:
        raise RuntimeError("exact M2009 source SHA drift")
    spec = importlib.util.spec_from_file_location("m2012_exact_m2009",
                                                  str(M2009_SOURCE))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import exact M2009")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCHEMA != (
            "m2009_ep34_decoder_d0_noreplace_resume_successor_source_r1_v1"):
        raise RuntimeError("M2009 schema drift")
    return module


P = _load_exact_m2009()
B = P.B
M2006 = P.M2006
M1704 = P.M1704
M2003 = P.M2003
M2012Error = P.M2009Error
require = P.require
rename_noreplace = P.rename_noreplace
classify_process_records = P.classify_process_records
_proc_record = P._proc_record
LEGACY_M2009_RUNTIME = (
    P.PRESTOP, P.ATTEMPT, P.PLAN, P.RESULT, P.FAILURE, P.STAGING_PARENT)


def identity():
    return {
        "source_sha256": _sha256(SOURCE),
        "test_sha256": _sha256(TEST),
        "source_contract_sha256": _sha256(SOURCE_CONTRACT),
        "m2009_source_sha256": M2009_SOURCE_SHA256,
        "m2009_test_sha256": M2009_TEST_SHA256,
        "m2009_contract_sha256": M2009_CONTRACT_SHA256,
        "m2010_review_sha256": M2010_REVIEW_SHA256,
        "m2010_manifest_sha256": M2010_MANIFEST_SHA256,
        "m2010_outer_file_sha256": M2010_OUTER_SHA256,
        "m1706_release_sha256": P.M1706_RELEASE_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256,
    }


def validate_m2010_failure():
    B.verify_sealed_tree(M2010_REVIEW, M2010_REVIEW_SHA256,
        M2010_MANIFEST_SHA256, M2010_OUTER_SHA256, False, "M2010")
    row = B.strict_json(M2010_REVIEW / "review.json")
    require(row.get("score_over_100") == 86 and
            row.get("severity_counts") == {"p0": 0, "p1": 1, "p2": 1}
            and [item.get("id") for item in row.get("p1", [])] == [
                "P1_PARTIAL_IMPORT_WORK_STILL_STRANDS_MANUAL_RESUME"]
            and [item.get("id") for item in row.get("p2", [])] == [
                "P2_LIVE_PROCESS_RECEIPT_IS_NOT_REREAD_BEFORE_PUBLISH"]
            and row.get("authorization", {}).get(
                "successor_source_authoring") is True
            and row.get("authorization", {}).get(
                "process_identity_capture") == 0
            and row.get("authorization", {}).get(
                "m2011_release_authoring") is False,
            "M2010 disposition drift")


def _absent(path, label):
    paths = (Path(path), Path(str(path) + ".sha256"),
             Path(str(path) + ".sha256.seal.sha256"))
    require(all(not os.path.lexists(str(item)) for item in paths),
            label + " exists")


def validate_source_stage():
    P.regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    P.regular_exact(M2009_SOURCE, M2009_SOURCE_SHA256, "exact M2009 source")
    P.regular_exact(M2009_TEST, M2009_TEST_SHA256, "exact M2009 test")
    P.regular_exact(M2009_CONTRACT, M2009_CONTRACT_SHA256,
                    "exact M2009 contract")
    validate_m2010_failure()
    B.verify_double_sealed_file(SOURCE_CONTRACT, "M2012 source contract")
    _absent(FORBIDDEN_M2011, "forbidden M2011 release")
    require(not FUTURE_REVIEW.exists(), "future M2013 review exists")
    _absent(FUTURE_RELEASE, "future M2014 release")
    runtime = (PRESTOP, Path(str(PRESTOP) + ".work"), ATTEMPT, PLAN,
               Path(str(PLAN) + ".work"), RESULT,
               Path(str(RESULT) + ".work"), FAILURE,
               Path(str(FAILURE) + ".work"), STAGING_PARENT)
    require(all(not os.path.lexists(str(path)) for path in runtime),
            "future M2012 runtime artifact exists")
    require(all(not os.path.lexists(str(path)) for path in LEGACY_M2009_RUNTIME),
            "legacy M2009 runtime artifact exists")
    return {"identity": identity(), "m2010": "one_p1_one_p2_bound",
            "process_capture": False, "archive_open": False,
            "merge": False, "reducer": False}


def reread_process_records(initial_rows, reader=None):
    """Re-read and exactly match all five classified process identities."""
    if reader is None:
        reader = _proc_record
    reread = classify_process_records(
        [reader(row["pid"]) for row in initial_rows])
    require(reread == initial_rows,
            "live process identity changed before receipt publication")
    return reread


def capture_process_identity(pids):
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("identity") == identity() and
            review.get("authorization") == {
                "process_identity_capture": 1,
                "m2014_release_authoring": 1, "archive_open": 0,
                "merge": 0, "reducer": 0, "payload_opens": 0,
                "gpu_runs": 0, "eda_runs": 0},
            "M2013 process-capture authority drift")
    rows = classify_process_records([_proc_record(pid) for pid in pids])
    work = Path(str(PRESTOP) + ".work")
    work.mkdir(parents=True, mode=0o700)
    receipt = {"schema": SCHEMA, "status":
        "SEALED_LIVE_M1704_PROCESS_IDENTITY__STOP_PENDING",
        "source_sha256": _sha256(SOURCE), "processes": rows,
        "captured_all_five_live": True, "reread_before_publish": True,
        "archive_open": 0, "merge": False, "reducer": False}
    (work / "result.json").write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    B.seal_work_tree(work)
    B.verify_sealed_tree(work, allow_ignored_pycache=False,
                         label="M2012 process identity work")
    # This is deliberately the final operation before no-replace publish.
    reread_process_records(rows)
    rename_noreplace(work, PRESTOP)
    return receipt


def inspect_import_work_topology(work):
    """Return complete/incomplete; reject links, specials and alien names."""
    work = Path(work)
    try:
        mode = work.lstat().st_mode
    except OSError as error:
        raise M2012Error("missing import-work") from error
    require(stat.S_ISDIR(mode) and not work.is_symlink(),
            "import-work must be a real directory")
    observed = set()
    for entry in os.scandir(str(work)):
        require(entry.name in ALLOWED_IMPORT_FILES,
                "unexpected import-work entry")
        info = entry.stat(follow_symlinks=False)
        require(stat.S_ISREG(info.st_mode) and not entry.is_symlink(),
                "import-work entry must be regular non-symlink")
        observed.add(entry.name)
    return "complete" if observed == ALLOWED_IMPORT_FILES else "incomplete"


def _ensure_quarantine_root():
    if not os.path.lexists(str(QUARANTINE_ROOT)):
        QUARANTINE_ROOT.mkdir(parents=True, mode=0o700)
    mode = QUARANTINE_ROOT.lstat().st_mode
    require(stat.S_ISDIR(mode) and not QUARANTINE_ROOT.is_symlink(),
            "quarantine root topology drift")


def quarantine_partial_import_work(work, target):
    """Preserve a partial copy under the first no-replace quarantine slot."""
    require(inspect_import_work_topology(work) == "incomplete",
            "only incomplete normal import-work may be quarantined")
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
    raise M2012Error("quarantine namespace exhausted")


def _verified_tree_matches_plan(tree, plan_row, label):
    tree = Path(tree)
    seal = B.verify_sealed_tree(tree, allow_ignored_pycache=False,
                                label=label)
    row = B.strict_json(tree / "result.json")
    require(_sha256(tree / "result.json") ==
            plan_row["result_json_sha256"] and
            seal["manifest_sha256"] == plan_row["manifest_sha256"] and
            M2006.canonical_sha(M2006.exact_receipt_core(row)) ==
            plan_row["deterministic_core_sha256"],
            label + " does not match verified plan")
    return row, seal


def _promote_result_resumable(source, target, plan_row):
    import_work = Path(str(target) + ".m2012_import_work")
    if os.path.lexists(str(import_work)):
        topology = inspect_import_work_topology(import_work)
        if topology == "complete":
            # A complete but corrupt/mismatched orphan is malicious evidence:
            # reject and preserve it; never relabel it as a normal partial copy.
            _verified_tree_matches_plan(import_work, plan_row,
                                        "M2012 complete import orphan")
        else:
            quarantine_partial_import_work(import_work, target)
            require(not os.path.lexists(str(import_work)),
                    "partial import-work quarantine failed")
    if not os.path.lexists(str(import_work)):
        # Revalidate the immutable staging member after any recovery move and
        # before the sole fresh copy.  copytree is exclusive at this pathname.
        _verified_tree_matches_plan(source, plan_row,
                                    "M2012 staged source before fresh copy")
        shutil.copytree(str(source), str(import_work), symlinks=False)
    _verified_tree_matches_plan(import_work, plan_row, "M2012 import work")
    rename_noreplace(import_work, target)


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
        "_promote_result_resumable": _promote_result_resumable,
    }
    for name, value in bindings.items():
        setattr(P, name, value)


_activate_successor_runtime()
validate_runtime_release = P.validate_runtime_release
merge_and_reduce = P.merge_and_reduce
manual_resume_from_plan = P.manual_resume_from_plan


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "repairs": {
            "partial_import_work_noreplace_quarantine": True,
            "fresh_copy_after_staging_reverification": True,
            "complete_mismatched_orphan_preserved_and_rejected": True,
            "live_process_identity_reread_immediately_before_publish": True,
            "renameat2_noreplace_all_canonical_publish": True},
        "inherited_m2009": {
            "single_archive_fd": True, "all_before_mutate": True,
            "sealed_plan": True, "explicit_m1706": True,
            "exact_minus_rss": True, "campaign_archive_open_count": 1,
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
            "status": "PASS_M2012_SOURCE_PREFLIGHT__NO_RUNTIME_ACTION",
            "authorities": validate_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
