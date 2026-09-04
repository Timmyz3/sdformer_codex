#!/usr/bin/env python3
"""Manual recovery of exactly three interrupted M1681 D0 shards.

The original M1681 attempts for ordinals 7560..7562 are immutable and remain
the attempts named by the canonical receipts.  Their empty work directories
are preserved with RENAME_NOREPLACE before the exact frozen M1681 payload,
scheduler, metric, and receipt path is re-entered.  This is a narrowly released
manual recovery, not an automatic retry and not new shard/attempt budget.

No reducer or broader decoder claim is included.  Execution requires an
independent M2094 source review, a double-sealed M2095 release, and a detached
non-interactive session leader.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import traceback


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
CONTRACT = HW / (
    "contracts/m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "source_contract_r1_20260904.json")
M2090_SOURCE = HERE / (
    "run_m2090_ep34_decoder_d0_fresh_tail_process_isolated_resume.py")
M2091_REVIEW = HW / (
    "reviews/m2091_m2090_ep34_decoder_d0_fresh_tail_process_isolated_"
    "resume_source_hammer_r1_20260904")
M2092_RELEASE = HW / (
    "contracts/m2092_m2091_m2090_ep34_decoder_d0_fresh_tail_"
    "process_isolated_resume_release_r1_20260904.json")
FUTURE_REVIEW = HW / (
    "reviews/m2094_m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "source_hammer_r1_20260904")
FUTURE_RELEASE = HW / (
    "contracts/m2095_m2094_m2093_ep34_decoder_d0_three_orphan_manual_"
    "recovery_release_r1_20260904.json")
ATTEMPT = HW / (
    "results/.m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "attempt_consumed")
RESULT = HW / (
    "results/m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "r1_20260904")
WORK = HW / (
    "results/.m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "r1_20260904.work")
FAILURE = HW / (
    "results/m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "r1_20260904.failed_no_retry")
QUARANTINE = HW / (
    "recovery_quarantine/m2093_decoder_d0_three_orphan_original_empty_work")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m2093_ep34_decoder_d0_three_orphan_manual_recovery_r1_v1"
STATUS = "SOURCE_ONLY__M2094_REVIEW_AND_M2095_RELEASE_REQUIRED"
REVIEW_STATUS = (
    "PASS_M2094_M2093_DECODER_D0_THREE_ORPHAN_MANUAL_RECOVERY_SOURCE__"
    "AUTHORIZE_M2095_RELEASE_ONLY")
RELEASE_SCHEMA = (
    "m2095_m2094_m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_M2093_ONE_DETACHED_THREE_ORPHAN_MANUAL_RECOVERY")
M2090_SHA256 = (
    "23b5c41ac50a13de8a3c2e7e5f46c666de3ed7326f629c6d40fc4b4f577017c7")
M2091_REVIEW_SHA256 = (
    "37bd79c50ca33807b38372f10cf80e67ae1f894813298ee98b2f117d1aa1630b")
M2091_MANIFEST_SHA256 = (
    "5d4ca74abc1793668d5569f3fbc6999fb56fc843a582b7f78b3afa93509d03da")
M2091_OUTER_SHA256 = (
    "51849581c3007a1dba96b83c0ede18932ebd64de4ebc2a64f9f1dd92227d9976")
M2092_SHA256 = (
    "9e88449cc0e41b20cc151225381bb8cb3f17f665778b8371c707cacb8132cc09")
M1706_SHA256 = (
    "43c7096fe90263abf7593d41c3222675bc9153ca4529436b3a57405c550fe7e0")
M1681_SHA256 = (
    "006535679b38e2aa207fadde05e9207d2e72dae0464315dceea4a3c96da77a6f")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
ORDINALS = (7560, 7561, 7562)


class M2093Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M2093Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_m2090():
    regular_exact(M2090_SOURCE, M2090_SHA256, "exact M2090 source")
    spec = importlib.util.spec_from_file_location("m2093_exact_m2090",
                                                  str(M2090_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M2090")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m2090_ep34_decoder_d0_fresh_tail_process_isolated_resume_r1_v1",
            "M2090 schema drift")
    return module


M2090 = load_m2090()
B = M2090.B


def _identity():
    return {"source_sha256": sha256(SOURCE),
        "contract_sha256": sha256(CONTRACT),
        "m2090_source_sha256": M2090_SHA256,
        "m2091_review_sha256": M2091_REVIEW_SHA256,
        "m2091_manifest_sha256": M2091_MANIFEST_SHA256,
        "m2091_outer_file_sha256": M2091_OUTER_SHA256,
        "m2092_release_sha256": M2092_SHA256,
        "m1706_release_sha256": M1706_SHA256,
        "m1681_source_sha256": M1681_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def _absent(path, label):
    require(not os.path.lexists(str(path)), label + " exists")


def _expected_attempt(ordinal):
    return {"schema": B.SCHEMA, "shard_ordinal": ordinal,
        "shard": B.G.shard_descriptor(ordinal),
        "source_sha256": M1681_SHA256,
        "release_sha256": M1706_SHA256,
        "automatic_retry": False,
        "payload_opened_before_attempt": False}


def _validate_orphan(ordinal):
    paths = B.namespace_paths(ordinal)
    mode = paths["attempt"].lstat().st_mode
    require(stat.S_ISREG(mode) and not paths["attempt"].is_symlink() and
            stat.S_IMODE(mode) == 0o400,
            "original attempt topology drift")
    require(B.strict_json(paths["attempt"]) == _expected_attempt(ordinal),
            "original attempt semantic identity drift")
    require(paths["work"].is_dir() and not paths["work"].is_symlink() and
            not any(paths["work"].iterdir()),
            "original interrupted work is not an empty regular directory")
    require(not os.path.lexists(str(paths["result"])) and
            not os.path.lexists(str(paths["failure"])),
            "orphan already has result/failure")
    return {"ordinal": ordinal,
        "attempt_sha256": sha256(paths["attempt"]),
        "attempt_mode": "0400", "empty_work": True}


def validate_source_topology():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    B.verify_double_sealed_file(CONTRACT, "M2093 source contract")
    regular_exact(M2090_SOURCE, M2090_SHA256, "exact M2090 source")
    B.verify_sealed_tree(M2091_REVIEW, M2091_REVIEW_SHA256,
        M2091_MANIFEST_SHA256, M2091_OUTER_SHA256, False, "M2091")
    regular_exact(M2092_RELEASE, M2092_SHA256, "exact M2092 release")
    B.verify_double_sealed_file(M2092_RELEASE, "M2092 release")
    regular_exact(B.SOURCE, M1681_SHA256, "exact M1681 source")
    rows = [_validate_orphan(ordinal) for ordinal in ORDINALS]
    for path, label in ((ATTEMPT, "M2093 attempt"), (RESULT, "M2093 result"),
                        (WORK, "M2093 work"), (FAILURE, "M2093 failure"),
                        (QUARANTINE, "M2093 quarantine")):
        _absent(path, label)
    return rows


def _validate_future_gate():
    seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M2094")
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and review.get("identity") == _identity() and
            review.get("authorization") == {
                "m2095_release_authoring": 1,
                "manual_recovery_execution": 0,
                "reducer_execution": 0},
            "M2094 review authority drift")
    release_sha = B.verify_double_sealed_file(FUTURE_RELEASE, "M2095")
    release = B.strict_json(FUTURE_RELEASE)
    identity = dict(_identity(),
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=seal["manifest_sha256"],
        review_outer_file_sha256=seal["outer_file_sha256"])
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == identity and
            release.get("authorization") == {
                "detached_launcher_runs": 1,
                "manual_recovery_shard_runs": 3,
                "payload_opens": 3,
                "new_m1681_shard_attempt_writes": 0,
                "outer_orchestration_attempt_writes": 1,
                "automatic_retry": False,
                "reducer_runs": 0,
                "gpu_runs": 0,
                "eda_runs": 0} and
            release.get("ordinals") == list(ORDINALS) and
            release.get("claim_boundary") == {
                "manual_recovery_only": True,
                "exact_m1681_compute_and_receipt_schema": True,
                "full_d0_result": False, "full_decoder": False,
                "system_speedup": False, "paper_result": False},
            "M2095 release drift")
    return release_sha


def _validate_detached_launch():
    require(os.environ.get("M2093_MANUAL_RECOVERY") == "1",
            "explicit manual-recovery token missing")
    require(os.getsid(0) == os.getpid(),
            "manual recovery must be a setsid session leader")
    require(not any(os.isatty(descriptor) for descriptor in (0, 1, 2)),
            "manual recovery must not retain a terminal")


def _consume_outer_attempt(release_sha, orphans):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(ATTEMPT), flags, 0o400)
    try:
        row = {"schema": SCHEMA, "source_sha256": sha256(SOURCE),
            "release_sha256": release_sha, "ordinals": list(ORDINALS),
            "original_orphans": orphans,
            "new_m1681_shard_attempt_writes": 0,
            "automatic_retry": False, "payload_opened_before_attempt": False}
        payload = (json.dumps(row, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(str(ATTEMPT), 0o400)
    return sha256(ATTEMPT)


def _recover_one(ordinal, release_sha):
    paths = B.namespace_paths(ordinal)
    original_attempt_sha = sha256(paths["attempt"])
    quarantine = QUARANTINE / ("shard_{:04d}_original_empty_work".format(
        ordinal))
    M2090._rename_noreplace(paths["work"], quarantine)
    paths["work"].mkdir(mode=0o700)
    published = False
    try:
        B.G.R.validate_authorities(True)
        shard = B.G.shard_descriptor(ordinal)
        record = B.G.selected_record(shard)
        payload = (B.G.R.M1521_ROOT / record["positive_output"]).resolve()
        require(payload.parent ==
                (B.G.R.M1521_ROOT / "payloads").resolve(),
                "canonical payload path escaped payload directory")
        rss = B.G.P.RssGate()
        plane = B.ImmutableTimestepPlane(payload, record["shape"],
            record["positive_output_sha256"], shard["timestep"])
        rss.sample()
        metrics = B._schedule_actual_shard(shard, plane, rss)
        rss.sample()
        row = {"schema": B.RESULT_SCHEMA, "status": B.RESULT_STATUS,
            "source_sha256": M1681_SHA256,
            "release_sha256": M1706_SHA256,
            "attempt_sha256": original_attempt_sha,
            "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
            "resource_manifest_sha256": B.G.RESOURCE_SHA256,
            "shard_ordinal": ordinal, "shard": shard,
            "configuration_order": list(B.CONFIGS), "metrics": metrics,
            "integer_ratio_inputs":
                B.G.validate_three_configuration_metrics(metrics, shard),
            "payload_fd_sha256": plane.opened_sha256,
            "payload_fd_size": plane.opened_size,
            "rss": rss.summary(), "automatic_retry": False,
            "shard_isolated": True, "monolithic_full_call": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result": False,
            "independent_result_hammer_pending": True,
            "manual_recovery": {"source_sha256": sha256(SOURCE),
                "release_sha256": release_sha,
                "original_attempt_preserved": True,
                "original_empty_work_quarantine":
                    str(quarantine.relative_to(HW)),
                "new_m1681_shard_attempt_written": False}}
        B.validate_shard_receipt(row, ordinal, original_attempt_sha,
                                 M1706_SHA256)
        (paths["work"] / "result.json").write_text(json.dumps(
            row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        B.seal_work_tree(paths["work"])
        B.verify_sealed_tree(paths["work"], allow_ignored_pycache=False,
                             label="M2093 recovered shard work")
        M2090._rename_noreplace(paths["work"], paths["result"])
        published = True
        return {"ordinal": ordinal, "attempt_sha256": original_attempt_sha,
            "result_manifest_sha256": sha256(paths["result"] / "SHA256SUMS"),
            "payload_fd_sha256": plane.opened_sha256}
    except BaseException:
        if paths["work"].is_dir() and not os.path.lexists(
                str(paths["failure"])):
            (paths["work"] / "recovery_failure.json").write_text(json.dumps({
                "schema": SCHEMA, "ordinal": ordinal,
                "status": "FAILED_MANUAL_RECOVERY_NO_RETRY",
                "traceback": traceback.format_exc()}, indent=2,
                sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            B.seal_work_tree(paths["work"])
            M2090._rename_noreplace(paths["work"], paths["failure"])
        raise
    finally:
        require(published or not paths["work"].exists(),
                "unpublished recovered-shard work remains")


def execute():
    _validate_detached_launch()
    release_sha = _validate_future_gate()
    orphans = validate_source_topology()
    attempt_sha = _consume_outer_attempt(release_sha, orphans)
    QUARANTINE.mkdir(parents=True, mode=0o700)
    WORK.mkdir(mode=0o700)
    published = False
    try:
        recovered = [_recover_one(ordinal, release_sha)
                     for ordinal in ORDINALS]
        for ordinal in ORDINALS:
            B.verify_sealed_shard(ordinal)
        receipt = {"schema": SCHEMA,
            "status": "PASS_M2093_THREE_ORPHAN_MANUAL_RECOVERY",
            "source_sha256": sha256(SOURCE),
            "release_sha256": release_sha, "attempt_sha256": attempt_sha,
            "ordinals": list(ORDINALS), "recovered": recovered,
            "original_attempts_preserved": True,
            "new_m1681_shard_attempt_writes": 0,
            "reducer_executed": False, "full_d0_result": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result": False, "independent_result_hammer_pending": True}
        (WORK / "result.json").write_text(json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        B.seal_work_tree(WORK)
        B.verify_sealed_tree(WORK, allow_ignored_pycache=False,
                             label="M2093 result work")
        M2090._rename_noreplace(WORK, RESULT)
        published = True
        return receipt
    except BaseException as error:
        if WORK.is_dir() and not os.path.lexists(str(FAILURE)):
            (WORK / "failure.json").write_text(json.dumps({
                "schema": SCHEMA, "status": "FAILED_NO_RETRY",
                "error": repr(error), "traceback": traceback.format_exc(),
                "automatic_retry": False}, indent=2, sort_keys=True,
                allow_nan=False) + "\n", encoding="utf-8")
            B.seal_work_tree(WORK)
            M2090._rename_noreplace(WORK, FAILURE)
        raise
    finally:
        require(published or not WORK.exists(),
                "unpublished M2093 work remains")


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "ordinals": list(ORDINALS),
        "reuse": "exact M1681 payload/scheduler/metric/receipt schema",
        "preservation": "original attempts and empty work evidence",
        "claim_boundary": {"source_only": True,
            "manual_recovery_only": True, "new_algorithm": False,
            "automatic_retry": False, "new_m1681_attempt_budget": False,
            "reducer_execution": False, "full_d0_result": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        output = execute()
    elif args.preflight:
        output = {"schema": SCHEMA,
            "status": "PASS_M2093_SOURCE_PREFLIGHT_NO_EXECUTION",
            "identity": _identity(), "orphans": validate_source_topology(),
            "execution": False}
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
