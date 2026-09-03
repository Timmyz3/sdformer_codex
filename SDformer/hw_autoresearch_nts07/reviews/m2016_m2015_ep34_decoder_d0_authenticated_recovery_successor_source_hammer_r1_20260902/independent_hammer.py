#!/usr/bin/env python3
"""M2016 independent, source-only hammer for M2015.

Every executable mutation is confined to TemporaryDirectory namespaces.  The
production process set, remote archive, shard/payload namespace, merge,
reducer, GPU, and EDA tools are never touched.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / (
    "system_simulator/scripts/"
    "build_m2015_ep34_decoder_d0_authenticated_recovery_successor_source.py")
EXPECTED_SOURCE_SHA256 = (
    "a60da7c35121ab15d069bd9006837f386c3b795179afc553b24d6290886ed5fb")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


require(sha256(SOURCE) == EXPECTED_SOURCE_SHA256, "M2015 source SHA drift")
spec = importlib.util.spec_from_file_location("m2016_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except (M.M2015Error, RuntimeError, OSError, ValueError, KeyError,
            AssertionError):
        return True
    raise AssertionError("expected fail-closed rejection")


def exact_receipt(ordinal=4500):
    return {
        "schema": "synthetic_shard", "status": "synthetic_complete",
        "source_sha256": "1" * 64, "release_sha256": "2" * 64,
        "attempt_sha256": "3" * 64,
        "checkpoint_sha256": M.B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": M.B.G.RESOURCE_SHA256,
        "shard_ordinal": ordinal, "shard": [ordinal, ordinal + 1],
        "configuration_order": ["A1"], "metrics": {"cycles": 1},
        "integer_ratio_inputs": {"numerator": 1, "denominator": 1},
        "payload_fd_sha256": "4" * 64, "payload_fd_size": 1,
        "rss": 123, "automatic_retry": False, "shard_isolated": True,
        "monolithic_full_call": True, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}


def make_sealed_tree(path, row=None):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700)
    row = exact_receipt() if row is None else row
    (path / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    M.B.seal_work_tree(path)
    seal = M.B.verify_sealed_tree(
        path, allow_ignored_pycache=False, label="synthetic sealed tree")
    return row, seal


def plan_row(path, row, seal):
    return {"ordinal": row["shard_ordinal"],
        "attempt_sha256": row["attempt_sha256"],
        "result_json_sha256": sha256(Path(path) / "result.json"),
        "manifest_sha256": seal["manifest_sha256"],
        "deterministic_core_sha256": M.M2006.canonical_sha(
            M.M2006.exact_receipt_core(row))}


def write_review(path, score, severities, sealed=True):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700)
    row = {"status": M.REVIEW_STATUS, "score_over_100": score,
        "severity_counts": severities, "identity": M.identity(),
        "authorization": {"process_identity_capture": 1,
            "m2017_release_authoring": 1, "archive_open": 0,
            "merge": 0, "reducer": 0, "payload_opens": 0,
            "gpu_runs": 0, "eda_runs": 0}}
    (path / "review.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    if sealed:
        M.B.seal_work_tree(path)
    return row


def process_record(pid, ppid, raw, start=None):
    return {"pid": pid, "ppid": ppid,
        "starttime_ticks": pid * 10 if start is None else start,
        "cmdline_raw_hex": raw.hex(),
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
        "cmdline_text": raw.replace(b"\0", b" ").decode().strip(),
        "cwd": str(M.HW.parent)}


def process_records():
    launcher = (b"bash build_m1704_ep34_decoder_d0_execution_authority_"
                b"adapter_source.py range(1,8700)\0")
    return [process_record(10, 1, launcher),
            process_record(11, 10, b"python3\0-\0"),
            process_record(12, 11, b"python3\0-\0"),
            process_record(13, 11, b"python3\0-\0"),
            process_record(14, 11, b"python3\0-\0")]


def main():
    positive = {}
    findings = {}
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)

        # The future review must be a fully sealed tree, have score >=95,
        # zero findings in all three severity classes, exact identity, and
        # exact narrow authority.
        old_review = M.FUTURE_REVIEW
        try:
            M.FUTURE_REVIEW = root / "review_unsealed"
            write_review(M.FUTURE_REVIEW, 100,
                         {"p0": 0, "p1": 0, "p2": 0}, False)
            expect_failure(M.validate_capture_review)
            M.FUTURE_REVIEW = root / "review_score_94"
            write_review(M.FUTURE_REVIEW, 94,
                         {"p0": 0, "p1": 0, "p2": 0})
            expect_failure(M.validate_capture_review)
            for severity in ("p0", "p1", "p2"):
                M.FUTURE_REVIEW = root / ("review_nonzero_" + severity)
                counts = {"p0": 0, "p1": 0, "p2": 0}
                counts[severity] = 1
                write_review(M.FUTURE_REVIEW, 100, counts)
                expect_failure(M.validate_capture_review)
            M.FUTURE_REVIEW = root / "review_exact_95"
            write_review(M.FUTURE_REVIEW, 95,
                         {"p0": 0, "p1": 0, "p2": 0})
            row, seal = M.validate_capture_review()
            require(row["score_over_100"] == 95 and
                    len(seal["manifest_sha256"]) == 64,
                    "valid sealed review did not pass")
            positive["sealed_score95_zero_severity_review_gate"] = True
        finally:
            M.FUTURE_REVIEW = old_review

        # Immutable staged source/plan used by all import recovery cases.
        staged = root / "staged"
        staged_row, staged_seal = make_sealed_tree(staged)
        expected = plan_row(staged, staged_row, staged_seal)
        old_m_quarantine = M.QUARANTINE_ROOT
        old_q_quarantine = M.Q.QUARANTINE_ROOT
        M.QUARANTINE_ROOT = root / "quarantine"
        M.Q.QUARANTINE_ROOT = M.QUARANTINE_ROOT
        try:
            # A strict subset left by copytree is quarantined and a fresh
            # sealed copy is published.
            subset_target = root / "subset_target"
            subset_work = Path(str(subset_target) + ".m2015_import_work")
            subset_work.mkdir()
            (subset_work / "result.json").write_text("{\"partial\":true}\n")
            M._promote_result_resumable(staged, subset_target, expected)
            quarantined = sorted(M.QUARANTINE_ROOT.iterdir())
            require(subset_target.is_dir() and not subset_work.exists() and
                    len(quarantined) == 1 and
                    (quarantined[0] / "result.json").exists(),
                    "strict-subset recovery failed")
            M.Q._verified_tree_matches_plan(
                subset_target, expected, "strict-subset recovered target")
            positive["incomplete_file_copy_recovers_and_preserves"] = True

            # The M2013 boundary: all three allowed pathnames exist, but the
            # final seal is truncated.  M2015 classifies it as a normal
            # partial copy, then calls M2012's helper.  That helper rechecks
            # pathname topology and only permits an incomplete name set, so
            # it rejects before moving the orphan.  Two calls demonstrate
            # that the sole resume path remains stranded.
            truncated_target = root / "truncated_target"
            truncated_work = Path(
                str(truncated_target) + ".m2015_import_work")
            truncated_work.mkdir()
            shutil.copyfile(str(staged / "result.json"),
                            str(truncated_work / "result.json"))
            shutil.copyfile(str(staged / "SHA256SUMS"),
                            str(truncated_work / "SHA256SUMS"))
            (truncated_work / "SHA256SUMS.seal.sha256").write_bytes(b"0")
            require(M.classify_import_orphan(truncated_work) ==
                    "partial_unsealed", "M2015 classifier boundary drift")
            require(M.Q.inspect_import_work_topology(truncated_work) ==
                    "complete", "M2012 helper topology boundary drift")
            first = expect_failure(lambda: M._promote_result_resumable(
                staged, truncated_target, expected))
            second = expect_failure(lambda: M._promote_result_resumable(
                staged, truncated_target, expected))
            require(first and second and truncated_work.is_dir() and
                    not truncated_target.exists(),
                    "truncated-seal stranded-resume finding did not reproduce")
            findings[
                "truncated_seal_recovery_rejected_by_reused_quarantine_helper"] = True

            # A correctly sealed but plan-mismatching import orphan is
            # authenticated, preserved in place, and rejected.
            mismatch_target = root / "mismatch_target"
            mismatch_work = Path(
                str(mismatch_target) + ".m2015_import_work")
            wrong_row, wrong_seal = make_sealed_tree(
                mismatch_work, exact_receipt(4501))
            require(M.classify_import_orphan(mismatch_work) ==
                    "authenticated_sealed", "sealed orphan not authenticated")
            expect_failure(lambda: M._promote_result_resumable(
                staged, mismatch_target, expected))
            require(mismatch_work.is_dir() and not mismatch_target.exists(),
                    "sealed mismatched evidence was not preserved/rejected")
            # Confirm the mismatched evidence remained byte-identical.
            require(sha256(mismatch_work / "result.json") ==
                    sha256(root / "mismatch_target.m2015_import_work" /
                           "result.json") and wrong_row["shard_ordinal"] == 4501 and
                    len(wrong_seal["manifest_sha256"]) == 64,
                    "sealed mismatch evidence changed")
            positive[
                "authenticated_sealed_plan_mismatch_preserved_rejected"] = True
        finally:
            M.QUARANTINE_ROOT = old_m_quarantine
            M.Q.QUARANTINE_ROOT = old_q_quarantine

        # Drive capture_process_identity only in a temporary namespace.  A
        # stable synthetic five-process topology must be read exactly twice;
        # PID reuse on the second pass must reject before publication.
        old_review = M.FUTURE_REVIEW
        old_prestop = M.PRESTOP
        old_reader = M._proc_record
        try:
            M.FUTURE_REVIEW = root / "capture_review"
            write_review(M.FUTURE_REVIEW, 100,
                         {"p0": 0, "p1": 0, "p2": 0})
            records = process_records()
            by_pid = dict((row["pid"], dict(row)) for row in records)
            calls = {"count": 0}

            def stable_reader(pid):
                calls["count"] += 1
                return dict(by_pid[pid])

            M.PRESTOP = root / "stable_process_receipt"
            M._proc_record = stable_reader
            receipt = M.capture_process_identity([10, 11, 12, 13, 14])
            require(calls["count"] == 10 and M.PRESTOP.is_dir() and
                    receipt["reread_before_publish"] is True,
                    "five-process exact second read missing")
            M.B.verify_sealed_tree(
                M.PRESTOP, allow_ignored_pycache=False,
                label="synthetic stable process receipt")

            changed_calls = {"count": 0}

            def changed_reader(pid):
                changed_calls["count"] += 1
                row = dict(by_pid[pid])
                if changed_calls["count"] > 5 and pid == 10:
                    row["starttime_ticks"] += 1
                return row

            M.PRESTOP = root / "changed_process_receipt"
            M._proc_record = changed_reader
            expect_failure(lambda: M.capture_process_identity(
                [10, 11, 12, 13, 14]))
            require(changed_calls["count"] == 10 and
                    not M.PRESTOP.exists() and
                    Path(str(M.PRESTOP) + ".work").is_dir(),
                    "PID-reuse seam was published or not reread")
            positive["five_process_records_reread_and_pid_reuse_rejected"] = True
        finally:
            M.FUTURE_REVIEW = old_review
            M.PRESTOP = old_prestop
            M._proc_record = old_reader

        # Effective runtime publication resolves through P and is rebound to
        # M2015.  Q's quarantine helper is also rebound, but Q's other legacy
        # path constants remain M2012 values despite M2015's broad comment.
        runtime_names = ("PRESTOP", "ATTEMPT", "PLAN", "RESULT", "FAILURE",
                         "STAGING_PARENT", "QUARANTINE_ROOT")
        require(all(getattr(M.P, name) == getattr(M, name)
                    for name in runtime_names),
                "P effective runtime path escaped M2015")
        require(M.Q.P is M.P and
                M.Q.QUARANTINE_ROOT == M.QUARANTINE_ROOT,
                "Q effective helper binding escaped M2015")
        require(M.P._promote_result_resumable is M._promote_result_resumable and
                M.P.capture_process_identity is M.capture_process_identity,
                "P callable binding escaped M2015")
        positive["effective_p_and_q_helper_paths_bound_to_m2015"] = True
        stale_q = [name for name in runtime_names
                   if getattr(M.Q, name) != getattr(M, name)]
        require(stale_q == ["PRESTOP", "ATTEMPT", "PLAN", "RESULT",
                            "FAILURE", "STAGING_PARENT"],
                "unexpected Q legacy-path population")
        findings["q_module_retains_inert_m2012_runtime_path_constants"] = True

    require(set(findings) == {
        "truncated_seal_recovery_rejected_by_reused_quarantine_helper",
        "q_module_retains_inert_m2012_runtime_path_constants"},
        "finding set drift")
    output = {"status": "FAIL_M2016_INDEPENDENT_SOURCE_HAMMER",
        "positive": positive, "findings": findings,
        "production_process_capture": False, "remote_access": False,
        "archive_open": False, "canonical_payload_open": False,
        "merge": False, "reducer": False, "gpu": False, "eda": False}
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
