#!/usr/bin/env python3
"""No-production regression for M2021 complete-name partial recovery."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_m2021_ep34_decoder_d0_complete_partial_recovery_successor_source.py")
spec = importlib.util.spec_from_file_location("m2021_test_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except (M.M2021Error, RuntimeError, OSError, ValueError, KeyError,
            AssertionError):
        return
    raise AssertionError("expected fail-closed rejection")


def exact_receipt(ordinal=4500):
    return {"schema": "synthetic_shard", "status": "synthetic_complete",
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
    path.mkdir(parents=True, mode=0o700)
    row = exact_receipt() if row is None else row
    (path / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    M.B.seal_work_tree(path)
    seal = M.B.verify_sealed_tree(
        path, allow_ignored_pycache=False, label="synthetic tree")
    return row, seal


def plan_row(path, row, seal):
    return {"ordinal": row["shard_ordinal"],
        "attempt_sha256": row["attempt_sha256"],
        "result_json_sha256": M.sha256(path / "result.json"),
        "manifest_sha256": seal["manifest_sha256"],
        "deterministic_core_sha256": M.M2006.canonical_sha(
            M.M2006.exact_receipt_core(row))}


def write_review(path, score=100, severity=None, sealed=True):
    severity = {"p0": 0, "p1": 0, "p2": 0} if severity is None else severity
    path.mkdir(parents=True, mode=0o700)
    row = {"status": M.REVIEW_STATUS, "score_over_100": score,
        "severity_counts": severity, "identity": M.identity(),
        "authorization": M._review_authorization()}
    (path / "review.json").write_text(json.dumps(row) + "\n")
    if sealed:
        M.B.seal_work_tree(path)


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
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        staged = root / "staged"
        row, seal = make_sealed_tree(staged)
        expected = plan_row(staged, row, seal)

        old_quarantine = M.QUARANTINE_ROOT
        M.QUARANTINE_ROOT = root / "quarantine"
        try:
            subset_target = root / "subset_target"
            subset_work = Path(str(subset_target) + ".m2021_import_work")
            subset_work.mkdir()
            (subset_work / "result.json").write_text("{}\n")
            assert M._promote_result_resumable(
                staged, subset_target, expected) == "published"
            M.Q._verified_tree_matches_plan(
                subset_target, expected, "subset recovered")

            truncated_target = root / "truncated_target"
            truncated_work = Path(str(truncated_target) + ".m2021_import_work")
            truncated_work.mkdir()
            shutil.copyfile(str(staged / "result.json"),
                            str(truncated_work / "result.json"))
            shutil.copyfile(str(staged / "SHA256SUMS"),
                            str(truncated_work / "SHA256SUMS"))
            (truncated_work / "SHA256SUMS.seal.sha256").write_bytes(b"0")
            assert M.classify_import_orphan(truncated_work) == "partial_unsealed"
            assert M.Q.inspect_import_work_topology(truncated_work) == "complete"
            assert M._promote_result_resumable(
                staged, truncated_target, expected) == "published"
            assert not truncated_work.exists()
            M.Q._verified_tree_matches_plan(
                truncated_target, expected, "truncated recovered")
            before = sorted(M.QUARANTINE_ROOT.iterdir())
            assert M._promote_result_resumable(
                staged, truncated_target, expected) == "already_published"
            assert sorted(M.QUARANTINE_ROOT.iterdir()) == before

            mismatch_target = root / "mismatch_target"
            mismatch_work = Path(str(mismatch_target) + ".m2021_import_work")
            make_sealed_tree(mismatch_work, exact_receipt(4501))
            expect_failure(lambda: M._promote_result_resumable(
                staged, mismatch_target, expected))
            assert mismatch_work.is_dir() and not mismatch_target.exists()
            assert len(list(M.QUARANTINE_ROOT.iterdir())) == 2
        finally:
            M.QUARANTINE_ROOT = old_quarantine

        old_review, old_prestop, old_reader = (
            M.FUTURE_REVIEW, M.PRESTOP, M._proc_record)
        try:
            M.FUTURE_REVIEW = root / "unsealed_review"
            write_review(M.FUTURE_REVIEW, sealed=False)
            expect_failure(M.validate_capture_review)
            M.FUTURE_REVIEW = root / "low_review"
            write_review(M.FUTURE_REVIEW, score=94)
            expect_failure(M.validate_capture_review)
            M.FUTURE_REVIEW = root / "good_review"
            write_review(M.FUTURE_REVIEW)
            records = process_records()
            table = dict((item["pid"], item) for item in records)
            calls = {"n": 0}

            def stable_reader(pid):
                calls["n"] += 1
                return dict(table[pid])

            M.PRESTOP = root / "process_receipt"
            M._proc_record = stable_reader
            M.capture_process_identity([10, 11, 12, 13, 14])
            assert calls["n"] == 10 and M.PRESTOP.is_dir()
        finally:
            M.FUTURE_REVIEW, M.PRESTOP, M._proc_record = (
                old_review, old_prestop, old_reader)

    for module in (M.R, M.Q, M.P):
        for name in M.RUNTIME_NAMES:
            assert getattr(module, name) == getattr(M, name)
        assert module._promote_result_resumable is M._promote_result_resumable
        assert module.capture_process_identity is M.capture_process_identity
    assert M.P.validate_runtime_release is M.validate_runtime_release
    text = SOURCE.read_text()
    assert "shutil.rmtree" not in text and "os.replace" not in text
    print(json.dumps({"status": "PASS_M2021_SOURCE_TEST",
        "subset_recovery": True, "three_name_truncated_recovery": True,
        "idempotent_second_promote": True,
        "sealed_plan_mismatch_preserved": True,
        "all_nested_runtime_paths_bound": True,
        "sealed_review_gate": True, "five_process_double_read": True,
        "production_process_capture": False, "archive_open": False,
        "merge": False, "reducer": False, "gpu": False, "eda": False},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
