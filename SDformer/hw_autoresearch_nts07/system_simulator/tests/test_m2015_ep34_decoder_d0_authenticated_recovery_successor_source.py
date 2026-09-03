#!/usr/bin/env python3
"""No-production regression for M2015 authenticated recovery."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_m2015_ep34_decoder_d0_authenticated_recovery_successor_source.py")
spec = importlib.util.spec_from_file_location("m2015_test_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except (M.M2015Error, OSError, KeyError, ValueError):
        return
    raise AssertionError("expected fail-closed rejection")


def write_review(path, score, severities, sealed):
    path.mkdir()
    row = {"status": M.REVIEW_STATUS, "score_over_100": score,
        "severity_counts": severities, "identity": M.identity(),
        "authorization": {"process_identity_capture": 1,
            "m2017_release_authoring": 1, "archive_open": 0,
            "merge": 0, "reducer": 0, "payload_opens": 0,
            "gpu_runs": 0, "eda_runs": 0}}
    (path / "review.json").write_text(json.dumps(row) + "\n")
    if sealed:
        M.B.seal_work_tree(path)


def main():
    old_review = M.FUTURE_REVIEW
    try:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            M.FUTURE_REVIEW = root / "unsealed"
            write_review(M.FUTURE_REVIEW, 100,
                         {"p0": 0, "p1": 0, "p2": 0}, False)
            expect_failure(M.validate_capture_review)

            M.FUTURE_REVIEW = root / "low_score"
            write_review(M.FUTURE_REVIEW, 0,
                         {"p0": 0, "p1": 0, "p2": 0}, True)
            expect_failure(M.validate_capture_review)

            M.FUTURE_REVIEW = root / "nonzero"
            write_review(M.FUTURE_REVIEW, 100,
                         {"p0": 0, "p1": 1, "p2": 0}, True)
            expect_failure(M.validate_capture_review)

            M.FUTURE_REVIEW = root / "valid"
            write_review(M.FUTURE_REVIEW, 100,
                         {"p0": 0, "p1": 0, "p2": 0}, True)
            row, seal = M.validate_capture_review()
            assert row["score_over_100"] == 100
            assert len(seal["manifest_sha256"]) == 64

            truncated = root / "truncated"
            truncated.mkdir()
            (truncated / "result.json").write_text("{}\n")
            (truncated / "SHA256SUMS").write_text("broken\n")
            (truncated / "SHA256SUMS.seal.sha256").write_text("short\n")
            assert M.classify_import_orphan(truncated) == "partial_unsealed"

            unexpected = root / "unexpected"
            unexpected.mkdir()
            (unexpected / "alien").write_text("x")
            expect_failure(lambda: M.classify_import_orphan(unexpected))
    finally:
        M.FUTURE_REVIEW = old_review

    desc = M.describe()
    assert desc["repairs"][
        "three_names_invalid_seal_is_quarantined_partial"]
    assert desc["repairs"][
        "future_review_tree_authenticated_before_first_pid_read"]
    text = SOURCE.read_text()
    assert "shutil.rmtree" not in text and "os.replace" not in text
    print(json.dumps({"status": "PASS_M2015_SOURCE_TEST",
        "unsealed_review_rejected": True, "low_score_rejected": True,
        "nonzero_severity_rejected": True, "sealed_review_passed": True,
        "three_names_truncated_seal_is_partial": True,
        "unexpected_orphan_rejected": True,
        "production_process_capture": False, "archive_open": False,
        "merge": False, "reducer": False, "gpu": False, "eda": False},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
