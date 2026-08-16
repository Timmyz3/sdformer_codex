#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_local5_numeric_execution_chain_v1.py")
SPEC = importlib.util.spec_from_file_location("execution_chain", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExecutionChainAuditTest(unittest.TestCase):
    def test_run_is_selected_before_later_skip(self) -> None:
        rows = [
            {"batch_name": "v2", "execution": "SKIP_ALREADY_SEALED"},
            {"batch_name": "v1", "execution": "RUN"},
        ]
        self.assertEqual(
            MODULE.select_proven_execution(rows)["execution"], "RUN"
        )

    def test_resume_is_selected_when_no_run_receipt(self) -> None:
        rows = [
            {"batch_name": "v2", "execution": "SKIP_ALREADY_SEALED"},
            {"batch_name": "v1", "execution": "RESUME_INCOMPLETE_SHARD"},
        ]
        self.assertEqual(
            MODULE.select_proven_execution(rows)["execution"],
            "RESUME_INCOMPLETE_SHARD",
        )

    def test_skip_only_chain_is_explicit_gap(self) -> None:
        rows = [{"batch_name": "v1", "execution": "SKIP_ALREADY_SEALED"}]
        self.assertIsNone(MODULE.select_proven_execution(rows))

    def test_empty_chain_is_explicit_gap(self) -> None:
        self.assertIsNone(MODULE.select_proven_execution([]))

    def test_failed_and_staging_names_are_detectable(self) -> None:
        names = ["batch.failed_numpy211", "batch.staging.42"]
        self.assertTrue(all(".failed" in name or ".staging" in name for name in names))

    def test_collect_receipts_ignores_failed_directory_before_parsing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            old_results = MODULE.RESULTS
            try:
                MODULE.RESULTS = Path(directory)
                failed = MODULE.RESULTS / "local5_numeric_samples3_6_batch.failed"
                failed.mkdir()
                (failed / "sample3.receipt.json").write_text("not json")
                rows = MODULE.collect_receipts(
                    3, MODULE.RESULTS / "sample3",
                    {"complete": "0", "report": "0", "archive": "0"},
                )
                self.assertEqual(rows, [])
            finally:
                MODULE.RESULTS = old_results


if __name__ == "__main__":
    unittest.main()
