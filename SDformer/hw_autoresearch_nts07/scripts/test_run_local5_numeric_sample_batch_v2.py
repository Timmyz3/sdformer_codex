#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_local5_numeric_sample_batch_v2.py")
SPEC = importlib.util.spec_from_file_location("numeric_batch_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NumericBatchV2Test(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_runtime_environment_is_exact_and_self_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            row = MODULE.freeze_runtime_environment(path)
            self.assertEqual(row["python_version"], "3.12.3")
            self.assertEqual(row["numpy_version"], "1.26.4")
            self.assertEqual(
                row["executable_sha256"],
                self._sha(Path(row["resolved_executable"])),
            )

    def test_parent_batch_preserves_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            output = root / "sample7"
            plan = {
                "schema": "local5_numeric_sample_batch_plan_v1",
                "samples": [7],
            }
            self._write_json(parent / "plan.json", plan)
            row = {
                "sample": 7,
                "status": "PASS",
                "output": str(output),
                "execution": "RESUME_INCOMPLETE_SHARD",
            }
            self._write_json(parent / "sample7.receipt.json", row)
            complete = {
                "schema": "local5_numeric_sample_batch_complete_v1",
                "status": "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0",
                "formal_g0": "DENY",
                "samples_requested": [7],
                "samples_passed": [7],
                "samples_failed": [],
                "plan_sha256": self._sha(parent / "plan.json"),
                "rows": [row],
            }
            self._write_json(parent / "complete.json", complete)
            binding = MODULE.bind_parent_batch(parent, [7], {7: output})
            self.assertEqual(
                binding["rows"][0]["execution"], "RESUME_INCOMPLETE_SHARD"
            )

    def test_parent_batch_rejects_output_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            self._write_json(parent / "plan.json", {
                "schema": "local5_numeric_sample_batch_plan_v1",
                "samples": [7],
            })
            row = {
                "sample": 7,
                "status": "PASS",
                "output": str(root / "wrong"),
                "execution": "RUN",
            }
            self._write_json(parent / "sample7.receipt.json", row)
            self._write_json(parent / "complete.json", {
                "schema": "local5_numeric_sample_batch_complete_v1",
                "status": "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0",
                "formal_g0": "DENY",
                "samples_requested": [7],
                "samples_passed": [7],
                "samples_failed": [],
                "plan_sha256": self._sha(parent / "plan.json"),
                "rows": [row],
            })
            with self.assertRaises(ValueError):
                MODULE.bind_parent_batch(parent, [7], {7: root / "expected"})

    def test_source_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_text("v1\n", encoding="utf-8")
            digest = self._sha(source)
            MODULE.require_source_sha(source, digest)
            source.write_text("v2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.require_source_sha(source, digest)


if __name__ == "__main__":
    unittest.main()
