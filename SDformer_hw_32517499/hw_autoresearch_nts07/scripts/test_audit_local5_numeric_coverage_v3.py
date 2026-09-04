#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_local5_numeric_coverage_v3.py")
SPEC = importlib.util.spec_from_file_location("numeric_audit_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NumericCoverageAuditV3Test(unittest.TestCase):
    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write(path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def _runtime(self) -> dict[str, str]:
        executable = Path("/usr/bin/python3.12")
        numpy_file = Path("/usr/lib/python3/dist-packages/numpy/__init__.py")
        return {
            "schema": "local5_numeric_batch_runtime_environment_v1",
            "status": "FROZEN_EXACT_RUNTIME",
            "resolved_executable": str(executable),
            "executable_sha256": self._sha(executable),
            "python_version": "3.12.3",
            "numpy_version": "1.26.4",
            "numpy_file": str(numpy_file),
            "numpy_file_sha256": self._sha(numpy_file),
        }

    def test_runtime_binding_accepts_exact_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self._runtime()
            self._write(root / "runtime_environment.json", runtime)
            digest = self._sha(root / "runtime_environment.json")
            plan = {
                "runtime_environment": runtime,
                "runtime_environment_sha256": digest,
            }
            complete = dict(plan)
            row = MODULE.validate_runtime_environment(root, plan, complete)
            self.assertEqual(row["numpy_version"], "1.26.4")

    def test_runtime_binding_rejects_plan_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self._runtime()
            self._write(root / "runtime_environment.json", runtime)
            digest = self._sha(root / "runtime_environment.json")
            with self.assertRaises(ValueError):
                MODULE.validate_runtime_environment(
                    root,
                    {
                        "runtime_environment": runtime | {"numpy_version": "2.1.1"},
                        "runtime_environment_sha256": digest,
                    },
                    {
                        "runtime_environment": runtime,
                        "runtime_environment_sha256": digest,
                    },
                )

    def test_parent_binding_preserves_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            sample_root = root / "sample71"
            self._write(parent / "plan.json", {"schema": "plan"})
            receipt = {
                "sample": 71,
                "output": str(sample_root),
                "execution": "RESUME_INCOMPLETE_SHARD",
            }
            self._write(parent / "sample71.receipt.json", receipt)
            self._write(parent / "complete.json", {
                "status": "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0",
                "formal_g0": "DENY",
                "samples_failed": [],
                "plan_sha256": self._sha(parent / "plan.json"),
                "rows": [receipt],
            })
            binding = {
                "schema": "local5_numeric_parent_batch_binding_v1",
                "root": str(parent),
                "plan_sha256": self._sha(parent / "plan.json"),
                "complete_sha256": self._sha(parent / "complete.json"),
                "rows": [{
                    "sample": 71,
                    "execution": "RESUME_INCOMPLETE_SHARD",
                    "sample_receipt_sha256": self._sha(
                        parent / "sample71.receipt.json"
                    ),
                }],
            }
            row = MODULE.validate_parent_batch(binding, 71, sample_root)
            self.assertEqual(row["execution"], "RESUME_INCOMPLETE_SHARD")


if __name__ == "__main__":
    unittest.main()
