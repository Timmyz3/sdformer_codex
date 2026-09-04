#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("run_local5_numeric_sample_batch_v1.py")
SPEC = importlib.util.spec_from_file_location("numeric_batch_v1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NumericBatchV1Test(unittest.TestCase):
    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_valid_fixture(self, root: Path, release_sha: str) -> None:
        (root / "shard").mkdir()
        release = root / "release"
        release.mkdir()
        manifest = release / "release_manifest.json"
        manifest.write_text("manifest")
        self.assertEqual(self._sha(manifest), release_sha)
        release_complete = release / "release_complete.json"
        release_complete.write_text("complete")
        archive = root / "shard/acc32_miter_shard.npz"
        topology = MODULE.EXPECTED_TOPOLOGY
        heads_array = np.asarray([row[2] for row in topology], dtype=np.uint8)
        offsets = np.zeros(MODULE.WINDOWS_PER_SAMPLE + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(heads_array.astype(np.int64) * 450 * 32)
        values = np.arange(MODULE.SCALARS_PER_SAMPLE, dtype=np.int32)
        np.savez_compressed(
            archive,
            schema_version=np.asarray([4], dtype=np.uint16),
            window_sample=np.full(12, 7, dtype=np.uint16),
            window_stage=np.asarray([row[0] for row in topology], dtype=np.uint8),
            window_block=np.asarray([row[1] for row in topology], dtype=np.uint8),
            window_token=np.arange(12, dtype=np.uint16),
            window_weight=np.arange(12, dtype=np.uint16),
            window_heads=heads_array,
            window_value_offsets=offsets,
            expected_acc32=values,
            actual_acc32=values.copy(),
        )
        binding = {
            "schema": "local5_erep_numeric_sample_release_binding_v1",
            "status": "PASS_RELEASE_BOUND_NOT_G0",
            "formal_g0": "DENY",
            "release_manifest": str(manifest),
            "release_manifest_sha256": release_sha,
            "release_complete": str(release_complete),
            "release_complete_sha256": self._sha(release_complete),
        }
        binding_path = root / "release_binding.json"
        binding_path.write_text(json.dumps(binding))
        windows = [
            {
                "sample": 7,
                "stage": stage,
                "block": block,
                "heads": heads,
                "window": index,
                "weight": index,
                "mismatch_count": 0,
                "max_abs_error": 0,
            }
            for index, (stage, block, heads) in enumerate(MODULE.EXPECTED_TOPOLOGY)
        ]
        report = {
            "schema": "local5_erep_numeric_sample_shard_v1",
            "status": "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0",
            "formal_g0": "DENY",
            "sample": 7,
            "window_count": 12,
            "final_acc32_scalar_count": 1_987_200,
            "mismatch_count": 0,
            "max_abs_error": 0,
            "release_manifest_sha256": release_sha,
            "archive": str(archive),
            "archive_sha256": self._sha(archive),
            "total_regression_cycles": 123,
            "windows": windows,
        }
        report_path = root / "shard/numeric_shard_report.json"
        report_path.write_text(json.dumps(report))
        for name, targets in (
            ("window_receipt_sha256.txt", [report_path]),
            ("result_sha256.txt", [archive, report_path]),
        ):
            (root / name).write_text("".join(
                f"{self._sha(target)}  {target}\n" for target in targets
            ))
        complete = {
            "schema": "local5_erep_numeric_sample_shard_complete_v1",
            "status": "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0",
            "formal_g0": "DENY",
            "sample": 7,
            "release_binding_sha256": self._sha(binding_path),
            "result_sha256_file_sha256": self._sha(root / "result_sha256.txt"),
        }
        complete_path = root / "complete.json"
        complete_path.write_text(json.dumps(complete))
        (root / "receipt_sha256.txt").write_text(
            f"{self._sha(root / 'result_sha256.txt')}  {root / 'result_sha256.txt'}\n"
            f"{self._sha(complete_path)}  {complete_path}\n"
        )

    def test_validate_sample_accepts_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_sha = hashlib.sha256(b"manifest").hexdigest()
            self._write_valid_fixture(root, release_sha)
            row = MODULE.validate_sample(7, root, release_sha)
            self.assertEqual(row["acc32_scalars"], 1_987_200)

    def test_validate_sample_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_sha = hashlib.sha256(b"manifest").hexdigest()
            self._write_valid_fixture(root, release_sha)
            report_path = root / "shard/numeric_shard_report.json"
            report = json.loads(report_path.read_text())
            report["mismatch_count"] = 1
            report["max_abs_error"] = 1
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                MODULE.validate_sample(7, root, release_sha)

    def test_archive_miter_directly_rejects_numeric_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_sha = hashlib.sha256(b"manifest").hexdigest()
            self._write_valid_fixture(root, release_sha)
            archive = root / "shard/acc32_miter_shard.npz"
            with np.load(archive, allow_pickle=False) as value:
                arrays = {name: value[name].copy() for name in value.files}
            arrays["actual_acc32"][17] += 1
            np.savez_compressed(archive, **arrays)
            with self.assertRaises(ValueError):
                MODULE.validate_acc32_archive(
                    archive,
                    7,
                    json.loads(
                        (root / "shard/numeric_shard_report.json").read_text()
                    )["windows"],
                )

    def test_execution_mode_run_resume_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "sample"
            self.assertEqual(MODULE.sample_execution_mode(output), "RUN")
            output.mkdir()
            self.assertEqual(
                MODULE.sample_execution_mode(output), "RESUME_INCOMPLETE_SHARD"
            )
            (output / "shard").mkdir()
            (output / "complete.json").write_text("{}")
            (output / "shard/numeric_shard_report.json").write_text("{}")
            self.assertEqual(
                MODULE.sample_execution_mode(output), "SKIP_ALREADY_SEALED"
            )

    def test_require_source_sha_accepts_frozen_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "launcher.sh"
            source.write_text("#!/bin/sh\nexit 0\n")
            MODULE.require_source_sha(source, self._sha(source))

    def test_require_source_sha_rejects_live_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "launcher.sh"
            source.write_text("v1\n")
            frozen_sha = self._sha(source)
            source.write_text("v2\n")
            with self.assertRaises(ValueError):
                MODULE.require_source_sha(source, frozen_sha)


if __name__ == "__main__":
    unittest.main()
