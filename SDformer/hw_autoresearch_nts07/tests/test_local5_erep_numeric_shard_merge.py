from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.local5_erep_integrated_cross_head_actual import sha256
from scripts.local5_erep_numeric_shard_merge import (
    BLOCKS,
    _load_expected,
    _validate_complete,
)


class NumericShardMergeTest(unittest.TestCase):
    def make_expected(self, root: Path) -> dict[str, int]:
        directory = root / "software_expected"
        directory.mkdir()
        identity = {"sample": 0, "stage": 0, "block": 0, "heads": 3}
        plan = {
            "schema": "local5_projection_task_plan_v1",
            "scope": "formal_numeric_sample_shard_not_g0",
            **identity,
            "window": 94,
            "out_dim": 32,
            "tasks": [
                {"input_group_index": head, "output_tile": tile}
                for tile in range(3)
                for head in range(3)
            ],
        }
        plan_path = directory / "task_plan.json"
        expected_path = directory / "software_expected.npz"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        np.savez(
            expected_path,
            schema_version=np.asarray([1], dtype=np.uint16),
            expected_acc32=np.zeros(3 * 450 * 32, dtype=np.int32),
        )
        sources = [
            Path("scripts/local5_erep_numeric_window_expected.py").resolve(),
            Path("scripts/local5_erep_formal_canary_expected.py").resolve(),
        ]
        receipt = {
            "schema": "local5_erep_numeric_window_expected_v1",
            "status": "PASS_NUMERIC_WINDOW_EXPECTED_NOT_G0",
            "formal_g0": "DENY",
            "identity": {**identity, "window": 94},
            "task_plan_sha256": sha256(plan_path),
            "software_expected_sha256": sha256(expected_path),
            "expected_scalar_count": 3 * 450 * 32,
            "numpy_version": np.__version__,
            "source_bindings": [
                {"file": str(path), "sha256": sha256(path)} for path in sources
            ],
        }
        (directory / "software_expected_receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        return identity

    def test_block_topology_is_frozen(self) -> None:
        self.assertEqual(len(BLOCKS), 12)
        self.assertEqual(sum(row[2] for row in BLOCKS), 138)
        self.assertEqual({row[2] for row in BLOCKS}, {3, 6, 12, 24})

    def test_expected_contract_accepts_canonical_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = self.make_expected(root)
            plan, expected = _load_expected(root, identity)
            self.assertEqual(plan["window"], 94)
            self.assertEqual(expected.shape, (3 * 450 * 32,))

    def test_expected_contract_rejects_wrong_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = self.make_expected(root)
            plan_path = root / "software_expected/task_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["scope"] = "canary"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 合同"):
                _load_expected(root, identity)

    def test_complete_marker_binds_exact_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "a.bin"
            artifact.write_bytes(b"a")
            identity = {
                "sample": 0, "stage": 0, "block": 0, "heads": 3, "window": 94
            }
            receipt = {
                "schema": "local5_erep_numeric_window_complete_v1",
                "status": "SEALED_READY_FOR_MITER_NOT_G0",
                "formal_g0": "DENY",
                "identity": identity,
                "artifact_sha256": {"artifact": sha256(artifact)},
            }
            marker = root / "window_complete.json"
            marker.write_text(json.dumps(receipt), encoding="utf-8")
            _validate_complete(marker, identity, {"artifact": artifact})
            receipt["artifact_sha256"]["shadow"] = sha256(artifact)
            marker.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "集合不精确"):
                _validate_complete(marker, identity, {"artifact": artifact})


if __name__ == "__main__":
    unittest.main()
