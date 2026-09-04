#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_local5_active_tcfm5_postg0 import (
    analyze_descriptor_chunk,
    file_sha256,
    verify_bound_file,
)
from scripts.analyze_motion_temporal_equivalence import (
    evaluate_temporal_equivalence,
    profile_contract,
)


class Local5ActiveTcfm5Test(unittest.TestCase):
    def test_five_roles_are_conflict_free_only_under_topology_coloring(self):
        result = analyze_descriptor_chunk(
            gates=np.asarray([[7, 7, 7, 7, 7]], dtype=np.uint16),
            valid_mask=np.asarray([0b11111], dtype=np.uint8),
            k_bitmap=np.asarray([0b11], dtype=np.uint64),
            plane=np.asarray([0], dtype=np.uint8),
            source_y=np.asarray([2], dtype=np.uint16),
            source_x=np.asarray([2], dtype=np.uint16),
            height=5,
            width=5,
        )
        self.assertEqual(int(result["product_terms"][0]), 2)
        self.assertEqual(int(result["destination_updates"][0]), 10)
        self.assertEqual(int(result["tcfm5_cycles"][0]), 2)
        self.assertGreater(int(result["linear5_cycles"][0]), 2)

    def test_equal_gate_roles_share_one_product(self):
        result = analyze_descriptor_chunk(
            gates=np.asarray([[9, 9, 0, 3, 3]], dtype=np.uint16),
            valid_mask=np.asarray([0b11011], dtype=np.uint8),
            k_bitmap=np.asarray([0b101], dtype=np.uint64),
            plane=np.asarray([0], dtype=np.uint8),
            source_y=np.asarray([1], dtype=np.uint16),
            source_x=np.asarray([1], dtype=np.uint16),
            height=3,
            width=3,
        )
        self.assertEqual(int(result["product_terms"][0]), 4)
        self.assertEqual(int(result["destination_updates"][0]), 8)

    def test_bound_input_sha_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"bound")
            expected = file_sha256(path)
            self.assertEqual(verify_bound_file(path, expected, "input"), path.resolve())
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA绑定失效"):
                verify_bound_file(path, expected, "input")


class MotionTemporalEquivalenceTest(unittest.TestCase):
    def test_profile_contract_uses_runtime_identity(self):
        profile = {
            "experiment": "fullres",
            "samples": 100,
            "eval_protocol": {
                "resolution": [480, 640],
                "crop": None,
                "window_size": [2, 15, 15],
                "tokens_per_window": 450,
                "bn_policy": "no_running",
            },
            "artifact_identity": {
                "config_path": "/tmp/config.yml",
                "config_sha256": "a" * 64,
                "checkpoint_path": "/tmp/checkpoint.pth",
                "checkpoint_sha256": "b" * 64,
            },
            "summary": {"h60_records": [{"tokens": 450}, {"tokens": 450}]},
        }
        result = profile_contract(profile, Path("profile.json"))
        self.assertEqual(result["temporal_tokens"], 450)
        self.assertEqual(result["window_size"], [2, 15, 15])
        self.assertEqual(result["h60_records"], 2)

    def test_profile_contract_rejects_temporal_identity_mismatch(self):
        profile = {
            "samples": 1,
            "eval_protocol": {"tokens_per_window": 162},
            "artifact_identity": {
                "config_path": "c",
                "config_sha256": "d",
                "checkpoint_path": "e",
                "checkpoint_sha256": "f",
            },
            "summary": {"h60_records": [{"tokens": 450}]},
        }
        with self.assertRaisesRegex(ValueError, "temporal token"):
            profile_contract(profile, Path("profile.json"))

    def test_descriptor_and_delta_work(self):
        result = evaluate_temporal_equivalence(
            {
                "pair_total": 4,
                "pair_empty": 1,
                "pair_score_equal_h67": 3,
                "update_histogram": [1, 2, 1] + [0] * 30,
            }
        )
        self.assertEqual(result["pair_active"], 3)
        self.assertEqual(result["pair_score_unequal"], 1)
        self.assertEqual(result["baseline_active_score_descriptors"], 6)
        self.assertEqual(result["compressed_active_score_descriptors"], 4)
        self.assertGreater(result["width_results"][2]["speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
