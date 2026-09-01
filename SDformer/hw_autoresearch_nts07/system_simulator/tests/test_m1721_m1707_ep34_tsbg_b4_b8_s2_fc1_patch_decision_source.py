#!/usr/bin/env python3
"""Synthetic/source-only tests for the M1721 decision analyzer."""
from __future__ import print_function

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / (
    "scripts/analyze_m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_"
    "decision_source.py")
SPEC = importlib.util.spec_from_file_location("m1721_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1721SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("NumPy is required for vector-kernel tests")
        cls.np = np

    def test_reference_lru_is_persistent(self):
        misses, cache, hits = M._reference_lru([0, 1, 2, 3, 0, 4], 4)
        self.assertEqual(misses, 5)
        self.assertEqual(hits, [0])
        self.assertEqual(cache, (2, 3, 0, 4))

    def test_vector_lru_matches_scalar_reference_random(self):
        np = self.np
        rng = np.random.RandomState(1721)
        for capacity in M.BUNDLES:
            for output_tiles in (1, 2, 4):
                active = rng.rand(97, 11) < 0.31
                stats = M.exact_lru_entity_stats(
                    active, output_tiles, capacity, np)
                accesses = []
                for token in range(active.shape[0]):
                    for tile in range(output_tiles):
                        for group in np.flatnonzero(active[token]).tolist():
                            accesses.append(tile * active.shape[1] + group)
                misses, _cache, hits = M._reference_lru(
                    accesses, capacity)
                self.assertEqual(stats["accesses"], len(accesses))
                self.assertEqual(stats["misses"], misses)
                self.assertEqual(stats["hits"], len(hits))

    def test_tsbg_uses_same_capacity_ordinary_lru(self):
        np = self.np
        active = np.ones((8, 3), dtype=np.bool_)
        nnz = np.ones((8, 3), dtype=np.int16)
        point = M.tsbg_pair_metrics(active, nnz, 2, 6144, 0, 4, np)
        self.assertEqual(point["ordinary_lru_capacity_rows"], 4)
        self.assertGreater(point["baseline_weight_row_fetches"],
                           point["candidate_weight_row_fetches"])
        self.assertEqual(point["compute_issue_cycles"], 8 * 3 * 2)

    def test_fetch_ratio_cannot_be_called_cycle_speedup(self):
        np = self.np
        active = np.ones((8, 9), dtype=np.bool_)
        nnz = np.full((8, 9), 16, dtype=np.int16)
        point = M.tsbg_pair_metrics(active, nnz, 1, 1, 0, 8, np)
        fetch_ratio = float(point["baseline_weight_fetch_bytes"]) / float(
            point["candidate_weight_fetch_bytes"])
        cycle_ratio = float(point["baseline_roofline_cycles"]) / float(
            point["candidate_roofline_cycles"])
        self.assertGreater(fetch_ratio, cycle_ratio)
        self.assertEqual(cycle_ratio, 1.0)

    def test_s2_zero_is_exact_bypass(self):
        np = self.np
        active = np.array([[True, False], [True, True]], dtype=np.bool_)
        nnz = np.array([[1, 0], [2, 1]], dtype=np.int16)
        magnitude = np.array([[1, 0], [3, 1]], dtype=np.int32)
        point = M.s2_fc1_pair_metrics(
            active, nnz, magnitude, 32, [1, 2], 0.0, np)
        self.assertEqual(point["dropped_blocks"], 0)
        self.assertEqual(point["metadata_bytes"], 0)
        self.assertEqual(point["saved_nonzero_products"], 0)

    def test_s2_positive_uses_real_magnitude_and_reports_work(self):
        np = self.np
        active = np.array([[True, True], [True, True]], dtype=np.bool_)
        nnz = np.array([[1, 8], [2, 16]], dtype=np.int16)
        magnitude = np.array([[1, 30], [4, 100]], dtype=np.int32)
        point = M.s2_fc1_pair_metrics(
            active, nnz, magnitude, 32, [1, 2], 0.01, np)
        self.assertEqual(point["threshold_abs_code_sum"], 20)
        self.assertEqual(point["dropped_blocks"], 4)
        self.assertEqual(point["kept_blocks"], 4)
        self.assertEqual(point["saved_nonzero_products"], (1 + 2) * 32)
        self.assertGreater(point["max_accumulated_abs_output_code_debt_per_token"], 0)

    def test_s2_geometry_or_value_drift_rejected(self):
        np = self.np
        active = np.ones((2, 2), dtype=np.bool_)
        nnz = np.ones((2, 3), dtype=np.int16)
        magnitude = np.ones((2, 2), dtype=np.int32)
        with self.assertRaises(M.M1721Error):
            M.s2_fc1_pair_metrics(
                active, nnz, magnitude, 32, [1, 1], 0.01, np)

    def _make_tree(self, root):
        names = ["RUN_COMPLETE.txt", "capture_manifest.json", "fc_frames.bin",
                 "layers.json", "patch_s1_histogram_debt.jsonl.zlib",
                 "preload_permit_receipt.json", "sample_order.json",
                 M.M1707_RECEIPT]
        for name in names:
            (root / name).write_bytes((name + "\n").encode("utf-8"))
        sums = "".join("{}  {}\n".format(M.sha256(root / name), name)
                       for name in sorted(names))
        (root / "SHA256SUMS").write_text(sums, encoding="ascii")
        (root / "SHA256SUMS.seal.sha256").write_text(
            "{}  SHA256SUMS\n".format(M.sha256(root / "SHA256SUMS")),
            encoding="ascii")

    def test_exact_tree_seal_accepts_complete_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)
            row = M.verify_tree(root)
            self.assertEqual(len(row["members"]), 8)

    def test_tree_member_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)
            (root / "fc_frames.bin").write_bytes(b"mutation")
            with self.assertRaises(M.M1721Error):
                M.verify_tree(root)

    def test_tree_unsealed_extra_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)
            (root / "extra.bin").write_bytes(b"extra")
            with self.assertRaises(M.M1721Error):
                M.verify_tree(root)

    def test_tree_symlink_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)
            target = root / "real.bin"
            target.write_bytes(b"real")
            link = root / "link.bin"
            os.symlink(str(target), str(link))
            old = (root / "SHA256SUMS").read_text(encoding="ascii")
            (root / "SHA256SUMS").write_text(
                old + "{}  link.bin\n".format(M.sha256(target)), encoding="ascii")
            (root / "SHA256SUMS.seal.sha256").write_text(
                "{}  SHA256SUMS\n".format(M.sha256(root / "SHA256SUMS")),
                encoding="ascii")
            with self.assertRaises(M.M1721Error):
                M.verify_tree(root)

    def test_static_boundaries_are_explicit(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("fetch_ratio_is_cycle_speedup", text)
        self.assertIn("ordinary persistent same-capacity LRU-B", text)
        self.assertIn("BLOCKED_M1707_PATCH_IS_HISTOGRAM_ONLY", text)
        self.assertIn("FIXED_NO_GO_FROM_M1713", text)
        self.assertNotIn("import subprocess", text)
        self.assertNotIn("import socket", text)

    def test_source_self_check_is_inert(self):
        row = M.source_self_check()
        self.assertEqual(row["bundles"], [4, 8])
        self.assertFalse(row["analysis_executed"])
        self.assertEqual(row["gpu_runs"], 0)
        self.assertEqual(row["eda_runs"], 0)
        self.assertFalse(row["claim_boundary"]["paper_result"])

    def test_production_result_namespace_is_fresh(self):
        self.assertFalse(os.path.lexists(str(M.RESULT)))
        self.assertFalse(os.path.lexists(str(M.WORK)))


if __name__ == "__main__":
    unittest.main()
