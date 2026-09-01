#!/usr/bin/env python3
"""Source-only tests for the additive M1727 decision analyzer successor."""
from __future__ import print_function

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / (
    "scripts/analyze_m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_"
    "decision_source.py")
SPEC = importlib.util.spec_from_file_location("m1727_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1727SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("NumPy is required for source tests")
        cls.np = np

    def _identities(self):
        return {
            "source_sha256": M.sha256(M.SOURCE),
            "test_sha256": M.sha256(M.TEST),
            "contract_sha256": M.sha256(M.CONTRACT),
            "contract_sidecar_sha256": M.sha256(M.CONTRACT_SIDECAR),
            "contract_outer_seal_file_sha256": M.sha256(M.CONTRACT_OUTER),
            "m1721_source_sha256": M.M1721_SOURCE_SHA256,
            "m1725_failed_review_sha256": M.M1725_REVIEW_SHA256}

    def _seal_directory(self, root):
        members = sorted(path.relative_to(root).as_posix()
                         for path in root.rglob("*") if path.is_file() and
                         path.name not in
                         ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
        sums = root / "SHA256SUMS"
        sums.write_text("".join("{}  {}\n".format(M.sha256(root / name), name)
                                 for name in members), encoding="ascii")
        (root / "SHA256SUMS.seal.sha256").write_text(
            "{}  SHA256SUMS\n".format(M.sha256(sums)), encoding="ascii")

    def _seal_file(self, path):
        sidecar = Path(str(path) + ".sha256")
        outer = Path(str(path) + ".sha256.seal.sha256")
        sidecar.write_text("{}  {}\n".format(M.sha256(path), path.name),
                           encoding="ascii")
        outer.write_text("{}  {}\n".format(M.sha256(sidecar), sidecar.name),
                         encoding="ascii")

    def _make_authority(self, root):
        identities = self._identities()
        review_root = root / "review"
        review_root.mkdir()
        review = {"schema": M.REVIEW_SCHEMA, "status": M.REVIEW_STATUS,
            "identity": identities,
            "authorization": {"m1729_release_may_be_created": True,
                "analysis_run": False, "capture_verify": False},
            "claim_boundary": {"paper_result": False}}
        (review_root / "review.json").write_text(
            json.dumps(review, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        self._seal_directory(review_root)
        review_binding = M.validate_future_review(review_root, identities)
        release_path = root / "release.json"
        release_identity = dict(identities)
        release_identity.update({
            "m1728_review_sha256": review_binding["review_sha256"],
            "m1728_review_outer_seal_file_sha256":
                review_binding["outer_seal_file_sha256"]})
        release = {"schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
            "identity": release_identity,
            "authorization": {"analysis_runs": 1, "capture_verifications": 1,
                "result_publications": 1, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0},
            "claim_boundary": {"paper_result": False}}
        release_path.write_text(
            json.dumps(release, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        self._seal_file(release_path)
        return review_root, release_path, identities

    def test_exact_failed_predecessor_is_unchanged(self):
        self.assertEqual(M.sha256(M.M1721_SOURCE), M.M1721_SOURCE_SHA256)
        self.assertEqual(M.sha256(M.M1725_REVIEW), M.M1725_REVIEW_SHA256)

    def test_vector_lru_stays_equal_to_scalar(self):
        np = self.np
        rng = np.random.RandomState(1727)
        for capacity in M.BUNDLES:
            active = rng.rand(113, 13) < 0.37
            got = M.exact_lru_entity_stats(active, 3, capacity, np)
            accesses = []
            for token in range(active.shape[0]):
                for tile in range(3):
                    accesses.extend(tile * active.shape[1] + int(group)
                        for group in np.flatnonzero(active[token]).tolist())
            misses, _cache, hits = M._reference_lru(accesses, capacity)
            self.assertEqual((got["accesses"], got["misses"], got["hits"]),
                             (len(accesses), misses, len(hits)))

    def test_same_B_ordinary_lru_remains_the_comparator(self):
        np = self.np
        active = np.ones((8, 3), dtype=np.bool_)
        nnz = np.ones((8, 3), dtype=np.int16)
        point = M.tsbg_pair_metrics(active, nnz, 2, 6144, 0, 4, np)
        self.assertEqual(point["ordinary_lru_capacity_rows"], 4)
        self.assertGreater(point["baseline_weight_row_fetches"],
                           point["candidate_weight_row_fetches"])

    def test_tsbg_B4_state_cost_is_explicit_and_diagnostic(self):
        row = M.tsbg_resource_account(4)
        self.assertEqual(row["baseline_acc24_context_bytes_lower_bound"], 288)
        self.assertEqual(
            row["candidate_b_token_acc24_context_bytes_lower_bound"], 1152)
        self.assertEqual(row["baseline_source_fifo_bytes_lower_bound"], 16)
        self.assertEqual(row["candidate_b_token_source_fifo_bytes_lower_bound"],
                         64)
        self.assertEqual(row["candidate_incremental_state_bytes_lower_bound"],
                         912)
        self.assertFalse(row["full_area_energy_pricing_complete"])
        self.assertFalse(row["same_resource_claim"])
        self.assertTrue(row["screening_only"])

    def test_tsbg_B8_state_and_weight_scope_are_explicit(self):
        row = M.tsbg_resource_account(8)
        self.assertEqual(
            row["candidate_b_token_acc24_context_bytes_lower_bound"], 2304)
        self.assertEqual(row["candidate_b_token_source_fifo_bytes_lower_bound"],
                         128)
        self.assertEqual(row["captured_weight_bytes_per_element_screening"], 4)
        self.assertFalse(row["hardware_weight_quantization_authority"])
        self.assertFalse(row["context_tag_and_broadcast_control_priced"])

    def test_s2_sum_debt_counts_32_output_channels(self):
        np = self.np
        point = M.s2_fc1_pair_metrics(
            np.array([[True]], dtype=np.bool_),
            np.array([[1]], dtype=np.int16),
            np.array([[1]], dtype=np.int32), 32, [1, 1], 0.01, np)
        self.assertEqual(point["sum_abs_output_code_debt"], 32)
        self.assertTrue(point[
            "sum_abs_output_code_debt_includes_output_channel_multiplicity"])

    def test_s2_sum_debt_counts_partial_tail_block(self):
        np = self.np
        point = M.s2_fc1_pair_metrics(
            np.array([[True]], dtype=np.bool_),
            np.array([[1]], dtype=np.int16),
            np.array([[1]], dtype=np.int32), 17, [1, 2], 0.01, np)
        self.assertEqual(point["sum_abs_output_code_debt"], 18)

    def test_s2_zero_budget_remains_exact(self):
        np = self.np
        point = M.s2_fc1_pair_metrics(
            np.array([[True]], dtype=np.bool_),
            np.array([[1]], dtype=np.int16),
            np.array([[1]], dtype=np.int32), 32, [1, 1], 0.0, np)
        self.assertEqual(point["sum_abs_output_code_debt"], 0)
        self.assertEqual(point["dropped_blocks"], 0)

    def test_valid_future_review_and_release_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            review, release, identities = self._make_authority(Path(tmp))
            binding = M.validate_future_review(review, identities)
            row = M.validate_future_release(release, binding, identities)
            self.assertEqual(len(row["release_sha256"]), 64)

    def test_resealed_review_identity_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review, _release, identities = self._make_authority(root)
            row = json.loads((review / "review.json").read_text())
            row["identity"]["source_sha256"] = "0" * 64
            (review / "review.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n")
            self._seal_directory(review)
            with self.assertRaises(M.M1727Error):
                M.validate_future_review(review, identities)

    def test_resealed_release_budget_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review, release, identities = self._make_authority(root)
            binding = M.validate_future_review(review, identities)
            row = json.loads(release.read_text())
            row["authorization"]["analysis_runs"] = 2
            release.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            self._seal_file(release)
            with self.assertRaises(M.M1727Error):
                M.validate_future_release(release, binding, identities)

    def test_no_authority_fails_before_capture_sentinel(self):
        class NoAuthority(Exception):
            pass
        touched = [False]
        old_authority = M.verify_analysis_authority
        old_capture = M.BASE.verify_capture_identity
        def deny():
            raise NoAuthority("no release")
        def capture(_root):
            touched[0] = True
            raise AssertionError("capture must not be touched")
        M.verify_analysis_authority = deny
        M.BASE.verify_capture_identity = capture
        try:
            with self.assertRaises(NoAuthority):
                M.run_analysis()
            self.assertFalse(touched[0])
        finally:
            M.verify_analysis_authority = old_authority
            M.BASE.verify_capture_identity = old_capture

    def test_authority_gate_precedes_namespace_and_capture_in_source(self):
        text = SOURCE.read_text(encoding="utf-8")
        begin = text.index("def run_analysis():")
        body = text[begin:text.index("def source_self_check():", begin)]
        self.assertLess(body.index("verify_analysis_authority()"),
                        body.index("os.path.lexists"))
        self.assertNotIn("verify_capture_identity", body)

    def test_static_boundaries_and_no_execution_imports(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("M1728", text)
        self.assertIn("M1729", text)
        self.assertIn("hardware_weight_quantization_authority", text)
        self.assertIn("sum_abs_output_code_debt_includes_output_channel", text)
        self.assertNotIn("import subprocess", text)
        self.assertNotIn("import socket", text)

    def test_source_self_check_is_inert(self):
        row = M.source_self_check()
        self.assertEqual(row["s2_32_output_unit_debt"], 32)
        self.assertFalse(row["analysis_executed"])
        self.assertFalse(row["capture_touched"])
        self.assertFalse(row["claim_boundary"]["paper_result"])

    def test_production_namespaces_are_fresh(self):
        self.assertFalse(os.path.lexists(str(M.RESULT)))
        self.assertFalse(os.path.lexists(str(M.WORK)))


if __name__ == "__main__":
    unittest.main()
