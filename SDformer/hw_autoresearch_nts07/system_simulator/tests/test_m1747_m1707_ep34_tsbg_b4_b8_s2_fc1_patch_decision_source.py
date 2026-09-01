#!/usr/bin/env python3
"""Source-only tests for the additive M1747 TSBG schema successor."""
from __future__ import print_function

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / (
    "scripts/analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_"
    "decision_source.py")
SPEC = importlib.util.spec_from_file_location("m1747_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1747SourceTest(unittest.TestCase):
    def _seal_directory(self, root):
        members = sorted(path.relative_to(root).as_posix()
                         for path in root.rglob("*") if path.is_file() and
                         path.name not in ("SHA256SUMS",
                                           "SHA256SUMS.seal.sha256"))
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
        identities = M.source_identities()
        review_root = root / "review"
        review_root.mkdir()
        review = {"schema": M.REVIEW_SCHEMA, "status": M.REVIEW_STATUS,
            "identity": identities,
            "authorization": {"m1749_release_may_be_created": True,
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
            "m1748_review_sha256": review_binding["review_sha256"],
            "m1748_review_outer_seal_file_sha256":
                review_binding["outer_seal_file_sha256"]})
        release = {"schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
            "identity": release_identity,
            "authorization": {"analysis_runs": 1,
                "capture_verifications": 1, "result_publications": 1,
                "automatic_retry": False, "gpu_runs": 0, "eda_runs": 0,
                "all_other_runs": 0},
            "claim_boundary": {"paper_result": False}}
        release_path.write_text(
            json.dumps(release, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        self._seal_file(release_path)
        return review_root, release_path, identities

    def _canonical_samples(self):
        return {"schema": M.SAMPLE_ORDER_SCHEMA,
            "samples": [{"global_sample_id": index} for index in range(40)],
            "identity": {"checkpoint_sha256": M.BASE.CHECKPOINT_SHA256}}

    def test_exact_predecessor_and_consumed_authority_are_bound(self):
        self.assertEqual(M.sha256(M.M1727_SOURCE), M.M1727_SOURCE_SHA256)
        self.assertEqual(M.sha256(M.M1727_TEST), M.M1727_TEST_SHA256)
        self.assertEqual(M.sha256(M.M1727_CONTRACT), M.M1727_CONTRACT_SHA256)
        self.assertEqual(M.sha256(M.M1729_RELEASE), M.M1729_RELEASE_SHA256)

    def test_failure_receipt_is_exact_and_double_sealed(self):
        self.assertEqual(M.sha256(M.FAILED_RECEIPT),
                         M.FAILED_RECEIPT_SHA256)
        self.assertEqual(M.sha256(M.FAILED_RECEIPT_SIDECAR),
                         M.FAILED_RECEIPT_SIDECAR_SHA256)
        self.assertEqual(M.sha256(M.FAILED_RECEIPT_OUTER),
                         M.FAILED_RECEIPT_OUTER_SHA256)
        M.verify_failure_and_capture_review()

    def test_m1744_exact_triple_is_live(self):
        binding = M.verify_sealed_directory(M.M1744_REVIEW, "M1744 review")
        self.assertEqual(binding, {"review_sha256": M.M1744_REVIEW_SHA256,
            "manifest_sha256": M.M1744_MANIFEST_SHA256,
            "outer_seal_file_sha256": M.M1744_OUTER_SHA256})

    def test_only_canonical_schema_is_adapted(self):
        original = self._canonical_samples()
        adapted = M.adapt_sample_order_document(original)
        self.assertEqual(original["schema"], M.SAMPLE_ORDER_SCHEMA)
        self.assertEqual(adapted["schema"], M.LEGACY_SAMPLE_ORDER_SCHEMA)
        self.assertEqual(adapted["samples"], original["samples"])
        self.assertEqual(adapted["identity"], original["identity"])

    def test_legacy_or_arbitrary_schema_is_rejected(self):
        for schema in (M.LEGACY_SAMPLE_ORDER_SCHEMA, "near_match", ""):
            value = self._canonical_samples()
            value["schema"] = schema
            with self.assertRaises(M.M1747Error):
                M.adapt_sample_order_document(value)

    def test_sample_count_order_and_checkpoint_mutations_are_rejected(self):
        mutations = []
        short = self._canonical_samples()
        short["samples"] = short["samples"][:-1]
        mutations.append(short)
        permuted = self._canonical_samples()
        permuted["samples"][7]["global_sample_id"] = 8
        mutations.append(permuted)
        checkpoint = self._canonical_samples()
        checkpoint["identity"]["checkpoint_sha256"] = "0" * 64
        mutations.append(checkpoint)
        for value in mutations:
            with self.assertRaises(M.M1747Error):
                M.adapt_sample_order_document(value)

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
            path = review / "review.json"
            row = json.loads(path.read_text())
            row["identity"]["canonical_sample_order_sha256"] = "0" * 64
            path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            self._seal_directory(review)
            with self.assertRaises(M.M1747Error):
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
            with self.assertRaises(M.M1747Error):
                M.validate_future_release(release, binding, identities)

    def test_no_authority_fails_before_capture_sentinel(self):
        class NoAuthority(Exception):
            pass
        touched = [False]
        old_authority = M.verify_analysis_authority
        old_capture = M.BASE.verify_capture_identity
        def deny():
            raise NoAuthority("no M1749")
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

    def test_exact_m1727_algorithms_and_gates_are_reused(self):
        self.assertIs(M.BASE.tsbg_pair_metrics, M.M1727.tsbg_pair_metrics)
        self.assertIs(M.BASE.s2_fc1_pair_metrics, M.M1727.s2_fc1_pair_metrics)
        self.assertIs(M.BASE.DecisionAccumulator.finalize_tsbg_rows,
                      M.M1727.finalize_tsbg_rows)
        self.assertEqual(M.BASE.BUNDLES, (4, 8))
        self.assertEqual(M.BASE.S2_EPSILON_RATIO,
                         (0.0, 0.01, 0.02, 0.05, 0.10))

    def test_authority_gate_precedes_namespaces_and_capture(self):
        text = SOURCE.read_text(encoding="utf-8")
        begin = text.index("def run_analysis():")
        body = text[begin:text.index("def source_self_check():", begin)]
        self.assertLess(body.index("verify_analysis_authority()"),
                        body.index("os.path.lexists"))
        self.assertNotIn("verify_capture_identity", body)

    def test_static_no_execution_or_network_imports(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", text)
        self.assertNotIn("import socket", text)
        self.assertNotIn("requests", text)
        self.assertNotIn("paramiko", text)

    def test_source_self_check_is_inert(self):
        row = M.source_self_check()
        self.assertFalse(row["algorithm_changed"])
        self.assertFalse(row["gates_changed"])
        self.assertFalse(row["claim_boundary_changed"])
        self.assertFalse(row["capture_touched"])
        self.assertFalse(row["analysis_executed"])
        self.assertFalse(row["claim_boundary"]["paper_result"])

    def test_production_namespaces_are_fresh(self):
        self.assertFalse(os.path.lexists(str(M.RESULT)))
        self.assertFalse(os.path.lexists(str(M.WORK)))


if __name__ == "__main__":
    unittest.main()
