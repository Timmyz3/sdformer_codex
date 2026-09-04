#!/usr/bin/env python3
"""Synthetic-only tests for the M1643 source-only paired evaluator."""
from __future__ import print_function

import copy
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / (
    "scripts/build_m1643_motion_ep34_s2_ccbs16_paired_evaluation_source.py")
SPEC = importlib.util.spec_from_file_location("m1643_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def digest(character):
    return character * 64


def make_document(tsbg=False):
    baseline = []
    for ordinal in range(40):
        sequence = "seq_a" if ordinal < 20 else "seq_b"
        blocks = []
        for block_ordinal in range(2):
            blocks.append({
                "block_id": "b%d" % block_ordinal,
                "source_group": 16,
                "output_tile": 16,
                "weight_bytes": 256,
                "compute_ops": 256,
                "psum_bytes": 32,
                "service_cycles": 20,
                "first_weight_fetch_cycle": 10 + 20 * block_ordinal,
                "first_compute_cycle": 12 + 20 * block_ordinal,
                "first_psum_cycle": 18 + 20 * block_ordinal,
            })
        baseline.append({
            "sequence": sequence,
            "sample_ordinal": ordinal,
            "global_sample_id": "sample_%02d" % ordinal,
            "aee": 1.0,
            "cycle_count": 100,
            "blocks": blocks,
        })

    points = []
    for epsilon in M.EPSILON_AXIS:
        rows = []
        for sample in baseline:
            decisions = []
            if epsilon > 0.0:
                decisions = [
                    {"block_id": "b0", "source_group": 16,
                     "output_tile": 16, "drop": True,
                     "decision_cycle": 9},
                    {"block_id": "b1", "source_group": 16,
                     "output_tile": 16, "drop": False,
                     "decision_cycle": 20},
                ]
            rows.append({
                "sequence": sample["sequence"],
                "sample_ordinal": sample["sample_ordinal"],
                "global_sample_id": sample["global_sample_id"],
                "aee": 1.0 if epsilon == 0.0 else 1.01,
                "cycle_count": 100 if epsilon == 0.0 else 80,
                "decisions": decisions,
            })
        points.append({"epsilon": epsilon, "samples": rows})

    return {
        "schema": M.INPUT_SCHEMA,
        "capture": {
            "producer": "M1624",
            "result_namespace": M.M1624_RESULT_NAMESPACE,
            "source_contract_sha256": M.M1624_SOURCE_CONTRACT_SHA256,
            "release_sha256": M.M1626_RELEASE_SHA256,
            "checkpoint_sha256": M.CHECKPOINT_SHA256,
            "sample_order_sha256": M.SAMPLE_ORDER_SHA256,
            "samples": 40,
            "reduced_binary": True,
            "result_manifest_sha256": digest("a"),
            "result_outer_seal_file_sha256": digest("b"),
            "different_author_result_review_sha256": digest("c"),
            "different_author_result_review_status":
                "PASS_M1624_REDUCED_BINARY_RESULT__PAIRED_EVALUATION_ONLY",
        },
        "tsbg": {
            "admitted": tsbg,
            "admission_receipt_sha256": digest("d") if tsbg else None,
        },
        "baseline_identity": {
            "mode": (M.BASELINE_WITH_TSBG if tsbg else
                     M.BASELINE_WITHOUT_TSBG),
            "same_resource": True,
            "same_cohort": True,
            "sample_order_sha256": M.SAMPLE_ORDER_SHA256,
            "checkpoint_sha256": M.CHECKPOINT_SHA256,
            "cycle_model_sha256": digest("e"),
            "resource_model_sha256": digest("f"),
            "baseline_receipt_sha256": digest("1"),
            "includes_admitted_tsbg": tsbg,
            "component_speedup_multiplication_allowed": False,
        },
        "baseline_samples": baseline,
        "epsilon_points": points,
    }


class M1643SourceTest(unittest.TestCase):
    def assert_rejected(self, document):
        with self.assertRaises(M.M1643Error):
            M.evaluate_paired_document(document)

    def test_source_self_check_is_inert(self):
        row = M.source_self_check()
        self.assertEqual(row["samples"], 40)
        self.assertTrue(row["claim_boundary"]["source_only"])
        self.assertFalse(row["claim_boundary"]["actual_payload"])

    def test_valid_point_uses_ratio_of_sums_and_internal_savings(self):
        result = M.evaluate_paired_document(make_document())
        point = result["points"][1]
        self.assertAlmostEqual(point["cycle_account"][
            "local_same_resource_speedup"], 1.25)
        self.assertEqual(point["internally_derived_savings"]["weight_bytes"],
                         40 * 256)
        self.assertEqual(point["internally_derived_savings"]["compute_ops"],
                         40 * 256)
        self.assertEqual(point["internally_derived_savings"]["psum_bytes"],
                         40 * 32)
        self.assertTrue(point["passes_fixed_gate"])

    def test_epsilon_zero_is_exact_bypass(self):
        result = M.evaluate_paired_document(make_document())
        point = result["points"][0]
        self.assertEqual(point["metadata_account"]["metadata_bytes"], 0)
        self.assertEqual(point["internally_derived_savings"]["total_blocks"], 0)
        self.assertEqual(point["cycle_account"]["local_same_resource_speedup"], 1.0)

    def test_epsilon_zero_drop_or_aee_drift_rejected(self):
        document = make_document()
        document["epsilon_points"][0]["samples"][0]["decisions"] = [{
            "block_id": "b0", "source_group": 16, "output_tile": 16,
            "drop": True, "decision_cycle": 0}]
        self.assert_rejected(document)
        document = make_document()
        document["epsilon_points"][0]["samples"][0]["aee"] = 1.0001
        self.assert_rejected(document)

    def test_metadata_is_uint16_per_16x16_block(self):
        point = M.evaluate_paired_document(make_document())["points"][1]
        self.assertEqual(point["metadata_account"]["metadata_bytes"], 160)
        self.assertAlmostEqual(point["metadata_account"][
            "metadata_to_baseline_weight_bytes"], 2.0 / 256.0)

    def test_wrong_geometry_rejected(self):
        document = make_document()
        document["epsilon_points"][1]["samples"][0]["decisions"][0][
            "source_group"] = 8
        self.assert_rejected(document)

    def test_late_decision_rejected(self):
        document = make_document()
        document["epsilon_points"][1]["samples"][0]["decisions"][0][
            "decision_cycle"] = 10
        self.assert_rejected(document)

    def test_incomplete_decision_cover_rejected(self):
        document = make_document()
        document["epsilon_points"][1]["samples"][0]["decisions"].pop()
        self.assert_rejected(document)

    def test_ratio_of_sums_is_not_mean_of_ratios(self):
        document = make_document()
        point = document["epsilon_points"][1]
        document["baseline_samples"][0]["cycle_count"] = 1000
        document["epsilon_points"][0]["samples"][0]["cycle_count"] = 1000
        point["samples"][0]["cycle_count"] = 500
        result = M.evaluate_paired_document(document)["points"][1]
        expected = float(4900) / float(3620)
        self.assertAlmostEqual(result["cycle_account"][
            "local_same_resource_speedup"], expected)
        self.assertNotAlmostEqual(expected, (2.0 + 39 * 1.25) / 40.0)

    def test_overall_and_sequence_aee_are_paired_internally(self):
        document = make_document()
        rows = document["epsilon_points"][1]["samples"]
        for index, row in enumerate(rows):
            row["aee"] = 1.03 if index < 20 else 1.0
        point = M.evaluate_paired_document(document)["points"][1]
        self.assertAlmostEqual(point["overall_aee_delta"], 0.015)
        self.assertAlmostEqual(point["per_sequence_aee_delta"]["seq_a"], 0.03)
        self.assertAlmostEqual(point["per_sequence_aee_delta"]["seq_b"], 0.0)
        self.assertTrue(point["gates"]["overall_aee_delta_le_0p02"])
        self.assertTrue(point["gates"]["every_sequence_aee_delta_le_0p03"])

    def test_aee_gate_failure_is_reported_not_hidden(self):
        document = make_document()
        for row in document["epsilon_points"][1]["samples"]:
            row["aee"] = 1.031
        point = M.evaluate_paired_document(document)["points"][1]
        self.assertFalse(point["passes_fixed_gate"])
        self.assertFalse(point["gates"]["overall_aee_delta_le_0p02"])
        self.assertFalse(point["gates"]["every_sequence_aee_delta_le_0p03"])

    def test_speed_gate_failure_is_reported(self):
        document = make_document()
        for row in document["epsilon_points"][1]["samples"]:
            row["cycle_count"] = 90
        point = M.evaluate_paired_document(document)["points"][1]
        self.assertFalse(point["gates"][
            "local_same_resource_ratio_of_sums_cycles_ge_1p15"])

    def test_metadata_gate_failure_is_reported(self):
        document = make_document()
        for sample in document["baseline_samples"]:
            for block in sample["blocks"]:
                block["weight_bytes"] = 64
        point = M.evaluate_paired_document(document)["points"][1]
        self.assertFalse(point["gates"][
            "metadata_le_2pct_baseline_weight_bytes"])

    def test_exact_cohort_and_order_are_mandatory(self):
        document = make_document()
        document["epsilon_points"][1]["samples"][0], \
            document["epsilon_points"][1]["samples"][1] = \
            document["epsilon_points"][1]["samples"][1], \
            document["epsilon_points"][1]["samples"][0]
        self.assert_rejected(document)
        document = make_document()
        document["baseline_samples"].pop()
        self.assert_rejected(document)

    def test_tsbg_admission_forces_tsbg_baseline_and_no_multiplication(self):
        result = M.evaluate_paired_document(make_document(tsbg=True))
        self.assertEqual(result["baseline_mode"], M.BASELINE_WITH_TSBG)
        self.assertFalse(result["component_speedup_multiplication_allowed"])
        document = make_document(tsbg=True)
        document["baseline_identity"]["mode"] = M.BASELINE_WITHOUT_TSBG
        self.assert_rejected(document)
        document = make_document(tsbg=True)
        document["baseline_identity"][
            "component_speedup_multiplication_allowed"] = True
        self.assert_rejected(document)

    def test_capture_and_axis_are_fixed(self):
        document = make_document()
        document["capture"]["samples"] = 39
        self.assert_rejected(document)
        document = make_document()
        document["epsilon_points"][1]["epsilon"] = 0.011
        self.assert_rejected(document)

    def test_source_has_no_payload_or_execution_imports(self):
        text = SOURCE.read_text(encoding="utf-8")
        for token in ("import numpy", "import torch", "import subprocess",
                      "import socket", "np.load", "fromfile(", "Path(",
                      "open("):
            self.assertNotIn(token, text)
        self.assertIn("only --source-self-check is available", text)


if __name__ == "__main__":
    unittest.main()
