#!/usr/bin/env python3
"""Different-author, synthetic-only hammer for the frozen M1643 source.

This test never opens M1624 payload/result files.  It imports only the frozen
in-memory accounting kernel and feeds independently constructed dictionaries.
"""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "build_m1643_motion_ep34_s2_ccbs16_paired_evaluation_source.py")
CONTRACT = REPO / (
    "hw_autoresearch_nts07/contracts/"
    "m1643_motion_ep34_s2_ccbs16_paired_evaluation_source_contract_r1_20260901.json")
EXPECTED_SOURCE_SHA256 = (
    "3d4c53292337a83a17436b6d8030dffd8eb48c25ed334f4b8894495c8bd6fe5d")
EXPECTED_CONTRACT_SHA256 = (
    "37ce4e1b7428c7b4205def96e9b92c0c58bf6692dba772828ce70f24926947fb")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SPEC = importlib.util.spec_from_file_location("m1643_independent", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def digest(character):
    return character * 64


def document(tsbg=False, blocks_per_sample=3):
    """Build an independent 40-sample/two-sequence exact paired fixture."""
    baseline = []
    for ordinal in range(40):
        sequence = "independent_a" if ordinal < 13 else "independent_b"
        blocks = []
        for block_ordinal in range(blocks_per_sample):
            first = 20 + block_ordinal * 30
            blocks.append({
                "block_id": "blk_%d" % block_ordinal,
                "source_group": 16,
                "output_tile": 16,
                "weight_bytes": 256,
                "compute_ops": 256,
                "psum_bytes": 48,
                "service_cycles": 24,
                "first_weight_fetch_cycle": first,
                "first_compute_cycle": first + 2,
                "first_psum_cycle": first + 7,
            })
        baseline.append({
            "sequence": sequence,
            "sample_ordinal": ordinal,
            "global_sample_id": "independent_%02d" % ordinal,
            "aee": 2.0,
            "cycle_count": 230 + ordinal,
            "blocks": blocks,
        })

    points = []
    for epsilon in (0.0, 0.01, 0.02, 0.05, 0.10):
        rows = []
        for sample in baseline:
            decisions = []
            if epsilon > 0.0:
                for block_ordinal in range(blocks_per_sample):
                    decisions.append({
                        "block_id": "blk_%d" % block_ordinal,
                        "source_group": 16,
                        "output_tile": 16,
                        "drop": block_ordinal == 0,
                        "decision_cycle": 19 + block_ordinal * 30,
                    })
            rows.append({
                "sequence": sample["sequence"],
                "sample_ordinal": sample["sample_ordinal"],
                "global_sample_id": sample["global_sample_id"],
                "aee": 2.0 if epsilon == 0.0 else 2.01,
                "cycle_count": (sample["cycle_count"] if epsilon == 0.0
                                else sample["cycle_count"] - 45),
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
            "mode": M.BASELINE_WITH_TSBG if tsbg else M.BASELINE_WITHOUT_TSBG,
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


class IndependentM1644Hammer(unittest.TestCase):
    maxDiff = None

    def reject(self, row):
        with self.assertRaises(M.M1643Error):
            M.evaluate_paired_document(row)

    def test_01_frozen_hashes_and_constants(self):
        self.assertEqual(sha256(SOURCE), EXPECTED_SOURCE_SHA256)
        self.assertEqual(sha256(CONTRACT), EXPECTED_CONTRACT_SHA256)
        self.assertEqual((M.SOURCE_GROUP, M.OUTPUT_TILE, M.WEIGHTS_PER_BLOCK),
                         (16, 16, 256))
        self.assertEqual(M.METADATA_BYTES_PER_BLOCK, 2)
        self.assertEqual(tuple(M.EPSILON_AXIS),
                         (0.0, 0.01, 0.02, 0.05, 0.10))

    def test_02_valid_independent_fixture(self):
        result = M.evaluate_paired_document(document())
        self.assertEqual(len(result["points"]), 5)
        self.assertFalse(result["paper_admission"])
        self.assertTrue(result["paired_speedup_only"])
        self.assertFalse(result["component_speedup_multiplication_allowed"])
        self.assertTrue(result["points"][1]["passes_fixed_gate"])

    def test_03_epsilon_zero_is_literal_bypass(self):
        result = M.evaluate_paired_document(document())["points"][0]
        self.assertEqual(result["metadata_account"]["metadata_bytes"], 0)
        self.assertEqual(result["internally_derived_savings"]["total_blocks"], 0)
        self.assertEqual(result["cycle_account"]["local_same_resource_speedup"], 1.0)
        mutations = []
        for field, value in (("aee", 2.0000001), ("cycle_count", 229)):
            row = document()
            row["epsilon_points"][0]["samples"][0][field] = value
            mutations.append(row)
        row = document()
        row["epsilon_points"][0]["samples"][0]["decisions"] = [{
            "block_id": "blk_0", "source_group": 16,
            "output_tile": 16, "drop": False, "decision_cycle": 0}]
        mutations.append(row)
        for row in mutations:
            self.reject(row)

    def test_04_exact_geometry_and_complete_cover(self):
        mutations = []
        for field, value in (("source_group", 8), ("source_group", 32),
                             ("output_tile", 8), ("output_tile", 32)):
            row = document()
            row["epsilon_points"][1]["samples"][0]["decisions"][0][field] = value
            mutations.append(row)
        row = document()
        row["epsilon_points"][1]["samples"][0]["decisions"].pop()
        mutations.append(row)
        row = document()
        row["epsilon_points"][1]["samples"][0]["decisions"].append(copy.deepcopy(
            row["epsilon_points"][1]["samples"][0]["decisions"][0]))
        mutations.append(row)
        row = document()
        row["epsilon_points"][1]["samples"][0]["decisions"][0]["block_id"] = "alien"
        mutations.append(row)
        for row in mutations:
            self.reject(row)

    def test_05_drop_must_precede_each_resource(self):
        fields = ("first_weight_fetch_cycle", "first_compute_cycle",
                  "first_psum_cycle")
        for earliest in fields:
            row = document()
            block = row["baseline_samples"][0]["blocks"][0]
            for field in fields:
                block[field] = 50
            block[earliest] = 19
            row["epsilon_points"][1]["samples"][0]["decisions"][0][
                "decision_cycle"] = 19
            self.reject(row)

    def test_06_keep_may_be_late_but_drop_may_not(self):
        row = document()
        row["epsilon_points"][1]["samples"][0]["decisions"][1][
            "decision_cycle"] = 100000
        result = M.evaluate_paired_document(row)
        self.assertTrue(result["points"][1]["passes_fixed_gate"])
        row["epsilon_points"][1]["samples"][0]["decisions"][1]["drop"] = True
        self.reject(row)

    def test_07_candidate_saved_fields_have_no_authority(self):
        clean = M.evaluate_paired_document(document())["points"][1]
        row = document()
        for sample in row["epsilon_points"][1]["samples"]:
            sample["saved_weight_bytes"] = 10 ** 30
            sample["saved_compute_ops"] = 10 ** 30
            sample["saved_psum_bytes"] = 10 ** 30
            for decision in sample["decisions"]:
                decision["saved_weight_bytes"] = 10 ** 30
        dirty = M.evaluate_paired_document(row)["points"][1]
        self.assertEqual(dirty["internally_derived_savings"],
                         clean["internally_derived_savings"])

    def test_08_metadata_is_two_bytes_for_every_positive_decision(self):
        point = M.evaluate_paired_document(document())["points"][1]
        self.assertEqual(point["metadata_account"]["metadata_bytes"],
                         40 * 3 * 2)
        self.assertAlmostEqual(
            point["metadata_account"]["metadata_to_baseline_weight_bytes"],
            2.0 / 256.0)
        self.assertEqual(point["metadata_account"]["encoding"],
                         "one_uint16_bound_per_positive_epsilon_block")

    def test_09_ratio_of_sums_not_mean_of_ratios(self):
        row = document()
        row["baseline_samples"][0]["cycle_count"] = 10000
        row["epsilon_points"][0]["samples"][0]["cycle_count"] = 10000
        row["epsilon_points"][1]["samples"][0]["cycle_count"] = 4000
        point = M.evaluate_paired_document(row)["points"][1]
        baseline_sum = sum(x["cycle_count"] for x in row["baseline_samples"])
        candidate_sum = sum(x["cycle_count"] for x in
                            row["epsilon_points"][1]["samples"])
        mean_ratios = sum(
            float(base["cycle_count"]) / float(candidate["cycle_count"])
            for base, candidate in zip(row["baseline_samples"],
                                       row["epsilon_points"][1]["samples"])) / 40
        self.assertAlmostEqual(point["cycle_account"]["local_same_resource_speedup"],
                               float(baseline_sum) / candidate_sum)
        self.assertNotAlmostEqual(point["cycle_account"]["local_same_resource_speedup"],
                                  mean_ratios)

    def test_10_paired_aee_overall_and_per_sequence(self):
        row = document()
        for index, candidate in enumerate(row["epsilon_points"][1]["samples"]):
            candidate["aee"] = 2.03 if index < 13 else 2.0
        point = M.evaluate_paired_document(row)["points"][1]
        self.assertAlmostEqual(point["overall_aee_delta"], 13 * 0.03 / 40)
        self.assertAlmostEqual(point["per_sequence_aee_delta"]["independent_a"], 0.03)
        self.assertAlmostEqual(point["per_sequence_aee_delta"]["independent_b"], 0.0)
        self.assertTrue(point["gates"]["overall_aee_delta_le_0p02"])
        self.assertTrue(point["gates"]["every_sequence_aee_delta_le_0p03"])

    def test_11_gate_boundaries_are_fixed_and_conjunctive(self):
        self.assertEqual(M.OVERALL_AEE_DELTA_MAX, 0.02)
        self.assertEqual(M.PER_SEQUENCE_AEE_DELTA_MAX, 0.03)
        self.assertEqual(M.METADATA_TO_BASELINE_WEIGHT_BYTES_MAX, 0.02)
        self.assertEqual(M.LOCAL_SAME_RESOURCE_SPEEDUP_MIN, 1.15)
        gate_names = (
            "overall_aee_delta_le_0p02",
            "every_sequence_aee_delta_le_0p03",
            "metadata_le_2pct_baseline_weight_bytes",
            "local_same_resource_ratio_of_sums_cycles_ge_1p15")
        for gate in gate_names:
            row = document()
            if gate == gate_names[0]:
                for sample in row["epsilon_points"][1]["samples"]:
                    sample["aee"] = 2.0201
            elif gate == gate_names[1]:
                for index, sample in enumerate(row["epsilon_points"][1]["samples"]):
                    sample["aee"] = 2.0301 if index < 13 else 1.99
            elif gate == gate_names[2]:
                for sample in row["baseline_samples"]:
                    for block in sample["blocks"]:
                        block["weight_bytes"] = 99
            else:
                for base, sample in zip(row["baseline_samples"],
                                        row["epsilon_points"][1]["samples"]):
                    sample["cycle_count"] = int(base["cycle_count"] / 1.149)
                    if float(base["cycle_count"]) / sample["cycle_count"] >= 1.15:
                        sample["cycle_count"] += 1
            point = M.evaluate_paired_document(row)["points"][1]
            self.assertFalse(point["gates"][gate])
            self.assertFalse(point["passes_fixed_gate"])

    def test_12_exact_population_order_and_stratification(self):
        mutations = []
        row = document(); row["baseline_samples"].pop(); mutations.append(row)
        row = document(); row["epsilon_points"][1]["samples"].pop(); mutations.append(row)
        row = document(); row["epsilon_points"][1]["samples"][0], row["epsilon_points"][1]["samples"][1] = row["epsilon_points"][1]["samples"][1], row["epsilon_points"][1]["samples"][0]; mutations.append(row)
        row = document(); row["baseline_samples"][1]["global_sample_id"] = row["baseline_samples"][0]["global_sample_id"]; row["baseline_samples"][1]["sequence"] = row["baseline_samples"][0]["sequence"]; row["baseline_samples"][1]["sample_ordinal"] = row["baseline_samples"][0]["sample_ordinal"]; mutations.append(row)
        row = document()
        for sample in row["baseline_samples"]:
            sample["sequence"] = "only_one"
        for point in row["epsilon_points"]:
            for sample in point["samples"]:
                sample["sequence"] = "only_one"
        mutations.append(row)
        for row in mutations:
            self.reject(row)

    def test_13_capture_identity_mutations_are_rejected(self):
        mutations = {
            "producer": "M1623",
            "result_namespace": "wrong",
            "source_contract_sha256": digest("0"),
            "release_sha256": digest("0"),
            "checkpoint_sha256": digest("0"),
            "sample_order_sha256": digest("0"),
            "samples": 39,
            "reduced_binary": False,
            "result_manifest_sha256": "bad",
            "result_outer_seal_file_sha256": "bad",
            "different_author_result_review_sha256": "bad",
            "different_author_result_review_status": "PASS_BUT_WRONG",
        }
        for field, value in mutations.items():
            row = document(); row["capture"][field] = value; self.reject(row)

    def test_14_baseline_identity_mutations_are_rejected(self):
        fields = {
            "mode": "WEAK_BASELINE",
            "same_resource": False,
            "same_cohort": False,
            "sample_order_sha256": digest("0"),
            "checkpoint_sha256": digest("0"),
            "cycle_model_sha256": "bad",
            "resource_model_sha256": "bad",
            "baseline_receipt_sha256": "bad",
            "includes_admitted_tsbg": True,
            "component_speedup_multiplication_allowed": True,
        }
        for field, value in fields.items():
            row = document(); row["baseline_identity"][field] = value; self.reject(row)

    def test_15_tsbg_admission_forces_receipt_and_baseline(self):
        result = M.evaluate_paired_document(document(tsbg=True))
        self.assertEqual(result["baseline_mode"], M.BASELINE_WITH_TSBG)
        self.assertTrue(result["tsbg_admitted"])
        mutations = []
        row = document(tsbg=True); row["tsbg"]["admission_receipt_sha256"] = None; mutations.append(row)
        row = document(tsbg=True); row["baseline_identity"]["mode"] = M.BASELINE_WITHOUT_TSBG; mutations.append(row)
        row = document(tsbg=True); row["baseline_identity"]["includes_admitted_tsbg"] = False; mutations.append(row)
        row = document(tsbg=True); row["baseline_identity"]["component_speedup_multiplication_allowed"] = True; mutations.append(row)
        for row in mutations:
            self.reject(row)

    def test_16_epsilon_axis_is_exact_and_ordered(self):
        mutations = []
        row = document(); row["epsilon_points"][1]["epsilon"] = 0.011; mutations.append(row)
        row = document(); row["epsilon_points"].pop(); mutations.append(row)
        row = document(); row["epsilon_points"][1], row["epsilon_points"][2] = row["epsilon_points"][2], row["epsilon_points"][1]; mutations.append(row)
        row = document(); row["epsilon_points"][0]["epsilon"] = 1e-10; mutations.append(row)
        for row in mutations:
            self.reject(row)

    def test_17_type_nan_inf_and_negative_hammer(self):
        mutations = []
        row = document(); row["baseline_samples"][0]["aee"] = float("nan"); mutations.append(row)
        row = document(); row["epsilon_points"][1]["samples"][0]["aee"] = float("inf"); mutations.append(row)
        row = document(); row["baseline_samples"][0]["cycle_count"] = True; mutations.append(row)
        row = document(); row["epsilon_points"][1]["samples"][0]["cycle_count"] = 0; mutations.append(row)
        row = document(); row["baseline_samples"][0]["blocks"][0]["psum_bytes"] = -1; mutations.append(row)
        row = document(); row["epsilon_points"][1]["samples"][0]["decisions"][0]["decision_cycle"] = -1; mutations.append(row)
        row = document(); row["epsilon_points"][1]["samples"][0]["decisions"][0]["drop"] = 1; mutations.append(row)
        for row in mutations:
            self.reject(row)

    def test_18_source_remains_inert_and_claim_closed(self):
        source_text = SOURCE.read_text(encoding="utf-8")
        forbidden = ("import numpy", "import torch", "import subprocess",
                     "import socket", "np.load", "torch.load", "fromfile(",
                     "os.system", "Popen(")
        for token in forbidden:
            self.assertNotIn(token, source_text)
        with self.assertRaises(M.M1643Error):
            M.main([])
        self.assertEqual(M.main(["--source-self-check"]), 0)
        check = M.source_self_check()
        self.assertTrue(check["claim_boundary"]["source_only"])
        for key in ("actual_payload", "payload_loader", "aee_result",
                    "cycle_result", "performance_claim", "paper_result",
                    "gpu", "dse", "rtl", "eda", "release"):
            self.assertFalse(check["claim_boundary"][key])


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentM1644Hammer)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "schema": "m1644_m1643_independent_hammer_runtime_summary_r1_v1",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "runtime": sys.version.split()[0],
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "input_mutations_or_boundary_attacks": 77,
        "payload_opened": False,
        "gpu_runs": 0,
        "dse_runs": 0,
        "rtl_runs": 0,
        "eda_runs": 0,
        "release": False,
    }
    print("M1644_RUNTIME_SUMMARY=" + json.dumps(summary, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
