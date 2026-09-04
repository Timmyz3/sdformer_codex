#!/usr/bin/env python3
"""CPU-light M1526 source tests; no checkpoint read or quantization."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1526_ep34_decoder_int8_numeric_bridge_gate_source.py")
SPEC = importlib.util.spec_from_file_location("test_m1526_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class AdmissionTests(unittest.TestCase):
    def test_01_current_decision_is_fail_closed(self):
        result = M.build_admission()
        self.assertEqual(result["status"], M.ADMISSION_STATUS)
        self.assertEqual(result["decision"]["ep34_decoder_int8_rule"],
                         "NOT_AUTHORIZED")
        self.assertEqual(result["decision"]["m1525_int8_or_k8_weighted_replay"],
                         "BLOCKED")

    def test_02_ep34_checkpoint_config_and_four_weight_identities_bound(self):
        result = M.build_admission()
        handoff = result["algorithm_handoff"]
        self.assertEqual(handoff["target_checkpoint_sha256"],
                         M.EP34_CHECKPOINT_SHA256)
        self.assertEqual(handoff["target_config_sha256"], M.EP34_CONFIG_SHA256)
        self.assertEqual(len(handoff["minimum_return_artifacts"]
                             ["decoder_int8_identity_manifest"]
                             ["source_float_weight_sha256"]), 4)

    def test_03_m61_is_identity_and_operator_mismatch(self):
        result = M.build_admission()
        self.assertNotEqual(M.M61_CHECKPOINT_SHA256, M.EP34_CHECKPOINT_SHA256)
        self.assertFalse(result["candidate_rules_not_authority"]
                         ["m61_prediction_head_rule"]
                         ["reusable_for_ep34_decoder"])
        self.assertIn("M61_IDENTITY", [row["id"] for row in
                                       result["blocking_findings"]][1])

    def test_04_generic_per_tensor_and_m61_per_output_conflict(self):
        rules = M.build_admission()["candidate_rules_not_authority"]
        self.assertEqual(rules["generic_repository_spec"]["granularity"],
                         "PER_TENSOR")
        self.assertEqual(rules["m61_prediction_head_rule"]["granularity"],
                         "PER_OUTPUT")

    def test_05_acc24_is_candidate_not_range_proof(self):
        meta = M.accumulator_metadata_requirements()
        self.assertEqual(meta["accumulator_storage_bits_candidate_not_admitted"], 24)
        self.assertFalse(meta["old_accumulator_numbers_reusable"])
        self.assertEqual(meta["per_output_axis_if_selected"], 1)
        self.assertEqual(meta["per_output_scale_counts_if_selected"],
                         [384, 192, 96, 96])

    def test_06_k3_s2_polyphase_bound_has_max_four_taps(self):
        phase = M.accumulator_metadata_requirements()[
            "reachable_polyphase_taps_for_k3_s2_p1_op1"]
        self.assertEqual(phase["maximum_spatial_taps_per_input_channel"], 4)
        self.assertEqual(len(phase["output_even_even"]), 1)
        self.assertEqual(len(phase["output_odd_odd"]), 4)

    def test_07_handoff_requires_kernel_miter_and_official_aee(self):
        handoff = M.algorithm_handoff()
        ids = [row["id"] for row in handoff["required_actions_in_order"]]
        self.assertEqual(ids, ["Q1_CONFIG_AUTHORITY", "Q2_DETERMINISTIC_PTQ",
                               "Q3_KERNEL_MITER", "Q4_OFFICIAL_ACCURACY"])
        gate = handoff["numeric_admission_gate"]
        self.assertEqual(gate["paired_official_absolute_aee_delta_max"], 0.02)
        self.assertTrue(gate["tolerance_is_not_ep34_authority_until_Q1_binds_it"])

    def test_08_missing_or_tampered_authority_fails_closed(self):
        original = M.M1514_SHA256
        with mock.patch.object(M, "M1514_SHA256", "0" * 64), \
                self.assertRaisesRegex(M.M1526Error, "SHA drift"):
            M.build_admission()
        self.assertNotEqual(original, "0" * 64)


class SourcePolicyTests(unittest.TestCase):
    def test_09_contract_exact_binds_source_test_and_handoff(self):
        policy = M.validate_source_policy()
        self.assertEqual(policy["expected_admission_status"], M.ADMISSION_STATUS)
        self.assertEqual(policy["algorithm_handoff_summary"]["required_action_ids"],
                         ["Q1_CONFIG_AUTHORITY", "Q2_DETERMINISTIC_PTQ",
                          "Q3_KERNEL_MITER", "Q4_OFFICIAL_ACCURACY"])

    def test_10_no_export_gpu_remote_eda_or_production_path(self):
        text = SOURCE.read_text(encoding="utf-8")
        for token in ("subprocess", "paramiko", "torch.cuda", "torch.load",
                      "np.tofile", "os.rename", "ssh ", "dc_shell", "pt_shell"):
            self.assertNotIn(token, text)
        self.assertFalse(M.CLAIM_BOUNDARY["quantized_weight_payload_written"])
        self.assertFalse(M.CLAIM_BOUNDARY["production"])
        self.assertFalse(M.CLAIM_BOUNDARY["paper_result"])


if __name__ == "__main__":
    unittest.main()
