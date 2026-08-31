#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1281_decoder_cycle_traffic_surrogate_calibration_source.py")
SPEC = importlib.util.spec_from_file_location("m1281_source_under_test", str(SOURCE))
M1281 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M1281)


class M1281SyntheticFixtureTests(unittest.TestCase):
    def test_exact_fixture_passes_calibration_but_never_annexes_fixture(self):
        result = M1281.calibrate_payload(M1281.synthetic_payload(), synthetic_fixture=True)
        self.assertTrue(result["calibration_only"])
        self.assertTrue(result["cycle_surrogate"]["error_gate_pass"])
        self.assertFalse(result["cycle_surrogate"]["analytical_cycle_annex_allowed"])
        self.assertFalse(result["claim_boundary"]["analytical_cycle_annex"])
        self.assertFalse(result["claim_boundary"]["system_speedup_admitted"])
        self.assertFalse(result["claim_boundary"]["paper_ppa_ready"])

    def test_noisy_fixture_misses_point_one_percent_gate(self):
        result = M1281.calibrate_payload(
            M1281.synthetic_payload(noisy=True), synthetic_fixture=True)
        self.assertFalse(result["cycle_surrogate"]["error_gate_pass"])
        self.assertFalse(result["cycle_surrogate"]["analytical_cycle_annex_allowed"])
        self.assertTrue(result["status"].startswith("STOP_"))

    def test_traffic_conservation(self):
        payload = M1281.synthetic_payload()
        first = payload["calls"][0]
        expected = M1281.expected_traffic(first["group_count"],
                                          first["active_source_terms"])
        for key in ("descriptor_bytes", "weight_bytes", "psum_read_bytes",
                    "compute_count", "psum_write_bytes", "commit_bytes"):
            self.assertEqual(first[key], expected[key])

    def test_unsealed_input_rejected(self):
        payload = M1281.synthetic_payload()
        payload["authority"]["result_hammer_pass"] = False
        with self.assertRaises(M1281.CalibrationError):
            M1281.calibrate_payload(payload, synthetic_fixture=True)

    def test_claim_promotion_rejected(self):
        payload = M1281.synthetic_payload()
        payload["claim_boundary"]["system_speedup_admitted"] = True
        with self.assertRaises(M1281.CalibrationError):
            M1281.calibrate_payload(payload, synthetic_fixture=True)

    def test_full_selftest(self):
        receipt = M1281.run_self_test()
        self.assertEqual(receipt["attack_cases_rejected"], 15)
        self.assertFalse(receipt["live_work_prefix_opened"])


if __name__ == "__main__":
    unittest.main()
