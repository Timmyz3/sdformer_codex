import unittest

from scripts.run_h81_rqtb_g0_profile import apply_mvsec_gate, split_empty_active


class H81EmptyActiveSplitTest(unittest.TestCase):
    def test_split_and_conservation(self):
        result = split_empty_active(
            {
                "pairs": 100,
                "pair_empty": 60,
                "score_equal_ttx": 90,
            }
        )
        self.assertEqual(result["nonempty_pairs"], 40)
        self.assertEqual(result["nonempty_equal_pairs"], 30)
        self.assertAlmostEqual(result["nonempty_equal_ratio"], 0.75)
        self.assertTrue(all(result["conservation"].values()))

    def test_rejects_empty_not_subset_of_equal(self):
        with self.assertRaises(ValueError):
            split_empty_active(
                {
                    "pairs": 100,
                    "pair_empty": 60,
                    "score_equal_ttx": 59,
                }
            )

    def test_mvsec_failure_is_fail_closed(self):
        report = {
            "status": "G0_PASS_G1_BLOCKED_BY_SELECTOR_MVSEC_AND_FAIR_RTL",
            "blocking_gates": [
                "algorithm selector has not selected H81",
                "H81 MVSEC is missing",
            ],
        }
        receipt = {
            "status": "FAIL_H81_MVSEC_ALL_SEQUENCE_GATE",
            "failing_sequences": ["indoor_flying1"],
        }
        apply_mvsec_gate(report, receipt)
        self.assertEqual(
            report["status"],
            "G0_PASS_G1_BLOCKED_BY_SELECTOR_AND_MVSEC_FAIL",
        )
        self.assertIn(
            "H81 MVSEC all-sequence gate failed: indoor_flying1",
            report["blocking_gates"],
        )


if __name__ == "__main__":
    unittest.main()
