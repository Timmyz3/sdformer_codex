#!/usr/bin/env python3
"""Local5 H24 phase 结构合同的解析计数单元测试。"""

from __future__ import annotations

import unittest

from generate_local5_h24_phase_structure_contract_v1 import expected_event_counts


class PhaseStructureContractTest(unittest.TestCase):
    def test_calibration_trace_rows(self) -> None:
        self.assertEqual(sum(expected_event_counts(3, 2).values()), 862_507)
        self.assertEqual(sum(expected_event_counts(6, 2).values()), 3_190_783)
        self.assertEqual(sum(expected_event_counts(12, 2).values()), 12_244_663)

    def test_h24_baseline_and_candidate(self) -> None:
        baseline = expected_event_counts(24, 0)
        candidate = expected_event_counts(24, 2)
        self.assertNotIn("weight_response_stall", baseline)
        self.assertEqual(candidate["weight_response_stall"], 1_179_648)
        self.assertEqual(sum(baseline.values()), 46_762_087)
        self.assertEqual(sum(candidate.values()), 47_941_735)
        self.assertEqual(candidate["tx_state"], 1_038_577)
        self.assertEqual(candidate["acc_state"], 15_897_601)
        self.assertEqual(candidate["head_state"], 26_586_433)


if __name__ == "__main__":
    unittest.main()
