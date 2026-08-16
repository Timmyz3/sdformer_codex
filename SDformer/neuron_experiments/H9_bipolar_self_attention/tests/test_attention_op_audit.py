from __future__ import annotations

import sys
import unittest
from pathlib import Path


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))


class AttentionOperationAuditTest(unittest.TestCase):
    def test_historical_int8_grid_requires_more_than_eight_code_bits(self):
        from run_h60_family_deploy_eval import quant_grid

        score = quant_grid(-2.0, 2.0, 1.0 / 128.0)
        gate = quant_grid(0.0, 2.0, 1.0 / 128.0)
        self.assertEqual(score["levels"], 513)
        self.assertEqual(score["minimum_code_bits"], 10)
        self.assertEqual(gate["levels"], 257)
        self.assertEqual(gate["minimum_code_bits"], 9)

    def test_candidate_incremental_counts(self):
        from audit_attention_candidate_ops import counts_for_record

        record = {"batch_windows": 2, "num_heads": 3, "tokens": 4, "head_dim": 8}
        base = counts_for_record(record, {})
        self.assertEqual(base["tx_lane_logic"], 192)
        self.assertEqual(base["tx_popcount_add"], 168)
        self.assertEqual(base["incremental_logic"], 0)

        h67 = counts_for_record(record, {"binary_motion_xor_alpha": 0.25})
        self.assertEqual(h67["incremental_logic"], 192 + 24)
        self.assertEqual(h67["incremental_add"], 168 + 24)

        h69 = counts_for_record(record, {"score_scale": 8.0})
        self.assertEqual(h69["incremental_logic"], 24)

        h70 = counts_for_record(record, {"event_temperature_enabled": True})
        self.assertEqual(h70["incremental_logic"], 192 + 48)
        self.assertEqual(h70["incremental_add"], 168)

        h71 = counts_for_record(record, {"context_broadcast_enabled": True})
        self.assertEqual(h71["incremental_add"], 144 + 192)
        self.assertEqual(h71["incremental_mac"], 48)
        self.assertEqual(h71["incremental_logic"], 192)

        h66a = counts_for_record(record, {"mode": "binary_alpha_xnor_matrix_shiftmax"})
        self.assertGreater(h66a["incremental_logic"], base["tx_lane_logic"])
        self.assertGreater(h66a["incremental_add"], base["tx_popcount_add"])
        self.assertGreater(h66a["incremental_mac"], 0)

        h66c = counts_for_record(record, {"mode": "binary_axnor_temporal_pair_shiftmax"})
        h66d = counts_for_record(record, {"mode": "binary_axnor_local5_shiftmax"})
        self.assertGreater(h66c["incremental_logic"], 0)
        self.assertGreater(h66d["incremental_logic"], h66c["incremental_logic"])
        self.assertGreater(h66d["incremental_mac"], h66c["incremental_mac"])

        h66b = counts_for_record(record, {"mode": "hamming_binary_direct"})
        self.assertGreater(h66b["incremental_logic"], 0)
        self.assertGreater(h66b["incremental_add"], base["tx_popcount_add"])

        h73 = counts_for_record(record, {"mode": "binary_de9_match_code"})
        h74 = counts_for_record(record, {"mode": "binary_mc49_match_code"})
        h75 = counts_for_record(record, {"mode": "binary_ax17_match_code"})
        self.assertEqual(h73["incremental_mac"], 24 * 18 * 8)
        self.assertEqual(h74["incremental_mac"], 24 * 49 * 8)
        self.assertEqual(h75["incremental_mac"], 24 * 17 * 8)
        self.assertGreater(h74["incremental_logic"], h73["incremental_logic"])
        self.assertGreater(h74["incremental_add"], h73["incremental_add"])
        self.assertLess(h75["incremental_logic"], h73["incremental_logic"])

        h76 = counts_for_record(record, {"mode": "binary_pc9_patch_match_code"})
        h77 = counts_for_record(record, {"mode": "binary_lc4_match_code"})
        h78 = counts_for_record(record, {"mode": "binary_g4_match_code"})
        self.assertEqual(h76["incremental_mac"], 24 * 9 * 8)
        self.assertEqual(h77["incremental_mac"], 24 * 9 * 8)
        self.assertEqual(h78["incremental_mac"], 24 * 36 * 8)
        self.assertGreater(h76["incremental_add"], h77["incremental_add"])
        self.assertGreater(h78["incremental_mac"], h76["incremental_mac"])

        h79 = counts_for_record(record, {"mode": "binary_cf10_match_code"})
        h80 = counts_for_record(record, {"mode": "binary_dn9_match_code"})
        self.assertEqual(h79["incremental_mac"], 24 * (9 * 8 + 2))
        self.assertEqual(h80["incremental_mac"], 24 * (9 * 8 + 9))
        self.assertGreater(h80["incremental_mac"], h79["incremental_mac"])


if __name__ == "__main__":
    unittest.main()
