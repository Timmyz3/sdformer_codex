import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resolution_tile_term_ledger.py"
SPEC = importlib.util.spec_from_file_location("resolution_tile_term_ledger", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ResolutionTileTermLedgerTest(unittest.TestCase):
    def test_crop_w9_matches_frozen_profile_geometry(self):
        case = MODULE.build_case("crop-w9", 288, 384, 9)
        self.assertEqual(case.rows_per_frame, 6720)
        self.assertEqual(case.scheduled_token_slots_per_frame, 1_088_640)
        self.assertEqual(
            [stage.rows_per_frame for stage in case.stages],
            [2640, 1440, 2160, 480],
        )
        self.assertEqual(case.local5_valid_edges_per_window, 738)
        self.assertAlmostEqual(case.local5_invalid_candidate_ratio, 72 / 810)

    def test_full_w15_preserves_row_count_but_scales_token_work(self):
        crop = MODULE.build_case("crop-w9", 288, 384, 9)
        full = MODULE.build_case("full-w15", 480, 640, 15)
        self.assertEqual(full.rows_per_frame, crop.rows_per_frame)
        self.assertEqual(full.tokens_per_row, 450)
        self.assertAlmostEqual(
            full.scheduled_token_slots_per_frame
            / crop.scheduled_token_slots_per_frame,
            25 / 9,
        )
        self.assertEqual(full.scs_counter_bits, 9)
        self.assertEqual(full.local5_three_row_k_bytes_per_head, 360)
        self.assertEqual(full.local5_valid_edges_per_window, 2130)
        self.assertAlmostEqual(full.local5_invalid_candidate_ratio, 120 / 2250)

    def test_full_w9_scales_descriptor_count(self):
        full = MODULE.build_case("full-w9", 480, 640, 9)
        self.assertEqual(full.rows_per_frame, 19_980)
        self.assertEqual(full.tokens_per_row, 162)


if __name__ == "__main__":
    unittest.main()
