import json
import unittest
from pathlib import Path

from scripts.model_motion_reversible_quotient_bundle import build_model


ROOT = Path(__file__).resolve().parents[1]


class MotionReversibleQuotientBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compact = json.loads(
            (ROOT / "results/profile100_compact_arch_stats_20260714.json").read_text()
        )
        tesc = json.loads(
            (ROOT / "results/motion_temporal_equivalence_20260803/report.json").read_text()
        )
        cls.result = build_model(compact, tesc)

    def test_modes_cover_all_pairs(self):
        counts = self.result["counts"]
        self.assertEqual(
            sum(row["pairs"] for row in counts["modes"].values()),
            counts["pair_total"],
        )

    def test_known_mode_counts(self):
        modes = self.result["counts"]["modes"]
        self.assertEqual(modes["zz_equal"]["pairs"], 45_196_672)
        self.assertEqual(modes["one_equal"]["pairs"], 5_786_710)
        self.assertEqual(modes["both_equal"]["pairs"], 2_738_242)

    def test_slot_reduction_matches_equal_rate(self):
        counts = self.result["counts"]
        stream = self.result["stream"]
        equal_rate = counts["score_equal"] / counts["pair_total"]
        self.assertAlmostEqual(
            stream["quotient_slot_reduction_vs_fixed_ttb"], equal_rate / 2
        )

    def test_active_entries_match_active_k_tokens(self):
        self.assertEqual(
            self.result["counts"]["active_k_tokens"],
            self.result["backend"]["baseline_scs_active_entries"],
        )

    def test_admission_is_not_predeclared(self):
        self.assertFalse(self.result["admission"]["fullres_t450_profile_present"])

    def test_raw_profile_schema_is_accepted(self):
        compact = json.loads(
            (ROOT / "results/profile100_compact_arch_stats_20260714.json").read_text()
        )
        raw_shape = {
            "summary": {
                "binary_temporal_pairs": compact["models"]["H67"][
                    "binary_temporal_pairs"
                ],
                "h60_records": [{"tokens": 450}],
            }
        }
        tesc = json.loads(
            (ROOT / "results/motion_temporal_equivalence_20260803/report.json").read_text()
        )
        result = build_model(raw_shape, tesc)
        self.assertEqual(result["source"]["temporal_tokens"], 450)
        self.assertFalse(result["admission"]["fullres_t450_profile_present"])

    def test_fullres_admission_requires_complete_identity(self):
        compact = json.loads(
            (ROOT / "results/profile100_compact_arch_stats_20260714.json").read_text()
        )
        raw_shape = {
            "summary": {
                "binary_temporal_pairs": compact["models"]["H67"][
                    "binary_temporal_pairs"
                ],
                "h60_records": [{"tokens": 450}],
            }
        }
        tesc = json.loads(
            (ROOT / "results/motion_temporal_equivalence_20260803/report.json").read_text()
        )
        tesc["source"] = {
            "profile": tesc["profile"],
            "resolution": [480, 640],
            "crop": None,
            "window_size": [2, 15, 15],
            "temporal_tokens": 450,
            "samples": 100,
            "h60_records": 1200,
            "bn_policy": "no_running",
            "config_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
        }
        result = build_model(raw_shape, tesc)
        self.assertTrue(result["admission"]["fullres_t450_profile_present"])

    def test_fullres_admission_rejects_tesc_profile_mismatch(self):
        compact = json.loads(
            (ROOT / "results/profile100_compact_arch_stats_20260714.json").read_text()
        )
        tesc = json.loads(
            (ROOT / "results/motion_temporal_equivalence_20260803/report.json").read_text()
        )
        tesc["source"] = {
            "profile": "/different/profile.json",
            "temporal_tokens": None,
        }
        with self.assertRaisesRegex(ValueError, "路径不一致"):
            build_model(compact, tesc)


if __name__ == "__main__":
    unittest.main()
