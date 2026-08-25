from __future__ import print_function

import importlib.util
import json
import pathlib
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[2]
VALIDATOR_PATH = (
    REPO
    / "hw_autoresearch_nts07/dc_handoff/scripts/"
    "validate_m37_r13_independent_hammer_review.py"
)
SPEC = importlib.util.spec_from_file_location("m37_r13_review_validator", str(VALIDATOR_PATH))
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class M37R13IndependentHammerValidatorTest(unittest.TestCase):
    def write(self, root, name, text):
        path = pathlib.Path(root) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_canonical_review_rebuild(self):
        rebuilt = VALIDATOR.audit()
        canonical = VALIDATOR.strict_json(VALIDATOR.REVIEW)
        self.assertEqual(rebuilt, canonical)
        self.assertEqual(rebuilt["review_score_0_to_100"], 94)
        self.assertEqual(rebuilt["p0_count"], 0)
        self.assertEqual(rebuilt["p1_count"], 0)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "bad.json", '{"status": 1, "status": 1}\n')
            with self.assertRaises(VALIDATOR.ReviewFailure):
                VALIDATOR.strict_json(path)

    def test_nonfinite_json_constant_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "bad.json", '{"metric": NaN}\n')
            with self.assertRaises(VALIDATOR.ReviewFailure):
                VALIDATOR.strict_json(path)

    def test_manifest_byte_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = self.write(root, "target.txt", "frozen\n")
            digest = VALIDATOR.sha256(target)
            manifest = self.write(root, "manifest.sha256", "{}  ./target.txt\n".format(digest))
            self.assertEqual(len(VALIDATOR.manifest_rows(manifest)), 1)
            target.write_text("mutated\n", encoding="utf-8")
            with self.assertRaises(VALIDATOR.ReviewFailure):
                VALIDATOR.manifest_rows(manifest)

    def test_historical_manifest_can_be_parsed_without_following_live_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = self.write(root, "live.txt", "new\n")
            manifest = self.write(root, "historical.sha256", "{}  {}\n".format(
                "0" * 64, target))
            rows = VALIDATOR.manifest_rows(manifest, verify_targets=False)
            self.assertEqual(rows[0][0], "0" * 64)
            with self.assertRaises(VALIDATOR.ReviewFailure):
                VALIDATOR.manifest_rows(manifest, verify_targets=True)

    def test_area_parser_separates_architecture_from_metric_summary(self):
        text = """Design : dut
Number of cells: 11
Number of combinational cells: 7
Number of sequential cells: 4
Number of macros/black boxes: 0
Combinational area: 7.250000
Noncombinational area: 4.500000
Net Interconnect area: undefined (Wire load has zero net area)
Total cell area: 11.750000
tcbn28hpcplusbwp35p140ssg0p9v125c
"""
        with tempfile.TemporaryDirectory() as directory:
            metrics = VALIDATOR.area_metrics(self.write(directory, "area.rpt", text))
        self.assertEqual(metrics["total_cell_area_um2"], 11.75)
        self.assertTrue(metrics["zero_net_area_text"])
        self.assertEqual(metrics["macro_blackbox_cells"], 0)

    def test_timing_parser_uses_worst_of_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.write(
                directory,
                "timing.rpt",
                "slack (MET) 0.2000\nslack (VIOLATED) -0.0100\nslack (MET) 0.1000\n",
            )
            self.assertEqual(VALIDATOR.min_slack(report), (-0.01, 3))

    def test_register_sum_rejects_width_drift(self):
        good = "| foo_q_reg | Flip-flop | 3 | N |\n| bar_q_reg | Flip-flop | 5 | Y |\n"
        self.assertEqual(sum(VALIDATOR.register_widths(good).values()), 8)
        bad = good + "| foo_q_reg | Flip-flop | 4 | N |\n"
        with self.assertRaises(VALIDATOR.ReviewFailure):
            VALIDATOR.register_widths(bad)

    def test_no_unmatched_phrase_cannot_spoof_raw_matching_summary(self):
        report_only = "No unmatched points.\n"
        with self.assertRaises(VALIDATOR.ReviewFailure):
            VALIDATOR.unmatched_value(report_only, r"compare points")
        raw = (
            " 0(0) Unmatched reference(implementation) compare points\n"
            " 0(0) Unmatched reference(implementation) primary inputs, black-box outputs\n"
            " 1(0) Unmatched reference(implementation) unread points\n"
        )
        self.assertEqual(VALIDATOR.unmatched_value(raw, r"compare points"), 0)
        self.assertEqual(VALIDATOR.unmatched_value(raw, r"unread points"), 1)

    def test_independent_multiplier_pattern_rejects_mapped_operator(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = self.write(directory, "clean.v", "FA1D0 U1 (.A(a), .B(b));\n")
            dirty = self.write(directory, "dirty.v", "GTECH_MULT U2 (.A(a), .B(b));\n")
            self.assertEqual(VALIDATOR.physical_multiplier_hits([clean]), [])
            self.assertEqual(len(VALIDATOR.physical_multiplier_hits([dirty])), 1)

    def test_warning_ledger_keeps_unclassified_warnings(self):
        lines, codes = VALIDATOR.warning_ledger(
            "Warning: first (TIM-134)\nWarning: unclassified warning\nInformation: ignored\n"
        )
        self.assertEqual(len(lines), 2)
        self.assertEqual(codes, {"TIM-134": 1, "UNCLASSIFIED": 1})


if __name__ == "__main__":
    unittest.main()
