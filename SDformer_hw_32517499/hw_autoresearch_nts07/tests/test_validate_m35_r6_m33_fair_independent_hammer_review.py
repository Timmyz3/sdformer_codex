from __future__ import print_function

import importlib.util
import pathlib
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[2]
VALIDATOR_PATH = (
    REPO / "hw_autoresearch_nts07/dc_handoff/scripts/"
    "validate_m35_r6_m33_fair_independent_hammer_review.py"
)
SPEC = importlib.util.spec_from_file_location("m35_r6_review_validator", str(VALIDATOR_PATH))
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class M35R6M33FairIndependentHammerReviewTest(unittest.TestCase):
    def write(self, root, name, text):
        path = pathlib.Path(root) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_canonical_review_rebuilds_to_strict_no_go(self):
        rebuilt = VALIDATOR.audit()
        canonical = VALIDATOR.strict_json(VALIDATOR.REVIEW)
        self.assertEqual(rebuilt, canonical)
        self.assertEqual(rebuilt["review_score_0_to_100"], 80)
        self.assertEqual((rebuilt["p0_count"], rebuilt["p1_count"],
                          rebuilt["p2_count"]), (0, 1, 5))
        self.assertEqual(
            rebuilt["review_verdict"],
            "NO_GO_EXACT_SHA_RELEASE_REPAIR_CONTRACT_AND_FRESH_RERUN")

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

    def test_manifest_target_byte_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = self.write(root, "target.txt", "frozen\n")
            manifest = self.write(root, "manifest.sha256", "{}  ./target.txt\n".format(
                VALIDATOR.sha256(target)))
            self.assertEqual(len(VALIDATOR.manifest_rows(manifest, root)), 1)
            target.write_text("mutated\n", encoding="utf-8")
            with self.assertRaises(VALIDATOR.ReviewFailure):
                VALIDATOR.manifest_rows(manifest, root)

    def test_manifest_duplicate_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = self.write(root, "target.txt", "frozen\n")
            row = "{}  ./target.txt\n".format(VALIDATOR.sha256(target))
            manifest = self.write(root, "manifest.sha256", row + row)
            with self.assertRaises(VALIDATOR.ReviewFailure):
                VALIDATOR.manifest_rows(manifest, root)

    def test_nonzero_and_malformed_rc_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, content in enumerate(("1\n", "0", "zero\n")):
                path = self.write(directory, "bad{}.rc".format(index), content)
                with self.assertRaises(VALIDATOR.ReviewFailure):
                    VALIDATOR.require_zero_rc(path)

    def test_timing_parser_uses_worst_of_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.write(
                directory, "timing.rpt",
                "slack (MET) 0.2000\nslack (VIOLATED) -0.0100\n"
                "slack (MET) 0.1000\n")
            self.assertEqual(VALIDATOR.min_slack(report), (-0.01, 3))

    def test_area_parser_keeps_zero_wireload_boundary(self):
        text = """Design : dut
Number of ports: 5
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

    def test_indented_warning_and_unclassified_warning_are_not_dropped(self):
        lines, codes = VALIDATOR.warning_ledger(
            "Warning: high fanout (TIM-134)\n"
            "    Warning: 0 (347) undriven nets (FM-399)\n"
            "Warning: unlinked power cells\nInformation: ignored\n")
        self.assertEqual(len(lines), 3)
        self.assertEqual(codes, {"FM-399": 1, "TIM-134": 1, "UNCLASSIFIED": 1})

    def test_no_unmatched_phrase_accepts_clean_report_rejects_injection(self):
        VALIDATOR.require_empty_formality_report(
            "Report : unmatched_points\nNo unmatched points.\n1\n",
            "No unmatched points.")
        with self.assertRaises(VALIDATOR.ReviewFailure):
            VALIDATOR.require_empty_formality_report(
                "No unmatched points.\nUnmatched compare point\n",
                "No unmatched points.")

    def test_mapped_multiplier_signature_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = self.write(directory, "clean.v", "FA1D0 U1 (.A(a), .B(b));\n")
            dirty = self.write(directory, "dirty.v", "GTECH_MULT U2 (.A(a), .B(b));\n")
            self.assertEqual(VALIDATOR.physical_multiplier_hits([clean]), [])
            self.assertEqual(len(VALIDATOR.physical_multiplier_hits([dirty])), 1)

    def test_contract_builder_mismatch_cannot_be_hidden(self):
        mismatch = {
            "contract_declared_sha256": "old",
            "launch_manifest_sha256": "fixed",
            "snapshot_file_sha256": "fixed",
            "live_file_sha256": "fixed",
        }
        VALIDATOR.require_identity_mismatch(mismatch)
        repaired_or_hidden = dict(mismatch, contract_declared_sha256="fixed")
        with self.assertRaises(VALIDATOR.ReviewFailure):
            VALIDATOR.require_identity_mismatch(repaired_or_hidden)


if __name__ == "__main__":
    unittest.main()
