import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/analyze_m37_phase_decoupled_csd_reconstruct.py"
SPEC = importlib.util.spec_from_file_location("m37", str(SCRIPT))
M37 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M37)


class M37Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = M37.build()

    def test_full_int8_domain_exact_and_bounded(self):
        audit = self.result["signed_int8_coefficient_audit"]
        self.assertEqual(audit["values"], 256)
        self.assertLessEqual(audit["maximum_terms"], 4)
        self.assertTrue(audit["all_constructive_identities_exact"])
        for row in audit["rows"]:
            self.assertEqual(M37.reconstruct(
                [(term["sign"], term["shift"]) for term in row["terms"]]
            ), row["value"])

    def test_phase_schedule_math(self):
        schedule = self.result["phase_schedule_sensitivity"]
        self.assertEqual(schedule["reduction_cycles_per_tile"], 5)
        self.assertEqual(schedule["reconstruction_cycles_per_tile"], 5)
        self.assertEqual(schedule["serialized_cycles_per_tile"], 10)
        self.assertEqual(schedule["overlapped_steady_state_tile_ii_target"], 5)
        self.assertEqual(schedule["serialized_t10_cycles"], 73183500)
        self.assertEqual(schedule["overlapped_arithmetic_issue_cycles_with_ideal_phase_fill"], 36591755)
        self.assertEqual(schedule["added_hardware_target"]["independent_csd_coefficient_ops_per_cycle"], 96)
        self.assertEqual(schedule["added_hardware_target"]["worst_case_signed_shift_add_terms_per_cycle"], 384)

    def test_sensitivity_crosses_three_but_is_not_admitted(self):
        rows = {row["line"]: row for row in
            self.result["phase_schedule_sensitivity"]["rows"]}
        self.assertEqual(rows["local"]["proposal_cycles_sensitivity"], 186010489)
        self.assertEqual(rows["motion"]["proposal_cycles_sensitivity"], 183799569)
        self.assertTrue(rows["local"]["crosses_3x_sensitivity"])
        self.assertTrue(rows["motion"]["crosses_3x_sensitivity"])
        self.assertFalse(self.result["admission"]["system_cycles_admitted"])
        self.assertFalse(self.result["admission"]["speedup_admitted"])
        self.assertFalse(self.result["admission"]["headline_admitted"])

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m37.json"
            output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                M37.write_output(output, {"must_not": "overwrite"})
            self.assertEqual(output.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
