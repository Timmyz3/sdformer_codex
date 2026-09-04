#!/opt/anaconda3/bin/python3
"""CPU-only tests for the additive M2225 source repair; no EDA is run."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_m2225_ep34_tsbg_matched_power_repair_one_shot.py"
CONTRACT = HW / "contracts/m2225_ep34_tsbg_matched_power_source_repair_contract_r1_20260904.json"
STRUCT = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"
POWER = HW / "system_simulator/scripts/parse_m2117_m2018_tsbg_rtl_saifmap_power.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestM2225(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load(RUNNER, "m2225_runner_test")
        cls.source = RUNNER.read_text()
        cls.contract = json.loads(CONTRACT.read_text())

    def test_01_helper_exact_identity_is_preflighted(self):
        self.assertEqual(self.runner.sha(STRUCT), self.runner.STRUCT_HELPER_SHA)
        self.assertEqual(self.runner.sha(POWER), self.runner.POWER_HELPER_SHA)
        self.assertIn('need(sha(STRUCT_HELPER) == STRUCT_HELPER_SHA', self.source)
        self.assertIn('need(sha(POWER_HELPER) == POWER_HELPER_SHA', self.source)

    def test_02_helper_mutations_fail_before_contract_or_eda(self):
        for field in ("STRUCT_HELPER_SHA", "POWER_HELPER_SHA"):
            original = getattr(self.runner, field)
            try:
                setattr(self.runner, field, "0" * 64)
                with self.assertRaisesRegex(self.runner.Failure, "helper identity"):
                    self.runner.source_validation(require_review=False)
            finally:
                setattr(self.runner, field, original)

    def test_03_helpers_are_contract_inventory_members(self):
        inventory = self.contract["source_inventory"]
        for path, digest in (
            (STRUCT, self.runner.STRUCT_HELPER_SHA),
            (POWER, self.runner.POWER_HELPER_SHA),
        ):
            rel = path.relative_to(ROOT).as_posix()
            self.assertEqual(inventory[rel], digest)

    def test_04_actual_dc_and_ptpx_corners_are_separate(self):
        mapping = self.contract["mapping_and_power"]
        self.assertEqual(mapping["dc_max_corner"], "SSG0P9V125C")
        self.assertEqual(mapping["dc_min_corner"], "FFG1P05VM40C")
        self.assertEqual(mapping["ptpx_corner"], "TT0P9V25C")
        self.assertTrue(mapping["dc_to_ptpx_is_mixed_corner"])
        self.assertIn("ssg0p9v125c.db", self.source)
        self.assertIn("ffg1p05vm40c.db", self.source)
        self.assertIn("tt0p9v25c.db", self.source)
        self.assertIn('"M2217_OPERATING_CONDITION": "ssg0p9v125c"', self.source)

    def test_05_sram_numbers_and_labels_are_unchanged(self):
        model = self.contract["external_weight_sram_model"]
        self.assertEqual(model["dynamic_energy_pj_per_actual_accepted_bank_activation"], 22.213)
        self.assertAlmostEqual(model["leakage_power_mw_each_axis"], 3.826774326764422)
        self.assertTrue(model["mixed_corner_model_must_be_labeled"])

    def test_06_fresh_review_attempt_and_result_identities(self):
        self.assertIn("m2226_m2225", self.runner.REVIEW.as_posix())
        self.assertIn("m2227_m2225", self.runner.RESULT.as_posix())
        self.assertIn(".m2227_m2225", self.runner.ATTEMPT.as_posix())
        self.assertIn('review["identity"]["m2172_helper_sha256"]', self.source)
        self.assertIn('review["identity"]["m2117_helper_sha256"]', self.source)
        self.assertIn("PASS_RAW_M2227_PENDING_M2228", self.source)
        self.assertNotIn("results/m2219", self.source)
        self.assertNotIn(".m2219_", self.source)

    def test_07_old_attempt_is_neither_input_nor_output(self):
        inventory = self.contract["source_inventory"]
        self.assertFalse(any("m2219" in key for key in inventory))
        self.assertFalse(self.contract["execution_authority"]["reuse_or_consume_m2219"])

    def test_08_execution_budget_is_unchanged_and_no_retry(self):
        self.assertEqual(self.contract["execution_budget"], {
            "license_queries": 1, "vcs_compiles": 2, "simv_runs": 6,
            "diagnostic_saif_files": 6, "measurement_saif_files": 6,
            "dc_runs": 2, "ptpx_runs": 6, "automatic_retry": False,
            "p1_serial": True, "reuse_m2203_raw": False,
        })


if __name__ == "__main__":
    unittest.main()
