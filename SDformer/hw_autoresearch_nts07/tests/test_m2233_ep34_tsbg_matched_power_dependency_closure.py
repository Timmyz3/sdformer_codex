#!/opt/anaconda3/bin/python3
"""M2233 CPU-only complete local import closure tests. No EDA is invoked."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
HW = REPO / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_m2233_ep34_tsbg_matched_power_repair_one_shot.py"
CONTRACT = HW / "contracts/m2233_ep34_tsbg_matched_power_source_repair_contract_r1_20260905.json"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestM2233(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load(RUNNER, "m2233_test_runner")
        cls.contract = json.loads(CONTRACT.read_text())

    def helpers(self):
        return (
            ("STRUCT_HELPER", "STRUCT_HELPER_SHA", "m2172_helper_sha256"),
            ("POWER_HELPER", "POWER_HELPER_SHA", "m2117_helper_sha256"),
            ("BASE_HELPER", "BASE_HELPER_SHA", "m2160_helper_sha256"),
        )

    def test_01_all_three_helpers_in_inventory_and_exact(self):
        for path_name, sha_name, _ in self.helpers():
            path = getattr(self.runner, path_name)
            digest = getattr(self.runner, sha_name)
            self.assertEqual(self.runner.sha(path), digest)
            self.assertEqual(self.contract["source_inventory"][path.relative_to(REPO).as_posix()], digest)

    def test_02_all_three_mutations_rejected_before_contract_or_tool_checks(self):
        for _, sha_name, _ in self.helpers():
            with patch.object(self.runner, sha_name, "0" * 64), patch.object(
                    self.runner, "validate_dc_launcher", side_effect=AssertionError("TOOL_GATE_REACHED")):
                with self.assertRaisesRegex(self.runner.Failure, "helper identity"):
                    self.runner.source_validation(False)

    def test_03_runtime_local_import_closure_is_fully_inventory_bound(self):
        original_spec = importlib.util.spec_from_file_location
        imported = []
        def traced_spec(name, path, *args, **kwargs):
            imported.append(Path(path).resolve())
            return original_spec(name, path, *args, **kwargs)
        with patch.object(importlib.util, "spec_from_file_location", traced_spec):
            load(self.runner.PARSER, "m2233_test_parser_closure")
        expected = {self.runner.PARSER, self.runner.STRUCT_HELPER,
                    self.runner.BASE_HELPER, self.runner.POWER_HELPER}
        self.assertEqual(set(imported), expected)
        for path in imported:
            self.assertEqual(self.contract["source_inventory"][path.relative_to(REPO).as_posix()], self.runner.sha(path))

    def review_fixture(self):
        return {
            "status": "PASS_M2234_M2233_MATCHED_POWER_SOURCE_REPAIR_RELEASE",
            "score_over_100": 95,
            "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
            "authorization": self.contract["execution_budget"],
            "identity": {
                "runner_sha256": self.runner.sha(RUNNER),
                "contract_sha256": self.runner.sha(CONTRACT),
                **{key: getattr(self.runner, sha_name) for _, sha_name, key in self.helpers()},
            },
        }

    def reviewed_gate(self, review):
        strict_original = self.runner.strict_json
        seal_original = self.runner.verify_seal
        def strict(path):
            return review if path == self.runner.REVIEW / "review.json" else strict_original(path)
        def seal(path):
            return {} if path == self.runner.REVIEW else seal_original(path)
        with patch.object(self.runner, "strict_json", strict), patch.object(self.runner, "verify_seal", seal):
            return self.runner.source_validation(True)

    def test_04_complete_source_review_binding_passes(self):
        self.assertEqual(len(self.reviewed_gate(self.review_fixture())["source_inventory"]), 29)

    def test_05_every_review_helper_mutation_is_rejected(self):
        for _, _, key in self.helpers():
            review = self.review_fixture()
            review["identity"][key] = "0" * 64
            with self.assertRaisesRegex(self.runner.Failure, "helper binding"):
                self.reviewed_gate(review)

    def test_06_review_score_94_is_rejected(self):
        review = self.review_fixture()
        review["score_over_100"] = 94
        with self.assertRaisesRegex(self.runner.Failure, "M2234 release"):
            self.reviewed_gate(review)

    def test_07_result_emits_complete_helper_and_corner_identity(self):
        source = RUNNER.read_text()
        for _, _, key in self.helpers():
            self.assertIn('"' + key + '": sha(', source)
        self.assertIn('result["implementation_corners"]', source)
        self.assertIn('"dc_to_ptpx_is_mixed_corner": True', source)
        self.assertIn("FIXED_THREE_WINDOW_WEIGHTED_INDEX__NOT_POPULATION_MEAN", source)
        self.assertIn('"aggregate_is_2880_workload_population_mean": False', source)
        self.assertIn('"aggregate_is_frame_energy": False', source)

    def test_08_namespace_is_fresh_and_budget_unchanged(self):
        for path in (self.runner.RESULT, self.runner.ATTEMPT, self.runner.LOCK):
            self.assertIn("m2235_m2233", str(path))
        source = RUNNER.read_text()
        for old in ("results/m2219", ".m2219_", "results/m2227", ".m2227_"):
            self.assertNotIn(old, source)
        self.assertEqual(self.contract["execution_budget"], {
            "license_queries": 1, "vcs_compiles": 2, "simv_runs": 6,
            "diagnostic_saif_files": 6, "measurement_saif_files": 6,
            "dc_runs": 2, "ptpx_runs": 6, "automatic_retry": False,
            "p1_serial": True, "reuse_m2203_raw": False,
        })

    def test_09_original_m2225_sources_and_docs359_unchanged(self):
        old = json.loads((HW / "contracts/m2225_ep34_tsbg_matched_power_source_repair_contract_r1_20260904.json").read_text())
        for rel, digest in old["source_inventory"].items():
            self.assertEqual(self.runner.sha(REPO / rel), digest)
        self.assertEqual(self.runner.sha(self.runner.DOC359), self.runner.DOC_SHA)

    def test_10_model_and_corners_unchanged(self):
        model = self.contract["external_weight_sram_model"]
        self.assertEqual(model["dynamic_energy_pj_per_actual_accepted_bank_activation"], 22.213)
        self.assertAlmostEqual(model["leakage_power_mw_each_axis"], 3.826774326764422)
        mapping = self.contract["mapping_and_power"]
        self.assertEqual((mapping["dc_max_corner"], mapping["dc_min_corner"], mapping["ptpx_corner"]),
                         ("SSG0P9V125C", "FFG1P05VM40C", "TT0P9V25C"))
        self.assertTrue(mapping["dc_to_ptpx_is_mixed_corner"])


if __name__ == "__main__":
    unittest.main()
