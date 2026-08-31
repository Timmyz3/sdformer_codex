#!/usr/bin/env python3
"""Source-only negative tests for the M798 decoder production repair."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "scripts/execute_m798_m785_decoder_physical_residency_production.py"
RUNNER = HERE.parent / "scripts/run_m798_m785_decoder_physical_residency_one_shot.sh"


def load_driver():
    spec = importlib.util.spec_from_file_location("m798_test_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M798 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M798 = load_driver()


class M798RepairTests(unittest.TestCase):
    def test_synthetic_self_test(self):
        result = M798.self_test()
        self.assertEqual(
            result["status"],
            "PASS_M798_REPAIRED_DRIVER_SYNTHETIC_SELF_TEST")
        self.assertTrue(result["duplicate_json_rejected"])
        self.assertTrue(result["d1_headline_perturbation_invariant"])
        self.assertTrue(result["atomic_destination_race_rejected"])
        self.assertIsNone(result["production_cycles"])

    def test_duplicate_authorization_key_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m798_dup_") as directory:
            path = Path(directory) / "release.json"
            path.write_text(
                '{"launch_now":true,"launch_now":false,"release":true}\n',
                encoding="utf-8")
            with self.assertRaises(M798.Failure):
                M798.strict_json(path)

    def test_duplicate_canonical_path_key_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m798_dup_path_") as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(
                '{"canonical":{"result":"a","result":"b"}}\n',
                encoding="utf-8")
            with self.assertRaises(M798.Failure):
                M798.strict_json(path)

    def test_d1_does_not_change_headline_ratio(self):
        baseline = {
            M798.HEADLINE_DENOMINATOR: {
                "headline_total_cycles": 900,
                "total_cycles": 1000,
            },
            M798.HEADLINE_NUMERATOR: {
                "headline_total_cycles": 600,
                "total_cycles": 800,
            },
        }
        expected = M798.headline_ratio(baseline)
        baseline[M798.HEADLINE_DENOMINATOR]["total_cycles"] += 99999999
        baseline[M798.HEADLINE_NUMERATOR]["total_cycles"] += 1
        self.assertEqual(M798.headline_ratio(baseline), expected)
        self.assertEqual(expected, 1.5)

    def test_atomic_destination_race_is_no_replace(self):
        with tempfile.TemporaryDirectory(prefix="m798_race_") as directory:
            parent = Path(directory)
            stage = parent / "result.stage.1"
            result = parent / "result"
            stage.mkdir()
            (stage / "sentinel").write_text("stage\n", encoding="utf-8")
            result.mkdir()
            (result / "attacker").write_text("collision\n", encoding="utf-8")
            with self.assertRaises(M798.Failure):
                M798._rename_noreplace(stage, result)
            self.assertEqual((stage / "sentinel").read_text(), "stage\n")
            self.assertEqual((result / "attacker").read_text(), "collision\n")
            self.assertFalse((result / stage.name).exists())

    def test_runner_uses_explicit_atomic_publication_and_root_postcheck(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--publish-no-replace", text)
        self.assertIn("renameat2(RENAME_NOREPLACE)", text)
        self.assertIn('"${m798_result}/result.json"', text)
        self.assertIn('"${m798_result}/detailed_rows.json"', text)
        self.assertIn('"${m798_result}/SHA256SUMS"', text)
        self.assertIn('"${m798_result}/SHA256SUMS.seal.sha256"', text)
        self.assertNotIn('mv -- "${m798_stage}" "${m798_result}"', text)


if __name__ == "__main__":
    unittest.main()
