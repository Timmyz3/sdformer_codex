#!/usr/bin/env python3

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_m32_threshold_carry_late_scale_r3.py"
)
SPEC = importlib.util.spec_from_file_location("m32_r3", str(SCRIPT))
M32 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M32)


class M32ThresholdCarryLateScaleR3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = M32.build_report(M32.DEFAULT_CONTRACT)
        _contract, cls.paths, cls.hashes = M32.load_contract(
            M32.DEFAULT_CONTRACT
        )
        cls.r2 = json.loads(cls.paths["r2_report"].read_text(encoding="utf-8"))
        cls.manifest = json.loads(
            cls.paths["dataflow_manifest"].read_text(encoding="utf-8")
        )
        cls.rows = M32.read_jsonl(cls.paths["dataflow_rows"])
        cls.profile = json.loads(
            cls.paths["postrun_profile"].read_text(encoding="utf-8")
        )
        with cls.paths["sample_workload"].open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            cls.workload = list(csv.DictReader(handle))

    def test_scoped_semantic_admission_only(self):
        self.assertTrue(self.report["semantic_admission"])
        self.assertFalse(self.report["headline_admitted"])
        admission = self.report["admission"]
        self.assertTrue(admission["semantic_admission"])
        for field in (
            "semantic_generalization_admitted", "fixed_point_admitted",
            "rtl_admitted", "system_cycle_admitted", "performance_admitted",
            "ppa_admitted", "power_energy_admitted", "headline_admitted",
        ):
            self.assertFalse(admission[field])
        candidates = self.report["candidate_census"]["candidates"]
        self.assertEqual(len(candidates), 10)
        self.assertTrue(all(row["semantic_admission"] for row in candidates))

    def test_dynamic_ten_by_ten_population(self):
        audit = self.report["dynamic_dataflow_audit"]
        self.assertEqual(audit["samples"], 10)
        self.assertEqual(audit["candidate_pairs"], 10)
        self.assertEqual(audit["records"], 100)
        self.assertEqual(
            audit["ordered_sample_identity_sha256"],
            "70dd1b5bff849b411800a9cf8d25fe75fc66a51e4ca9026943e03f7f4b4cc274",
        )
        self.assertTrue(audit["performance_use_forbidden"])

    def test_identity_boolean_drift_fails_closed(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["same_tensor_object"] = False
        with self.assertRaisesRegex(ValueError, "row admission drift"):
            M32.audit_dynamic_identity(
                self.r2, self.manifest, rows, self.profile,
                self.workload, self.hashes,
            )

    def test_manifest_population_drift_fails_closed(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["records"] = 99
        with self.assertRaisesRegex(ValueError, "manifest admission drift"):
            M32.audit_dynamic_identity(
                self.r2, manifest, self.rows, self.profile,
                self.workload, self.hashes,
            )

    def test_contract_hash_drift_fails_closed(self):
        contract = json.loads(
            M32.DEFAULT_CONTRACT.read_text(encoding="utf-8")
        )
        contract["inputs"]["runtime_profile_source"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                M32.build_report(path)


if __name__ == "__main__":
    unittest.main()
