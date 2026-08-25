#!/usr/bin/env python3
"""Regression and adversarial tests for M39-r2 fail-closed evidence."""

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
SCRIPT = HW_ROOT / "system_simulator/scripts/analyze_m39_remaining_bottleneck_r2.py"
CONTRACT = HW_ROOT / "contracts/m39_remaining_bottleneck_input_contract_r2_20260822.json"

SPEC = importlib.util.spec_from_file_location("m39_r2", str(SCRIPT))
M39 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M39)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class M39R2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = M39.build(CONTRACT)
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def write_contract(self, root, payload):
        path = root / "contract.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def mutate_receipt(self, root, key, receipt_mutator):
        contract = copy.deepcopy(self.contract)
        original_receipt = Path(M39.resolve(contract["inputs"][key]["path"]))
        receipt = json.loads(original_receipt.read_text(encoding="utf-8"))
        receipt_mutator(receipt)
        receipt_path = root / (key + ".json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        contract["inputs"][key] = {"path": str(receipt_path), "sha256": digest(receipt_path)}
        return self.write_contract(root, contract)

    def forge_vcs_log(self, root, receipt, vcs_key, forged_text):
        vcs = receipt[vcs_key]
        original = Path(vcs["directory"])
        forged = root / (vcs_key + "_forged")
        forged.mkdir()
        for name in ("input_sha256.txt", "compile.log", "vectors.txt"):
            shutil.copy2(str(original / name), str(forged / name))
        (forged / "sim.log").write_text(forged_text, encoding="utf-8")
        output = forged / "output_sha256.txt"
        output.write_text(
            "{}  {}\n{}  {}\n{}  {}\n".format(
                digest(forged / "compile.log"), forged / "compile.log",
                digest(forged / "sim.log"), forged / "sim.log",
                digest(forged / "vectors.txt"), forged / "vectors.txt"),
            encoding="utf-8")
        vcs["directory"] = str(forged)
        vcs["input_ledger_sha256"] = digest(forged / "input_sha256.txt")
        vcs["output_ledger_sha256"] = digest(output)
        vcs["compile_log_sha256"] = digest(forged / "compile.log")
        vcs["sim_log_sha256"] = digest(forged / "sim.log")
        vcs["vector_sha256"] = digest(forged / "vectors.txt")

    def test_base_status_is_blocked_not_pass(self):
        self.assertEqual(
            self.result["status"],
            "BLOCKED_BY_STALE_M38_R2_EXPLORATORY_ONLY_REANCHOR_REQUIRED_AFTER_M38_R3")
        admission = self.result["admission"]
        self.assertFalse(admission["m38_r2_current_rebuild_admitted"])
        self.assertFalse(admission["m38_r2_recursive_anchor_admitted"])
        self.assertFalse(admission["conditional_h67_compute_dse_admitted"])
        self.assertFalse(admission["system_speedup_admitted"])
        self.assertFalse(admission["headline_admitted"])

    def test_recursive_synopsys_evidence_counts(self):
        audit = self.result["recursive_evidence_audit"]
        self.assertEqual(audit["m33_flat_r2"]["formality_compare_points"], [655, 0, 0])
        self.assertEqual(audit["m33_flat_r2"]["formality_snapshot_files"], 21)
        self.assertEqual(audit["m35_r7"]["formality_compare_points"], [2333, 0, 0])
        self.assertEqual(audit["m35_r7"]["formality_snapshot_files"], 20)
        self.assertIn("LIVE_WRAPPER_DRIFT_IGNORED",
                      audit["m35_r7"]["formality_authority"])
        self.assertEqual(audit["m38_r2"]["current_rebuild_error"],
                         "receipt live source drift for unified_core_rtl")

    def test_dse_numbers_and_local5_fail_close(self):
        self.assertEqual(self.result["remaining_cycle_ledger"]["fixed_compute_cycles"],
                         620868243)
        self.assertEqual(self.result["attention_and_trace_completeness"]["local5_ep44"],
                         "MISSING_UNKNOWN_NONZERO_AT_LEAST_120_CALLS")
        dse = self.result["conditional_dse"]
        self.assertEqual(dse["m38_conditional_ideal"]["conditional_t10_ii"], 5)
        self.assertFalse(dse["m38_conditional_ideal"]["system_speedup_admitted"])
        ten = {(row["line"], row["late_scale_implementation"]): row
               for row in dse["ten_consumer_rows"]}
        self.assertEqual(ten[("Local", "M33_shared96")][
            "conditional_cycles_after_substitution"], 189817484)
        self.assertEqual(ten[("Motion", "M35_zero_mul_sidecar")][
            "conditional_cycles_after_substitution"], 183799564)
        four = {(row["line"], row["late_scale_implementation"]): row
                for row in dse["four_bottleneck_rows"]}
        self.assertEqual(four[("Local", "M33_shared96")]["replacement"]["total_cycles"],
                         17071010)
        self.assertEqual(four[("Motion", "M35_zero_mul_sidecar")][
            "conditional_cycles_after_substitution"], 202666647)

    def test_2p7_thresholds_are_exact_rationals(self):
        rows = self.result["conditional_dse"]["ten_consumer_rows"]
        local = next(row for row in rows if row["line"] == "Local"
                     and row["late_scale_implementation"] == "M33_shared96")
        gate = local["target_gates"][0]
        self.assertEqual(gate["target_speedup"], {"numerator": 27, "denominator": 10})
        self.assertEqual(gate["target_cycle_ceiling"],
                         {"numerator": 2069560810, "denominator": 9})
        self.assertEqual(gate["maximum_scope_replacement_cycles"],
                         {"numerator": 606455551, "denominator": 9})
        self.assertNotIsInstance(gate["target_cycle_ceiling"], float)

    def test_forged_minimal_m33_log_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def mutate(receipt):
                self.forge_vcs_log(root, receipt, "vcs_r2",
                                   "M33_UQ_PASS packets=2048 valid_scalar_products=4608 "
                                   "digit_reconstruction_checks=8192\n")

            contract = self.mutate_receipt(root, "m33_receipt", mutate)
            with self.assertRaisesRegex(ValueError, "M33 simulator marker missing"):
                M39.build(contract)

    def test_forged_minimal_m35_log_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def mutate(receipt):
                self.forge_vcs_log(root, receipt, "vcs_r6",
                                   "M35_PASS packets=5120 valid_products=23680 "
                                   "consecutive_full_rate=630\n")

            contract = self.mutate_receipt(root, "m35_receipt", mutate)
            with self.assertRaisesRegex(ValueError, "M35 simulator marker missing"):
                M39.build(contract)

    def test_contract_top_level_population_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mutated = copy.deepcopy(self.contract)
            mutated["forged"] = True
            with self.assertRaisesRegex(ValueError, "contract population drift"):
                M39.build(self.write_contract(root, mutated))

    def test_contract_identity_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mutated = copy.deepcopy(self.contract)
            mutated["identity"] += "_forged"
            with self.assertRaisesRegex(ValueError, "identity drift"):
                M39.build(self.write_contract(root, mutated))

    def test_contract_claim_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mutated = copy.deepcopy(self.contract)
            mutated["claim_boundary"] += " FORGED"
            with self.assertRaisesRegex(ValueError, "claim boundary drift"):
                M39.build(self.write_contract(root, mutated))

    def test_prosperity_boundary_is_fail_closed(self):
        assessment = self.result["prosperity_phi_adapter_assessment"]["Prosperity"]
        self.assertEqual(assessment["real_domain_evidence_authority"],
                         "M32_ONLY_EXACT_REAL_DOMAIN_NUMBERS")
        self.assertEqual(assessment["fixed_point_and_accuracy"], "UNADMITTED")
        self.assertEqual(assessment["official_repository_commit"],
                         "6ee1c6f1cb419fcf942f2eda63db84ca28248f4b")
        self.assertEqual(assessment["repository_file_sha256"],
                         "NOT_MEASURED_DO_NOT_INFER")


if __name__ == "__main__":
    unittest.main()
