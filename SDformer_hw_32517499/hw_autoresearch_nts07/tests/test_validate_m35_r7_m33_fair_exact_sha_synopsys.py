from __future__ import print_function

import copy
import importlib.util
import pathlib
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[2]
VALIDATOR_PATH = (
    REPO / "hw_autoresearch_nts07/dc_handoff/scripts/"
    "validate_m35_r6_m33_fair_exact_sha_synopsys.py"
)
CONTRACT_PATH = (
    REPO / "hw_autoresearch_nts07/contracts/"
    "m35_r7_m33_fair_exact_sha_synopsys_contract_r1_20260823.json"
)
MANIFEST_PATH = (
    REPO / "hw_autoresearch_nts07/contracts/"
    "m35_r7_m33_fair_exact_sha_launch_manifest_r1_20260823.json"
)
SPEC = importlib.util.spec_from_file_location("m35_r7_producer_validator", str(VALIDATOR_PATH))
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class M35R7ProducerValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = VALIDATOR.read_json(CONTRACT_PATH)
        manifest = VALIDATOR.read_json(MANIFEST_PATH)
        cls.entries = dict((entry["snapshot"], entry["sha256"])
                           for entry in manifest["entries"])

    def write(self, root, name, text):
        path = pathlib.Path(root) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_frozen_contract_all_sha_crosscheck_passes(self):
        result = VALIDATOR.validate_contract_against_launch(
            copy.deepcopy(self.contract), dict(self.entries))
        self.assertTrue(result["all_contract_sha_values_launch_backed"])
        self.assertEqual(result["receipt_builder_sha256"],
                         VALIDATOR.EXPECTED_FIXED_BUILDER)
        self.assertEqual(result["declared_snapshot_member_count"], 22)

    def test_old_failed_r5_builder_pin_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["common_synopsys_flow"]["receipt_builder_sha256"] = (
            "1e2ecc527d1255314cf05e4002a1e77c6870d2a9c29be107d2fccefed4d278d9")
        with self.assertRaises(ValueError):
            VALIDATOR.validate_contract_against_launch(mutated, dict(self.entries))

    def test_validator_pin_drift_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["common_synopsys_flow"]["independent_run_validator_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            VALIDATOR.validate_contract_against_launch(mutated, dict(self.entries))

    def test_runner_pin_drift_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["common_synopsys_flow"]["snapshot_runner_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            VALIDATOR.validate_contract_against_launch(mutated, dict(self.entries))

    def test_rtl_pin_drift_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["m35_admission"]["candidate_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            VALIDATOR.validate_contract_against_launch(mutated, dict(self.entries))

    def test_constraint_pin_drift_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["common_synopsys_flow"]["common_sdc_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            VALIDATOR.validate_contract_against_launch(mutated, dict(self.entries))

    def test_launch_member_omission_rejected(self):
        mutated_entries = dict(self.entries)
        del mutated_entries[VALIDATOR.PATHS["formality_tcl"]]
        with self.assertRaises(ValueError):
            VALIDATOR.validate_contract_against_launch(
                copy.deepcopy(self.contract), mutated_entries)

    def test_contract_exact_snapshot_map_omission_rejected(self):
        mutated = copy.deepcopy(self.contract)
        del mutated["exact_snapshot_sha256"][VALIDATOR.PATHS["m33_filelist"]]
        with self.assertRaises(ValueError):
            VALIDATOR.validate_contract_against_launch(mutated, dict(self.entries))

    def test_contract_unbacked_sha_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["repair_provenance"]["injected_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            VALIDATOR.validate_contract_against_launch(mutated, dict(self.entries))

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = self.write(directory, "bad.json", '{"schema": 1, "schema": 1}\n')
            with self.assertRaises(ValueError):
                VALIDATOR.read_json(bad)

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = self.write(directory, "bad.json", '{"value": NaN}\n')
            with self.assertRaises(ValueError):
                VALIDATOR.read_json(bad)

    def test_sha_manifest_duplicate_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            row = "{}  ./same\n".format("0" * 64)
            bad = self.write(directory, "bad.sha256", row + row)
            with self.assertRaises(ValueError):
                VALIDATOR.parse_sha_manifest(bad)


if __name__ == "__main__":
    unittest.main()
