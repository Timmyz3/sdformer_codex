#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fixture-only tests for M1401; never reads or writes the canonical result."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/hammer_m1401_m1349_motion_ep34_live105_capture_result_source.py"
SPEC = importlib.util.spec_from_file_location("m1401_result_hammer", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def one_sequence():
    rows = []
    for category, count in M.EXPECTED_COUNTS.items():
        if category == "atlif":
            names = list(M.M1349.EXPECTED_ATLIF_NAMES)
        else:
            names = [f"{category}.unit_{index:03d}" for index in range(count)]
        for name in names:
            rows.append({"sample_id": 0, "category": category, "name": name,
                         "input": {}, "payload": {}})
    assert len(rows) == 259
    return rows


def ordered_fixture():
    sequence = one_sequence()
    return [{**row, "sample_id": sample} for sample in range(40) for row in sequence]


def admission_fixture():
    return {
        "schema": "m1343_final_capture_admission_r1_v1", "status": "PASS",
        "ordered": 10360, "attention": 480, "payload_files": 640,
        "execution": 7360, "operator_rows": 79, "atlif_live_rows": 105,
        "atlif_static": 105, "dead_sn_v": [],
        "atlif_names_sha256": M.ATLIF_NAMES_SHA256,
        "claim_boundary": {"capture_only": True, "paper_result": False,
                           "cycles": False, "speedup": False,
                           "energy": False, "ppa": False},
    }


def manifest_fixture():
    return {
        "schema": "m1343_motion_ep34_live105_unified_hardware_capture_r1_v1",
        "status": "CAPTURE_COMPLETE__FRESH_M1343_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
        "identity": {
            "checkpoint_load_audit": {"missing_count": 0, "unexpected_count": 0},
            "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "selection": {"selected": {"candidate_id": "resume_ep34", "epoch": 34,
                "checkpoint": {"sha256": M.CHECKPOINT_SHA256},
                "configuration": {"sha256": M.CONFIG_SHA256},
                "profile": {"sha256": M.PROFILE_SHA256, "samples": 825,
                    "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12}}}},
        },
        "m1227_runtime_contract": {"final_selection_identity": {
            "epoch": 34, "checkpoint_sha256": M.CHECKPOINT_SHA256,
            "config_sha256": M.CONFIG_SHA256, "profile_sha256": M.PROFILE_SHA256,
            "selection_sha256": M.SELECTION_SHA256}},
        "m1343_runtime_contract": {
            "static_modules": 259, "static_atlif": 105,
            "live_modules_per_sample": 259, "live_atlif": 105,
            "dead_sn_v": [], "dead_calls_per_sample": 0,
            "atlif_names_sha256": M.ATLIF_NAMES_SHA256,
            "ordered_records": 10360, "attention_records": 480,
            "payload_files": 640},
        "claim_boundary": {"capture_only": True, "accuracy": False,
            "cycles": False, "speedup": False, "system_speedup": False,
            "energy": False, "rtl": False, "ppa": False,
            "fresh_result_hammer_required": True},
    }


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ordered = ordered_fixture()

    def test_01_valid_ordered_population(self):
        audit = M.validate_ordered(self.ordered)
        self.assertEqual(audit["ordered_rows"], 10360)
        self.assertTrue(audit["all_sample_sequences_equal"])

    def test_02_missing_or_extra_row_rejected(self):
        for mutant in (self.ordered[:-1], self.ordered + [copy.deepcopy(self.ordered[-1])]):
            with self.assertRaises(M.M1401Error): M.validate_ordered(mutant)

    def test_03_sample_id_type_and_value_rejected(self):
        for value in (True, "0", 1):
            mutant = copy.deepcopy(self.ordered); mutant[0]["sample_id"] = value
            with self.assertRaises(M.M1401Error): M.validate_ordered(mutant)

    def test_04_category_count_and_unknown_rejected(self):
        for value in ("unknown", "fc1"):
            mutant = copy.deepcopy(self.ordered); mutant[0]["category"] = value
            with self.assertRaises(M.M1401Error): M.validate_ordered(mutant)

    def test_05_cross_sample_sequence_drift_rejected(self):
        mutant = copy.deepcopy(self.ordered)
        mutant[259 + 1]["name"] += ".drift"
        with self.assertRaises(M.M1401Error): M.validate_ordered(mutant)

    def test_06_atlif_reorder_duplicate_rename_rejected(self):
        start = sum(count for category, count in M.EXPECTED_COUNTS.items()
                    if category != "atlif" and list(M.EXPECTED_COUNTS).index(category) <
                    list(M.EXPECTED_COUNTS).index("atlif"))
        variants = []
        swap = copy.deepcopy(self.ordered)
        swap[start], swap[start + 1] = swap[start + 1], swap[start]
        variants.append(swap)
        duplicate = copy.deepcopy(self.ordered)
        duplicate[start + 1]["name"] = duplicate[start]["name"]
        variants.append(duplicate)
        rename = copy.deepcopy(self.ordered)
        rename[start]["name"] += ".renamed"
        variants.append(rename)
        for mutant in variants:
            with self.assertRaises(M.M1401Error): M.validate_ordered(mutant)

    def test_07_missing_input_or_payload_rejected(self):
        for key in ("input", "payload"):
            mutant = copy.deepcopy(self.ordered); mutant[0].pop(key)
            with self.assertRaises(M.M1401Error): M.validate_ordered(mutant)

    def test_08_admission_exact(self):
        M.validate_admission(admission_fixture())

    def test_09_admission_mutations_rejected(self):
        for key, value in (("ordered", 9880), ("atlif_live_rows", 93),
                           ("dead_sn_v", ["x"]), ("payload_files", 639)):
            mutant = admission_fixture(); mutant[key] = value
            with self.assertRaises(M.M1401Error): M.validate_admission(mutant)
        mutant = admission_fixture(); mutant["extra"] = 1
        with self.assertRaises(M.M1401Error): M.validate_admission(mutant)

    def test_10_manifest_exact_identity(self):
        M.validate_manifest(manifest_fixture())

    def test_11_manifest_identity_mutations_rejected(self):
        paths = [
            ("identity.selection.selected.epoch", 33),
            ("identity.selection.selected.checkpoint.sha256", "0" * 64),
            ("identity.selection.selected.profile.samples", 824),
            ("m1227_runtime_contract.final_selection_identity.selection_sha256", "0" * 64),
            ("m1343_runtime_contract.live_modules_per_sample", 247),
            ("m1343_runtime_contract.atlif_names_sha256", "0" * 64),
            ("claim_boundary.cycles", True),
        ]
        for dotted, value in paths:
            mutant = manifest_fixture(); cursor = mutant
            parts = dotted.split(".")
            for part in parts[:-1]: cursor = cursor[part]
            cursor[parts[-1]] = value
            with self.assertRaises(M.M1401Error): M.validate_manifest(mutant)

    def test_12_source_authorities_and_namespace_absence(self):
        M.verify_authority_dir(M.M1349_AUTHOR, M.M1349_AUTHOR_SEAL,
            "PASS_SOURCE_AUTHOR__DIFFERENT_AUTHOR_BLIND_REQUIRED")
        M.verify_authority_dir(M.M1353_BLIND, M.M1353_BLIND_SEAL,
            "PASS_SOURCE__FRESH_RELEASE_AUTHOR_MAY_BE_AUTHORED")
        M.canonical_absent()


if __name__ == "__main__":
    unittest.main(verbosity=2)
