from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "check_m1313_motion_ep34_final_unified_capture_production_launch.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


M = load("m1313_static_checker_under_test", CHECKER)


class M1313ProductionLaunchAuthorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = M.strict_json(CONTRACT)

    def validate(self, value=None, exists=lambda _path: False, sample_bytes=False):
        return M.validate_contract(
            self.contract if value is None else value,
            check_sample_bytes=sample_bytes,
            namespace_exists=exists,
        )

    def rejected(self, mutation) -> None:
        value = copy.deepcopy(self.contract)
        mutation(value)
        with self.assertRaises(M.AuditError):
            self.validate(value)

    def test_01_exact_author_contract_passes_and_does_not_execute_capture(self):
        result = self.validate()
        self.assertEqual((result["selected_candidate_id"], result["selected_epoch"]),
                         ("resume_ep34", 34))
        self.assertEqual(result["samples"], 40)
        self.assertFalse(result["automatic_retry"])
        self.assertFalse(result["remote_gpu_capture_executed"])

    def test_02_all_forty_sample_bytes_match_the_frozen_cohort(self):
        result = self.validate(sample_bytes=True)
        self.assertEqual(result["cohort_compact_sha256"], M.COHORT_SHA256)

    def test_03_m1249_release_identity_and_files_are_exact(self):
        self.assertEqual(self.contract["release_identity"], M.M1249_RELEASE)
        for key, value in M.M1249_RELEASE.items():
            if key.endswith("_path"):
                self.assertEqual(sha(ROOT / value),
                                 M.M1249_RELEASE[key.replace("_path", "_sha256")])
        for key in M.M1249_RELEASE:
            if key.endswith("_sha256"):
                self.rejected(lambda row, key=key: row["release_identity"].__setitem__(
                    key, "0" * 64))

    def test_04_each_m1243_identity_is_exact_and_mutation_is_rejected(self):
        for name, expected in M.M1243.items():
            with self.subTest(name=name):
                self.assertEqual(self.contract["inputs"][name], expected)
                self.rejected(lambda row, name=name: row["inputs"][name].__setitem__(
                    "sha256", "0" * 64))

    def test_05_canonical_result_path_is_required_and_staged_path_is_rejected(self):
        self.assertEqual(self.contract["inputs"]["final_selection_result"],
                         M.FINAL_SELECTION)
        self.rejected(lambda row: row["inputs"]["final_selection_result"].__setitem__(
            "result_path",
            "hw_autoresearch_nts07/system_handoff/incoming/m1306_remote_selection_result_20260830/"
            "hw_autoresearch_nts07/results/"
            "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830"))

    def test_06_m1312_entry_and_all_seal_fields_are_exact(self):
        self.assertEqual(self.contract["inputs"]["final_selection_result_hammer"], M.M1312)
        for key in ("manifest_sha256", "outer_file_sha256", "review_sha256"):
            with self.subTest(key=key):
                self.rejected(lambda row, key=key: row["inputs"][
                    "final_selection_result_hammer"].__setitem__(key, "0" * 64))

    def test_07_selected_member_manifest_outer_and_tuple_are_bound(self):
        for key in ("manifest_sha256", "outer_file_sha256", "selection_sha256"):
            with self.subTest(key=key):
                self.rejected(lambda row, key=key: row["inputs"][
                    "final_selection_result"].__setitem__(key, "0" * 64))
        self.assertEqual(M.EXPECTED_AUTHORITY["selected_candidate_id"], "resume_ep34")
        self.assertEqual(M.EXPECTED_AUTHORITY["selected_epoch"], 34)

    def test_08_cohort_is_exact_m1182_m1210_population_and_order(self):
        samples = self.contract["cohort"]["samples"]
        self.assertEqual(len(samples), 40)
        self.assertEqual(M.compact_sha(samples), M.COHORT_SHA256)
        self.assertEqual(samples, M.strict_json(M.M1182)["cohort"]["samples"])
        self.assertEqual(samples, M.strict_json(M.M1210)["cohort"]["samples"])

    def test_09_cohort_sha_order_and_population_attacks_fail_closed(self):
        self.rejected(lambda row: row["cohort"]["samples"][0].__setitem__(
            "sha256", "0" * 64))
        self.rejected(lambda row: row["cohort"]["samples"].__setitem__(
            slice(0, 2), list(reversed(row["cohort"]["samples"][0:2]))))
        self.rejected(lambda row: row["cohort"]["samples"].pop())

    def test_10_retry_and_each_namespace_attack_fail_closed(self):
        self.rejected(lambda row: row["one_shot"].__setitem__("automatic_retry", True))
        self.rejected(lambda row: row["one_shot"].__setitem__("attempt_marker", "wrong"))
        self.rejected(lambda row: row["output"].__setitem__("path", "wrong"))
        self.rejected(lambda row: row["production_log"].__setitem__("path", "wrong"))

    def test_11_each_occupied_namespace_fails_closed(self):
        for target in (M.RESULT, M.ATTEMPT, M.LOG):
            with self.subTest(target=target):
                with self.assertRaises(M.AuditError):
                    self.validate(exists=lambda path, target=str(target): path == target)

    def test_12_top_level_and_nested_extra_keys_fail_closed(self):
        self.rejected(lambda row: row.__setitem__("claim_boundary", {}))
        self.rejected(lambda row: row["inputs"].__setitem__("extra", True))
        self.rejected(lambda row: row["release_identity"].__setitem__("extra", True))

    def test_13_pinned_predecessor_seals_and_docs359_are_unchanged(self):
        for path, expected in M.PINNED_SEALS.items():
            with self.subTest(path=path):
                self.assertEqual(sha(ROOT / path), expected)

    def test_14_checker_is_read_only_and_has_no_execution_stack(self):
        text = CHECKER.read_text(encoding="utf-8")
        for forbidden in ("import subprocess", "import torch", "paramiko", "ssh ",
                          "dc_shell", "vcs -full64", "execute_once(", "run_capture("):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("write_text", text)
        self.assertNotIn("write_bytes", text)

    def test_15_contract_is_only_authoring_not_evidence_or_metric(self):
        self.assertEqual(set(self.contract), M.TOP_KEYS)
        self.assertNotIn("hardware_speedup", json.dumps(self.contract))
        self.assertNotIn("system_speedup", json.dumps(self.contract))
        self.assertNotIn("hardware_energy", json.dumps(self.contract))


if __name__ == "__main__":
    unittest.main(verbosity=2)
