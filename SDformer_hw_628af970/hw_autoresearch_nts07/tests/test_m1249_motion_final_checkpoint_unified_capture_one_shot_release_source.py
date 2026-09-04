from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1249_motion_final_checkpoint_unified_hardware_one_shot_release_r1.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1249_motion_final_checkpoint_unified_capture_one_shot_release_source_contract_r1_20260830.json")
OLD_TEST = ROOT / (
    "hw_autoresearch_nts07/tests/"
    "test_m1233_motion_final_checkpoint_unified_capture_selection_interface_source.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


M = load("m1249_release_under_test", SOURCE)
T = load("m1249_m1233_fixture", OLD_TEST)


class Lease:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class M1249ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.base = T.M1233SelectionInterfaceTest(
            "test_03_exact_m1234_shape_passes_and_keyerror_is_regressed")
        self.base.setUp()

    def tearDown(self):
        self.base.tearDown()

    def production(self, path):
        return {
            "schema": M.PRODUCTION_SCHEMA,
            "status": M.PRODUCTION_STATUS,
            "contract_path": str(path.relative_to(M.ROOT)),
            "release_identity": {
                "source_path": str(SOURCE.relative_to(M.ROOT)),
                "source_sha256": sha(SOURCE),
                "test_path": str(Path(__file__).resolve().relative_to(M.ROOT)),
                "test_sha256": sha(Path(__file__).resolve()),
                "source_contract_path": str(CONTRACT.relative_to(M.ROOT)),
                "source_contract_sha256": sha(CONTRACT),
            },
            "inputs": {
                "m1243_source": {
                    "path": str(M.M1243_SOURCE.relative_to(M.ROOT)),
                    "sha256": M.M1243_SOURCE_SHA256,
                },
                "m1243_test": {
                    "path": str(M.M1243_TEST.relative_to(M.ROOT)),
                    "sha256": M.M1243_TEST_SHA256,
                },
                "m1243_source_contract": {
                    "path": str(M.M1243_CONTRACT.relative_to(M.ROOT)),
                    "sha256": M.M1243_CONTRACT_SHA256,
                },
                "m1244_source_hammer": copy.deepcopy(M.M1244_ENTRY),
                "final_selection_result": self.base.selection_entry,
                "final_selection_result_hammer": self.base.hammer_entry,
            },
            "cohort": {"samples": []},
            "one_shot": {
                "attempt_marker": str(M.CANONICAL_ATTEMPT.relative_to(M.ROOT)),
                "automatic_retry": False,
            },
            "output": {"path": str(M.CANONICAL_RESULT.relative_to(M.ROOT))},
            "production_log": {"path": str(M.CANONICAL_LOG.relative_to(M.ROOT))},
        }

    def validate(self, mutate=None):
        with tempfile.TemporaryDirectory(prefix=".m1249_launch_", dir=M.HW / "contracts") as name:
            path = Path(name) / "launch.json"
            contract = self.production(path)
            if mutate is not None:
                mutate(contract)
            path.write_text(json.dumps(contract) + "\n", encoding="utf-8")

            expected = {
                str(M.CANONICAL_ATTEMPT.relative_to(M.ROOT)): M.CANONICAL_ATTEMPT,
                str(M.CANONICAL_RESULT.relative_to(M.ROOT)): M.CANONICAL_RESULT,
                str(M.CANONICAL_LOG.relative_to(M.ROOT)): M.CANONICAL_LOG,
            }

            def mapped(value, missing_leaf=False):
                self.assertTrue(missing_leaf)
                return expected[value]

            with mock.patch.object(M.R1, "validate_m1224", return_value={}), \
                 mock.patch.object(M.R1, "validate_cohort", return_value=[]), \
                 mock.patch.object(M.R1, "safe_repo_path", side_effect=mapped):
                return M.validate_production_launch(contract, path)

    def assert_rejected(self, mutate):
        with self.assertRaises(M.M1249Error):
            self.validate(mutate)

    def test_01_exact_m1243_and_m1244_pins(self):
        self.assertEqual(sha(M.M1243_SOURCE), M.M1243_SOURCE_SHA256)
        self.assertEqual(sha(M.M1243_TEST), M.M1243_TEST_SHA256)
        self.assertEqual(sha(M.M1243_CONTRACT), M.M1243_CONTRACT_SHA256)
        value = M.M1243.verify_source_hammer(M.M1244_ENTRY)
        self.assertTrue(value["production_capture"])

    def test_02_import_is_lazy_for_heavy_stack(self):
        code = (
            "import importlib.util,sys;"
            "s=importlib.util.spec_from_file_location('isolated_m1249',{!r});"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "print(int('torch' in sys.modules),int('numpy' in sys.modules))"
        ).format(str(SOURCE))
        self.assertEqual(subprocess.check_output([sys.executable, "-c", code]).decode().strip(),
                         "0 0")

    def test_03_source_contract_cannot_launch(self):
        source = json.loads(CONTRACT.read_text(encoding="utf-8"))
        with self.assertRaises(M.M1249Error):
            M.validate_production_launch(source, CONTRACT)

    def test_04_valid_future_m1237_entry_passes(self):
        binding = self.validate()
        self.assertEqual(binding["identity"]["candidate_id"], "resume_ep32")
        self.assertTrue(binding["identity"]["m1244_source_hammer"]["production_capture"])

    def test_05_missing_or_extra_m1244_hammer_is_rejected(self):
        self.assert_rejected(lambda row: row["inputs"].pop("m1244_source_hammer"))
        self.assert_rejected(lambda row: row["inputs"]["m1244_source_hammer"].__setitem__(
            "extra", True))

    def test_06_each_m1244_seal_sha_mutation_is_rejected(self):
        for key in ("manifest_sha256", "outer_file_sha256", "review_sha256"):
            with self.subTest(key=key):
                self.assert_rejected(lambda row, key=key: row["inputs"][
                    "m1244_source_hammer"].__setitem__(key, "0" * 64))

    def test_07_each_m1243_identity_mutation_is_rejected(self):
        for key in ("m1243_source", "m1243_test", "m1243_source_contract"):
            with self.subTest(key=key):
                self.assert_rejected(lambda row, key=key: row["inputs"][key].__setitem__(
                    "sha256", "0" * 64))

    def test_08_future_m1237_entry_is_exact_and_sealed(self):
        self.assert_rejected(lambda row: row["inputs"].pop("final_selection_result_hammer"))
        self.assert_rejected(lambda row: row["inputs"][
            "final_selection_result_hammer"].__setitem__("extra", True))
        for key in ("manifest_sha256", "outer_file_sha256", "review_sha256"):
            with self.subTest(key=key):
                self.assert_rejected(lambda row, key=key: row["inputs"][
                    "final_selection_result_hammer"].__setitem__(key, "0" * 64))

    def test_09_all_three_namespace_mutations_are_rejected(self):
        mutations = (
            lambda row: row["one_shot"].__setitem__("attempt_marker", "wrong"),
            lambda row: row["output"].__setitem__("path", "wrong"),
            lambda row: row["production_log"].__setitem__("path", "wrong"),
        )
        for mutation in mutations:
            self.assert_rejected(mutation)

    def test_10_namespaces_are_fresh_disjoint_and_absent(self):
        values = {M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG}
        self.assertEqual(len(values), 3)
        prior = {
            M.M1243.CANONICAL_RESULT, M.M1243.CANONICAL_ATTEMPT, M.M1243.CANONICAL_LOG,
            M.M1243.P.CANONICAL_RESULT, M.M1243.P.CANONICAL_ATTEMPT, M.M1243.P.CANONICAL_LOG,
        }
        self.assertFalse(values & prior)
        for path in values:
            self.assertFalse(path.exists())

    def test_11_capture_semantics_are_exact_aliases(self):
        for name in ("EXPECTED_STATIC_COUNTS", "EXPECTED_LIVE_COUNTS", "DEAD_SN_V",
                     "audit_call_matrix", "audit_attention_population",
                     "validate_payload_population", "atomic_sample_snapshot",
                     "final_validate_and_seal"):
            self.assertIs(getattr(M, name), getattr(M.M1243, name))
        self.assertEqual(sum(M.EXPECTED_STATIC_COUNTS.values()), 259)
        self.assertEqual(sum(M.EXPECTED_LIVE_COUNTS.values()), 247)

    def test_12_production_contract_shape_is_exact(self):
        self.assert_rejected(lambda row: row.__setitem__("extra", True))
        self.assert_rejected(lambda row: row["inputs"].__setitem__("extra", True))

    def test_13_release_source_identity_is_exact(self):
        self.assert_rejected(lambda row: row["release_identity"].__setitem__(
            "source_sha256", "0" * 64))
        self.assert_rejected(lambda row: row["release_identity"].__setitem__(
            "test_sha256", "0" * 64))
        self.assert_rejected(lambda row: row["release_identity"].__setitem__(
            "source_contract_sha256", "0" * 64))

    def test_14_attempt_is_after_all_preflight(self):
        substrate = types.SimpleNamespace(exclusive_gpu_lease=lambda lease: Lease())
        with mock.patch.object(M, "validate_production_launch",
                               side_effect=M.M1249Error("preflight")), \
             mock.patch.object(M, "consume_attempt") as consume:
            with self.assertRaises(M.M1249Error):
                M.execute_once({}, Path(__file__), substrate)
            consume.assert_not_called()

    def test_15_attempt_is_exclusive_and_no_retry(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("os.O_EXCL", text)
        self.assertIn("0o400", text)
        self.assertEqual(M.ATTEMPT_TOKEN,
                         "M1249_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
        self.assert_rejected(lambda row: row["one_shot"].__setitem__(
            "automatic_retry", True))

    def test_16_run_capture_restores_predecessor_namespace(self):
        old = M.M1243.CANONICAL_RESULT
        seen = {}
        with mock.patch.object(M.M1243, "run_capture", side_effect=lambda *a, **k: (
                seen.update(namespace=M.M1243.CANONICAL_RESULT) or M.CANONICAL_RESULT)):
            self.assertEqual(M.run_capture({}, {}, substrate=object()), M.CANONICAL_RESULT)
        self.assertEqual(seen["namespace"], M.CANONICAL_RESULT)
        self.assertEqual(M.M1243.CANONICAL_RESULT, old)

    def test_17_no_production_launch_contract_exists(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertFalse(contract["production_launch_contract_created"])
        self.assertTrue(contract["future_M1237_result_hammer_required"])

    def test_18_no_remote_gpu_or_release_was_executed(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import subprocess", "paramiko", "ssh ",
                          "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["author_execution"], {
            "remote": False, "gpu": False, "checkpoint": False, "capture": False,
            "release": False, "eda": False,
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
