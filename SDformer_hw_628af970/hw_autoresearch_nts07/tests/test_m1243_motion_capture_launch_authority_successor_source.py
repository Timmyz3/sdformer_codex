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
    "capture_m1243_motion_final_checkpoint_unified_hardware_launch_authority_r3.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1243_motion_capture_launch_authority_successor_source_contract_r1_20260830.json")
OLD_TEST = ROOT / "hw_autoresearch_nts07/tests/test_m1233_motion_final_checkpoint_unified_capture_selection_interface_source.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


M = load("m1243_launch_authority_under_test", SOURCE)
T = load("m1243_m1233_fixture", OLD_TEST)


class M1243LaunchAuthorityTest(unittest.TestCase):
    def setUp(self):
        self.base = T.M1233SelectionInterfaceTest(
            "test_03_exact_m1234_shape_passes_and_keyerror_is_regressed")
        self.base.setUp()
        self.hammer_tmp = tempfile.TemporaryDirectory(prefix=".m1243_source_hammer_",
                                                       dir=M.HW / "reviews")
        self.hammer_root = Path(self.hammer_tmp.name)
        self.hammer_entry = self.write_hammer(self.hammer_value())

    def tearDown(self):
        self.hammer_tmp.cleanup()
        self.base.tearDown()

    def hammer_value(self):
        return {
            "schema": M.SOURCE_HAMMER_SCHEMA,
            "status": M.SOURCE_HAMMER_STATUS,
            "source_authority": {
                "source_path": str(SOURCE.relative_to(M.ROOT)),
                "source_sha256": sha(SOURCE),
                "contract_path": str(CONTRACT.relative_to(M.ROOT)),
                "contract_sha256": sha(CONTRACT),
                "test_path": str(Path(__file__).resolve().relative_to(M.ROOT)),
                "test_sha256": sha(Path(__file__).resolve()),
            },
            "independence": {"different_author": True},
            "authorization": {"production_capture": True},
        }

    def write_hammer(self, value):
        for name in ("review.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"):
            path = self.hammer_root / name
            if path.exists():
                path.unlink()
        review = self.hammer_root / "review.json"
        review.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        M.R1.write_double_seal(self.hammer_root)
        return {
            "path": str(self.hammer_root.relative_to(M.ROOT)),
            "manifest_sha256": sha(self.hammer_root / "SHA256SUMS"),
            "outer_file_sha256": sha(self.hammer_root / "SHA256SUMS.seal.sha256"),
            "review_sha256": sha(review),
        }

    def assert_hammer_rejected(self, entry=None):
        with self.assertRaises(M.M1243Error):
            M.verify_source_hammer(self.hammer_entry if entry is None else entry)

    def launch(self, include_hammer=True):
        with tempfile.TemporaryDirectory(prefix=".m1243_launch_", dir=M.HW / "contracts") as name:
            path = Path(name) / "launch.json"
            inputs = {
                "launcher": {"path": str(SOURCE.relative_to(M.ROOT)), "sha256": sha(SOURCE)},
                "source_contract": {"path": str(CONTRACT.relative_to(M.ROOT)),
                                    "sha256": sha(CONTRACT)},
                "final_selection_result": self.base.selection_entry,
                "final_selection_result_hammer": self.base.hammer_entry,
            }
            if include_hammer:
                inputs["source_hammer"] = self.hammer_entry
            contract = {
                "schema": M.LAUNCH_SCHEMA, "status": M.LAUNCH_STATUS,
                "contract_path": str(path.relative_to(M.ROOT)), "inputs": inputs,
                "cohort": {"samples": []},
                "one_shot": {"attempt_marker": "attempt"},
                "output": {"path": "result"},
                "production_log": {"path": "log"},
            }
            path.write_text(json.dumps(contract) + "\n", encoding="utf-8")

            def mapped(value, missing_leaf=False):
                return {"attempt": M.CANONICAL_ATTEMPT, "result": M.CANONICAL_RESULT,
                        "log": M.CANONICAL_LOG}[value]

            with mock.patch.object(M.R1, "validate_m1224", return_value={}), \
                 mock.patch.object(M.R1, "validate_cohort", return_value=[]), \
                 mock.patch.object(M.R1, "safe_repo_path", side_effect=mapped):
                return M.validate_launch_contract(contract, path)

    def test_01_selection_and_capture_are_exact_frozen_aliases(self):
        self.assertIs(M.validate_final_selection, M.P.validate_final_selection)
        for name in ("EXPECTED_STATIC_COUNTS", "EXPECTED_LIVE_COUNTS", "DEAD_SN_V",
                     "audit_call_matrix", "audit_attention_population",
                     "validate_payload_population", "atomic_sample_snapshot",
                     "final_validate_and_seal"):
            self.assertIs(getattr(M, name), getattr(M.P, name))
        self.assertEqual(sum(M.EXPECTED_STATIC_COUNTS.values()), 259)
        self.assertEqual(sum(M.EXPECTED_LIVE_COUNTS.values()), 247)

    def test_02_import_is_lazy_for_heavy_stack(self):
        code = (
            "import importlib.util,sys;"
            "s=importlib.util.spec_from_file_location('isolated_m1243',{!r});"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "print(int('torch' in sys.modules),int('numpy' in sys.modules),"
            "int('m1227_sealed_m1174' in sys.modules))"
        ).format(str(SOURCE))
        self.assertEqual(subprocess.check_output([sys.executable, "-c", code]).decode().strip(),
                         "0 0 0")

    def test_03_valid_sealed_different_author_hammer_passes(self):
        value = M.verify_source_hammer(self.hammer_entry)
        self.assertTrue(value["production_capture"])

    def test_04_source_only_contract_cannot_launch(self):
        source = json.loads(CONTRACT.read_text(encoding="utf-8"))
        with self.assertRaises(M.M1243Error):
            M.validate_launch_contract(source, CONTRACT)

    def test_05_missing_source_hammer_is_rejected(self):
        with self.assertRaises(M.M1243Error):
            self.launch(include_hammer=False)

    def test_06_source_hammer_entry_shape_is_exact(self):
        self.assert_hammer_rejected({**self.hammer_entry, "extra": True})
        entry = dict(self.hammer_entry)
        del entry["review_sha256"]
        self.assert_hammer_rejected(entry)

    def test_07_every_source_hammer_seal_sha_is_bound(self):
        for key in ("manifest_sha256", "outer_file_sha256", "review_sha256"):
            with self.subTest(key=key):
                self.assert_hammer_rejected({**self.hammer_entry, key: "0" * 64})

    def test_08_source_hammer_schema_and_status_are_fixed(self):
        for key in ("schema", "status"):
            value = self.hammer_value()
            value[key] = "wrong"
            self.assert_hammer_rejected(self.write_hammer(value))

    def test_09_every_source_contract_test_cross_sha_is_bound(self):
        for key in sorted(M.SOURCE_AUTHORITY_KEYS):
            value = self.hammer_value()
            current = value["source_authority"][key]
            value["source_authority"][key] = str(current) + ".drift"
            with self.subTest(key=key):
                self.assert_hammer_rejected(self.write_hammer(value))

    def test_10_same_author_assertion_is_rejected(self):
        value = self.hammer_value()
        value["independence"] = {"different_author": False}
        self.assert_hammer_rejected(self.write_hammer(value))

    def test_11_production_capture_false_or_extra_authority_is_rejected(self):
        value = self.hammer_value()
        value["authorization"] = {"production_capture": False}
        self.assert_hammer_rejected(self.write_hammer(value))
        value = self.hammer_value()
        value["authorization"]["extra"] = True
        self.assert_hammer_rejected(self.write_hammer(value))

    def test_12_positive_launch_consumes_hammer_and_returns_authority(self):
        binding = self.launch()
        self.assertTrue(binding["identity"]["source_hammer"]["production_capture"])
        self.assertEqual(binding["identity"]["candidate_id"], "resume_ep32")

    def test_13_launch_source_and_contract_sha_mutations_are_rejected(self):
        canonical = self.hammer_value()
        canonical["source_authority"]["source_sha256"] = "0" * 64
        self.hammer_entry = self.write_hammer(canonical)
        with self.assertRaises(M.M1243Error):
            self.launch()

    def test_14_delegate_changes_only_fresh_result_namespace(self):
        seen = {}

        def run_capture(contract, binding, predecessor=None, substrate=None):
            seen.update(contract=contract, binding=binding, predecessor=predecessor,
                        substrate=substrate, result=module.CANONICAL_RESULT)
            return module.CANONICAL_RESULT

        old = Path("old")
        module = types.SimpleNamespace(CANONICAL_RESULT=old, run_capture=run_capture)
        substrate = object()
        output = M.run_capture({"c": 1}, {"b": 2}, predecessor=module,
                               substrate=substrate)
        self.assertEqual(output, M.CANONICAL_RESULT)
        self.assertEqual(module.CANONICAL_RESULT, old)
        self.assertIs(seen["predecessor"], M.R1)
        self.assertIs(seen["substrate"], substrate)

    def test_15_namespaces_are_fresh_and_contract_hashes_are_exact(self):
        self.assertNotEqual(M.CANONICAL_RESULT, M.P.CANONICAL_RESULT)
        self.assertFalse(M.CANONICAL_RESULT.exists())
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["source"]["sha256"], sha(SOURCE))
        self.assertEqual(contract["test"]["sha256"], sha(Path(__file__).resolve()))

    def test_16_source_only_no_gpu_remote_or_release(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import subprocess", "paramiko", "ssh ",
                          "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertFalse(contract["claim_boundary"]["production_authorized"])
        self.assertFalse(contract["claim_boundary"]["capture_complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
