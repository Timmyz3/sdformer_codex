#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1325_motion_ep34_runtime_projection_successor_r1.py")


def load_source():
    spec = importlib.util.spec_from_file_location("test_m1325_source", str(SOURCE))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_source()


class M1325Tests(unittest.TestCase):
    def setUp(self):
        self.m1313 = M.strict_json(M.M1313_CONTRACT)

    def test_01_m1324_forensic_and_source_policy_exact(self):
        review = M.verify_m1324_forensic()
        policy = M.validate_source_policy()
        self.assertEqual(review["independence"], {"different_author": True})
        self.assertFalse(policy["production_authorized"])

    def test_02_ast_audit_exact_four_direct_keys(self):
        self.assertEqual(M.frozen_m1227_direct_contract_keys(),
                         {"contract_path", "capture", "cohort", "output"})

    def test_03_runtime_projection_exact_four_keys(self):
        runtime = M.build_runtime_contract(self.m1313)
        self.assertEqual(set(runtime), M.RUNTIME_KEYS)
        self.assertEqual(runtime["capture"], {"attention_windows_per_call": 100})
        self.assertEqual(runtime["cohort"], self.m1313["cohort"])
        self.assertEqual(runtime["output"]["path"],
                         str(M.CANONICAL_RESULT.relative_to(M.ROOT)))

    def test_04_projection_is_deep_copy_and_m1313_is_unchanged(self):
        before = copy.deepcopy(self.m1313)
        runtime = M.build_runtime_contract(self.m1313)
        runtime["cohort"]["samples"][0]["bytes"] = 1
        self.assertEqual(self.m1313, before)

    def test_05_missing_extra_and_wrong_capture_rejected(self):
        runtime = M.build_runtime_contract(self.m1313)
        for mutation in (lambda row: row.pop("capture"),
                         lambda row: row.update(extra=True),
                         lambda row: row["capture"].update(attention_windows_per_call=99)):
            changed = copy.deepcopy(runtime)
            mutation(changed)
            with self.assertRaises(M.M1325Error):
                M.validate_runtime_contract(changed, self.m1313)

    def test_06_nonexact_m1313_and_noncanonical_contract_path_rejected(self):
        changed = copy.deepcopy(self.m1313)
        changed["status"] = "changed"
        with self.assertRaisesRegex(M.M1325Error, "exact M1313"):
            M.build_runtime_contract(changed)
        with self.assertRaisesRegex(M.M1325Error, "canonical M1325"):
            M.build_runtime_contract(self.m1313, M.ROOT / "other.json")

    def test_07_actual_chain_receives_projection_and_new_output(self):
        runtime = M.build_runtime_contract(self.m1313)
        binding = {
            "policy": {}, "verified_samples": [], "identity": {}, "selection": {},
            "checkpoint_path": Path("checkpoint"), "config_path": Path("config"),
        }
        observed = {}

        def fake_m1227(contract, child_binding, r1=None):
            observed["contract"] = copy.deepcopy(contract)
            observed["binding_is_same"] = child_binding is binding
            observed["substrate"] = r1
            observed["m1227_result"] = M.M1227.CANONICAL_RESULT
            return M.M1227.CANONICAL_RESULT

        substrate = object()
        old_m1249 = M.M1319.M1249.CANONICAL_RESULT
        with mock.patch.object(M.M1227, "run_capture", side_effect=fake_m1227):
            output = M.delegate_for_future_release(runtime, binding, substrate)
        self.assertEqual(observed["contract"], runtime)
        self.assertTrue(observed["binding_is_same"])
        self.assertIs(observed["substrate"], substrate)
        self.assertEqual(observed["m1227_result"], M.CANONICAL_RESULT)
        self.assertEqual(output, M.CANONICAL_RESULT)
        self.assertEqual(M.M1319.M1249.CANONICAL_RESULT, old_m1249)

    def test_08_projection_identity_path_is_read_only(self):
        binding = {"identity": {"m1319_projection": "exact"}}
        with mock.patch.object(M.M1319, "validate_exact_m1313_m1314",
                               return_value=(copy.deepcopy(self.m1313), binding)) as validate, \
             mock.patch.object(M.M1319, "execute_once") as execute, \
             mock.patch.object(M.M1319.M1249, "consume_attempt") as consume:
            runtime, observed = M.validate_identity_and_project()
        self.assertEqual(set(runtime), M.RUNTIME_KEYS)
        self.assertIs(observed, binding)
        validate.assert_called_once_with(M.M1313_CONTRACT, M.M1314_ENTRY)
        execute.assert_not_called()
        consume.assert_not_called()

    def test_09_namespaces_are_new_and_pairwise_distinct(self):
        paths = {M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG}
        self.assertEqual(len(paths), 3)
        self.assertTrue(all("m1325" in path.name for path in paths))
        self.assertFalse(any("m1249" in str(path) for path in paths))

    def test_10_source_has_no_attempt_or_production_cli(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("M1249.consume_attempt", source)
        self.assertNotIn("M1319.execute_once", source)
        self.assertNotIn("exclusive_gpu_lease", source)
        self.assertIn("--source-self-check", source)
        self.assertNotIn("--run", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
