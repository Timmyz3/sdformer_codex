#!/usr/bin/env python3
"""Dual-runtime regression for the source-only M1638 configuration binding."""
from __future__ import print_function

import copy
import importlib.util
from pathlib import Path
import unittest


TESTS = Path(__file__).resolve().parent
SOURCE = (TESTS.parent / "scripts" /
          "build_m1638_decoder_compact_l2_session_configuration_bound_successor_source.py")
BASE_TEST = TESTS / "test_m1628_decoder_compact_l2_retained_ledger_successor_source.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_TEST, "m1638_bound_m1628_tests")
base.SOURCE = SOURCE


class M1638CompatibilityAndRepairTests(base.M1628Tests):
    @classmethod
    def setUpClass(cls):
        cls.m = base.load_source()

    def test_01_describe_and_static_self_test_are_source_only(self):
        description = self.m.describe()
        self.assertEqual(description["status"],
            "SOURCE_ONLY__M1629_CONFIGURATION_RELABEL_P1_REPAIRED__NO_PAYLOAD_NO_EXECUTION")
        self.assertTrue(description["repair"]["m1628_behavior_inherited"])
        self.assertTrue(description["repair"]["hidden_initial_configuration_bound"])
        self.assertEqual(description["repair"]["exact_bundle_coverage_policy"],
                         [[True, True], [False, False], [False, False]])
        for field in ("actual_runner_source", "actual_payload", "l2_execution",
                      "l3", "pilot", "production", "cycles", "traffic",
                      "energy", "speedup", "rtl", "eda", "paper_result"):
            self.assertFalse(description["authorization"][field])
        self.assertFalse(description["future_gate"]["review_present"])
        self.assertFalse(description["future_gate"]["release_present"])
        result = self.m.static_self_test()
        self.assertEqual(result["status"],
                         "PASS_M1638_CONFIGURATION_BOUND_SOURCE_STATIC_ONLY")
        self.assertEqual(result["configuration_relabel_rejected"], 2)
        self.assertEqual(result["attacks_rejected"], 4)

    def test_19_three_actual_dense_sessions_cannot_be_relabelled(self):
        actual_dense = [self.completed_miter(self.m.CONFIGS[0])
                        for _index in range(3)]
        first = actual_dense[0].finish()
        self.assertEqual(first.as_dict()["configuration"], self.m.CONFIGS[0])
        for miter, target in zip(actual_dense[1:], self.m.CONFIGS[1:]):
            miter.configuration = target
            self.rejects(miter.finish)

    def test_20_binding_is_checked_at_request_state_payload_and_finish(self):
        m = self.m

        request_miter = m.CanonicalPrefixMiter(m.CONFIGS[0])
        request_miter.configuration = m.CONFIGS[1]
        self.rejects(lambda: self.request(request_miter, "commit", 0))

        state_miter = m.CanonicalPrefixMiter(m.CONFIGS[0])
        self.populate_destination(state_miter)
        state = m.synthetic_state(state_miter, True)
        state_miter.configuration = m.CONFIGS[1]
        self.rejects(lambda: state_miter.accept_destination_pair(
            state, copy.deepcopy(state)))

        payload_miter = self.completed_miter(m.CONFIGS[0])
        payload_miter.configuration = m.CONFIGS[1]
        self.rejects(payload_miter._finish_payload)

        finish_miter = self.completed_miter(m.CONFIGS[0])
        finish_miter.configuration = m.CONFIGS[2]
        self.rejects(finish_miter.finish)

        rows = [self.completed_miter(configuration).finish()
                for configuration in m.CONFIGS]
        policy = [(row.as_dict()["dense_cache_covered"],
                   row.as_dict()["dense_psum_1rw_covered"])
                  for row in rows]
        self.assertEqual(policy,
                         [(True, True), (False, False), (False, False)])
        self.assertTrue(m.validate_bundle(rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
