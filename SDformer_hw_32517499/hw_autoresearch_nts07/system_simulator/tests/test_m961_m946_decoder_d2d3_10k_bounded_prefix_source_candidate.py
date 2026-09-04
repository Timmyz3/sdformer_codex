#!/usr/bin/env python3
"""No-prefix tests for the source-only M961 workflow."""

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = (HERE.parent / "scripts" /
          "execute_m961_m946_decoder_d2d3_10k_bounded_prefix_r1.py")
RUNNER = (HERE.parent / "scripts" /
          "run_m961_m946_decoder_d2d3_10k_bounded_prefix_r1_one_shot.sh")
CONTRACT = (HW / "contracts" /
            "m961_m946_decoder_d2d3_10k_bounded_prefix_source_contract_r1_20260829.json")
SPEC = importlib.util.spec_from_file_location("m961_test_driver", DRIVER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import M961 driver")
M961 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M961
SPEC.loader.exec_module(M961)


def fake_row(layer="D2", *, commits=0, transactions=1):
    return {
        "row_identity": {"layer": layer, "numerical_route":
                         "EXACT_BINARY_SUPPORT"},
        "prefix": 10000,
        "elapsed_seconds": 1.0,
        "process_max_rss_kib": 2048,
        "exact_miter": {
            "status": "PASS_M768_M861_M890_M896_EXACT_MITER",
            "expanded_request_count": 10000,
            "compressed_transaction_count": transactions,
            "commit_requests_in_prefix": commits,
            "combined_live_event_state_bytes": 8192,
        },
    }


class M961SourceCandidateTest(unittest.TestCase):
    def test_source_contract_and_namespace(self):
        value = M961.validate_source_contract(CONTRACT, RUNNER)
        self.assertEqual(
            value["status"],
            "PASS_M961_SOURCE_CONTRACT__NO_EXECUTION_AUTHORIZED")
        self.assertTrue(value["result_absent"])
        self.assertTrue(value["attempt_absent"])

    def test_source_self_test_executes_no_prefix(self):
        value = M961.source_self_test()
        self.assertEqual(
            value["status"],
            "PASS_M961_SOURCE_SELF_TEST__NO_PREFIX_EXECUTED")
        self.assertFalse(value["prefix_executed"])
        self.assertFalse(value["full_row_authorized"])

    def test_d2_d3_100k_projection_is_source_fetch_only(self):
        for layer, full_count in (("D2", 231600), ("D3", 465600)):
            value = M961.project_100k(fake_row(layer))
            self.assertEqual(value["source_fetch_full_request_count"],
                             full_count)
            self.assertTrue(value["future_100k_stays_inside_source_fetch"])
            self.assertFalse(value["future_100k_contributor_mapper_covered"])
            self.assertFalse(value["future_100k_commit_covered"])
            self.assertFalse(value["automatic_100k_authorized"])
            self.assertFalse(value["full_row_authorized"])

    def test_projection_rejects_commit_or_extra_transaction(self):
        with self.assertRaises(RuntimeError):
            M961.project_100k(fake_row(commits=1))
        with self.assertRaises(RuntimeError):
            M961.project_100k(fake_row(transactions=2))

    def test_release_absent_and_cannot_validate(self):
        self.assertFalse(M961.FUTURE_RELEASE.exists())
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            M961.validate_release(
                M961.FUTURE_RELEASE, RUNNER, "0" * 64,
                M961.RELEASE_HAMMER, ("0" * 64,) * 3)

    def test_canonical_scope_and_no_d1(self):
        self.assertEqual(M961.PREFIX_10K, 10000)
        self.assertEqual(M961.PREFIX_100K, 100000)
        self.assertEqual(set(M961.SOURCE_FETCH_REQUESTS), {"D2", "D3"})
        self.assertNotIn("D1", M961.SOURCE_FETCH_REQUESTS)
        paths = M961.canonical_paths()
        self.assertNotEqual(paths["result"], paths["attempt"])


if __name__ == "__main__":
    unittest.main()
