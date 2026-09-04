#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1583_ep34_decoder_one_process_one_config_source.py"


def load_source():
    spec = importlib.util.spec_from_file_location("m1583_test_module", str(SOURCE))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M1583")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_source()


def clean_row(config="DENSE_TYPED_K8"):
    return {
        "configuration": config,
        "resource_manifest_sha256": M.RESOURCE_SHA256,
        "total_cycles": 101,
        "request_count": 3,
        "kind_counts": {"compute": 2, "commit": 1},
        "byte_counts": {"compute": 288, "commit": 3},
        "transaction_address_sha256": "a" * 64,
        "commit_sequence_sha256": "b" * 64,
        "streaming": {"materialized_transaction_list": False,
                      "destinations": 7, "timesteps": 10},
        "schema": "upstream",
        "pilot_call_ordinal": 0,
        "module_ordinal": 0,
        "timesteps": 10,
        "diagnostic_only": True,
        "paper_result": False,
        "product_capture": False,
        "production": False,
        "payload_fd_sha256": "c" * 64,
        "payload_fd_size": 4096,
        "m1573_rss": {"gate_calls": 2,
                      "baseline_current_rss_kib": 100,
                      "baseline_peak_rss_kib": 120,
                      "max_current_rss_kib": 130,
                      "max_peak_rss_kib": 140,
                      "absolute_limit_kib": M.RSS_LIMIT_KIB,
                      "fresh_exec_required": True},
        "fresh_exec_required": True,
    }


class M1583Tests(unittest.TestCase):
    def test_01_description_is_source_only(self):
        value = M.describe()
        self.assertEqual(value["status"], M.STATUS)
        self.assertTrue(value["fresh_interpreter_per_configuration"])
        self.assertFalse(value["claim_boundary"]["actual_execution"])
        self.assertFalse(value["claim_boundary"]["cycles"])

    def test_02_one_shot_token_precedes_second_call(self):
        calls = []
        worker = M._build_one_shot(lambda config: calls.append(config) or clean_row(config))
        self.assertEqual(worker("DENSE_TYPED_K8")["total_cycles"], 101)
        with self.assertRaises(M.M1583Error):
            worker("BIT_TYPED_K8")
        self.assertEqual(calls, ["DENSE_TYPED_K8"])

    def test_03_product_rejected_before_bound_entry(self):
        calls = []
        worker = M._build_one_shot(lambda config: calls.append(config) or clean_row(config))
        with self.assertRaises(M.M1583Error):
            worker(M.FORBIDDEN_CONFIG)
        self.assertEqual(calls, [])

    def test_04_identity_and_dual_rss_mutations_rejected(self):
        mutations = [
            ("configuration", "BIT_TYPED_K8"),
            ("resource_manifest_sha256", "d" * 64),
            ("total_cycles", 0),
            ("request_count", 0),
            ("transaction_address_sha256", "x" * 64),
            ("fresh_exec_required", False),
        ]
        for key, value in mutations:
            row = clean_row()
            row[key] = value
            with self.assertRaises(M.M1583Error, msg=key):
                M.validate_result("DENSE_TYPED_K8", row)
        for key, value in (("gate_calls", 0),
                           ("max_peak_rss_kib", M.RSS_LIMIT_KIB)):
            row = clean_row()
            row["m1573_rss"][key] = value
            with self.assertRaises(M.M1583Error, msg=key):
                M.validate_result("DENSE_TYPED_K8", row)

    def test_05_request_conservation_rejected(self):
        row = clean_row()
        row["kind_counts"]["commit"] = 2
        with self.assertRaises(M.M1583Error):
            M.validate_result("DENSE_TYPED_K8", row)

    def test_06_frozen_identities(self):
        self.assertEqual(M.sha256(M.M1573_PATH), M.M1573_SHA256)
        self.assertEqual(M.sha256(M.M1577 / "review.json"),
                         M.M1577_REVIEW_SHA256)
        self.assertEqual(M.sha256(M.DOCS359), M.DOCS359_SHA256)


if __name__ == "__main__":
    unittest.main()
