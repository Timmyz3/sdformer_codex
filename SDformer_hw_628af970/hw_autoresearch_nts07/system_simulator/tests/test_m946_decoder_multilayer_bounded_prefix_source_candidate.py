#!/usr/bin/env python3
"""Lightweight tests for the fail-closed M946 source candidate."""

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = (HERE.parent / "scripts" /
          "analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py")
SPEC = importlib.util.spec_from_file_location("m946_test_source", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import M946 source")
M946 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M946
SPEC.loader.exec_module(M946)


class M946SourceCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authority = M946.validate_frozen_authority()
        _, _, _, cls.records, _, _ = M946._context()

    def test_authority_and_layer_routes(self):
        self.assertEqual(self.authority["status"],
                         "PASS_M946_FROZEN_AUTHORITY")
        expected = {
            "D1": (1, "COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE"),
            "D2": (2, "EXACT_BINARY_SUPPORT"),
            "D3": (3, "EXACT_BINARY_SUPPORT"),
        }
        for layer, (module_index, route) in expected.items():
            record = M946.select_record(self.records, layer, 0)
            self.assertEqual(int(record["module_index"]), module_index)
            self.assertEqual(M946.ROUTE_BY_LAYER[layer], route)

    def test_refuses_d0_and_unsealed_sample(self):
        with self.assertRaises(M946.Failure):
            M946.select_record(self.records, "D0", 0)
        with self.assertRaises(M946.Failure):
            M946.select_record(self.records, "D2", 10)

    def test_three_real_1k_prefixes_exact(self):
        for layer in M946.ALLOWED_LAYERS:
            output = M946.run_bounded_prefix(
                layer, 0, "A1_OSG", 0, 1000)
            self.assertEqual(
                output["status"],
                "PASS_M946_BOUNDED_PREFIX_EXACT_PREFLIGHT__NO_FULL_ROW")
            self.assertEqual(
                output["exact_miter"]["status"],
                "PASS_M768_M861_M890_M896_EXACT_MITER")
            self.assertEqual(
                output["exact_miter"]["expanded_request_count"], 1000)
            self.assertFalse(output["claim_boundary"]["paper_citable"])
            self.assertFalse(output["claim_boundary"]["full_row_authorized"])
            self.assertFalse(output["claim_boundary"]["decoder_complete"])

    def test_synthetic_1k_exact(self):
        result = M946.M896.exact_miter(
            M946.M890.synthetic_transactions(1000), include_old=True)
        self.assertEqual(
            result["status"],
            "PASS_EXACT_M768_M861_M890_RUN_GTLS_MITER")

    def test_projection_never_authorizes_full_row(self):
        short = M946.projection("D1", 1000, 1.0, 1024, 4096)
        gate = M946.projection("D1", 100000, 1.0, 1024, 4096)
        self.assertFalse(short["prefix_is_authoritative_100k_gate"])
        self.assertFalse(short["full_row_authorized"])
        self.assertTrue(gate["prefix_is_authoritative_100k_gate"])
        self.assertFalse(gate["full_row_authorized"])

    def test_cli_refuses_full_row_production_and_output(self):
        for argv in (["--run-full-row"], ["--run-production"],
                     ["--output", "/tmp/forbidden.json"]):
            with self.assertRaises(M946.Failure):
                M946.main(argv)


if __name__ == "__main__":
    unittest.main()
