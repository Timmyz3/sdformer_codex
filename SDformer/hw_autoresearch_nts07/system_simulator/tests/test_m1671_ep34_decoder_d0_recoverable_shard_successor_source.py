#!/usr/bin/env python3
"""Source-only tests for M1671; canonical payload bytes are never opened."""
from __future__ import print_function

import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest


TESTS = Path(__file__).resolve().parent
SOURCE = TESTS.parent / "scripts/build_m1671_ep34_decoder_d0_recoverable_shard_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1671_author_test", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1671Tests(unittest.TestCase):
    def rejects(self, function):
        with self.assertRaises(M.M1671Error):
            function()

    def test_01_grid_is_complete_and_unique(self):
        row = M.validate_grid()
        self.assertEqual(row, {"calls": 30, "timesteps": 300,
            "destinations": 360000, "shards": 8700,
            "gap_count": 0, "overlap_count": 0})
        self.assertEqual(M.SHARDS_PER_TIMESTEP, 29)
        self.assertEqual(M.TOTAL_SHARDS, 8700)

    def test_02_first_and_last_shard_boundaries(self):
        first = M.shard_descriptor(0)
        last = M.shard_descriptor(M.TOTAL_SHARDS - 1)
        self.assertEqual((first["call_ordinal"], first["timestep"],
                          first["destination_start"],
                          first["destination_stop_exclusive"]),
                         (0, 0, 0, 42))
        self.assertEqual((last["call_ordinal"], last["timestep"],
                          last["destination_start"],
                          last["destination_stop_exclusive"],
                          last["destination_count"]),
                         (116, 9, 1176, 1200, 24))
        self.rejects(lambda: M.shard_descriptor(-1))
        self.rejects(lambda: M.shard_descriptor(M.TOTAL_SHARDS))

    def test_03_authority_and_population_preflight_is_source_only(self):
        row = M.validate_authorities()
        self.assertEqual(row["checkpoint_sha256"], M.CHECKPOINT_SHA256)
        self.assertEqual(row["resource_manifest_sha256"], M.RESOURCE_SHA256)
        self.assertEqual(row["grid"]["shards"], 8700)
        self.assertFalse(row["actual_payload"])
        self.assertFalse(row["actual_execution"])

    def test_04_exact_m1656_and_m1666_boundaries(self):
        result = M.verify_result_tree()
        review = M.verify_m1666()
        self.assertFalse(result["full_decoder"])
        self.assertFalse(result["paper_result"])
        self.assertEqual(review["verdict"], "PASS_PREFIX_DIAGNOSTIC_ONLY")
        self.assertFalse(review["authorization"]["l3_expansion"])

    def test_05_selected_records_are_d0_and_manifest_bound(self):
        for ordinal in (0, M.TOTAL_SHARDS - 1, 4350):
            shard = M.shard_descriptor(ordinal)
            row = M.selected_record(shard)
            self.assertEqual(row["global_call_ordinal"],
                             shard["call_ordinal"])
            self.assertEqual(row["module_ordinal"], 0)
            self.assertEqual(tuple(row["shape"]), M.R.INPUT_SHAPES[0])

    def test_06_synthetic_exact_request_and_destination_miters(self):
        row = M.synthetic_shard()
        self.assertEqual(row["status"],
            "PASS_M1671_SYNTHETIC_SHARD__NO_PAYLOAD_NO_EXECUTION")
        self.assertEqual([metric["configuration"]
                          for metric in row["metrics"]], list(M.CONFIGS))
        self.assertTrue(all(metric["per_request_miter"] and
                            metric["per_destination_miter"]
                            for metric in row["metrics"]))
        self.assertFalse(row["actual_payload"])
        self.assertFalse(row["actual_execution"])
        self.assertEqual(row["attempt_writes"], 0)

    def test_07_coordinate_carries_actual_timestep_and_destination(self):
        config = M.CONFIGS[2]
        row = M.R.request(config + ":m0:t7:d811:ob2:g3:typed_desc",
            config, "external_read", [0], [0], 16)
        coordinate = M.actual_coordinate(config, row, 19, 7, 811, 2)
        self.assertEqual(coordinate[1], 0)
        self.assertEqual(coordinate[2], 7)
        self.assertEqual(coordinate[4], 811)
        self.assertEqual(coordinate[5], 2)
        self.assertEqual(coordinate[6], 3)
        self.rejects(lambda: M.actual_coordinate(
            config, row, 19, 6, 811, 2))

    def test_08_three_configuration_result_mutations_are_rejected(self):
        shard = M.shard_descriptor(M.TOTAL_SHARDS - 1)
        base = []
        for config in M.CONFIGS:
            base.append({"configuration": config,
                "resource_manifest_sha256": M.RESOURCE_SHA256,
                "per_request_miter": True, "per_destination_miter": True,
                "shard_reset_boundary": True, "total_cycles": 100,
                "request_count": 10,
                "kind_counts": {"commit": 24 * 4},
                "byte_counts": {"commit": 24 * 384},
                "packed_commit_sequence_sha256": "a" * 64})
        M.validate_three_configuration_metrics(base, shard)
        attacks = []
        for mutate in (
                lambda rows: rows.reverse(),
                lambda rows: rows[0].update(resource_manifest_sha256="0" * 64),
                lambda rows: rows[1].update(per_destination_miter=False),
                lambda rows: rows[2]["kind_counts"].update(commit=95),
                lambda rows: rows[2].update(
                    packed_commit_sequence_sha256="b" * 64)):
            rows = copy.deepcopy(base)
            mutate(rows)
            try:
                M.validate_three_configuration_metrics(rows, shard)
            except M.M1671Error:
                attacks.append(True)
        self.assertEqual(len(attacks), 5)

    def test_09_incomplete_reducer_and_forbidden_configuration_fail_closed(self):
        self.rejects(lambda: M.reduce_complete_shards([]))
        self.rejects(lambda: M.ShardSession(
            M.FORBIDDEN_CONFIG, M.shard_descriptor(0), M.P.RssGate()))

    def test_10_source_cli_has_no_payload_or_execution_mode(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and
                    node.name == "main")
        lines = SOURCE.read_text(encoding="utf-8").splitlines()
        text = "\n".join(lines[main.lineno - 1:])
        self.assertNotIn("--run", text)
        self.assertNotIn("--execute", text)
        self.assertNotIn("--reduce", text)
        row = M.describe()
        self.assertTrue(row["claim_boundary"]["source_only"])
        self.assertFalse(row["claim_boundary"]["actual_payload"])
        self.assertFalse(row["claim_boundary"]["execution"])
        self.assertFalse(row["claim_boundary"]["paper_result"])

    def test_11_expansion_boundaries_do_not_claim_decoder_or_system(self):
        row = M.describe()
        self.assertIn("exact source review", row["future_expansion"]["D2_D3"])
        self.assertIn("excluded", row["future_expansion"]["D1"])
        self.assertFalse(row["future_expansion"]["full_decoder"])
        self.assertFalse(row["future_expansion"]["system"])
        self.assertTrue(row["execution_model"]["attempt_before_payload"])
        self.assertFalse(row["execution_model"]["automatic_retry"])
        self.assertEqual(row["execution_model"]["reduction"],
                         "integer ratio-of-sums only")


if __name__ == "__main__":
    unittest.main()
