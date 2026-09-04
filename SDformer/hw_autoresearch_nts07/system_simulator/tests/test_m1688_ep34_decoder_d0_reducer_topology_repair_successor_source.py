#!/usr/bin/env python3
"""Payload-free tests for additive M1688 topology repair."""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


TESTS = Path(__file__).resolve().parent
SOURCE = TESTS.parent / "scripts/build_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1688_author_test", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_metric(configuration, shard, requests, cycles):
    commits = shard["destination_count"] * M.B.G.OUTPUT_BLOCKS
    row = {"configuration": configuration,
        "resource_manifest_sha256": M.B.G.RESOURCE_SHA256,
        "total_cycles": cycles, "request_count": requests,
        "kind_counts": {"commit": commits, "compute": requests - commits},
        "byte_counts": {"commit": commits * M.B.G.R.OUTPUT_COMMIT_BYTES,
                        "compute": 0},
        "packed_transaction_address_sha256": "1" * 64,
        "packed_commit_sequence_sha256": "2" * 64,
        "destination_state_chain_sha256": "3" * 64,
        "per_request_miter": True, "per_destination_miter": True,
        "shard_reset_boundary": True, "paper_result": False}
    row["final_state_sha256"] = M.B.metric_final_state(row, shard)
    return row


def make_receipt(ordinal, request_counts, attempt_sha):
    shard = M.B.G.shard_descriptor(ordinal)
    metrics = [make_metric(config, shard, requests, 200 - index * 20)
               for index, (config, requests) in enumerate(
                   zip(M.CONFIGS, request_counts))]
    return {"schema": M.B.RESULT_SCHEMA, "status": M.B.RESULT_STATUS,
        "source_sha256": M.B.sha256(M.B.SOURCE),
        "release_sha256": "4" * 64, "attempt_sha256": attempt_sha,
        "checkpoint_sha256": M.B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": M.B.G.RESOURCE_SHA256,
        "shard_ordinal": ordinal, "shard": shard,
        "configuration_order": list(M.CONFIGS), "metrics": metrics,
        "integer_ratio_inputs":
            M.B.G.validate_three_configuration_metrics(metrics, shard),
        "payload_fd_sha256": "6" * 64, "payload_fd_size": 576000,
        "rss": {"absolute_limit_kib": M.B.G.RSS_ABSOLUTE_LIMIT_KIB,
                "increment_limit_kib": M.B.G.RSS_INCREMENT_LIMIT_KIB,
                "gate_calls": 2},
        "automatic_retry": False, "shard_isolated": True,
        "monolithic_full_call": False, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}


def build_shard(root, ordinal, request_counts=(172, 173, 174)):
    root = Path(root)
    paths = {"result": root / ("result_" + str(ordinal)),
             "attempt": root / ("attempt_" + str(ordinal)),
             "work": root / ("work_" + str(ordinal)),
             "failure": root / ("failure_" + str(ordinal))}
    paths["attempt"].write_text("attempt {}\n".format(ordinal),
                                encoding="ascii")
    os.chmod(str(paths["attempt"]), 0o400)
    attempt_sha = digest(paths["attempt"])
    row = make_receipt(ordinal, request_counts, attempt_sha)
    paths["result"].mkdir()
    (paths["result"] / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    M.B.seal_work_tree(paths["result"])
    return paths, row


class M1688Tests(unittest.TestCase):
    def rejects(self, function):
        with self.assertRaises((M.M1688Error, M.B.M1681Error)):
            function()

    def bind(self, mapping):
        original = M.B.namespace_paths
        M.B.namespace_paths = lambda ordinal: mapping[ordinal]
        return original

    def test_01_m1682_disposition_and_source_stage(self):
        seal = M.verify_m1682_disposition()
        self.assertEqual(seal["manifest_sha256"], M.M1682_MANIFEST_SHA256)
        row = M.validate_source_stage()
        self.assertTrue(row["M1683_release_permanently_forbidden"])
        self.assertFalse(row["payload_opened"])
        self.assertFalse(row["execution"])

    def test_02_exact_topology_and_attempt_mode_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _row = build_shard(directory, 0)
            original = self.bind({0: paths})
            try:
                verified = M.verify_sealed_shard(0)
            finally:
                M.B.namespace_paths = original
            self.assertEqual(verified["ordinal"], 0)
            self.assertEqual(os.stat(str(paths["attempt"])).st_mode & 0o777,
                             0o400)

    def test_03_result_attempt_failure_attack_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _row = build_shard(directory, 0)
            paths["failure"].mkdir()
            original = self.bind({0: paths})
            try:
                self.rejects(lambda: M.verify_sealed_shard(0))
            finally:
                M.B.namespace_paths = original

    def test_04_result_attempt_work_attack_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _row = build_shard(directory, 0)
            paths["work"].mkdir()
            original = self.bind({0: paths})
            try:
                self.rejects(lambda: M.verify_sealed_shard(0))
            finally:
                M.B.namespace_paths = original

    def test_05_attempt_symlink_and_wrong_mode_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _row = build_shard(root, 0)
            target = root / "attempt_target"
            paths["attempt"].rename(target)
            paths["attempt"].symlink_to(target)
            original = self.bind({0: paths})
            try:
                self.rejects(lambda: M.verify_sealed_shard(0))
            finally:
                M.B.namespace_paths = original
        with tempfile.TemporaryDirectory() as directory:
            paths, _row = build_shard(directory, 0)
            os.chmod(str(paths["attempt"]), 0o600)
            original = self.bind({0: paths})
            try:
                self.rejects(lambda: M.verify_sealed_shard(0))
            finally:
                M.B.namespace_paths = original

    def test_06_reducer_preserves_344_346_348_request_conservation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths0, _row0 = build_shard(directory, 0)
            paths1, _row1 = build_shard(directory, 1)
            original_paths = self.bind({0: paths0, 1: paths1})
            old_total = M.B.G.TOTAL_SHARDS
            M.B.G.TOTAL_SHARDS = 2
            try:
                reduced = M.reduce_complete_sealed_shards()
            finally:
                M.B.G.TOTAL_SHARDS = old_total
                M.B.namespace_paths = original_paths
        actual = dict((config,
            reduced["configuration_totals"][config]["requests"])
                      for config in M.CONFIGS)
        self.assertEqual(actual, {"DENSE_TYPED_K8": 344,
            "BIT_EQUAL_SERVICE_K1X8": 346, "BIT_TYPED_K8": 348})
        self.assertTrue(reduced["exact_sibling_topology"])
        self.assertTrue(reduced["attempt_regular_nonsymlink_mode_0400"])

    def test_07_all_fifteen_existing_metric_and_result_attacks_remain_rejected(self):
        shard = M.B.G.shard_descriptor(0)
        base = [make_metric(config, shard, request, 200 - index * 20)
                for index, (config, request) in enumerate(zip(
                    M.CONFIGS, (172, 173, 174)))]
        M.B.validate_metric_bundle(base, shard)
        mutations = [
            lambda rows: rows[0].update(total_cycles=0),
            lambda rows: rows[0].update(request_count=0),
            lambda rows: rows[0].update(request_count=-1),
            lambda rows: rows[0].update(request_count=True),
            lambda rows: rows[0]["kind_counts"].update(compute=-1),
            lambda rows: rows[0]["kind_counts"].update(compute=5),
            lambda rows: rows[0]["byte_counts"].update(compute=-1),
            lambda rows: rows[0]["kind_counts"].update(commit=167),
            lambda rows: rows[0]["byte_counts"].update(commit=0),
            lambda rows: rows[0].update(
                packed_transaction_address_sha256="bad"),
            lambda rows: rows[1].update(
                packed_commit_sequence_sha256="7" * 64),
            lambda rows: rows[0].update(
                destination_state_chain_sha256="bad"),
            lambda rows: rows[0].update(final_state_sha256="0" * 64),
        ]
        rejected = 0
        for mutation in mutations:
            rows = copy.deepcopy(base); mutation(rows)
            try:
                M.B.validate_metric_bundle(rows, shard)
            except M.B.M1681Error:
                rejected += 1
        self.assertEqual(rejected, 13)
        with tempfile.TemporaryDirectory() as directory:
            paths, _row = build_shard(directory, 0)
            original = self.bind({0: paths})
            try:
                (paths["result"] / "unsealed").write_text("x",
                                                           encoding="ascii")
                self.rejects(lambda: M.verify_sealed_shard(0)); rejected += 1
                (paths["result"] / "unsealed").unlink()
                cache = paths["result"] / "__pycache__"; cache.mkdir()
                (cache / "x.pyc").write_bytes(b"x")
                self.rejects(lambda: M.verify_sealed_shard(0)); rejected += 1
            finally:
                M.B.namespace_paths = original
        self.assertEqual(rejected, 15)

    def test_08_reducer_calls_strong_verifier_not_m1681_weak_verifier(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        reducer = next(node for node in tree.body if
                       isinstance(node, ast.FunctionDef) and
                       node.name == "reduce_complete_sealed_shards")
        calls = [node for node in ast.walk(reducer)
                 if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Name) and
                            call.func.id == "verify_sealed_shard"
                            for call in calls))
        self.assertFalse(any(isinstance(call.func, ast.Attribute) and
                             isinstance(call.func.value, ast.Name) and
                             call.func.value.id == "B" and
                             call.func.attr == "verify_sealed_shard"
                             for call in calls))

    def test_09_source_cli_has_no_replay_or_reducer_mode(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(node for node in tree.body if
                    isinstance(node, ast.FunctionDef) and node.name == "main")
        text = "\n".join(source.splitlines()[main.lineno - 1:])
        self.assertNotIn("verify_sealed_shard(", text)
        self.assertNotIn("reduce_complete_sealed_shards(", text)
        row = M.describe()
        self.assertFalse(row["claim_boundary"]["actual_execution"])
        self.assertFalse(row["claim_boundary"]["paper_result"])
        self.assertEqual(row["numbering"]["forbidden_release"], "M1683")


if __name__ == "__main__":
    unittest.main()
