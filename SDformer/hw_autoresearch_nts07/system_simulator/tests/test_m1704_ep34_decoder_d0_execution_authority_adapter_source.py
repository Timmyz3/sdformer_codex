#!/usr/bin/env python3
"""Payload-free dual-runtime tests for M1704 authority adapter."""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TESTS = Path(__file__).resolve().parent
SOURCE = TESTS.parent / "scripts/build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py"
SPEC = importlib.util.spec_from_file_location("m1704_author_test", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_file(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(digest(path) + "  " + path.name + "\n", encoding="ascii")
    Path(str(path) + ".sha256.seal.sha256").write_text(
        digest(sidecar) + "  " + sidecar.name + "\n", encoding="ascii")


class FutureFixture(object):
    def __init__(self, root, review_mutation=None, release_mutation=None):
        root = Path(root)
        self.review = root / "review"
        self.review.mkdir()
        review = {"schema": "m1705_review_fixture",
            "status": M.REVIEW_STATUS, "score_over_100": 100,
            "p0_count": 0, "p1_count": 0,
            "identity": M._review_identity(),
            "authorization": {"release_authoring": True,
                "shard_execution": False, "payload_open": False,
                "reducer_execution": False, "automatic_retry": False}}
        if review_mutation:
            review_mutation(review)
        (self.review / "review.json").write_text(json.dumps(
            review, indent=2, sort_keys=True, allow_nan=False) + "\n")
        M.B.seal_work_tree(self.review)
        seal = M.B.verify_sealed_tree(self.review, allow_ignored_pycache=False,
                                      label="fixture review")
        self.release = root / "release.json"
        release = {"schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
            "identity": dict(M._review_identity(),
                review_sha256=digest(self.review / "review.json"),
                review_manifest_sha256=seal["manifest_sha256"],
                review_outer_file_sha256=seal["outer_file_sha256"]),
            "authorization": {"shard_runs": 8700, "payload_opens": 8700,
                "attempt_writes": 8700, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0},
            "fixed_grid": M.B.G.fixed_grid(),
            "namespace_examples": {"first": M.B.namespace_strings(0),
                "last": M.B.namespace_strings(M.B.G.TOTAL_SHARDS - 1)},
            "reducer": {"source": "M1688",
                "strong_exact_sibling_topology": True,
                "attempt_regular_nonsymlink_mode_0400": True},
            "claim_boundary": {"shard_isolated": True,
                "monolithic_full_call": False, "full_decoder": False,
                "system_speedup": False, "paper_result": False}}
        if release_mutation:
            release_mutation(release)
        self.release.write_text(json.dumps(
            release, indent=2, sort_keys=True, allow_nan=False) + "\n")
        seal_file(self.release)


class M1704Tests(unittest.TestCase):
    def rejects(self, function):
        with self.assertRaises((M.M1704Error, M.B.M1681Error,
                                OSError, KeyError)):
            function()

    def test_01_exact_m1688_m1689_and_source_stage(self):
        seal = M.verify_m1689()
        self.assertEqual(seal["manifest_sha256"], M.M1689_MANIFEST_SHA256)
        row = M.validate_source_stage()
        self.assertEqual(row["grid"]["shards"], 8700)
        self.assertTrue(row["m1683_release_permanently_forbidden"])
        self.assertFalse(row["payload_opened"])
        self.assertFalse(row["execution"])

    def test_02_future_review_release_exact_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = FutureFixture(directory)
            old = M.FUTURE_REVIEW, M.FUTURE_RELEASE
            M.FUTURE_REVIEW, M.FUTURE_RELEASE = fixture.review, fixture.release
            try:
                self.assertEqual(M.validate_future_review_and_release(),
                                 digest(fixture.release))
            finally:
                M.FUTURE_REVIEW, M.FUTURE_RELEASE = old

    def test_03_release_mutations_rejected(self):
        mutations = (
            lambda row: row["authorization"].update(shard_runs=8699),
            lambda row: row["authorization"].update(automatic_retry=True),
            lambda row: row["identity"].update(source_sha256="0" * 64),
            lambda row: row["reducer"].update(
                strong_exact_sibling_topology=False),
            lambda row: row["fixed_grid"].update(shards=8699),
            lambda row: row["namespace_examples"].update(first={}),
            lambda row: row["claim_boundary"].update(full_decoder=True),
        )
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as directory:
                fixture = FutureFixture(directory, release_mutation=mutation)
                old = M.FUTURE_REVIEW, M.FUTURE_RELEASE
                M.FUTURE_REVIEW, M.FUTURE_RELEASE = fixture.review, fixture.release
                try:
                    self.rejects(M.validate_future_review_and_release)
                finally:
                    M.FUTURE_REVIEW, M.FUTURE_RELEASE = old

    def test_04_review_authority_mutations_rejected(self):
        mutations = (
            lambda row: row["authorization"].update(shard_execution=True),
            lambda row: row.update(p1_count=1),
            lambda row: row["identity"].update(m1688_source_sha256="0" * 64),
        )
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as directory:
                fixture = FutureFixture(directory, review_mutation=mutation)
                old = M.FUTURE_REVIEW, M.FUTURE_RELEASE
                M.FUTURE_REVIEW, M.FUTURE_RELEASE = fixture.review, fixture.release
                try:
                    self.rejects(M.validate_future_review_and_release)
                finally:
                    M.FUTURE_REVIEW, M.FUTURE_RELEASE = old

    def test_05_execution_entry_rebinds_only_authority_and_restores(self):
        original_run = M.B._run_authorized_shard
        original_gate = M.B.validate_future_review_and_release
        reached = []
        def fake_run(ordinal):
            self.assertIs(M.B.validate_future_review_and_release,
                          M.validate_future_review_and_release)
            reached.append(ordinal)
            return {"ordinal": ordinal}
        M.B._run_authorized_shard = fake_run
        try:
            self.assertEqual(M._run_authorized_shard(17), {"ordinal": 17})
        finally:
            M.B._run_authorized_shard = original_run
        self.assertEqual(reached, [17])
        self.assertIs(M.B.validate_future_review_and_release, original_gate)
        self.rejects(lambda: M._run_authorized_shard(True))
        self.rejects(lambda: M._run_authorized_shard(8700))

    def test_06_reducer_delegates_only_to_m1688_strong_reducer(self):
        original = M.M1688.reduce_complete_sealed_shards
        reached = []
        M.M1688.reduce_complete_sealed_shards = lambda: reached.append(True) or {
            "status": "synthetic"}
        try:
            self.assertEqual(M.reduce_complete_sealed_shards(),
                             {"status": "synthetic"})
        finally:
            M.M1688.reduce_complete_sealed_shards = original
        self.assertEqual(reached, [True])
        tree = ast.parse(SOURCE.read_text())
        function = next(node for node in tree.body if
                        isinstance(node, ast.FunctionDef) and
                        node.name == "reduce_complete_sealed_shards")
        calls = [node for node in ast.walk(function)
                 if isinstance(node, ast.Call)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].func.value.id, "M1688")
        self.assertEqual(calls[0].func.attr, "reduce_complete_sealed_shards")

    def test_07_source_cli_has_no_execution_or_reducer_mode(self):
        source = SOURCE.read_text()
        tree = ast.parse(source)
        main = next(node for node in tree.body if
                    isinstance(node, ast.FunctionDef) and node.name == "main")
        text = "\n".join(source.splitlines()[main.lineno - 1:])
        self.assertNotIn("_run_authorized_shard(", text)
        self.assertNotIn("reduce_complete_sealed_shards(", text)
        row = M.describe()
        self.assertTrue(row["claim_boundary"]["source_only"])
        self.assertFalse(row["claim_boundary"]["paper_result"])

    def test_08_contract_and_source_identity(self):
        contract = M.B.strict_json(M.SOURCE_CONTRACT)
        self.assertEqual(contract["schema"], M.SCHEMA)
        self.assertEqual(contract["source"]["sha256"], digest(M.SOURCE))
        self.assertEqual(contract["test"]["sha256"], digest(M.TEST))
        self.assertEqual(contract["authorization"]["shard_execution"], False)
        self.assertEqual(contract["numbering"]["forbidden_release"], "M1683")
        self.assertEqual(contract["numbering"]["future_review"], "M1705")
        self.assertEqual(contract["numbering"]["future_release"], "M1706")


if __name__ == "__main__":
    unittest.main(verbosity=2)
