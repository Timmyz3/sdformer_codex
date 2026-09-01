#!/usr/bin/env python3
"""Payload-free regression for M1681 execution-closure source."""
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
SOURCE = TESTS.parent / "scripts/build_m1681_ep34_decoder_d0_shard_execution_closure_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1681_author_test", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_tree(root):
    root = Path(root)
    members = sorted(path.relative_to(root) for path in root.rglob("*")
                     if path.is_file() and path.name not in
                     ("SHA256SUMS", "SHA256SUMS.seal.sha256") and
                     "__pycache__" not in path.parts)
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(
        digest(root / member), member.as_posix()) for member in members),
        encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        digest(manifest) + "  SHA256SUMS\n", encoding="ascii")


def seal_file(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(digest(path) + "  " + path.name + "\n",
                       encoding="ascii")
    Path(str(path) + ".sha256.seal.sha256").write_text(
        digest(sidecar) + "  " + sidecar.name + "\n", encoding="ascii")


def metric(configuration, shard, cycles=100, requests=8):
    row = {"configuration": configuration,
        "resource_manifest_sha256": M.G.RESOURCE_SHA256,
        "total_cycles": cycles, "request_count": requests,
        "kind_counts": {"commit": shard["destination_count"] * 4,
                        "compute": requests - shard["destination_count"] * 4},
        "byte_counts": {
            "commit": shard["destination_count"] * 4 *
                      M.G.R.OUTPUT_COMMIT_BYTES,
            "compute": 0},
        "packed_transaction_address_sha256": "1" * 64,
        "packed_commit_sequence_sha256": "2" * 64,
        "destination_state_chain_sha256": "3" * 64,
        "per_request_miter": True, "per_destination_miter": True,
        "shard_reset_boundary": True, "paper_result": False}
    row["final_state_sha256"] = M.metric_final_state(row, shard)
    return row


def receipt(ordinal=0, release_sha="4" * 64, attempt_sha="5" * 64):
    shard = M.G.shard_descriptor(ordinal)
    # A three-destination synthetic population keeps request conservation sane.
    shard = dict(shard)
    metrics = [metric(config, shard,
                      cycles=200 - index * 20,
                      requests=shard["destination_count"] * 4 + 4)
               for index, config in enumerate(M.CONFIGS)]
    return {"schema": M.RESULT_SCHEMA, "status": M.RESULT_STATUS,
        "source_sha256": M.sha256(M.SOURCE), "release_sha256": release_sha,
        "attempt_sha256": attempt_sha,
        "checkpoint_sha256": M.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": M.G.RESOURCE_SHA256,
        "shard_ordinal": ordinal, "shard": shard,
        "configuration_order": list(M.CONFIGS), "metrics": metrics,
        "integer_ratio_inputs":
            M.G.validate_three_configuration_metrics(metrics, shard),
        "payload_fd_sha256": "6" * 64, "payload_fd_size": 576000,
        "rss": {"absolute_limit_kib": M.G.RSS_ABSOLUTE_LIMIT_KIB,
                "increment_limit_kib": M.G.RSS_INCREMENT_LIMIT_KIB,
                "gate_calls": 2},
        "automatic_retry": False, "shard_isolated": True,
        "monolithic_full_call": False, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}


class FutureFixture(object):
    def __init__(self, root, review_mutation=None, release_mutation=None):
        root = Path(root)
        self.review = root / "review"
        self.release = root / "release.json"
        self.review.mkdir()
        row = {"status": M.REVIEW_STATUS, "score_over_100": 99,
            "p0_count": 0, "p1_count": 0,
            "identity": M._review_identity(),
            "authorization": {"release_authoring": True,
                "shard_execution": False, "payload_open": False,
                "automatic_retry": False}}
        if review_mutation:
            review_mutation(row)
        (self.review / "review.json").write_text(json.dumps(
            row, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        seal_tree(self.review)
        seal = M.verify_sealed_tree(self.review, label="fixture review")
        identity = dict(M._review_identity(),
            review_sha256=digest(self.review / "review.json"),
            review_manifest_sha256=seal["manifest_sha256"],
            review_outer_file_sha256=seal["outer_file_sha256"])
        release = {"schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
            "identity": identity,
            "authorization": {"shard_runs": 8700, "payload_opens": 8700,
                "attempt_writes": 8700, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0},
            "fixed_grid": M.G.fixed_grid(),
            "namespace_examples": {"first": M.namespace_strings(0),
                "last": M.namespace_strings(M.G.TOTAL_SHARDS - 1)},
            "claim_boundary": {"shard_isolated": True,
                "monolithic_full_call": False, "full_decoder": False,
                "system_speedup": False, "paper_result": False}}
        if release_mutation:
            release_mutation(release)
        self.release.write_text(json.dumps(
            release, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        seal_file(self.release)


class M1681Tests(unittest.TestCase):
    def rejects(self, function):
        with self.assertRaises(M.M1681Error):
            function()

    def test_01_m1672_findings_and_grid_are_exact(self):
        seal = M.verify_m1672_no_go()
        self.assertEqual(seal["manifest_sha256"], M.M1672_MANIFEST_SHA256)
        self.assertEqual(M.G.validate_grid()["shards"], 8700)
        self.assertEqual(M.G.sha256(M.G.SOURCE), M.M1671_SOURCE_SHA256)

    def test_02_m1666_pycache_is_explicitly_ignored_not_evidence(self):
        seal = M.verify_m1666_with_explicit_pycache_policy()
        self.assertEqual(seal["ignored_unsealed_pycache"], [
            "__pycache__/independent_hammer.cpython-310.pyc",
            "__pycache__/independent_hammer.cpython-36.pyc"])
        self.assertNotEqual(seal["ignored_unsealed_pycache"], [])

    def test_03_fixed_namespaces_cover_first_and_last_shards(self):
        first = M.namespace_strings(0)
        last = M.namespace_strings(8699)
        self.assertIn("0000", first["result"])
        self.assertIn("8699", last["result"])
        self.assertEqual(len(set(first.values())), 4)
        self.assertEqual(len(set(last.values())), 4)

    def test_04_future_review_release_semantics_and_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = FutureFixture(directory)
            old = M.FUTURE_REVIEW, M.FUTURE_RELEASE
            M.FUTURE_REVIEW, M.FUTURE_RELEASE = fixture.review, fixture.release
            try:
                self.assertEqual(M.validate_future_review_and_release(),
                                 digest(fixture.release))
            finally:
                M.FUTURE_REVIEW, M.FUTURE_RELEASE = old
        for mutation in (
                lambda row: row["authorization"].update(shard_runs=8699),
                lambda row: row["authorization"].update(automatic_retry=True),
                lambda row: row.update(fixed_grid={}),
                lambda row: row["claim_boundary"].update(paper_result=True)):
            with tempfile.TemporaryDirectory() as directory:
                fixture = FutureFixture(directory, release_mutation=mutation)
                old = M.FUTURE_REVIEW, M.FUTURE_RELEASE
                M.FUTURE_REVIEW, M.FUTURE_RELEASE = fixture.review, fixture.release
                try:
                    self.rejects(M.validate_future_review_and_release)
                finally:
                    M.FUTURE_REVIEW, M.FUTURE_RELEASE = old

    def test_05_attempt_is_consumed_before_any_payload_or_runtime_gate(self):
        events = []
        class StopBeforePayload(RuntimeError):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"result": root / "result", "attempt": root / "attempt",
                     "work": root / "work", "failure": root / "failure"}
            originals = (M.validate_future_review_and_release,
                M.require_fresh_shard, M.consume_attempt,
                M.G.R.validate_authorities, M.namespace_paths)
            M.validate_future_review_and_release = lambda: (
                events.append("authority") or "a" * 64)
            M.require_fresh_shard = lambda _ordinal: (
                events.append("fresh") or paths)
            M.consume_attempt = lambda _ordinal, _release: (
                events.append("attempt") or "b" * 64)
            M.namespace_paths = lambda _ordinal: paths
            M.G.R.validate_authorities = lambda _full: (
                events.append("runtime_population") or
                (_ for _ in ()).throw(StopBeforePayload()))
            try:
                with self.assertRaises(StopBeforePayload):
                    M._run_authorized_shard(0)
            finally:
                (M.validate_future_review_and_release,
                 M.require_fresh_shard, M.consume_attempt,
                 M.G.R.validate_authorities, M.namespace_paths) = originals
            self.assertEqual(events,
                ["authority", "fresh", "attempt", "runtime_population"])
            self.assertTrue(paths["failure"].is_dir())
            self.assertFalse(paths["work"].exists())

    def test_06_authority_failure_reaches_no_attempt_or_payload_seam(self):
        reached = []
        original = M.validate_future_review_and_release
        M.validate_future_review_and_release = lambda: (
            (_ for _ in ()).throw(M.M1681Error("blocked")))
        old_fresh, old_attempt, old_runtime = (
            M.require_fresh_shard, M.consume_attempt, M.G.R.validate_authorities)
        M.require_fresh_shard = lambda *_args: reached.append("fresh")
        M.consume_attempt = lambda *_args: reached.append("attempt")
        M.G.R.validate_authorities = lambda *_args: reached.append("payload")
        try:
            self.rejects(lambda: M._run_authorized_shard(0))
        finally:
            M.validate_future_review_and_release = original
            M.require_fresh_shard, M.consume_attempt = old_fresh, old_attempt
            M.G.R.validate_authorities = old_runtime
        self.assertEqual(reached, [])

    def test_07_metric_rejects_negative_request_bytes_and_digest_drift(self):
        shard = M.G.shard_descriptor(0)
        base = metric(M.CONFIGS[0], shard,
                      requests=shard["destination_count"] * 4 + 4)
        M.validate_metric(base, M.CONFIGS[0], shard)
        mutations = (
            lambda row: row.update(request_count=-1),
            lambda row: row["byte_counts"].update(compute=-1),
            lambda row: row.update(
                packed_transaction_address_sha256="bad"),
            lambda row: row.update(destination_state_chain_sha256="0" * 64),
            lambda row: row.update(final_state_sha256="0" * 64),
        )
        for mutation in mutations:
            row = copy.deepcopy(base); mutation(row)
            self.rejects(lambda row=row: M.validate_metric(
                row, M.CONFIGS[0], shard))

    def test_08_atomic_seal_verifier_forbids_extra_and_pycache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"; root.mkdir()
            (root / "result.json").write_text("{}\n", encoding="ascii")
            M.seal_work_tree(root)
            M.verify_sealed_tree(root, label="synthetic result")
            extra = root / "extra"; extra.write_text("x", encoding="ascii")
            self.rejects(lambda: M.verify_sealed_tree(
                root, label="synthetic result"))
            extra.unlink()
            cache = root / "__pycache__"; cache.mkdir()
            (cache / "x.pyc").write_bytes(b"x")
            self.rejects(lambda: M.verify_sealed_tree(
                root, allow_ignored_pycache=False, label="synthetic result"))

    def test_08b_sealed_shard_receipt_binds_attempt_and_all_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"result": root / "result", "attempt": root / "attempt",
                     "work": root / "work", "failure": root / "failure"}
            paths["attempt"].write_text("attempt\n", encoding="ascii")
            attempt_sha = digest(paths["attempt"])
            row = receipt(0, attempt_sha=attempt_sha)
            paths["result"].mkdir()
            (paths["result"] / "result.json").write_text(json.dumps(
                row, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8")
            M.seal_work_tree(paths["result"])
            original = M.namespace_paths
            M.namespace_paths = lambda _ordinal: paths
            try:
                verified = M.verify_sealed_shard(0)
                self.assertEqual(verified["attempt_sha256"], attempt_sha)
                (paths["result"] / "unsealed").write_text("x", encoding="ascii")
                self.rejects(lambda: M.verify_sealed_shard(0))
            finally:
                M.namespace_paths = original

    def test_09_immutable_timestep_snapshot_uses_opened_fd_hash(self):
        shape = tuple(M.G.R.INPUT_SHAPES[0])
        size = 576000
        raw = bytearray(size)
        timestep_bytes = size // 10
        raw[3 * timestep_bytes] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.bitpack"
            path.write_bytes(bytes(raw))
            plane = M.ImmutableTimestepPlane(
                path, shape, digest(path), 3)
            self.assertEqual(plane.opened_sha256, digest(path))
            self.assertEqual(plane.bit(0, 0, 0), 1)
            self.assertEqual(plane.bit(0, 0, 1), 0)
            self.rejects(lambda: M.ImmutableTimestepPlane(
                path, shape, "0" * 64, 3))

    def test_10_reducer_is_sealed_only_and_ratio_of_sums(self):
        old_total = M.G.TOTAL_SHARDS
        old_verify = M.verify_sealed_shard
        rows = [receipt(0), receipt(1)]
        M.G.TOTAL_SHARDS = 2
        M.verify_sealed_shard = lambda ordinal: {
            "ordinal": ordinal, "row": rows[ordinal],
            "seal": {"manifest_sha256": str(ordinal) * 64},
            "attempt_sha256": rows[ordinal]["attempt_sha256"]}
        try:
            reduced = M.reduce_complete_sealed_shards()
        finally:
            M.G.TOTAL_SHARDS = old_total
            M.verify_sealed_shard = old_verify
        self.assertEqual(reduced["complete_shards"], 2)
        self.assertEqual(reduced["ratio_of_sums"]["dense_to_bit_typed"],
                         {"numerator": 400, "denominator": 320})
        self.assertFalse(reduced["full_decoder"])
        self.assertFalse(reduced["system_speedup"])

    def test_11_private_targets_not_reachable_from_cli(self):
        source_text = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        main = next(node for node in tree.body if
                    isinstance(node, ast.FunctionDef) and node.name == "main")
        main_text = "\n".join(source_text.splitlines()[main.lineno - 1:])
        self.assertNotIn("_run_authorized_shard(", main_text)
        self.assertNotIn("reduce_complete_sealed_shards(", main_text)
        row = M.describe()
        self.assertEqual(row["execution_closure"]["private_target"],
                         "_run_authorized_shard")
        self.assertFalse(row["claim_boundary"]["shard_execution"])
        self.assertFalse(row["claim_boundary"]["paper_result"])


if __name__ == "__main__":
    unittest.main()
