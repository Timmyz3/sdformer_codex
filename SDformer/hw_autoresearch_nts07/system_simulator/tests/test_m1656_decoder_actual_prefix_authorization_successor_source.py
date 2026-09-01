#!/usr/bin/env python3
"""Source-only regression for M1656; never opens the decoder payload."""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest


TESTS = Path(__file__).resolve().parent
SOURCE = TESTS.parent / "scripts/build_m1656_decoder_actual_prefix_authorization_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1656_author_test", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_tree(root):
    members = sorted(path for path in root.iterdir()
                     if path.is_file() and path.name not in
                     ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(digest(path), path.name)
                                for path in members), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        digest(manifest) + "  SHA256SUMS\n", encoding="ascii")


def seal_file(path):
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(digest(path) + "  " + path.name + "\n",
                       encoding="ascii")
    Path(str(path) + ".sha256.seal.sha256").write_text(
        digest(sidecar) + "  " + sidecar.name + "\n", encoding="ascii")


class AuthorityFixture(object):
    def __init__(self, root, release_mutation=None, review_mutation=None):
        self.root = Path(root)
        self.review = self.root / "review"
        self.release = self.root / "release.json"
        self.review.mkdir()
        review = {"status": M.REVIEW_STATUS, "score": 99,
            "p0_count": 0, "p1_count": 0,
            "identity": M._review_identity(),
            "authorization": {"release_authoring": True,
                "execution": False, "payload": False,
                "automatic_retry": False}}
        if review_mutation is not None:
            review_mutation(review)
        (self.review / "review.json").write_text(json.dumps(
            review, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (self.review / "review.md").write_text("synthetic authority\n",
                                               encoding="utf-8")
        seal_tree(self.review)
        identity = dict(M._review_identity(),
            review_sha256=digest(self.review / "review.json"),
            review_manifest_sha256=digest(self.review / "SHA256SUMS"),
            review_outer_file_sha256=digest(
                self.review / "SHA256SUMS.seal.sha256"))
        release = {"schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
            "identity": identity,
            "authorization": {"actual_prefix_runs": 1,
                "payload_opens": 1, "attempt_writes": 1,
                "automatic_retry": False, "gpu_runs": 0,
                "eda_runs": 0, "all_other_runs": 0},
            "namespaces": M._namespaces(),
            "fixed_population": M._fixed_population(),
            "claim_boundary": {"prefix_only": True,
                "cycles_pending_hammer": True,
                "bytes_pending_hammer": True,
                "product_capture": False, "l3": False,
                "full_decoder": False, "production": False,
                "paper_result": False}}
        if release_mutation is not None:
            release_mutation(release)
        self.release.write_text(json.dumps(
            release, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        seal_file(self.release)


class M1656Tests(unittest.TestCase):
    def rejects(self, function):
        with self.assertRaises(M.M1656Error):
            function()

    def bind_fixture(self, fixture):
        old = M.FUTURE_REVIEW, M.FUTURE_RELEASE
        M.FUTURE_REVIEW, M.FUTURE_RELEASE = fixture.review, fixture.release
        return old

    def test_01_fixed_population_and_new_namespaces(self):
        row = M.describe()
        self.assertEqual(row["status"], M.STATUS)
        self.assertEqual(row["fixed_population"], {
            "decoder_stage": "D0", "call_ordinal": 0,
            "module_ordinal": 0, "timestep": 0,
            "destinations": list(range(42)),
            "output_blocks": [0, 1, 2, 3],
            "configuration_order": list(M.P.CONFIGS)})
        M.require_fresh_namespaces()
        paths = (M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE)
        self.assertEqual(len(set(paths)), 4)
        self.assertTrue(all("m1656_" in path.name for path in paths))

    def test_02_exact_m1646_tree_status_and_p1_disposition(self):
        row = M.verify_m1646_no_go_and_disposition()
        self.assertEqual(row["review_sha256"], M.M1646_REVIEW_SHA256)
        self.assertEqual(row["manifest_sha256"], M.M1646_MANIFEST_SHA256)
        self.assertEqual(row["outer_file_sha256"],
                         M.M1646_OUTER_FILE_SHA256)
        self.assertEqual(row["p1_id"],
            "P1_PRESENCE_ONLY_PRIVATE_EXECUTION_AUTHORIZATION")
        self.assertTrue(row["successor_source_repair"])

    def test_03_static_exact_m1645_path_preserved_without_payload(self):
        row = M.static_self_test()
        self.assertEqual(row["configurations"], list(M.P.CONFIGS))
        self.assertEqual(row["distinct_sessions"], 3)
        self.assertEqual(row["commits_per_configuration"], 168)
        self.assertEqual(row["fixed_population"]["destinations"],
                         list(range(42)))
        self.assertFalse(row["actual_payload"])
        self.assertFalse(row["actual_execution"])
        self.assertEqual(row["attempt_writes"], 0)

    def test_03b_m1646_mismatch_and_tree_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied = root / "copied"
            shutil.copytree(str(M.M1646), str(copied))
            review = copied / "review.json"
            review.write_bytes(review.read_bytes() + b"\n")
            old = M.M1646
            M.M1646 = copied
            try:
                self.rejects(M.verify_m1646_no_go_and_disposition)
                link = root / "link"
                link.symlink_to(old, target_is_directory=True)
                M.M1646 = link
                self.rejects(M.verify_m1646_no_go_and_disposition)
            finally:
                M.M1646 = old

    def test_04_source_stage_future_authority_and_namespaces_absent(self):
        row = M.verify_pre_payload_authorities(require_future=False)
        self.assertIsNone(row["release"])
        for path in (M.FUTURE_REVIEW, M.FUTURE_RELEASE,
                     Path(str(M.FUTURE_RELEASE) + ".sha256"),
                     Path(str(M.FUTURE_RELEASE) +
                          ".sha256.seal.sha256"),
                     M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE):
            self.assertFalse(os.path.lexists(str(path)))

    def test_05_empty_and_symlink_authority_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_review = root / "empty_review"; empty_review.mkdir()
            empty_release = root / "empty_release"; empty_release.write_bytes(b"")
            old = M.FUTURE_REVIEW, M.FUTURE_RELEASE
            M.FUTURE_REVIEW, M.FUTURE_RELEASE = empty_review, empty_release
            try:
                self.rejects(M.validate_future_review_and_release)
                target = root / "review_target"; target.mkdir()
                link = root / "review_link"; link.symlink_to(target)
                M.FUTURE_REVIEW = link
                self.rejects(M.validate_future_review_and_release)
            finally:
                M.FUTURE_REVIEW, M.FUTURE_RELEASE = old

    def test_06_valid_double_sealed_future_authorities_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(directory)
            old = self.bind_fixture(fixture)
            try:
                release, release_sha = M.validate_future_review_and_release()
            finally:
                M.FUTURE_REVIEW, M.FUTURE_RELEASE = old
            self.assertEqual(release["schema"], M.RELEASE_SCHEMA)
            self.assertEqual(release_sha, digest(fixture.release))

    def test_07_release_semantic_mutations_are_rejected_under_fresh_seals(self):
        mutations = (
            lambda row: row.update(schema="wrong"),
            lambda row: row.update(status="wrong"),
            lambda row: row["identity"].update(checkpoint_sha256="0" * 64),
            lambda row: row["authorization"].update(actual_prefix_runs=2),
            lambda row: row["namespaces"].update(result="results/wrong"),
        )
        for mutation in mutations:
            with self.subTest(mutation=str(mutation)):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = AuthorityFixture(directory,
                                               release_mutation=mutation)
                    old = self.bind_fixture(fixture)
                    try:
                        self.rejects(M.validate_future_review_and_release)
                    finally:
                        M.FUTURE_REVIEW, M.FUTURE_RELEASE = old

    def test_08_release_file_and_sidecar_symlinks_are_rejected(self):
        for target_name in ("release", "sidecar", "outer"):
            with self.subTest(target=target_name):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = AuthorityFixture(directory)
                    if target_name == "release":
                        target = fixture.release
                    elif target_name == "sidecar":
                        target = Path(str(fixture.release) + ".sha256")
                    else:
                        target = Path(str(fixture.release) +
                                      ".sha256.seal.sha256")
                    saved = Path(str(target) + ".saved")
                    target.rename(saved); target.symlink_to(saved)
                    old = self.bind_fixture(fixture)
                    try:
                        self.rejects(M.validate_future_review_and_release)
                    finally:
                        M.FUTURE_REVIEW, M.FUTURE_RELEASE = old

    def test_09_pre_payload_authority_failure_blocks_every_later_seam(self):
        reached = []
        original_authority = M.verify_pre_payload_authorities
        original_attempt = M.consume_attempt
        original_runtime = M.P.R.validate_authorities
        original_select = M.P._selected_payload
        original_rss = M.P.RssGate
        M.verify_pre_payload_authorities = lambda **_kw: (_ for _ in ()).throw(
            M.M1656Error("authority rejected"))
        M.consume_attempt = lambda *_args: reached.append("attempt")
        M.P.R.validate_authorities = lambda *_args: reached.append("predecessor")
        M.P._selected_payload = lambda: reached.append("payload")
        M.P.RssGate = lambda: reached.append("rss")
        try:
            self.rejects(M._run_authorized_actual_prefix)
        finally:
            M.verify_pre_payload_authorities = original_authority
            M.consume_attempt = original_attempt
            M.P.R.validate_authorities = original_runtime
            M.P._selected_payload = original_select
            M.P.RssGate = original_rss
        self.assertEqual(reached, [])

    def test_10_success_order_is_authority_attempt_predecessor_payload_then_rss(self):
        class StopBeforePayloadOpen(RuntimeError):
            pass
        events = []
        originals = (M.verify_pre_payload_authorities, M.consume_attempt,
                     M.P.R.validate_authorities, M.P._selected_payload,
                     M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            M.RESULT = root / "m1656_result"
            M.ATTEMPT = root / ".m1656_attempt"
            M.WORK = root / ".m1656_work"
            M.FAILURE = root / "m1656_failure"
            M.verify_pre_payload_authorities = lambda **_kw: (
                events.append("authority") or {"release_sha256": "1" * 64})
            M.consume_attempt = lambda *_args: events.append("attempt")
            M.P.R.validate_authorities = lambda *_args: events.append(
                "predecessor")
            def stop():
                events.append("payload_select")
                raise StopBeforePayloadOpen()
            M.P._selected_payload = stop
            try:
                with self.assertRaises(StopBeforePayloadOpen):
                    M._run_authorized_actual_prefix()
            finally:
                (M.verify_pre_payload_authorities, M.consume_attempt,
                 M.P.R.validate_authorities, M.P._selected_payload,
                 M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE) = originals
        self.assertEqual(events,
                         ["authority", "attempt", "predecessor",
                          "payload_select"])

    def test_11_cli_and_public_surface_do_not_execute_private_runner(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "_run_authorized_actual_prefix":
                    calls.append(node.lineno)
        self.assertEqual(calls, [])

    def test_12_source_contains_no_payload_or_rss_before_strong_gate(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and
                        node.name == "_run_authorized_actual_prefix")
        first = function.body[1] if (isinstance(function.body[0], ast.Expr)
                                    and isinstance(function.body[0].value,
                                                   ast.Str)) else function.body[0]
        self.assertIsInstance(first, ast.Assign)
        self.assertIsInstance(first.value, ast.Call)
        self.assertEqual(first.value.func.id,
                         "verify_pre_payload_authorities")


if __name__ == "__main__":
    unittest.main(verbosity=2)
