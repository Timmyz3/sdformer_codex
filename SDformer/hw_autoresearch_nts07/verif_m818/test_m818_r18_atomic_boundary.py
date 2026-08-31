#!/usr/bin/env python3
"""Source-only adversarial tests for M818 atomic publication boundaries."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
GUARD_PATH = HERE / "m818_c2_r18_atomic_guard.py"


def load_guard():
    spec = importlib.util.spec_from_file_location("m818_atomic_guard", GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M818 guard")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M818 = load_guard()


def sha_fields():
    return {
        "runner_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
        "candidate_sha256": "3" * 64,
        "release_sha256": "4" * 64,
        "final_hammer_outer_seal_sha256": "5" * 64,
    }


def failure_metadata(phase):
    value = {
        "schema": "m818_c2_r18_failure_receipt_v1",
        "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE_PERFORMANCE",
        "phase": phase,
        "return_code": 37,
        "attempt_consumed": phase == "POST_STAGE_INJECTED",
        "claim_boundary": {
            "failure_boundary_citable": True,
            "paper_performance_citable": False,
            "vcs_complete": False,
            "system_speedup": False,
        },
    }
    value.update(sha_fields())
    return value


class M818AtomicBoundaryTests(unittest.TestCase):
    def test_duplicate_authority_keys_rejected_at_all_depths(self):
        samples = [
            '{"status":"A","status":"B"}\n',
            '{"authorization":{"launch_now":true,"launch_now":false}}\n',
            '{"source_binding":{"runner_sha256":"a","runner_sha256":"b"}}\n',
        ]
        with tempfile.TemporaryDirectory(prefix="m818_dup.") as raw:
            root = Path(raw)
            for index, payload in enumerate(samples):
                path = root / "duplicate_{}.json".format(index)
                path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(M818.Failure, "duplicate JSON"):
                    M818.strict_json(path)

    def test_post_precheck_result_collision_is_no_replace_and_unpolluted(self):
        with tempfile.TemporaryDirectory(prefix="m818_result_collision.") as raw:
            root = Path(raw)
            stage = root / ".result.stage"
            stage.mkdir()
            (stage / "RUN_COMPLETE.txt").write_text("pending hammer\n",
                                                     encoding="utf-8")
            M818.seal_directory(stage)
            before = M818.verify_sealed_directory(stage)
            result = root / "result"
            result.mkdir()
            (result / "attacker").write_text("preserve\n", encoding="utf-8")
            with self.assertRaisesRegex(M818.Failure,
                                        "destination collision"):
                M818.publish_noreplace(stage, result)
            self.assertEqual((result / "attacker").read_text(encoding="utf-8"),
                             "preserve\n")
            self.assertFalse((result / stage.name).exists())
            self.assertEqual(M818.verify_sealed_directory(stage), before)

    def test_attempt_is_flat_double_sealed_and_collision_does_not_pollute(self):
        with tempfile.TemporaryDirectory(prefix="m818_attempt.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            identity = M818.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M818.create_attempt_stage(stage, identity)
            exact = {"attempt.json", "SHA256SUMS",
                     "SHA256SUMS.seal.sha256"}
            M818.verify_sealed_directory(stage, exact)
            attempt = root / "attempt"
            attempt.mkdir()
            (attempt / "attacker").write_text("immutable\n", encoding="utf-8")
            with self.assertRaisesRegex(M818.Failure,
                                        "destination collision"):
                M818.publish_noreplace(stage, attempt, exact)
            self.assertEqual({p.name for p in attempt.iterdir()}, {"attacker"})
            self.assertFalse((attempt / stage.name).exists())
            M818.verify_sealed_directory(stage, exact)

            clean = root / "clean_attempt"
            M818.publish_noreplace(stage, clean, exact)
            M818.verify_attempt(clean, identity)
            self.assertEqual({p.name for p in clean.iterdir()}, exact)

    def test_pre_and_post_stage_failures_publish_sealed_nonpaper_receipts(self):
        with tempfile.TemporaryDirectory(prefix="m818_failures.") as raw:
            root = Path(raw)
            for phase in ("PRE_STAGE_INJECTED", "POST_STAGE_INJECTED"):
                primary = "failure_{}".format(phase.lower())
                result = M818.write_failure_quarantine(
                    root, primary, failure_metadata(phase))
                path = Path(result["path"])
                self.assertEqual({p.name for p in path.iterdir()}, {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256",
                })
                M818.verify_sealed_directory(path, {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256",
                })
                receipt = M818.strict_json(path / "failure.json")
                self.assertTrue(receipt["claim_boundary"][
                    "failure_boundary_citable"])
                self.assertFalse(receipt["claim_boundary"][
                    "paper_performance_citable"])

    def test_failure_destination_collision_preserves_attacker_and_uses_fallback(self):
        with tempfile.TemporaryDirectory(prefix="m818_failure_collision.") as raw:
            root = Path(raw)
            primary = root / "failure_primary"
            primary.mkdir()
            (primary / "attacker").write_text("do not overwrite\n",
                                               encoding="utf-8")
            result = M818.write_failure_quarantine(
                root, primary.name, failure_metadata("PRE_STAGE_COLLISION"))
            self.assertGreaterEqual(result["collision_count"], 1)
            self.assertNotEqual(Path(result["path"]), primary)
            self.assertEqual((primary / "attacker").read_text(encoding="utf-8"),
                             "do not overwrite\n")
            self.assertFalse(any(p.name.startswith(".m818_failure_stage")
                                 for p in primary.iterdir()))
            M818.verify_sealed_directory(Path(result["path"]), {
                "failure.json", "driver.log", "SHA256SUMS",
                "SHA256SUMS.seal.sha256",
            })

    def test_guard_self_test(self):
        result = M818.self_test()
        self.assertEqual(result["status"],
                         "PASS_M818_ATOMIC_GUARD_SELF_TEST")
        self.assertTrue(result["duplicate_json_rejected"])
        self.assertTrue(result["renameat2_collision_rejected"])
        self.assertTrue(result["canonical_unpolluted"])

    def test_prepublication_failure_is_not_consumed(self):
        with tempfile.TemporaryDirectory(prefix="m818_prepublication.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            identity = M818.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M818.create_attempt_stage(stage, identity)
            state = M818.attempt_publication_state(
                root / "attempt", stage, identity,
                "ATTEMPT_ATOMIC_PUBLISH", False)
            self.assertFalse(state["attempt_consumed"])
            self.assertTrue(state["stage_exists"])
            self.assertFalse(state["canonical_exists"])
            self.assertEqual(state["authority"],
                             "NO_DURABLE_RENAME_EVIDENCE")

    def test_postrename_postverify_failure_is_consumed(self):
        with tempfile.TemporaryDirectory(prefix="m818_postrename.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            destination = root / "attempt"
            identity = M818.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M818.create_attempt_stage(stage, identity)
            published = M818.publish_attempt_noreplace(
                stage, destination, identity)
            self.assertTrue(published["rename_succeeded"])
            self.assertFalse(stage.exists())
            # Simulate a guard/process failure before the shell can assign its
            # in-memory flag.  The canonical exact identity is authoritative.
            state = M818.attempt_publication_state(
                destination, stage, identity,
                "ATTEMPT_POST_PUBLISH_VERIFY", False)
            self.assertTrue(state["attempt_consumed"])
            self.assertTrue(state["canonical_identity_verified"])
            self.assertEqual(state["authority"],
                             "CANONICAL_EXACT_IDENTITY")
            # Even if post-rename damage makes exact verification fail, the
            # moved-away stage plus occupied canonical path preserves the
            # conservative consumed fact during this explicit phase.
            (destination / "attempt.json").write_text("damaged\n",
                                                       encoding="utf-8")
            damaged = M818.attempt_publication_state(
                destination, stage, identity,
                "ATTEMPT_POST_PUBLISH_VERIFY", False)
            self.assertTrue(damaged["attempt_consumed"])
            self.assertFalse(damaged["canonical_identity_verified"])
            self.assertEqual(damaged["authority"],
                "CANONICAL_PRESENT_STAGE_MOVED_DURING_PUBLICATION_PHASE")

    def test_prepublication_collision_remains_not_consumed(self):
        with tempfile.TemporaryDirectory(prefix="m818_precollision.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            destination = root / "attempt"
            identity = M818.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M818.create_attempt_stage(stage, identity)
            destination.mkdir()
            (destination / "attacker").write_text("preserve\n",
                                                   encoding="utf-8")
            with self.assertRaisesRegex(M818.Failure,
                                        "destination collision"):
                M818.publish_attempt_noreplace(stage, destination, identity)
            state = M818.attempt_publication_state(
                destination, stage, identity,
                "ATTEMPT_ATOMIC_PUBLISH", False)
            self.assertFalse(state["attempt_consumed"])
            self.assertTrue(state["stage_exists"])
            self.assertFalse(state["canonical_identity_verified"])
            self.assertEqual((destination / "attacker").read_text(
                encoding="utf-8"), "preserve\n")

    def test_cli_failure_receipts_classify_pre_false_and_post_true(self):
        with tempfile.TemporaryDirectory(prefix="m818_cli_classify.") as raw:
            root = Path(raw)
            common = [
                "--runner-sha256", "1" * 64,
                "--contract-sha256", "2" * 64,
                "--candidate-sha256", "3" * 64,
                "--release-sha256", "4" * 64,
                "--final-hammer-outer-seal-sha256", "5" * 64,
            ]

            pre_stage = root / ".pre.stage"
            identity = M818.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M818.create_attempt_stage(pre_stage, identity)
            pre = subprocess.run([
                sys.executable, str(GUARD_PATH), "write-failure-quarantine",
                "--parent", str(root), "--primary-name", "pre_failure",
                "--phase", "ATTEMPT_ATOMIC_PUBLISH", "--return-code", "37",
                "--shell-published", "false", "--attempt-path",
                str(root / "pre_attempt"), "--attempt-stage", str(pre_stage),
            ] + common, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True)
            self.assertEqual(pre.returncode, 0, pre.stderr)
            pre_result = json.loads(pre.stdout)
            pre_receipt = M818.strict_json(
                Path(pre_result["path"]) / "failure.json")
            self.assertFalse(pre_receipt["attempt_consumed"])
            self.assertEqual(pre_receipt["attempt_publication"]["authority"],
                             "NO_DURABLE_RENAME_EVIDENCE")

            post_stage = root / ".post.stage"
            post_attempt = root / "post_attempt"
            M818.create_attempt_stage(post_stage, identity)
            M818.publish_attempt_noreplace(post_stage, post_attempt, identity)
            (post_attempt / "attempt.json").write_text(
                "injected post-rename damage\n", encoding="utf-8")
            post = subprocess.run([
                sys.executable, str(GUARD_PATH), "write-failure-quarantine",
                "--parent", str(root), "--primary-name", "post_failure",
                "--phase", "ATTEMPT_POST_PUBLISH_VERIFY",
                "--return-code", "38", "--shell-published", "false",
                "--attempt-path", str(post_attempt), "--attempt-stage",
                str(post_stage),
            ] + common, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True)
            self.assertEqual(post.returncode, 0, post.stderr)
            post_result = json.loads(post.stdout)
            post_receipt = M818.strict_json(
                Path(post_result["path"]) / "failure.json")
            self.assertTrue(post_receipt["attempt_consumed"])
            self.assertFalse(post_receipt["attempt_publication"][
                "canonical_identity_verified"])
            self.assertEqual(post_receipt["attempt_publication"]["authority"],
                "CANONICAL_PRESENT_STAGE_MOVED_DURING_PUBLICATION_PHASE")


if __name__ == "__main__":
    unittest.main()
