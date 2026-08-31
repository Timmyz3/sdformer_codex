#!/usr/bin/env python3
"""Source-only adversarial tests for M826 atomic publication boundaries."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
GUARD_PATH = HERE / "m826_c2_r20_atomic_guard.py"


def load_guard():
    spec = importlib.util.spec_from_file_location("m826_atomic_guard", GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M826 guard")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M826 = load_guard()


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
        "schema": "m826_c2_r20_failure_receipt_v1",
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


class M826AtomicBoundaryTests(unittest.TestCase):
    def test_duplicate_authority_keys_rejected_at_all_depths(self):
        samples = [
            '{"status":"A","status":"B"}\n',
            '{"authorization":{"launch_now":true,"launch_now":false}}\n',
            '{"source_binding":{"runner_sha256":"a","runner_sha256":"b"}}\n',
        ]
        with tempfile.TemporaryDirectory(prefix="m826_dup.") as raw:
            root = Path(raw)
            for index, payload in enumerate(samples):
                path = root / "duplicate_{}.json".format(index)
                path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(M826.Failure, "duplicate JSON"):
                    M826.strict_json(path)

    def test_nonfinite_json_constants_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m826_nonfinite.") as raw:
            root = Path(raw)
            for name, payload in (
                    ("nan", '{"value":NaN}\n'),
                    ("infinity", '{"value":Infinity}\n'),
                    ("negative_infinity", '{"value":-Infinity}\n')):
                path = root / (name + ".json")
                path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(M826.Failure, "non-finite JSON"):
                    M826.strict_json(path)

    def test_post_precheck_result_collision_is_no_replace_and_unpolluted(self):
        with tempfile.TemporaryDirectory(prefix="m826_result_collision.") as raw:
            root = Path(raw)
            stage = root / ".result.stage"
            stage.mkdir()
            (stage / "RUN_COMPLETE.txt").write_text("pending hammer\n",
                                                     encoding="utf-8")
            M826.seal_directory(stage)
            before = M826.verify_sealed_directory(stage)
            result = root / "result"
            result.mkdir()
            (result / "attacker").write_text("preserve\n", encoding="utf-8")
            with self.assertRaisesRegex(M826.Failure,
                                        "destination collision"):
                M826.publish_noreplace(stage, result)
            self.assertEqual((result / "attacker").read_text(encoding="utf-8"),
                             "preserve\n")
            self.assertFalse((result / stage.name).exists())
            self.assertEqual(M826.verify_sealed_directory(stage), before)

    def test_attempt_is_flat_double_sealed_and_collision_does_not_pollute(self):
        with tempfile.TemporaryDirectory(prefix="m826_attempt.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            identity = M826.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M826.create_attempt_stage(stage, identity)
            exact = {"attempt.json", "SHA256SUMS",
                     "SHA256SUMS.seal.sha256"}
            M826.verify_sealed_directory(stage, exact)
            attempt = root / "attempt"
            attempt.mkdir()
            (attempt / "attacker").write_text("immutable\n", encoding="utf-8")
            with self.assertRaisesRegex(M826.Failure,
                                        "destination collision"):
                M826.publish_noreplace(stage, attempt, exact)
            self.assertEqual({p.name for p in attempt.iterdir()}, {"attacker"})
            self.assertFalse((attempt / stage.name).exists())
            M826.verify_sealed_directory(stage, exact)

            clean = root / "clean_attempt"
            M826.publish_noreplace(stage, clean, exact)
            M826.verify_attempt(clean, identity)
            self.assertEqual({p.name for p in clean.iterdir()}, exact)

    def test_pre_and_post_stage_failures_publish_sealed_nonpaper_receipts(self):
        with tempfile.TemporaryDirectory(prefix="m826_failures.") as raw:
            root = Path(raw)
            for phase in ("PRE_STAGE_INJECTED", "POST_STAGE_INJECTED"):
                primary = "failure_{}".format(phase.lower())
                result = M826.write_failure_quarantine(
                    root, primary, failure_metadata(phase))
                path = Path(result["path"])
                self.assertEqual({p.name for p in path.iterdir()}, {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256",
                })
                M826.verify_sealed_directory(path, {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256",
                })
                receipt = M826.strict_json(path / "failure.json")
                self.assertTrue(receipt["claim_boundary"][
                    "failure_boundary_citable"])
                self.assertFalse(receipt["claim_boundary"][
                    "paper_performance_citable"])

    def test_failure_destination_collision_preserves_attacker_and_uses_fallback(self):
        with tempfile.TemporaryDirectory(prefix="m826_failure_collision.") as raw:
            root = Path(raw)
            primary = root / "failure_primary"
            primary.mkdir()
            (primary / "attacker").write_text("do not overwrite\n",
                                               encoding="utf-8")
            result = M826.write_failure_quarantine(
                root, primary.name, failure_metadata("PRE_STAGE_COLLISION"))
            self.assertGreaterEqual(result["collision_count"], 1)
            self.assertNotEqual(Path(result["path"]), primary)
            self.assertEqual((primary / "attacker").read_text(encoding="utf-8"),
                             "do not overwrite\n")
            self.assertFalse(any(p.name.startswith(".m826_failure_stage")
                                 for p in primary.iterdir()))
            M826.verify_sealed_directory(Path(result["path"]), {
                "failure.json", "driver.log", "SHA256SUMS",
                "SHA256SUMS.seal.sha256",
            })

    def test_guard_self_test(self):
        result = M826.self_test()
        self.assertEqual(result["status"],
                         "PASS_M826_ATOMIC_GUARD_SELF_TEST")
        self.assertTrue(result["duplicate_json_rejected"])
        self.assertTrue(result["renameat2_collision_rejected"])
        self.assertTrue(result["canonical_unpolluted"])

    def test_prepublication_failure_is_not_consumed(self):
        with tempfile.TemporaryDirectory(prefix="m826_prepublication.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            identity = M826.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M826.create_attempt_stage(stage, identity)
            state = M826.attempt_publication_state(
                root / "attempt", stage, identity,
                "ATTEMPT_ATOMIC_PUBLISH", False)
            self.assertFalse(state["attempt_consumed"])
            self.assertTrue(state["stage_exists"])
            self.assertFalse(state["canonical_exists"])
            self.assertEqual(state["authority"],
                             "NO_DURABLE_RENAME_EVIDENCE")

    def test_postrename_postverify_failure_is_consumed(self):
        with tempfile.TemporaryDirectory(prefix="m826_postrename.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            destination = root / "attempt"
            identity = M826.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M826.create_attempt_stage(stage, identity)
            published = M826.publish_attempt_noreplace(
                stage, destination, identity)
            self.assertTrue(published["rename_succeeded"])
            self.assertFalse(stage.exists())
            # Simulate a guard/process failure before the shell can assign its
            # in-memory flag.  The canonical exact identity is authoritative.
            state = M826.attempt_publication_state(
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
            damaged = M826.attempt_publication_state(
                destination, stage, identity,
                "ATTEMPT_POST_PUBLISH_VERIFY", False)
            self.assertTrue(damaged["attempt_consumed"])
            self.assertFalse(damaged["canonical_identity_verified"])
            self.assertEqual(damaged["authority"],
                "CANONICAL_PRESENT_STAGE_MOVED_DURING_PUBLICATION_PHASE")

    def test_prepublication_collision_remains_not_consumed(self):
        with tempfile.TemporaryDirectory(prefix="m826_precollision.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            destination = root / "attempt"
            identity = M826.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M826.create_attempt_stage(stage, identity)
            destination.mkdir()
            (destination / "attacker").write_text("preserve\n",
                                                   encoding="utf-8")
            with self.assertRaisesRegex(M826.Failure,
                                        "destination collision"):
                M826.publish_attempt_noreplace(stage, destination, identity)
            state = M826.attempt_publication_state(
                destination, stage, identity,
                "ATTEMPT_ATOMIC_PUBLISH", False)
            self.assertFalse(state["attempt_consumed"])
            self.assertTrue(state["stage_exists"])
            self.assertFalse(state["canonical_identity_verified"])
            self.assertEqual((destination / "attacker").read_text(
                encoding="utf-8"), "preserve\n")

    def test_preexisting_exact_identity_collision_remains_not_consumed(self):
        with tempfile.TemporaryDirectory(prefix="m826_exact_collision.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            prior_stage = root / ".prior.stage"
            destination = root / "attempt"
            identity = M826.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M826.create_attempt_stage(stage, identity)
            M826.create_attempt_stage(prior_stage, identity)
            M826.publish_attempt_noreplace(prior_stage, destination, identity)
            stage_before = M826.verify_attempt(stage, identity)
            destination_before = M826.verify_attempt(destination, identity)
            with self.assertRaisesRegex(M826.Failure,
                                        "destination collision"):
                M826.publish_attempt_noreplace(stage, destination, identity)
            state = M826.attempt_publication_state(
                destination, stage, identity,
                "ATTEMPT_ATOMIC_PUBLISH", False)
            self.assertFalse(state["attempt_consumed"])
            self.assertTrue(state["canonical_identity_verified"])
            self.assertTrue(state["stage_exists"])
            self.assertEqual(state["authority"],
                             "NO_DURABLE_RENAME_EVIDENCE")
            self.assertEqual(M826.verify_attempt(stage, identity), stage_before)
            self.assertEqual(M826.verify_attempt(destination, identity),
                             destination_before)

    def test_cli_four_failure_receipts_classify_collision_and_publish_boundary(self):
        with tempfile.TemporaryDirectory(prefix="m826_cli_classify.") as raw:
            root = Path(raw)
            common = [
                "--runner-sha256", "1" * 64,
                "--contract-sha256", "2" * 64,
                "--candidate-sha256", "3" * 64,
                "--release-sha256", "4" * 64,
                "--final-hammer-outer-seal-sha256", "5" * 64,
            ]

            identity = M826.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)

            def receipt(name, phase, attempt, stage, code):
                process = subprocess.run([
                    sys.executable, str(GUARD_PATH),
                    "write-failure-quarantine", "--parent", str(root),
                    "--primary-name", name, "--phase", phase,
                    "--return-code", str(code), "--shell-published", "false",
                    "--attempt-path", str(attempt), "--attempt-stage",
                    str(stage),
                ] + common, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                result = json.loads(process.stdout)
                path = Path(result["path"])
                M826.verify_sealed_directory(path, {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256",
                })
                return M826.strict_json(path / "failure.json")

            # 1. No canonical publication and source stage remains: false.
            pre_stage = root / ".pre.stage"
            M826.create_attempt_stage(pre_stage, identity)
            pre_receipt = receipt("pre_failure", "ATTEMPT_ATOMIC_PUBLISH",
                                  root / "pre_attempt", pre_stage, 37)
            self.assertFalse(pre_receipt["attempt_consumed"])
            self.assertEqual(pre_receipt["attempt_publication"]["authority"],
                             "NO_DURABLE_RENAME_EVIDENCE")

            # 2. A pre-existing exact destination collides and does not clobber
            # either side.  Exact identity alone cannot consume this stage.
            collision_stage = root / ".collision.stage"
            prior_stage = root / ".collision.prior"
            collision_attempt = root / "collision_attempt"
            M826.create_attempt_stage(collision_stage, identity)
            M826.create_attempt_stage(prior_stage, identity)
            M826.publish_attempt_noreplace(
                prior_stage, collision_attempt, identity)
            stage_before = M826.verify_attempt(collision_stage, identity)
            destination_before = M826.verify_attempt(
                collision_attempt, identity)
            with self.assertRaisesRegex(M826.Failure,
                                        "destination collision"):
                M826.publish_attempt_noreplace(
                    collision_stage, collision_attempt, identity)
            collision_receipt = receipt(
                "collision_failure", "ATTEMPT_ATOMIC_PUBLISH",
                collision_attempt, collision_stage, 38)
            self.assertFalse(collision_receipt["attempt_consumed"])
            self.assertTrue(collision_receipt["attempt_publication"][
                "canonical_identity_verified"])
            self.assertEqual(collision_receipt["attempt_publication"][
                "authority"], "NO_DURABLE_RENAME_EVIDENCE")
            self.assertEqual(M826.verify_attempt(collision_stage, identity),
                             stage_before)
            self.assertEqual(M826.verify_attempt(collision_attempt, identity),
                             destination_before)

            # 3. Rename succeeded and exact canonical is intact while the
            # shell latch is still false: moved stage proves true.
            exact_stage = root / ".exact.stage"
            exact_attempt = root / "exact_attempt"
            M826.create_attempt_stage(exact_stage, identity)
            M826.publish_attempt_noreplace(exact_stage, exact_attempt, identity)
            exact_receipt = receipt(
                "exact_failure", "ATTEMPT_POST_PUBLISH_VERIFY",
                exact_attempt, exact_stage, 39)
            self.assertTrue(exact_receipt["attempt_consumed"])
            self.assertTrue(exact_receipt["attempt_publication"][
                "canonical_identity_verified"])
            self.assertEqual(exact_receipt["attempt_publication"]["authority"],
                             "CANONICAL_EXACT_IDENTITY")

            # 4. Rename succeeded, canonical was damaged before postcheck,
            # and stage is gone: durable move evidence remains true.
            damaged_stage = root / ".damaged.stage"
            damaged_attempt = root / "damaged_attempt"
            M826.create_attempt_stage(damaged_stage, identity)
            M826.publish_attempt_noreplace(
                damaged_stage, damaged_attempt, identity)
            (damaged_attempt / "attempt.json").write_text(
                "injected post-rename damage\n", encoding="utf-8")
            damaged_receipt = receipt(
                "damaged_failure", "ATTEMPT_POST_PUBLISH_VERIFY",
                damaged_attempt, damaged_stage, 40)
            self.assertTrue(damaged_receipt["attempt_consumed"])
            self.assertFalse(damaged_receipt["attempt_publication"][
                "canonical_identity_verified"])
            self.assertEqual(damaged_receipt["attempt_publication"]["authority"],
                "CANONICAL_PRESENT_STAGE_MOVED_DURING_PUBLICATION_PHASE")


if __name__ == "__main__":
    unittest.main()
