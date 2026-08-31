#!/usr/bin/env python3
"""Source-only adversarial tests for M813 atomic publication boundaries."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
GUARD_PATH = HERE / "m813_c2_r17_atomic_guard.py"


def load_guard():
    spec = importlib.util.spec_from_file_location("m813_atomic_guard", GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M813 guard")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M813 = load_guard()


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
        "schema": "m813_c2_r17_failure_receipt_v1",
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


class M813AtomicBoundaryTests(unittest.TestCase):
    def test_duplicate_authority_keys_rejected_at_all_depths(self):
        samples = [
            '{"status":"A","status":"B"}\n',
            '{"authorization":{"launch_now":true,"launch_now":false}}\n',
            '{"source_binding":{"runner_sha256":"a","runner_sha256":"b"}}\n',
        ]
        with tempfile.TemporaryDirectory(prefix="m813_dup.") as raw:
            root = Path(raw)
            for index, payload in enumerate(samples):
                path = root / "duplicate_{}.json".format(index)
                path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(M813.Failure, "duplicate JSON"):
                    M813.strict_json(path)

    def test_post_precheck_result_collision_is_no_replace_and_unpolluted(self):
        with tempfile.TemporaryDirectory(prefix="m813_result_collision.") as raw:
            root = Path(raw)
            stage = root / ".result.stage"
            stage.mkdir()
            (stage / "RUN_COMPLETE.txt").write_text("pending hammer\n",
                                                     encoding="utf-8")
            M813.seal_directory(stage)
            before = M813.verify_sealed_directory(stage)
            result = root / "result"
            result.mkdir()
            (result / "attacker").write_text("preserve\n", encoding="utf-8")
            with self.assertRaisesRegex(M813.Failure,
                                        "destination collision"):
                M813.publish_noreplace(stage, result)
            self.assertEqual((result / "attacker").read_text(encoding="utf-8"),
                             "preserve\n")
            self.assertFalse((result / stage.name).exists())
            self.assertEqual(M813.verify_sealed_directory(stage), before)

    def test_attempt_is_flat_double_sealed_and_collision_does_not_pollute(self):
        with tempfile.TemporaryDirectory(prefix="m813_attempt.") as raw:
            root = Path(raw)
            stage = root / ".attempt.stage"
            identity = M813.attempt_identity(
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
            M813.create_attempt_stage(stage, identity)
            exact = {"attempt.json", "SHA256SUMS",
                     "SHA256SUMS.seal.sha256"}
            M813.verify_sealed_directory(stage, exact)
            attempt = root / "attempt"
            attempt.mkdir()
            (attempt / "attacker").write_text("immutable\n", encoding="utf-8")
            with self.assertRaisesRegex(M813.Failure,
                                        "destination collision"):
                M813.publish_noreplace(stage, attempt, exact)
            self.assertEqual({p.name for p in attempt.iterdir()}, {"attacker"})
            self.assertFalse((attempt / stage.name).exists())
            M813.verify_sealed_directory(stage, exact)

            clean = root / "clean_attempt"
            M813.publish_noreplace(stage, clean, exact)
            M813.verify_attempt(clean, identity)
            self.assertEqual({p.name for p in clean.iterdir()}, exact)

    def test_pre_and_post_stage_failures_publish_sealed_nonpaper_receipts(self):
        with tempfile.TemporaryDirectory(prefix="m813_failures.") as raw:
            root = Path(raw)
            for phase in ("PRE_STAGE_INJECTED", "POST_STAGE_INJECTED"):
                primary = "failure_{}".format(phase.lower())
                result = M813.write_failure_quarantine(
                    root, primary, failure_metadata(phase))
                path = Path(result["path"])
                self.assertEqual({p.name for p in path.iterdir()}, {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256",
                })
                M813.verify_sealed_directory(path, {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256",
                })
                receipt = M813.strict_json(path / "failure.json")
                self.assertTrue(receipt["claim_boundary"][
                    "failure_boundary_citable"])
                self.assertFalse(receipt["claim_boundary"][
                    "paper_performance_citable"])

    def test_failure_destination_collision_preserves_attacker_and_uses_fallback(self):
        with tempfile.TemporaryDirectory(prefix="m813_failure_collision.") as raw:
            root = Path(raw)
            primary = root / "failure_primary"
            primary.mkdir()
            (primary / "attacker").write_text("do not overwrite\n",
                                               encoding="utf-8")
            result = M813.write_failure_quarantine(
                root, primary.name, failure_metadata("PRE_STAGE_COLLISION"))
            self.assertGreaterEqual(result["collision_count"], 1)
            self.assertNotEqual(Path(result["path"]), primary)
            self.assertEqual((primary / "attacker").read_text(encoding="utf-8"),
                             "do not overwrite\n")
            self.assertFalse(any(p.name.startswith(".m813_failure_stage")
                                 for p in primary.iterdir()))
            M813.verify_sealed_directory(Path(result["path"]), {
                "failure.json", "driver.log", "SHA256SUMS",
                "SHA256SUMS.seal.sha256",
            })

    def test_guard_self_test(self):
        result = M813.self_test()
        self.assertEqual(result["status"],
                         "PASS_M813_ATOMIC_GUARD_SELF_TEST")
        self.assertTrue(result["duplicate_json_rejected"])
        self.assertTrue(result["renameat2_collision_rejected"])
        self.assertTrue(result["canonical_unpolluted"])


if __name__ == "__main__":
    unittest.main()
