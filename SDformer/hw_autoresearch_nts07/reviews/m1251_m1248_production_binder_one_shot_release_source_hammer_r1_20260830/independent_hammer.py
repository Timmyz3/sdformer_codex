#!/usr/bin/env python3
"""Independent M1251 hammer for the exact M1248 source-only release.

All mutations run in temporary fixtures.  A passing test means the hammer
observed the documented behavior; tests whose names contain ``gap`` prove a
release-blocking acceptance gap and therefore do not authorize production.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "scripts/run_m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source.py"
BASE_TEST = HW / "tests/test_run_m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source.py"
CONTRACT = HW / "contracts/m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source_contract_r1_20260830.json"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1251_exact_m1248", SOURCE)
T = load("m1251_m1248_base_tests", BASE_TEST)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __enter__(self):
        self.case = T.M1248ReleaseSourceTest("test_01_import_is_inert")
        self.case.setUp()
        return self.case

    def __exit__(self, exc_type, exc, traceback):
        self.case.tearDown()


def reseal_result(case) -> None:
    manifest = case.output / M.MANIFEST
    payloads = sorted(M.RESULT_PAYLOADS)
    manifest.write_text("".join(
        "{}  {}\n".format(sha(case.output / name), name) for name in payloads),
        encoding="utf-8")
    (case.output / M.OUTER).write_text(
        "{}  {}\n".format(sha(manifest), M.MANIFEST), encoding="utf-8")


class M1251IndependentHammer(unittest.TestCase):
    def test_01_exact_source_test_contract_and_dependency_pins(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["source"]["sha256"], sha(SOURCE))
        self.assertEqual(contract["test"]["sha256"], sha(BASE_TEST))
        for relative, expected in M.M1241_PINS.items():
            self.assertEqual(sha(HW.parent / relative), expected)
        self.assertEqual(sha(HW / "docs/359_DATE终局冻结_20260813.md"),
                         M.DOCS359_SHA256)

    def test_02_exact_four_profiles_four_checkpoints_two_configs_manifest(self):
        with Fixture() as case:
            paths = M.artifact_files(case.policy)
            self.assertEqual(len(paths), 11)
            self.assertEqual(len(set(paths)), 11)
            self.assertEqual(sum(path.name == "spike_profile.json" for path in paths), 4)
            self.assertEqual(sum(path.name.startswith("checkpoint_epoch") for path in paths), 4)
            self.assertEqual(sum(path.suffix == ".yml" for path in paths), 2)
            self.assertEqual(sum(path.suffix == ".json" and
                                 path.name != "spike_profile.json" for path in paths), 1)
            M.preflight(case.policy, case.interpreter, "3.10.20", case.repo)
            self.assertFalse(case.attempt.exists())

    def test_03_attempt_namespace_race_is_stopped_by_o_excl(self):
        with Fixture() as case:
            command = M.preflight(case.policy, case.interpreter, "3.10.20", case.repo)
            case.attempt.write_text("racer\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                M.consume_attempt(case.policy, command)
            self.assertEqual(case.attempt.read_text(encoding="utf-8"), "racer\n")

    def test_04_output_race_fails_closed_and_preserves_attempt(self):
        with Fixture() as case:
            def raced_output(command, cwd):
                case.output.mkdir()
                return subprocess.CompletedProcess(
                    command, 0, stdout=M.CHILD_TOKEN + "\n", stderr="")
            with self.assertRaises(M.ReleaseError):
                M.execute_once(case.policy, case.interpreter, "3.10.20", case.repo,
                               raced_output)
            self.assertTrue(case.attempt.is_file())
            self.assertTrue(case.log.is_file())

    def test_05_log_race_fails_closed_and_preserves_attempt(self):
        with Fixture() as case:
            def raced_log(command, cwd):
                case.publish_result()
                case.log.write_text("racer\n", encoding="utf-8")
                return subprocess.CompletedProcess(
                    command, 0, stdout=M.CHILD_TOKEN + "\n", stderr="")
            with self.assertRaises(FileExistsError):
                M.execute_once(case.policy, case.interpreter, "3.10.20", case.repo,
                               raced_log)
            self.assertTrue(case.attempt.is_file())
            self.assertEqual(case.log.read_text(encoding="utf-8"), "racer\n")

    def test_06_gap_m1241_source_can_drift_after_preflight_and_still_accept(self):
        with Fixture() as case:
            source = case.repo / M.M1241_REL
            original = source.read_bytes()
            def drift_after_attempt(command, cwd):
                self.assertTrue(case.attempt.is_file())
                source.write_bytes(original + b"post-preflight drift\n")
                case.publish_result()
                return subprocess.CompletedProcess(
                    command, 0, stdout=M.CHILD_TOKEN + "\n", stderr="")
            completed = M.execute_once(
                case.policy, case.interpreter, "3.10.20", case.repo,
                drift_after_attempt)
            self.assertEqual(completed.returncode, 0)
            self.assertNotEqual(sha(source), case.policy.m1241_pins[M.M1241_REL])

    def test_07_gap_candidate_can_drift_after_preflight_and_receipt_still_accept(self):
        with Fixture() as case:
            target = M.artifact_files(case.policy)[-2]
            before = sha(target)
            def drift_after_attempt(command, cwd):
                self.assertTrue(case.attempt.is_file())
                target.write_bytes(target.read_bytes() + b"post-preflight drift\n")
                case.publish_result()
                return subprocess.CompletedProcess(
                    command, 0, stdout=M.CHILD_TOKEN + "\n", stderr="")
            M.execute_once(case.policy, case.interpreter, "3.10.20", case.repo,
                           drift_after_attempt)
            self.assertNotEqual(sha(target), before)

    def test_08_gap_extra_overauthorization_in_claim_boundary_is_accepted(self):
        with Fixture() as case:
            case.publish_result(boundary_override={
                "power_or_energy": True,
                "paper_metric": True,
                "hardware_replay_complete": True,
            })
            result = M.verify_selection_receipt(case.output)
            self.assertTrue(result["claim_boundary"]["power_or_energy"])
            self.assertTrue(result["claim_boundary"]["paper_metric"])
            self.assertTrue(result["claim_boundary"]["hardware_replay_complete"])

    def test_09_gap_mismatched_candidate_epoch_pair_is_accepted(self):
        with Fixture() as case:
            case.publish_result()
            path = case.output / "final_checkpoint_selection.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["selected"] = {"candidate_id": "legacy_ep29", "epoch": 34}
            path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
            reseal_result(case)
            accepted = M.verify_selection_receipt(case.output)
            self.assertEqual(accepted["selected"],
                             {"candidate_id": "legacy_ep29", "epoch": 34})

    def test_10_schema_status_and_required_denials_remain_fail_closed(self):
        attacks = (
            ("schema", "wrong_schema"),
            ("status", "wrong_status"),
        )
        for key, value in attacks:
            with self.subTest(key=key), Fixture() as case:
                case.publish_result()
                path = case.output / "final_checkpoint_selection.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                result[key] = value
                path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
                reseal_result(case)
                with self.assertRaisesRegex(M.ReleaseError, "schema/status"):
                    M.verify_selection_receipt(case.output)
        for key in ("fresh_result_hammer_required", "hardware_rebind_authorized",
                    "hardware_speedup", "system_speedup"):
            with self.subTest(key=key), Fixture() as case:
                bad = False if key == "fresh_result_hammer_required" else True
                case.publish_result(boundary_override={key: bad})
                with self.assertRaisesRegex(M.ReleaseError, "claim boundary"):
                    M.verify_selection_receipt(case.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
