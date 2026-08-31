#!/usr/bin/env python3
"""Independent source-only hammer for M1253.

The acceptance probes intentionally demonstrate semantic receipt mutations that
M1253 currently accepts.  A passing hammer therefore supports a BLOCK verdict;
it is not production authorization.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
SOURCE = ROOT / "scripts/run_m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"
AUTHOR_TEST = ROOT / "tests/test_run_m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"
CONTRACT = ROOT / "contracts/m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor_source_contract_r1_20260830.json"
M1248_TEST = ROOT / "tests/test_run_m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source.py"

EXPECTED = {
    SOURCE: "491b4b9bfe2d268b184d538ca99b8811f962a39acc8a7947a627735f63f1fd30",
    AUTHOR_TEST: "55ffe2a8b8df9c452ada84b8dd06c6abaf1ba576a5d92a99dfc4d2ea1e0c0d0f",
    CONTRACT: "06c951085df50bf1776e84e2cadcd43a347070ac181bcf337d9b1c23e342262b",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1255_m1253_source", SOURCE)
T = load("m1255_m1253_author_tests", AUTHOR_TEST)


class Hammer(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = T.M1253ReleaseSuccessorTest(methodName="test_01_import_is_inert")
        self.fx.setUp()

    def tearDown(self) -> None:
        self.fx.tearDown()

    def reseal(self) -> None:
        manifest = self.fx.output / M.MANIFEST
        manifest.write_text("".join(
            "{}  {}\n".format(sha(self.fx.output / name), name)
            for name in sorted(M.RESULT_PAYLOADS)), encoding="utf-8")
        (self.fx.output / M.OUTER).write_text(
            "{}  {}\n".format(sha(manifest), M.MANIFEST), encoding="utf-8")

    def result_json(self) -> dict:
        return json.loads((self.fx.output / "final_checkpoint_selection.json").read_text())

    def write_result(self, result: dict) -> None:
        (self.fx.output / "final_checkpoint_selection.json").write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")

    def test_01_exact_reviewed_identity_and_pins(self) -> None:
        self.assertEqual({path: sha(path) for path in EXPECTED}, EXPECTED)
        contract = json.loads(CONTRACT.read_text())
        self.assertEqual(contract["source"]["sha256"], EXPECTED[SOURCE])
        self.assertEqual(contract["test"]["sha256"], EXPECTED[AUTHOR_TEST])
        self.assertEqual(sha(REPO / M.DOCS359_REL), M.DOCS359_SHA256)
        for rel, expected in {**M.M1248_PINS, **M.EXECUTION_PINS,
                              **M.M1241_AUX_PINS}.items():
            self.assertEqual(sha(REPO / rel), expected)

    def test_02_three_memfds_are_fully_sealed_and_are_the_only_passed_fds(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(tuple(int(value) for value in prepared.command[-4:-1]),
                             prepared.source_fds)
            required = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            for descriptor in prepared.source_fds:
                self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required,
                                 required)
                with self.assertRaises(PermissionError):
                    os.write(descriptor, b"mutation")
        finally:
            prepared.close()

    def test_03_eleven_snapshots_capture_mode_but_receipt_binding_drops_it(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            for snapshot in prepared.snapshots.values():
                self.assertIsInstance(snapshot.mode, int)
                self.assertNotIn("mode", snapshot.receipt_identity())
        finally:
            prepared.close()

    def test_04_post_prepare_permission_mode_drift_is_accepted(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            key = "legacy_ep29:checkpoint"
            target = Path(prepared.snapshots[key].absolute_path)
            os.chmod(target, 0o600)
            self.assertNotEqual(target.lstat().st_mode, prepared.snapshots[key].mode)
            self.fx.publish(prepared)
            # This acceptance is the finding: mode is not rebound to the receipt.
            M.verify_receipt(self.fx.output, prepared)
        finally:
            prepared.close()

    def test_05_top_level_extra_false_and_positive_claims_are_accepted(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.fx.publish(prepared)
            result = self.result_json()
            result["paper_metric"] = False
            result["hardware_speedup"] = True
            self.write_result(result)
            self.reseal()
            # Exact claim_boundary validation does not close the result root.
            M.verify_receipt(self.fx.output, prepared)
        finally:
            prepared.close()

    def test_06_nested_candidate_identity_receipt_splice_is_accepted(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.fx.publish(prepared)
            result = self.result_json()
            result["candidate_population"][0]["checkpoint"]["paper_metric"] = False
            self.write_result(result)
            self.reseal()
            # _exact_identity checks required fields but accepts unknown fields.
            M.verify_receipt(self.fx.output, prepared)
        finally:
            prepared.close()

    def test_07_rebind_target_payload_and_result_can_be_spliced_together(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.fx.publish(prepared)
            result = self.result_json()
            spliced = [{"id": "E0", "target": "spliced", "state_after_selection": "DONE"}]
            result["e0_e8_activation_dependent_invalidation_and_rebind_targets"] = spliced
            self.write_result(result)
            (self.fx.output / "e0_e8_activation_rebind_targets.json").write_text(
                json.dumps(spliced) + "\n", encoding="utf-8")
            self.reseal()
            # Equality between two receipt members is not authority binding.
            M.verify_receipt(self.fx.output, prepared)
        finally:
            prepared.close()

    def test_08_candidate_pair_and_order_mutations_fail(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.fx.publish(prepared, pair_override=("legacy_ep29", 34))
            with self.assertRaises(M.ReleaseError):
                M.verify_receipt(self.fx.output, prepared)
        finally:
            prepared.close()

    def test_09_nonminimum_selected_projection_mutation_fails(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.fx.publish(prepared, selected_override={
                "candidate_id": "resume_ep30", "epoch": 30})
            with self.assertRaisesRegex(M.ReleaseError, "exact minimum-AEE"):
                M.verify_receipt(self.fx.output, prepared)
        finally:
            prepared.close()

    def test_10_exact_claim_boundary_itself_rejects_extra_false_claim(self) -> None:
        prepared = M.prepare(self.fx.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.fx.publish(prepared, claim_override={"paper_metric": False})
            with self.assertRaisesRegex(M.ReleaseError, "exact claim boundary"):
                M.verify_receipt(self.fx.output, prepared)
        finally:
            prepared.close()

    def test_11_o_excl_attempt_and_no_retry_controls_hold(self) -> None:
        calls = []

        def failed(command, cwd, pass_fds):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 17, stdout="", stderr="fail")

        with self.assertRaises(M.ReleaseError):
            M.execute_once(self.fx.policy, self.fx.interpreter, "3.10.20",
                           self.fx.repo, failed)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.fx.attempt.exists())
        self.assertTrue(self.fx.log.exists())
        with self.assertRaises(M.ReleaseError):
            M.execute_once(self.fx.policy, self.fx.interpreter, "3.10.20",
                           self.fx.repo, failed)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Hammer)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "schema": "m1255_m1253_independent_hammer_output_r1_v1",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "controls_passed": result.wasSuccessful(),
        "interpretation": (
            "BLOCK: tests 03-07 prove fail-open identity/receipt acceptance; "
            "passing tests do not authorize production execution"),
    }, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
