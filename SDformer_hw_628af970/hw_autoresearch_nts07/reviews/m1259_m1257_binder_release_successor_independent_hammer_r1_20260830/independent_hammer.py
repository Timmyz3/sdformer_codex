#!/usr/bin/python3.12
"""Independent fail-closed hammer for the M1257 binder successor.

Temporary fixtures only.  This program neither invokes the production policy nor
touches remote/GPU/EDA state.  It deliberately reseals mutated fixture receipts
to distinguish schema/policy validation from checksum validation.
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


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"
AUTHOR_TEST = ROOT / "hw_autoresearch_nts07/tests/test_run_m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor_source_contract_r1_20260830.json"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "ce539d625c0583542dd795a0fdfacff2050c4475995b40371ce599109ce001b6"
AUTHOR_TEST_SHA = "2684a84d91cfdc09251d4cec76a10b55ebb811214eba464451994bdb4c179e49"
CONTRACT_SHA = "0a25fe22140a0401d0c13ef37d5ab3d9c16a2f02ab1b9f791d30b4ff013c0a8f"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = load("m1259_m1257_source", SOURCE)
A = load("m1259_m1257_author_fixture", AUTHOR_TEST)


class Hammer(unittest.TestCase):
    def setUp(self):
        self.fx = A.M1257ReleaseSuccessorTest(methodName="test_01_import_is_inert_and_predecessor_is_exact")
        self.fx.setUp()

    def tearDown(self):
        self.fx.tearDown()

    def reset_output(self):
        if self.fx.output.exists():
            for member in self.fx.output.iterdir():
                member.unlink()
            self.fx.output.rmdir()

    def prepared(self):
        return S.prepare(self.fx.policy, self.fx.fx.interpreter, "3.10.20", self.fx.fx.repo)

    def publish(self, prepared):
        return self.fx.publish(prepared)

    def mutate_json(self, name, callback):
        path = self.fx.output / name
        value = json.loads(path.read_text(encoding="utf-8"))
        callback(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def verify_fails(self, prepared, pattern=None):
        if pattern is None:
            with self.assertRaises(S.B.ReleaseError):
                S.verify_receipt(self.fx.output, prepared)
        else:
            with self.assertRaisesRegex(S.B.ReleaseError, pattern):
                S.verify_receipt(self.fx.output, prepared)

    def test_01_reviewed_bytes_and_import_are_inert(self):
        self.assertEqual(sha(SOURCE), SOURCE_SHA)
        self.assertEqual(sha(AUTHOR_TEST), AUTHOR_TEST_SHA)
        self.assertEqual(sha(CONTRACT), CONTRACT_SHA)
        self.assertEqual(sha(DOCS359), DOCS359_SHA)
        self.assertFalse(self.fx.fx.attempt.exists())
        self.assertFalse(self.fx.fx.log.exists())

    def test_02_three_sealed_memfds_and_exact_pass_fds_survive(self):
        prepared = self.prepared()
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(tuple(map(int, prepared.command[-4:-1])), prepared.source_fds)
            required = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            for descriptor in prepared.source_fds:
                self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS), required)
                with self.assertRaises(OSError):
                    os.write(descriptor, b"attack")
        finally:
            prepared.close()

    def test_03_full_mode_chain_and_closed_good_receipt(self):
        prepared = self.prepared()
        try:
            result = self.publish(prepared)
            verified = S.verify_receipt(self.fx.output, prepared)
            self.assertEqual(verified, result)
            self.assertEqual(set(result), S.RESULT_KEYS)
            self.assertEqual(len(result["candidate_population"]), 4)
            for row, candidate in zip(result["candidate_population"], prepared.policy.candidates):
                for field, snapshot_key in (
                    ("checkpoint", candidate.candidate_id + ":checkpoint"),
                    ("configuration", "config:" + candidate.config_key),
                    ("profile", candidate.candidate_id + ":profile"),
                ):
                    identity = row[field]
                    self.assertIn("mode", identity)
                    self.assertIs(type(identity["mode"]), int)
                    self.assertEqual(identity["mode"], prepared.snapshots[snapshot_key].mode)
            self.assertEqual(result["new_run_manifest"]["mode"], prepared.snapshots["manifest"].mode)
            self.assertIn("mode", result["selected"]["checkpoint"])
            selected = json.loads((self.fx.output / "selected_checkpoint_and_config.json").read_text())
            self.assertIn("mode", selected["checkpoint"])
            self.assertIn("mode", selected["configuration"])
            self.assertIn("mode", selected["profile"])
        finally:
            prepared.close()

    def test_04_coordinated_mode_splices_are_rejected(self):
        attacks = ("checkpoint_missing", "config_bool", "profile_changed", "manifest_extra")
        for attack in attacks:
            with self.subTest(attack=attack):
                prepared = self.prepared()
                try:
                    result = self.publish(prepared)
                    winner = result["selected"]["candidate_id"]
                    def change(root):
                        target = next(row for row in root["candidate_population"]
                                      if row["candidate_id"] == winner)
                        if attack == "checkpoint_missing":
                            target["checkpoint"].pop("mode")
                            root["selected"]["checkpoint"].pop("mode")
                        elif attack == "config_bool":
                            target["configuration"]["mode"] = True
                            root["selected"]["configuration"]["mode"] = True
                        elif attack == "profile_changed":
                            target["profile"]["mode"] ^= 0o111
                            root["selected"]["profile"]["mode"] ^= 0o111
                        else:
                            root["new_run_manifest"]["mode_alias"] = False
                    self.mutate_json("final_checkpoint_selection.json", change)
                    if attack != "manifest_extra":
                        def sidecar_change(sidecar):
                            field = attack.split("_", 1)[0]
                            if attack == "checkpoint_missing":
                                sidecar[field].pop("mode")
                            elif attack == "config_bool":
                                sidecar["configuration"]["mode"] = True
                            else:
                                sidecar["profile"]["mode"] ^= 0o111
                        self.mutate_json("selected_checkpoint_and_config.json", sidecar_change)
                    self.fx.reseal()
                    self.verify_fails(prepared)
                finally:
                    prepared.close()
                self.reset_output()

    def test_05_nested_extra_false_and_positive_claims_are_rejected(self):
        attacks = (
            ("checkpoint", "paper_metric", False),
            ("configuration", "hardware_speedup", True),
            ("profile", "power_or_energy", False),
            ("activity", "measured_energy", True),
            ("accuracy_metrics", "paper_metric", False),
            (None, "authority", False),
        )
        for field, key, value in attacks:
            with self.subTest(field=field, key=key, value=value):
                prepared = self.prepared()
                try:
                    self.publish(prepared)
                    def change(root):
                        row = root["candidate_population"][0]
                        (row if field is None else row[field])[key] = value
                    self.mutate_json("final_checkpoint_selection.json", change)
                    self.fx.reseal()
                    self.verify_fails(prepared)
                finally:
                    prepared.close()
                self.reset_output()

    def test_06_root_extra_false_positive_and_wrong_scalar_types_are_rejected(self):
        attacks = (
            {"paper_metric": False},
            {"hardware_speedup": True},
            {"schema_alias": S.B.RESULT_SCHEMA},
        )
        for extra in attacks:
            with self.subTest(extra=extra):
                prepared = self.prepared()
                try:
                    self.publish(prepared)
                    self.mutate_json("final_checkpoint_selection.json",
                                     lambda root: root.update(extra))
                    self.fx.reseal()
                    self.verify_fails(prepared, "exact root key")
                finally:
                    prepared.close()
                self.reset_output()

    def test_07_all_nine_E0_E8_rows_reject_coordinated_splices(self):
        exact = S.exact_rebind_targets()
        self.assertEqual([row["id"] for row in exact], ["E{}".format(i) for i in range(9)])
        self.assertTrue(all(set(row) == {"id", "target", "state_after_selection",
                                         "dependency", "reuse_rule"} for row in exact))
        for index in range(9):
            with self.subTest(index=index):
                prepared = self.prepared()
                try:
                    self.publish(prepared)
                    def change_result(root):
                        row = root["e0_e8_activation_dependent_invalidation_and_rebind_targets"][index]
                        row["target"] += "__COORDINATED_SPLICE"
                    self.mutate_json("final_checkpoint_selection.json", change_result)
                    def change_sidecar(rows):
                        rows[index]["target"] += "__COORDINATED_SPLICE"
                    self.mutate_json("e0_e8_activation_rebind_targets.json", change_sidecar)
                    self.fx.reseal()
                    self.verify_fails(prepared, "E0-E8 exact map")
                finally:
                    prepared.close()
                self.reset_output()

    def test_08_E0_E8_order_extra_key_and_population_splices_are_rejected(self):
        for attack in ("swap", "extra_false", "drop"):
            with self.subTest(attack=attack):
                prepared = self.prepared()
                try:
                    self.publish(prepared)
                    def mutate(rows):
                        if attack == "swap":
                            rows[3], rows[4] = rows[4], rows[3]
                        elif attack == "extra_false":
                            rows[5]["paper_metric"] = False
                        else:
                            rows.pop()
                    self.mutate_json(
                        "final_checkpoint_selection.json",
                        lambda root: mutate(root["e0_e8_activation_dependent_invalidation_and_rebind_targets"]))
                    self.mutate_json("e0_e8_activation_rebind_targets.json", mutate)
                    self.fx.reseal()
                    self.verify_fails(prepared, "E0-E8 exact map")
                finally:
                    prepared.close()
                self.reset_output()

    def test_09_candidate_order_pair_and_selected_projection_are_rejected(self):
        attacks = ("swap_rows", "epoch_bool", "nonminimum", "selected_extra")
        for attack in attacks:
            with self.subTest(attack=attack):
                prepared = self.prepared()
                try:
                    self.publish(prepared)
                    def change(root):
                        if attack == "swap_rows":
                            root["candidate_population"][0], root["candidate_population"][1] = (
                                root["candidate_population"][1], root["candidate_population"][0])
                        elif attack == "epoch_bool":
                            root["candidate_population"][0]["epoch"] = True
                        elif attack == "nonminimum":
                            loser = max(root["candidate_population"],
                                        key=lambda row: row["accuracy_metrics"]["AEE"])
                            root["selected"] = {key: loser[key] for key in (
                                "candidate_id", "epoch", "run_directory", "checkpoint",
                                "configuration", "profile", "accuracy_metrics", "activity")}
                        else:
                            root["selected"]["paper_metric"] = False
                    self.mutate_json("final_checkpoint_selection.json", change)
                    self.fx.reseal()
                    self.verify_fails(prepared)
                finally:
                    prepared.close()
                self.reset_output()

    def test_10_minimum_AEE_and_lowest_epoch_tie_are_recomputed(self):
        for attack in ("new_minimum", "new_tie"):
            with self.subTest(attack=attack):
                prepared = self.prepared()
                try:
                    self.publish(prepared)
                    def change(root):
                        rows = root["candidate_population"]
                        if attack == "new_minimum":
                            rows[-1]["accuracy_metrics"]["AEE"] = "0"
                        else:
                            common = rows[0]["accuracy_metrics"]["AEE"]
                            rows[1]["accuracy_metrics"]["AEE"] = common
                            root["selected"] = {key: rows[1][key] for key in (
                                "candidate_id", "epoch", "run_directory", "checkpoint",
                                "configuration", "profile", "accuracy_metrics", "activity")}
                    self.mutate_json("final_checkpoint_selection.json", change)
                    self.fx.reseal()
                    self.verify_fails(prepared, "selected projection")
                finally:
                    prepared.close()
                self.reset_output()

    def test_11_attempt_is_O_EXCL_and_failed_child_is_never_retried(self):
        calls = []
        def failed(command, cwd, pass_fds):
            calls.append((tuple(command), tuple(pass_fds)))
            return subprocess.CompletedProcess(command, 23, stdout="", stderr="hammer failure")
        with self.assertRaisesRegex(S.B.ReleaseError, "no retry"):
            S.execute_once(self.fx.policy, self.fx.fx.interpreter, "3.10.20",
                           self.fx.fx.repo, failed)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.fx.fx.attempt.exists())
        self.assertTrue(self.fx.fx.log.exists())
        self.assertEqual(self.fx.fx.attempt.stat().st_mode & 0o777, 0o400)
        self.assertEqual(self.fx.fx.log.stat().st_mode & 0o777, 0o400)
        with self.assertRaises(S.B.ReleaseError):
            S.execute_once(self.fx.policy, self.fx.fx.interpreter, "3.10.20",
                           self.fx.fx.repo, failed)
        self.assertEqual(len(calls), 1)

    def test_12_future_authority_remains_conditional_and_source_only(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        readiness = contract["current_readiness"]
        boundary = contract["claim_boundary"]
        self.assertFalse(readiness["production_execution_authorized"])
        self.assertFalse(readiness["all_four_real_strict_valid825_confirmed"])
        self.assertFalse(boundary["checkpoint_selected_now"])
        self.assertFalse(boundary["hardware_rebind_authorized"])
        for key in ("hardware_speedup", "system_speedup", "power_or_energy", "paper_metric"):
            self.assertFalse(boundary[key])
        self.assertIn("only after all four strict-valid825 artifacts exist",
                      contract["next_gate"])
        self.assertEqual(sha(DOCS359), DOCS359_SHA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
