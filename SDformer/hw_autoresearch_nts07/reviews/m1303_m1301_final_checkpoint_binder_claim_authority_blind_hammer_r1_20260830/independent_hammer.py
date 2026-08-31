#!/usr/bin/python3.12
"""Receipt-blind local hammer for M1301.

This hammer deliberately does not import or read the M1301 author receipt.  It
does not contact the remote host and cannot execute the production policy.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor.py"
AUTHOR_TEST = ROOT / "hw_autoresearch_nts07/tests/test_run_m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor.py"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor_source_contract_r1_20260830.json"
M1292_CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor_source_contract_r1_20260830.json"
M1297_SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor.py"
M1297_TEST = ROOT / "hw_autoresearch_nts07/tests/test_run_m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor.py"
M1297_CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor_source_contract_r1_20260830.json"
M1298 = ROOT / "hw_autoresearch_nts07/reviews/m1298_m1297_interpreter_entity_fd_bound_receipt_blind_hammer_r1_20260830"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "e8db73150c8d08ad52f4cf39d2013e1207c17db1192141fa002789b722203b4a"
AUTHOR_TEST_SHA = "de6381edde6f4722085c830c6960032fdd738e8f4fdb05fc76bf522927a48a30"
CONTRACT_SHA = "4aec2a68ac47a76bbef7b9ac773568ed0465d2d70fda4701c8a5e37fca7413ae"
M1292_CONTRACT_SHA = "b7dd35d704fc4c754ac76a1858febe2ce65ac23dc8576c30294737e354c21842"
M1297_SOURCE_SHA = "336195e40cf07cfa273be650f2edd0cf1c537c8c70b1c39e68515c087ca81899"
M1297_TEST_SHA = "1dddabbc1334ed98633e556898cf5df74a4b49089f70c253cae2cf5e408563de"
M1297_CONTRACT_SHA = "ace730ff38df4ba5025afb46edcd90e6913ef9806058570e2db2db04fdf35cb2"
M1298_MANIFEST_SHA = "c0f556d43be76e10d1518de44c8d4820292defed860f5ad0a8475b4d5c36b3a1"
M1298_OUTER_SHA = "638cdf4a83e3b05e1752faae99ac74b93a35e7b100a09657d6bb3efd1689bca2"
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


S = load("m1303_m1301_source", SOURCE)
A = load("m1303_m1301_author_fixture", AUTHOR_TEST)


class Hammer(unittest.TestCase):
    def setUp(self):
        self.fx = A.M1301ClaimAuthoritySuccessorTest(
            methodName="test_01_import_is_inert_and_exact_predecessor_triplet_is_pinned")
        self.fx.setUp()
        self.repo = self.fx.repo

    def tearDown(self):
        self.fx.tearDown()

    def test_01_reviewed_bytes_docs_and_predecessor_triplet_are_frozen(self):
        expected = ((SOURCE, SOURCE_SHA), (AUTHOR_TEST, AUTHOR_TEST_SHA),
                    (CONTRACT, CONTRACT_SHA), (M1292_CONTRACT, M1292_CONTRACT_SHA),
                    (M1297_SOURCE, M1297_SOURCE_SHA), (M1297_TEST, M1297_TEST_SHA),
                    (M1297_CONTRACT, M1297_CONTRACT_SHA), (DOCS359, DOCS359_SHA),
                    (M1298 / "SHA256SUMS", M1298_MANIFEST_SHA),
                    (M1298 / "SHA256SUMS.seal.sha256", M1298_OUTER_SHA))
        for path, wanted in expected:
            with self.subTest(path=path.name):
                self.assertEqual(sha(path), wanted)
        self.assertFalse(self.fx.old.old.old.fx.attempt.exists())

    def test_02_exact_seven_false_claims_and_no_paper_ppa_ready(self):
        inherited = json.loads(M1292_CONTRACT.read_text())["claim_boundary"]
        self.assertEqual(S.EXACT_CLAIM_BOUNDARY, inherited)
        self.assertEqual(set(S.EXACT_CLAIM_BOUNDARY), {
            "checkpoint_selected_now", "hardware_rebind_authorized",
            "hardware_speedup", "system_speedup", "power_or_energy",
            "paper_metric", "remote_execution_authorized"})
        self.assertNotIn("paper_ppa_ready", S.EXACT_CLAIM_BOUNDARY)
        self.assertTrue(all(type(v) is bool and v is False
                            for v in S.EXACT_CLAIM_BOUNDARY.values()))

    def test_03_claim_missing_extra_true_int_and_none_attacks_fail_closed(self):
        attacks = []
        for key in S.EXACT_CLAIM_BOUNDARY:
            bad = dict(S.EXACT_CLAIM_BOUNDARY); bad.pop(key); attacks.append(bad)
            for value in (True, 0, None):
                bad = dict(S.EXACT_CLAIM_BOUNDARY); bad[key] = value; attacks.append(bad)
        bad = dict(S.EXACT_CLAIM_BOUNDARY); bad["paper_ppa_ready"] = False; attacks.append(bad)
        for index, bad in enumerate(attacks):
            with self.subTest(index=index), self.assertRaises(S.B.ReleaseError):
                S.validate_claim_boundary(bad)

    def test_04_m1297_triplet_sha_drift_blocks_before_delegate(self):
        with tempfile.TemporaryDirectory() as td:
            drift = Path(td) / "m1297.py"; drift.write_bytes(M1297_SOURCE.read_bytes() + b"\n#drift\n")
            with mock.patch.object(S, "M1297_SOURCE", drift), \
                 mock.patch.object(S.M, "execute_once") as inherited:
                with self.assertRaises(S.B.ReleaseError):
                    S.execute_once(self.fx.old.policy, self.repo, self.fx.old.link,
                                   self.fx.old.real, self.fx.old.expected)
                inherited.assert_not_called()
        self.assertFalse(self.fx.old.old.old.fx.attempt.exists())

    def test_05_m1298_member_manifest_outer_and_authority_attacks_fail(self):
        target = self.repo / S.M1298_REL
        mutations = (
            ("review.md", lambda p: p.write_text("drift\n")),
            ("SHA256SUMS", lambda p: p.write_text("0" * 64 + "  review.json\n")),
            ("SHA256SUMS.seal.sha256", lambda p: p.write_text("0" * 64 + "  SHA256SUMS\n")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                shutil.rmtree(target); shutil.copytree(M1298, target)
                mutate(target / name)
                with self.assertRaises(S.B.ReleaseError):
                    S.verify_frozen_authorities(self.repo)

    def test_06_m1298_exact_seals_and_blocking_status_pass(self):
        S.verify_frozen_authorities(self.repo)
        value = json.loads((self.repo / S.M1298_REL / "review.json").read_text())
        self.assertEqual(value["schema"], S.M1298_SCHEMA)
        self.assertEqual(value["status"], S.M1298_STATUS)
        self.assertEqual(value["authority"]["exact_reviewed_byte_transfer"], "GO")
        self.assertEqual(value["authority"]["exactly_one_remote_production_execution"], "STOP")
        self.assertIs(value["authority"]["attempt_may_be_consumed_now"], False)

    def test_07_gate_order_is_m1301_authority_claim_then_m1297_executor(self):
        events = []
        completed = subprocess.CompletedProcess(["child"], 0, stdout="ok\n", stderr="")
        with mock.patch.object(S, "verify_frozen_authorities",
                               side_effect=lambda repo: events.append("m1301_authority")), \
             mock.patch.object(S, "validate_claim_boundary",
                               side_effect=lambda claim: events.append("claim") or claim), \
             mock.patch.object(S.M, "execute_once",
                               side_effect=lambda *a, **k: events.append("m1297_execute") or completed):
            self.assertIs(S.execute_once(self.fx.old.policy, self.repo, self.fx.old.link,
                                         self.fx.old.real, self.fx.old.expected), completed)
        self.assertEqual(events, ["m1301_authority", "claim", "m1297_execute"])

    def test_08_authority_or_claim_failure_never_delegates_or_consumes_attempt(self):
        for target in ("authority", "claim"):
            with self.subTest(target=target), \
                 mock.patch.object(S, "verify_frozen_authorities",
                                   side_effect=(S.B.ReleaseError("blocked") if target == "authority" else None)), \
                 mock.patch.object(S, "validate_claim_boundary",
                                   side_effect=(S.B.ReleaseError("blocked") if target == "claim" else lambda x: x)), \
                 mock.patch.object(S.M, "execute_once") as inherited:
                with self.assertRaises(S.B.ReleaseError):
                    S.execute_once(self.fx.old.policy, self.repo, self.fx.old.link,
                                   self.fx.old.real, self.fx.old.expected)
                inherited.assert_not_called()
                self.assertFalse(self.fx.old.old.old.fx.attempt.exists())

    def test_09_entity_fd_proc_passfds_snapshots_and_sources_are_preserved(self):
        prepared = S.M.prepare(self.fx.old.policy, self.repo, self.fx.old.link,
                               self.fx.old.real, copy.deepcopy(self.fx.old.expected))
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(len(prepared.pass_fds), 4)
            self.assertEqual(prepared.pass_fds,
                             prepared.source_fds + (prepared.interpreter.descriptor,))
            self.assertEqual(prepared.command[0],
                             "/proc/self/fd/{}".format(prepared.interpreter.descriptor))
        finally:
            prepared.close()

    def test_10_interpreter_swap_and_retained_fd_close_fail_before_attempt(self):
        prepared = S.M.prepare(self.fx.old.policy, self.repo, self.fx.old.link,
                               self.fx.old.real, copy.deepcopy(self.fx.old.expected))
        try:
            self.fx.old.link.unlink(); self.fx.old.link.symlink_to("/bin/false")
            with self.assertRaises(S.B.ReleaseError):
                S.M.revalidate_entity(prepared.interpreter, self.fx.old.expected)
            self.assertFalse(self.fx.old.old.old.fx.attempt.exists())
        finally:
            prepared.close()

    def test_11_attempt_is_O_EXCL_no_retry_and_binds_entity(self):
        prepared = S.M.prepare(self.fx.old.policy, self.repo, self.fx.old.link,
                               self.fx.old.real, copy.deepcopy(self.fx.old.expected))
        try:
            S.M.consume_attempt(prepared)
            body = self.fx.old.old.old.fx.attempt.read_text()
            self.assertIn("automatic_retry=false", body)
            self.assertIn("interpreter_entity_sha256=", body)
            with self.assertRaises(FileExistsError):
                S.M.consume_attempt(prepared)
        finally:
            prepared.close()

    def test_12_F1_F4_E0_E8_and_three_sealed_sources_remain_in_policy(self):
        policy = S.PRODUCTION_POLICY
        self.assertIs(policy, S.M.PRODUCTION_POLICY)
        pins = policy.base.execution_pins
        self.assertEqual(len(pins), 3)
        self.assertEqual(set(S.M.M.M.B.artifact_map(policy.base)),
                         set(S.M.M.M.B.artifact_map(S.M.PRODUCTION_POLICY.base)))
        targets = S.M.M.M.exact_rebind_targets()
        self.assertEqual([row["id"] for row in targets],
                         ["E{}".format(i) for i in range(9)])
        self.assertEqual(S.M.M.M.RESULT_KEYS, {
            "schema", "status", "new_run_manifest", "candidate_population",
            "selection_rule", "selected",
            "e0_e8_activation_dependent_invalidation_and_rebind_targets",
            "claim_boundary"})

    def test_13_zero_arg_local_inert_and_no_remote_gpu_eda(self):
        import inspect
        self.assertEqual(len(inspect.signature(S.main).parameters), 0)
        text = SOURCE.read_text()
        for forbidden in ("paramiko", "ssh ", "scp ", "rsync ", "import torch",
                          "nvidia-smi", "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(self.fx.old.old.old.fx.attempt.exists())

    def test_14_regression_m1297_main_authority_preflight_is_not_inherited(self):
        """Reproduce the blocking additive-successor regression.

        M1297.main calls M.verify_frozen_authorities before execute_once.  M1301
        delegates directly to M1297.execute_once, which does not call it.
        """
        completed = subprocess.CompletedProcess(["child"], 0, stdout="ok\n", stderr="")
        with mock.patch.object(S, "verify_frozen_authorities", return_value=None), \
             mock.patch.object(S, "validate_claim_boundary", side_effect=lambda x: x), \
             mock.patch.object(S.M, "execute_once", return_value=completed), \
             mock.patch.object(S.M.M, "verify_frozen_authorities",
                               side_effect=S.B.ReleaseError("M1257 triplet drift")) as inherited_gate:
            result = S.execute_once(self.fx.old.policy, self.repo, self.fx.old.link,
                                    self.fx.old.real, self.fx.old.expected)
        self.assertIs(result, completed)
        inherited_gate.assert_not_called()
        self.assertFalse(self.fx.old.old.old.fx.attempt.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
