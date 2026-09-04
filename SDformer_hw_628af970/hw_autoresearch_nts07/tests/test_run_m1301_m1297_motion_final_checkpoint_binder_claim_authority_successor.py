from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor.py"
OLD_TEST = ROOT / "tests/test_run_m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor.py"
CONTRACT = ROOT / "contracts/m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor_source_contract_r1_20260830.json"
M1298 = ROOT / "reviews/m1298_m1297_interpreter_entity_fd_bound_receipt_blind_hammer_r1_20260830"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


S = load("m1301_source_under_test", SOURCE)
T = load("m1301_m1297_fixture", OLD_TEST)


class M1301ClaimAuthoritySuccessorTest(unittest.TestCase):
    def setUp(self):
        self.old = T.M1297InterpreterEntitySuccessorTest(
            methodName="test_01_import_inert_and_m1292_frozen")
        self.old.setUp()
        self.repo = self.old.old.old.fx.repo
        destination = self.repo / S.M1298_REL
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(M1298, destination)

    def tearDown(self):
        self.old.tearDown()

    def test_01_import_is_inert_and_exact_predecessor_triplet_is_pinned(self):
        self.assertEqual(sha(S.M1297_SOURCE), S.M1297_SOURCE_SHA256)
        self.assertEqual(sha(S.M1297_TEST), S.M1297_TEST_SHA256)
        self.assertEqual(sha(S.M1297_CONTRACT), S.M1297_CONTRACT_SHA256)
        self.assertFalse(self.old.old.old.fx.attempt.exists())
        self.assertFalse(self.old.old.old.output.exists())

    def test_02_production_policy_is_same_object_and_entity_constants_unchanged(self):
        self.assertIs(S.PRODUCTION_POLICY, S.M.PRODUCTION_POLICY)
        self.assertEqual(S.M.TARGET_LINK, Path("/usr/bin/python3"))
        self.assertEqual(S.M.TARGET_REALPATH, Path("/usr/bin/python3.12"))
        self.assertEqual(S.M.TARGET_ENTITY, {
            "device": 1048625, "inode": 1347357695, "mode": 0x81ED,
            "size_bytes": 8020928, "mtime_sec": 1774292672,
            "sha256": "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7",
            "version": "3.12.3", "memfd_and_all_seals": True})

    def test_03_exact_m1292_claim_map_is_restored(self):
        predecessor = json.loads((ROOT / "contracts/m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor_source_contract_r1_20260830.json").read_text())["claim_boundary"]
        self.assertEqual(S.EXACT_CLAIM_BOUNDARY, predecessor)
        self.assertNotIn("paper_ppa_ready", S.EXACT_CLAIM_BOUNDARY)
        self.assertEqual(S.validate_claim_boundary(dict(S.EXACT_CLAIM_BOUNDARY)),
                         S.EXACT_CLAIM_BOUNDARY)

    def test_04_claim_missing_true_int_zero_and_extra_are_rejected(self):
        attacks = []
        value = dict(S.EXACT_CLAIM_BOUNDARY); value.pop("checkpoint_selected_now"); attacks.append(value)
        value = dict(S.EXACT_CLAIM_BOUNDARY); value["hardware_speedup"] = True; attacks.append(value)
        value = dict(S.EXACT_CLAIM_BOUNDARY); value["remote_execution_authorized"] = 0; attacks.append(value)
        value = dict(S.EXACT_CLAIM_BOUNDARY); value["paper_ppa_ready"] = False; attacks.append(value)
        for value in attacks:
            with self.assertRaises(S.B.ReleaseError):
                S.validate_claim_boundary(value)

    def test_05_m1298_double_seal_and_blocking_authority_are_internally_verified(self):
        S.verify_frozen_authorities(self.repo)
        review = json.loads((self.repo / S.M1298_REL / "review.json").read_text())
        self.assertEqual(review["status"], S.M1298_STATUS)
        self.assertEqual(review["authority"]["exactly_one_remote_production_execution"],
                         "STOP")
        self.assertFalse(review["authority"]["production_execution_authorized_now"])

    def test_06_m1298_member_or_outer_drift_is_rejected(self):
        root = self.repo / S.M1298_REL
        (root / "review.md").write_text("drift\n", encoding="utf-8")
        with self.assertRaises(S.B.ReleaseError):
            S.verify_frozen_authorities(self.repo)

    def test_07_execution_gate_checks_authority_and_claim_before_delegate(self):
        events = []
        completed = subprocess.CompletedProcess(["fd-child"], 0, stdout="ok\n", stderr="")
        with mock.patch.object(S, "verify_frozen_authorities",
                               side_effect=lambda repo: events.append("authority")), \
             mock.patch.object(S, "validate_claim_boundary",
                               side_effect=lambda value: events.append("claim") or dict(value)), \
             mock.patch.object(S.M, "execute_once",
                               side_effect=lambda *args, **kwargs: events.append("entity") or completed):
            result = S.execute_once(self.old.policy, self.repo, self.old.link,
                                    self.old.real, self.old.expected)
        self.assertIs(result, completed)
        self.assertEqual(events, ["authority", "claim", "entity"])

    def test_08_authority_failure_prevents_entity_executor_and_attempt(self):
        with mock.patch.object(S, "verify_frozen_authorities",
                               side_effect=S.B.ReleaseError("blocked")), \
             mock.patch.object(S.M, "execute_once") as inherited:
            with self.assertRaises(S.B.ReleaseError):
                S.execute_once(self.old.policy, self.repo, self.old.link,
                               self.old.real, self.old.expected)
            inherited.assert_not_called()
        self.assertFalse(self.old.old.old.fx.attempt.exists())

    def test_09_original_entity_prepare_and_pass_fds_semantics_survive(self):
        prepared = S.M.prepare(self.old.policy, self.repo, self.old.link,
                               self.old.real, copy.deepcopy(self.old.expected))
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(prepared.pass_fds,
                             prepared.source_fds + (prepared.interpreter.descriptor,))
            self.assertTrue(prepared.command[0].startswith("/proc/self/fd/"))
        finally:
            prepared.close()

    def test_10_main_is_zero_arg_and_uses_claim_gated_wrapper(self):
        completed = subprocess.CompletedProcess(["fd-child"], 0,
                                                stdout="child\n", stderr="")
        with mock.patch.object(S, "execute_once", return_value=completed) as execute, \
             mock.patch.object(sys, "argv", [str(SOURCE)]):
            self.assertEqual(S.main(), 0)
        execute.assert_called_once()

    def test_11_contract_identity_when_present_docs_and_scope(self):
        docs = ROOT / "docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(sha(docs), S.B.DOCS359_SHA256)
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("paramiko", "ssh ", "scp ", "rsync ", "import torch",
                          "nvidia-smi", "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        if CONTRACT.exists():
            value = json.loads(CONTRACT.read_text())
            self.assertEqual(value["source"]["sha256"], sha(SOURCE))
            self.assertEqual(value["test"]["sha256"], sha(Path(__file__).resolve()))
            self.assertEqual(value["claim_boundary"], S.EXACT_CLAIM_BOUNDARY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
