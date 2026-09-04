from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_m1306_m1301_motion_final_checkpoint_binder_inherited_authority_successor.py"
OLD_TEST = ROOT / "tests/test_run_m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor.py"
CONTRACT = ROOT / "contracts/m1306_m1301_motion_final_checkpoint_binder_inherited_authority_successor_source_contract_r1_20260830.json"
M1303 = ROOT / "reviews/m1303_m1301_final_checkpoint_binder_claim_authority_blind_hammer_r1_20260830"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


S = load("m1306_source_under_test", SOURCE)
T = load("m1306_m1301_fixture", OLD_TEST)


class M1306InheritedAuthoritySuccessorTest(unittest.TestCase):
    def setUp(self):
        self.old = T.M1301ClaimAuthoritySuccessorTest(
            methodName="test_01_import_is_inert_and_exact_predecessor_triplet_is_pinned")
        self.old.setUp()
        destination = self.old.repo / S.M1303_REL
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(M1303, destination)
        self.policy = self.old.old.policy

    def tearDown(self): self.old.tearDown()

    def test_01_import_inert_and_exact_m1301_triplet_pinned(self):
        self.assertEqual(sha(S.M1301_SOURCE), S.M1301_SOURCE_SHA256)
        self.assertEqual(sha(S.M1301_TEST), S.M1301_TEST_SHA256)
        self.assertEqual(sha(S.M1301_CONTRACT), S.M1301_CONTRACT_SHA256)
        self.assertFalse(self.old.old.old.old.fx.attempt.exists())

    def test_02_policy_entity_claims_and_execution_objects_do_not_drift(self):
        self.assertIs(S.PRODUCTION_POLICY, S.M.PRODUCTION_POLICY)
        self.assertEqual(S.EXACT_CLAIM_BOUNDARY, S.M.EXACT_CLAIM_BOUNDARY)
        self.assertNotIn("paper_ppa_ready", S.EXACT_CLAIM_BOUNDARY)
        self.assertIs(S.M.M.PRODUCTION_POLICY, S.PRODUCTION_POLICY)
        self.assertEqual(S.M.M.TARGET_ENTITY, {
            "device": 1048625, "inode": 1347357695, "mode": 0x81ED,
            "size_bytes": 8020928, "mtime_sec": 1774292672,
            "sha256": "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7",
            "version": "3.12.3", "memfd_and_all_seals": True})

    def test_03_m1303_stop_seal_is_verified(self):
        S.verify_frozen_authorities(self.old.repo)
        review = json.loads((self.old.repo / S.M1303_REL / "review.json").read_text())
        self.assertEqual(review["status"], S.M1303_STATUS)
        self.assertEqual(review["authority"]["exactly_one_remote_production_execution"],
                         "STOP")

    def test_04_m1303_member_drift_fails_closed(self):
        root = self.old.repo / S.M1303_REL
        (root / "review.md").write_text("drift\n", encoding="utf-8")
        with self.assertRaises(S.B.ReleaseError):
            S.verify_frozen_authorities(self.old.repo)

    def test_05_gate_order_is_m1306_m1301_claim_inherited_delegate(self):
        events = []
        completed = subprocess.CompletedProcess(["fd-child"], 0, stdout="ok\n", stderr="")
        with mock.patch.object(S, "verify_frozen_authorities",
                               side_effect=lambda repo: events.append("m1306")), \
             mock.patch.object(S.M, "verify_frozen_authorities",
                               side_effect=lambda repo: events.append("m1301")), \
             mock.patch.object(S.M, "validate_claim_boundary",
                               side_effect=lambda value: events.append("claim") or dict(value)), \
             mock.patch.object(S.M.M.M, "verify_frozen_authorities",
                               side_effect=lambda: events.append("m1257")), \
             mock.patch.object(S.M.M, "execute_once",
                               side_effect=lambda *a, **k: events.append("delegate") or completed):
            result = S.execute_once(self.policy, self.old.repo, self.old.old.link,
                                    self.old.old.real, self.old.old.expected)
        self.assertIs(result, completed)
        self.assertEqual(events, ["m1306", "m1301", "claim", "m1257", "delegate"])

    def test_06_inherited_gate_failure_called_once_and_delegate_zero(self):
        with mock.patch.object(S, "verify_frozen_authorities"), \
             mock.patch.object(S.M, "verify_frozen_authorities"), \
             mock.patch.object(S.M, "validate_claim_boundary",
                               side_effect=lambda value: dict(value)), \
             mock.patch.object(S.M.M.M, "verify_frozen_authorities",
                               side_effect=S.B.ReleaseError("M1257 drift")) as inherited, \
             mock.patch.object(S.M.M, "execute_once") as delegate:
            with self.assertRaisesRegex(S.B.ReleaseError, "M1257 drift"):
                S.execute_once(self.policy, self.old.repo, self.old.old.link,
                               self.old.old.real, self.old.old.expected)
        self.assertEqual(inherited.call_count, 1)
        self.assertEqual(delegate.call_count, 0)
        self.assertFalse(self.old.old.old.old.fx.attempt.exists())

    def test_07_m1301_authority_failure_prevents_inherited_and_delegate(self):
        with mock.patch.object(S, "verify_frozen_authorities"), \
             mock.patch.object(S.M, "verify_frozen_authorities",
                               side_effect=S.B.ReleaseError("M1301 drift")), \
             mock.patch.object(S.M.M.M, "verify_frozen_authorities") as inherited, \
             mock.patch.object(S.M.M, "execute_once") as delegate:
            with self.assertRaises(S.B.ReleaseError):
                S.execute_once(self.policy, self.old.repo, self.old.old.link,
                               self.old.old.real, self.old.old.expected)
        inherited.assert_not_called(); delegate.assert_not_called()

    def test_08_original_entity_fd_passfds_snapshots_and_sources_survive(self):
        prepared = S.M.M.prepare(self.policy, self.old.repo, self.old.old.link,
                                 self.old.old.real, self.old.old.expected)
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(prepared.pass_fds,
                             prepared.source_fds + (prepared.interpreter.descriptor,))
            self.assertTrue(prepared.command[0].startswith("/proc/self/fd/"))
        finally:
            prepared.close()

    def test_09_claim_attacks_still_fail_in_frozen_m1301(self):
        attacks = []
        value = dict(S.EXACT_CLAIM_BOUNDARY); value["remote_execution_authorized"] = True; attacks.append(value)
        value = dict(S.EXACT_CLAIM_BOUNDARY); value["paper_metric"] = 0; attacks.append(value)
        value = dict(S.EXACT_CLAIM_BOUNDARY); value["extra"] = False; attacks.append(value)
        for value in attacks:
            with self.assertRaises(S.B.ReleaseError):
                S.M.validate_claim_boundary(value)

    def test_10_main_zero_arg_uses_repaired_wrapper(self):
        completed = subprocess.CompletedProcess(["fd-child"], 0,
                                                stdout="child\n", stderr="")
        with mock.patch.object(S, "execute_once", return_value=completed) as execute, \
             mock.patch.object(sys, "argv", [str(SOURCE)]):
            self.assertEqual(S.main(), 0)
        execute.assert_called_once()

    def test_11_contract_docs_and_no_remote_gpu_eda(self):
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


if __name__ == "__main__": unittest.main(verbosity=2)
