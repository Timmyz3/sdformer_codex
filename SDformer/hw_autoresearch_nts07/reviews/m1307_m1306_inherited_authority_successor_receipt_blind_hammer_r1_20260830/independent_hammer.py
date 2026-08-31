#!/usr/bin/python3.12
"""Independent receipt-blind hammer for M1306.

The M1306 author receipt is intentionally neither opened nor imported.  All
execution is against disposable local fixtures; remote, production, GPU and EDA
paths are outside this program's scope.
"""
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
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1306_m1301_motion_final_checkpoint_binder_inherited_authority_successor.py"
AUTHOR_TEST = HW / "tests/test_run_m1306_m1301_motion_final_checkpoint_binder_inherited_authority_successor.py"
CONTRACT = HW / "contracts/m1306_m1301_motion_final_checkpoint_binder_inherited_authority_successor_source_contract_r1_20260830.json"
M1301_SOURCE = HW / "scripts/run_m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor.py"
M1301_TEST = HW / "tests/test_run_m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor.py"
M1301_CONTRACT = HW / "contracts/m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor_source_contract_r1_20260830.json"
M1303 = HW / "reviews/m1303_m1301_final_checkpoint_binder_claim_authority_blind_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "99b2f4f895b28bdc15ca3a3fa75e3364658751132cde4cc47cf01c778fc16548"
AUTHOR_TEST_SHA = "6154b311472a1f99a2a90bfea5461c78b06b43099251105f131400121de8ff5f"
CONTRACT_SHA = "478c005b7cd5db47971e9a0fa621f4901f0e087a170a2ccc6fd4a33ae404d4bf"
M1301_SOURCE_SHA = "e8db73150c8d08ad52f4cf39d2013e1207c17db1192141fa002789b722203b4a"
M1301_TEST_SHA = "de6381edde6f4722085c830c6960032fdd738e8f4fdb05fc76bf522927a48a30"
M1301_CONTRACT_SHA = "4aec2a68ac47a76bbef7b9ac773568ed0465d2d70fda4701c8a5e37fca7413ae"
M1303_MANIFEST_SHA = "8d2a938ebd475bca3b2a7dc0adbdc51c4848604d3255f1706234b266ce788b04"
M1303_OUTER_SHA = "67294688d5285a0836e8e401525e2835d7d87ea4aad6fb013d1914e52e8c2ff5"
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


S = load("m1307_m1306_source", SOURCE)
F = load("m1307_m1301_fixture", M1301_TEST)


class Hammer(unittest.TestCase):
    def setUp(self):
        self.fx = F.M1301ClaimAuthoritySuccessorTest(
            methodName="test_01_import_is_inert_and_exact_predecessor_triplet_is_pinned")
        self.fx.setUp()
        self.repo = self.fx.repo
        target = self.repo / S.M1303_REL
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(M1303, target)
        self.policy = self.fx.old.policy

    def tearDown(self):
        self.fx.tearDown()

    @property
    def attempt(self):
        return self.fx.old.old.old.fx.attempt

    def execute(self, **kwargs):
        return S.execute_once(self.policy, self.repo, self.fx.old.link,
                              self.fx.old.real, self.fx.old.expected, **kwargs)

    def test_01_reviewed_bytes_predecessor_docs_and_stop_seal_are_exact(self):
        exact = ((SOURCE, SOURCE_SHA), (AUTHOR_TEST, AUTHOR_TEST_SHA),
                 (CONTRACT, CONTRACT_SHA), (M1301_SOURCE, M1301_SOURCE_SHA),
                 (M1301_TEST, M1301_TEST_SHA), (M1301_CONTRACT, M1301_CONTRACT_SHA),
                 (M1303 / "SHA256SUMS", M1303_MANIFEST_SHA),
                 (M1303 / "SHA256SUMS.seal.sha256", M1303_OUTER_SHA),
                 (DOCS359, DOCS359_SHA))
        for path, wanted in exact:
            with self.subTest(path=path.name):
                self.assertEqual(sha(path), wanted)
        self.assertFalse(self.attempt.exists())

    def test_02_m1303_double_seal_status_and_denial_authority_are_exact(self):
        S.verify_frozen_authorities(self.repo)
        value = json.loads((self.repo / S.M1303_REL / "review.json").read_text())
        self.assertEqual(value["schema"], S.M1303_SCHEMA)
        self.assertEqual(value["status"], S.M1303_STATUS)
        self.assertEqual(value["authority"]["exactly_one_remote_production_execution"], "STOP")
        self.assertIs(value["authority"]["production_execution_authorized_now"], False)
        self.assertIs(value["authority"]["attempt_may_be_consumed_now"], False)
        self.assertIs(value["authority"]["checkpoint_selected_now"], False)

    def test_03_m1306_triplet_and_m1303_member_manifest_outer_attacks_fail(self):
        with tempfile.TemporaryDirectory() as td:
            drift = Path(td) / "m1301.py"
            drift.write_bytes(M1301_SOURCE.read_bytes() + b"\n# drift\n")
            with mock.patch.object(S, "M1301_SOURCE", drift):
                with self.assertRaises(S.B.ReleaseError):
                    S.verify_frozen_authorities(self.repo)
        target = self.repo / S.M1303_REL
        mutations = (
            ("review.md", "drift\n"),
            ("SHA256SUMS", "0" * 64 + "  review.json\n"),
            ("SHA256SUMS.seal.sha256", "0" * 64 + "  SHA256SUMS\n"),
        )
        for name, payload in mutations:
            with self.subTest(name=name):
                shutil.rmtree(target); shutil.copytree(M1303, target)
                (target / name).write_text(payload)
                with self.assertRaises(S.B.ReleaseError):
                    S.verify_frozen_authorities(self.repo)

    def test_04_exact_five_stage_order_and_single_delegate(self):
        events = []
        done = subprocess.CompletedProcess(["child"], 0, stdout="ok\n", stderr="")
        with mock.patch.object(S, "verify_frozen_authorities",
                               side_effect=lambda repo: events.append("m1306")), \
             mock.patch.object(S.M, "verify_frozen_authorities",
                               side_effect=lambda repo: events.append("m1301_m1297_m1298")), \
             mock.patch.object(S.M, "validate_claim_boundary",
                               side_effect=lambda value: events.append("claims") or value), \
             mock.patch.object(S.M.M.M, "verify_frozen_authorities",
                               side_effect=lambda: events.append("m1257")), \
             mock.patch.object(S.M.M, "execute_once",
                               side_effect=lambda *a, **k: events.append("delegate") or done) as delegate:
            self.assertIs(self.execute(), done)
        self.assertEqual(events, ["m1306", "m1301_m1297_m1298", "claims", "m1257", "delegate"])
        self.assertEqual(delegate.call_count, 1)

    def test_05_inherited_gate_failure_is_once_delegate_zero_attempt_false(self):
        with mock.patch.object(S, "verify_frozen_authorities"), \
             mock.patch.object(S.M, "verify_frozen_authorities"), \
             mock.patch.object(S.M, "validate_claim_boundary", side_effect=lambda x: x), \
             mock.patch.object(S.M.M.M, "verify_frozen_authorities",
                               side_effect=S.B.ReleaseError("M1257 drift")) as inherited, \
             mock.patch.object(S.M.M, "execute_once") as delegate:
            with self.assertRaisesRegex(S.B.ReleaseError, "M1257 drift"):
                self.execute()
        self.assertEqual(inherited.call_count, 1)
        self.assertEqual(delegate.call_count, 0)
        self.assertFalse(self.attempt.exists())

    def test_06_each_predelegate_gate_failure_stops_later_gates_and_attempt(self):
        stages = ("m1306", "m1301", "claims")
        for failed in stages:
            events = []
            def gate(name):
                def inner(*_a, **_k):
                    events.append(name)
                    if name == failed:
                        raise S.B.ReleaseError(name + " blocked")
                return inner
            with self.subTest(failed=failed), \
                 mock.patch.object(S, "verify_frozen_authorities", side_effect=gate("m1306")), \
                 mock.patch.object(S.M, "verify_frozen_authorities", side_effect=gate("m1301")), \
                 mock.patch.object(S.M, "validate_claim_boundary", side_effect=gate("claims")), \
                 mock.patch.object(S.M.M.M, "verify_frozen_authorities", side_effect=gate("m1257")) as inherited, \
                 mock.patch.object(S.M.M, "execute_once") as delegate:
                with self.assertRaises(S.B.ReleaseError):
                    self.execute()
                delegate.assert_not_called()
                self.assertFalse(self.attempt.exists())
                if failed != "claims":
                    self.assertEqual(inherited.call_count, 0)

    def test_07_exact_seven_false_claims_and_drift_attacks(self):
        wanted = {
            "checkpoint_selected_now": False,
            "hardware_rebind_authorized": False,
            "hardware_speedup": False,
            "system_speedup": False,
            "power_or_energy": False,
            "paper_metric": False,
            "remote_execution_authorized": False,
        }
        self.assertEqual(S.EXACT_CLAIM_BOUNDARY, wanted)
        self.assertNotIn("paper_ppa_ready", S.EXACT_CLAIM_BOUNDARY)
        attacks = []
        for key in wanted:
            bad = dict(wanted); bad.pop(key); attacks.append(bad)
            for value in (True, 0, None):
                bad = dict(wanted); bad[key] = value; attacks.append(bad)
        bad = dict(wanted); bad["paper_ppa_ready"] = False; attacks.append(bad)
        for index, bad in enumerate(attacks):
            with self.subTest(index=index), self.assertRaises(S.B.ReleaseError):
                S.M.validate_claim_boundary(bad)

    def test_08_policy_entity_candidate_and_execution_objects_are_identical(self):
        self.assertIs(S.PRODUCTION_POLICY, S.M.PRODUCTION_POLICY)
        self.assertIs(S.PRODUCTION_POLICY, S.M.M.PRODUCTION_POLICY)
        self.assertEqual(S.M.M.TARGET_LINK, Path("/usr/bin/python3"))
        self.assertEqual(S.M.M.TARGET_REALPATH, Path("/usr/bin/python3.12"))
        self.assertEqual(S.M.M.TARGET_ENTITY, {
            "device": 1048625, "inode": 1347357695, "mode": 0x81ED,
            "size_bytes": 8020928, "mtime_sec": 1774292672,
            "sha256": "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7",
            "version": "3.12.3", "memfd_and_all_seals": True})
        self.assertEqual(S.PRODUCTION_POLICY.base.candidates,
                         S.M.M.M.PRODUCTION_POLICY.base.candidates)
        self.assertEqual(S.PRODUCTION_POLICY.base.execution_pins,
                         S.M.M.M.PRODUCTION_POLICY.base.execution_pins)

    def test_09_entity_fd_proc_passfds_snapshots_sealed_sources_and_maps(self):
        prepared = S.M.M.prepare(self.policy, self.repo, self.fx.old.link,
                                 self.fx.old.real, copy.deepcopy(self.fx.old.expected))
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(len(prepared.pass_fds), 4)
            self.assertEqual(prepared.pass_fds,
                             prepared.source_fds + (prepared.interpreter.descriptor,))
            self.assertEqual(prepared.command[0],
                             "/proc/self/fd/{}".format(prepared.interpreter.descriptor))
            self.assertEqual([row["id"] for row in S.M.M.M.M.exact_rebind_targets()],
                             ["E{}".format(i) for i in range(9)])
        finally:
            prepared.close()

    def test_10_interpreter_path_swap_and_entity_field_attacks_fail_before_attempt(self):
        for key in S.M.M.ENTITY_KEYS:
            bad = copy.deepcopy(self.fx.old.expected)
            if key == "memfd_and_all_seals": bad[key] = False
            elif type(bad[key]) is int: bad[key] += 1
            else: bad[key] += "_drift"
            with self.subTest(key=key), self.assertRaises(S.B.ReleaseError):
                S.M.M.open_interpreter_entity(self.fx.old.link, self.fx.old.real, bad)
        prepared = S.M.M.prepare(self.policy, self.repo, self.fx.old.link,
                                 self.fx.old.real, copy.deepcopy(self.fx.old.expected))
        try:
            self.fx.old.link.unlink(); self.fx.old.link.symlink_to("/bin/false")
            with self.assertRaises(S.B.ReleaseError):
                S.M.M.revalidate_entity(prepared.interpreter, self.fx.old.expected)
            self.assertFalse(self.attempt.exists())
        finally:
            prepared.close()

    def test_11_attempt_binds_entity_is_O_EXCL_and_cannot_be_reused(self):
        prepared = S.M.M.prepare(self.policy, self.repo, self.fx.old.link,
                                 self.fx.old.real, copy.deepcopy(self.fx.old.expected))
        try:
            S.M.M.consume_attempt(prepared)
            body = self.attempt.read_text()
            self.assertIn("automatic_retry=false", body)
            self.assertIn("interpreter_entity_sha256=", body)
            self.assertIn("input_snapshot_sha256=", body)
            with self.assertRaises(FileExistsError):
                S.M.M.consume_attempt(prepared)
        finally:
            prepared.close()

    def test_12_failed_child_is_once_only_and_no_automatic_retry(self):
        calls = []
        def fail(command, cwd, pass_fds):
            calls.append((tuple(command), tuple(pass_fds)))
            return subprocess.CompletedProcess(command, 17, stdout="", stderr="fail")
        with self.assertRaisesRegex(S.B.ReleaseError, "no retry"):
            self.execute(runner=fail)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(Exception):
            self.execute(runner=fail)
        self.assertEqual(len(calls), 1)

    def test_13_main_zero_arg_import_inert_and_no_remote_gpu_eda(self):
        import inspect
        self.assertEqual(len(inspect.signature(S.main).parameters), 0)
        text = SOURCE.read_text()
        for forbidden in ("paramiko", "ssh ", "scp ", "rsync ", "import torch",
                          "nvidia-smi", "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(self.attempt.exists())

    def test_14_contract_identity_scope_and_docs_are_closed(self):
        value = json.loads(CONTRACT.read_text())
        self.assertEqual(value["source"]["sha256"], sha(SOURCE))
        self.assertEqual(value["test"]["sha256"], sha(AUTHOR_TEST))
        self.assertEqual(value["claim_boundary"], S.EXACT_CLAIM_BOUNDARY)
        self.assertEqual(value["protected_docs359_sha256"], sha(DOCS359))
        self.assertIs(value["authorization"]["attempt_consumed"], False)
        self.assertIs(value["authorization"]["checkpoint_selected_now"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
