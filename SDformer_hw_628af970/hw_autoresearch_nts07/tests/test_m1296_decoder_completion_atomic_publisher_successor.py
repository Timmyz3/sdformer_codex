#!/usr/bin/env python3
"""M1296 synthetic-only publisher tests; never open canonical/live state."""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m1296_decoder_completion_atomic_publisher_successor.py"
SPEC = importlib.util.spec_from_file_location("m1296_test_source", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
R = M.A.P.load_runner()


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1296_test.")
        parent = Path(self.temp.name)
        self.layout = M.A.P.Layout(parent, parent / M.A.P.RESULT_NAME,
            parent / M.A.P.ATTEMPT_NAME, parent / M.A.P.LOCK_NAME,
            parent / M.A.P.WORK_NAME, parent / M.NAME)

    def tearDown(self):
        self.temp.cleanup()

    def attempt(self):
        self.layout.attempt.mkdir()
        value = {"schema": "m1111dr2_decoder_production_attempt_v2",
            "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
            "maximum_attempts": 1, "automatic_retry": False,
            "canonical_payload_opened_before_attempt": False,
            "runner_sha256": M.A.P.RUNNER_SHA256,
            "contract_sha256": R.CONTRACT_ID[0]}
        (self.layout.attempt / "attempt.json").write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        R.atomic_seal(self.layout.attempt)

    def result(self):
        R.build_publish_self_test_candidate(self.layout.result)
        R.atomic_seal(self.layout.result)

    def publish(self, hook=None):
        return M._publish_once(self.layout, R, alive=lambda _: False,
                               cmdline=lambda _: b"", after_stage=hook)

    def test_01_predecessor_and_contract_are_frozen(self):
        self.assertEqual(M.sha256(M.OLD), M.OLD_SHA)
        self.assertEqual(M.sha256(M.OLD_CONTRACT), M.OLD_CONTRACT_SHA)
        M.verify_static_authorities()

    def test_02_public_publisher_is_zero_argument_and_no_capability_exists(self):
        self.assertEqual(list(inspect.signature(M.publish_canonical).parameters), [])
        self.assertFalse(any("capability" in name.lower() for name in vars(M)))

    def test_03_incomplete_consumes_no_marker_and_publishes_nothing(self):
        self.attempt()
        with self.assertRaises(Exception):
            self.publish()
        self.assertFalse((self.layout.parent / M.MARKER).exists())
        self.assertFalse(self.layout.annex.exists())

    def test_04_native_schema_token_seal_and_closed_claims(self):
        self.attempt(); self.result(); result = self.publish()
        self.assertFalse(result["table_a"]); self.assertFalse(result["replay"])
        payload = json.loads((self.layout.annex / "annex.json").read_text())
        self.assertEqual(payload["schema"], M.SCHEMA)
        self.assertEqual(payload["status"], M.STATUS)
        self.assertEqual(payload["claim_boundary"], M.EXPECTED_CLAIMS)
        self.assertEqual((self.layout.annex / "RUN_COMPLETE.txt").read_text(), M.TOKEN)
        self.assertTrue((self.layout.annex / M.SEAL / M.MANIFEST).is_file())
        self.assertFalse((self.layout.annex / M.A.P.ANNEX_SEAL).exists())
        marker = json.loads((self.layout.parent / M.MARKER).read_text())
        self.assertEqual(marker["state"], "COMMITTED")

    def test_05_second_publication_is_stopped_by_persistent_marker(self):
        self.attempt(); self.result(); self.publish()
        with self.assertRaises(FileExistsError):
            self.publish()
        marker = json.loads((self.layout.parent / M.MARKER).read_text())
        self.assertEqual(marker["state"], "COMMITTED")

    def test_06_alternate_destination_is_rejected(self):
        self.attempt(); self.result()
        alternate = M.A.P.Layout(self.layout.parent, self.layout.result,
            self.layout.attempt, self.layout.lock, self.layout.work,
            self.layout.parent / "alternate_annex")
        with self.assertRaisesRegex(Exception, "sole canonical"):
            M._publish_once(alternate, R, alive=lambda _: False, cmdline=lambda _: b"")
        self.assertFalse((self.layout.parent / M.MARKER).exists())

    def test_07_late_result_mutation_fails_and_retains_marker_and_stage(self):
        self.attempt(); self.result()
        def attack(_layout, snapshot, _stage):
            with (snapshot.root_path / M.A.P.PAYLOAD).open("a", encoding="utf-8") as stream:
                stream.write("\n")
        with self.assertRaisesRegex(Exception, "identity drift|seal drift"):
            self.publish(attack)
        self.assertFalse(self.layout.annex.exists())
        marker = json.loads((self.layout.parent / M.MARKER).read_text())
        self.assertEqual(marker["state"], "FAILED")
        self.assertFalse(marker["automatic_retry"])
        self.assertFalse(marker["automatic_rollback"])
        stages = list(self.layout.parent.glob("." + M.NAME + ".stage.*"))
        self.assertEqual(len(stages), 1)
        self.assertTrue((stages[0] / M.SEAL / M.MANIFEST).is_file())

    def test_08_result_root_replacement_is_rejected_after_stage(self):
        self.attempt(); self.result()
        def attack(layout, _snapshot, _stage):
            old = layout.parent / "old_result"
            os.rename(layout.result, old)
            R.build_publish_self_test_candidate(layout.result)
            R.atomic_seal(layout.result)
        with self.assertRaisesRegex(Exception, "root replacement"):
            self.publish(attack)
        self.assertFalse(self.layout.annex.exists())
        self.assertEqual(json.loads((self.layout.parent / M.MARKER).read_text())["state"],
                         "FAILED")

    def test_09_contract_claim_promotion_is_rejected(self):
        promoted = copy.deepcopy(M._exact_contract())
        promoted["claim_boundary"]["table_a"] = True
        path = Path(self.temp.name) / "promoted.json"
        path.write_text(json.dumps(promoted, sort_keys=True) + "\n")
        old = M.CONTRACT
        M.CONTRACT = path
        try:
            with self.assertRaisesRegex(Exception, "contract drift"):
                M.verify_static_authorities()
        finally:
            M.CONTRACT = old

    def test_10_final_claim_and_source_identity_promotions_are_rejected(self):
        self.attempt(); self.result()
        snap = M._snapshot(self.layout, R)
        try:
            gate = M._revalidate_snapshot(self.layout, R, snap)
            payload = M._build_payload(self.layout, R, gate, snap)
            for key in ("table_a", "full_network", "system_speedup", "paper_headline"):
                value = copy.deepcopy(payload); value["claim_boundary"][key] = True
                with self.assertRaisesRegex(Exception, "claim boundary"):
                    M._validate_payload(self.layout, R, gate, snap, value)
            value = copy.deepcopy(payload)
            value["source_result"]["payload_sha256"] = "0" * 64
            with self.assertRaisesRegex(Exception, "source identity"):
                M._validate_payload(self.layout, R, gate, snap, value)
        finally:
            snap.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
