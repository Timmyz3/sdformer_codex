#!/usr/bin/env python3
"""Synthetic-only tests for M1284; never call canonical/live main."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m1284_decoder_completion_gate_diagnostic_annex_successor.py"
SPEC = importlib.util.spec_from_file_location("m1284_test_source", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M; SPEC.loader.exec_module(M)
P = M.P; R = P.load_runner()


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1284_test.")
        parent = Path(self.temp.name)
        self.layout = P.Layout(parent, parent / P.RESULT_NAME, parent / P.ATTEMPT_NAME,
            parent / P.LOCK_NAME, parent / P.WORK_NAME, parent / M.ANNEX_NAME)

    def tearDown(self): self.temp.cleanup()

    def attempt(self, maximum=1):
        self.layout.attempt.mkdir()
        value = {"schema": "m1111dr2_decoder_production_attempt_v2",
            "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
            "maximum_attempts": maximum, "automatic_retry": False,
            "canonical_payload_opened_before_attempt": False,
            "runner_sha256": P.RUNNER_SHA256, "contract_sha256": R.CONTRACT_ID[0]}
        (self.layout.attempt / "attempt.json").write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        R.atomic_seal(self.layout.attempt)

    def result(self):
        R.build_publish_self_test_candidate(self.layout.result)
        R.atomic_seal(self.layout.result)

    def live(self, rows=3):
        source = self.layout.parent / "source"; R.build_publish_self_test_candidate(source)
        self.layout.work.mkdir()
        lines = (source / R.CALLS).read_text(encoding="utf-8").splitlines(True)
        (self.layout.work / R.CALLS).write_text("".join(lines[:rows]), encoding="utf-8")
        self.layout.lock.mkdir()
        (self.layout.lock / "owner.json").write_text(json.dumps({
            "pid": P.PRODUCER_PID, "maximum_attempts": 1,
            "automatic_retry": False}, sort_keys=True) + "\n", encoding="utf-8")

    def capability(self):
        return M.completion_capability(self.layout, R, alive=lambda _: False)

    def valid_gate_payload(self):
        cap = self.capability()
        gate = copy.deepcopy(cap.gate)
        return gate, M.build_annex(self.layout, R, gate)

    def test_01_predecessor_is_frozen(self):
        self.assertEqual(M.sha256(M.PREDECESSOR), M.PREDECESSOR_SHA256)
        self.assertEqual(M.sha256(M.PREDECESSOR_CONTRACT), M.PREDECESSOR_CONTRACT_SHA256)

    def test_02_bool_attempt_is_rejected(self):
        self.attempt(True); self.result()
        with self.assertRaisesRegex(Exception, "exact integer"):
            self.capability()

    def test_03_incomplete_issues_no_capability_or_output(self):
        self.attempt(); self.live()
        with self.assertRaises(P.Incomplete):
            M.completion_capability(self.layout, R, alive=lambda _: True,
                                    cmdline=lambda _: P.EXPECTED_CMDLINE)
        self.assertFalse(self.layout.annex.exists())

    def test_04_plain_object_is_not_capability(self):
        self.attempt(); self.result()
        with self.assertRaisesRegex(Exception, "capability"):
            M.publish_with_capability(self.layout, R, object())

    def test_05_valid_capability_publishes_diagnostic_only(self):
        self.attempt(); self.result(); cap = self.capability()
        out = M.publish_with_capability(self.layout, R, cap)
        self.assertFalse(out["table_a"]); self.assertFalse(out["replay"])
        payload = json.loads((self.layout.annex / "annex.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], M.ANNEX_SCHEMA)
        self.assertFalse(payload["claim_boundary"]["system_speedup"])

    def test_06_capability_is_single_use(self):
        self.attempt(); self.result(); cap = self.capability()
        M.publish_with_capability(self.layout, R, cap)
        with self.assertRaisesRegex(Exception, "unused completion capability"):
            M.publish_with_capability(self.layout, R, cap)

    def test_07_final_checkpoint_promotion_is_rejected(self):
        self.attempt(); self.result(); gate, payload = self.valid_gate_payload()
        payload["identity"]["checkpoint"] = "final"
        with self.assertRaisesRegex(Exception, "ep35 identity"):
            M.validate_annex(self.layout, R, gate["checked"], payload)

    def test_08_table_system_headline_promotions_are_rejected(self):
        self.attempt(); self.result(); gate, payload = self.valid_gate_payload()
        for key in ("table_a", "full_network", "system_speedup", "paper_headline"):
            value = copy.deepcopy(payload); value["claim_boundary"][key] = True
            with self.assertRaisesRegex(Exception, "annex claim"):
                M.validate_annex(self.layout, R, gate["checked"], value)

    def test_09_bool_module_count_is_rejected(self):
        self.attempt(); self.result(); gate, payload = self.valid_gate_payload()
        payload["module_breakdown"][0]["calls"] = True
        with self.assertRaisesRegex(Exception, "exact integer"):
            M.validate_annex(self.layout, R, gate["checked"], payload)

    def test_10_bool_row_counter_is_rejected(self):
        self.attempt(); self.result(); cap = self.capability()
        cap.gate["rows"][0]["transaction_count"] = True
        with self.assertRaisesRegex(Exception, "projection|exact integer"):
            M.publish_with_capability(self.layout, R, cap)

    def test_11_payload_argument_surface_is_closed(self):
        import inspect
        self.assertEqual(list(inspect.signature(M.publish_with_capability).parameters),
                         ["layout", "runner", "capability"])
        self.assertFalse(hasattr(M, "_issue_capability"))

    def test_12_result_claim_integer_zero_is_rejected_by_exact_bool(self):
        self.attempt(); self.result(); gate = P.completion_gate(
            self.layout, R, alive=lambda _: False)
        gate["checked"]["payload"]["claim_boundary"]["speedup_admitted"] = 0
        with self.assertRaisesRegex(Exception, "projection|exact boolean"):
            M.validate_complete_gate(self.layout, R, gate)


if __name__ == "__main__": unittest.main(verbosity=2)
