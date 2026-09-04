#!/usr/bin/env python3
"""M1278 tests use temporary synthetic M1111DR2 candidates only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m1278_decoder_completion_gate_and_diagnostic_annex.py"
SPEC = importlib.util.spec_from_file_location("m1278_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M; SPEC.loader.exec_module(M)
R = M.load_runner()


class M1278Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="m1278_test.")
        parent = Path(self.temp.name)
        self.layout = M.Layout(parent, parent / M.RESULT_NAME, parent / M.ATTEMPT_NAME,
                               parent / M.LOCK_NAME, parent / M.WORK_NAME,
                               parent / M.ANNEX_NAME)
        self.make_attempt()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_attempt(self) -> None:
        self.layout.attempt.mkdir()
        receipt = {"schema": "m1111dr2_decoder_production_attempt_v2",
            "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS", "maximum_attempts": 1,
            "automatic_retry": False, "canonical_payload_opened_before_attempt": False,
            "runner_sha256": M.RUNNER_SHA256, "contract_sha256": R.CONTRACT_ID[0]}
        (self.layout.attempt / "attempt.json").write_text(
            json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        R.atomic_seal(self.layout.attempt)

    def make_valid_result(self) -> None:
        R.build_publish_self_test_candidate(self.layout.result)
        R.atomic_seal(self.layout.result)

    def reseal_result(self) -> None:
        shutil.rmtree(self.layout.result / R.SEAL_DIR)
        R.atomic_seal(self.layout.result)

    def make_live(self, rows: int) -> None:
        source = self.layout.parent / "source"
        R.build_publish_self_test_candidate(source)
        self.layout.work.mkdir()
        lines = (source / R.CALLS).read_text(encoding="utf-8").splitlines(True)
        (self.layout.work / R.CALLS).write_text("".join(lines[:rows]), encoding="utf-8")
        self.layout.lock.mkdir()
        (self.layout.lock / "owner.json").write_text(json.dumps({
            "pid": M.PRODUCER_PID, "maximum_attempts": 1,
            "automatic_retry": False}, sort_keys=True) + "\n", encoding="utf-8")

    def live_gate(self):
        return M.completion_gate(self.layout, R, alive=lambda _: True,
                                 cmdline=lambda _: M.EXPECTED_CMDLINE)

    def test_01_incomplete_prefix_exits_without_annex(self) -> None:
        self.make_live(7)
        gate = self.live_gate()
        self.assertEqual(gate["state"], "INCOMPLETE")
        self.assertEqual(gate["rows"], 7)
        self.assertFalse(self.layout.annex.exists())

    def test_02_incomplete_duplicate_ordinal_is_rejected(self) -> None:
        self.make_live(3)
        lines = (self.layout.work / R.CALLS).read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[1]); row["global_call_ordinal"] = 0
        lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
        (self.layout.work / R.CALLS).write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(Exception, "identity/order"):
            self.live_gate()

    def test_03_complete_candidate_projects_diagnostic_annex(self) -> None:
        self.make_valid_result()
        gate = M.completion_gate(self.layout, R, alive=lambda _: False)
        gate["source_result"] = self.layout.result
        payload = M.annex_payload(gate)
        self.assertTrue(payload["claim_boundary"]["diagnostic_only"])
        self.assertFalse(payload["claim_boundary"]["table_a"])
        self.assertFalse(payload["claim_boundary"]["system_speedup"])
        self.assertEqual(sum(row["calls"] for row in payload["module_breakdown"]), 120)

    def test_04_wrong_d1_theta_is_rejected(self) -> None:
        self.make_valid_result()
        rows = (self.layout.result / R.CALLS).read_text(encoding="utf-8").splitlines()
        value = json.loads(rows[1]); value["d1_theta_word_uint32"] = 1065353216
        rows[1] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        (self.layout.result / R.CALLS).write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.reseal_result()
        with self.assertRaisesRegex(Exception, "theta|digest"):
            M.completion_gate(self.layout, R, alive=lambda _: False)

    def test_05_bad_result_seal_is_rejected(self) -> None:
        self.make_valid_result()
        with (self.layout.result / R.CALLS).open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(Exception, "seal|atomic member"):
            M.completion_gate(self.layout, R, alive=lambda _: False)

    def test_06_wrong_checkpoint_identity_is_rejected(self) -> None:
        self.make_valid_result()
        path = self.layout.result / R.PAYLOAD
        value = json.loads(path.read_text(encoding="utf-8"))
        value["identity"]["checkpoint_sha256"] = "0" * 64
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.reseal_result()
        with self.assertRaisesRegex(Exception, "identity"):
            M.completion_gate(self.layout, R, alive=lambda _: False)

    def test_07_live_owner_workdir_is_exact(self) -> None:
        self.make_live(1)
        self.layout.work.rename(self.layout.parent / (M.WORK_NAME + ".wrong"))
        with self.assertRaisesRegex(Exception, "exact live work"):
            self.live_gate()

    def test_08_published_annex_has_two_member_seal_and_no_table_a(self) -> None:
        self.make_valid_result()
        gate = M.completion_gate(self.layout, R, alive=lambda _: False)
        gate["source_result"] = self.layout.result
        result = M.publish_annex(self.layout, M.annex_payload(gate))
        self.assertFalse(result["table_a"])
        self.assertFalse(result["replay"])
        self.assertEqual(result["seal"]["members"], 2)
        payload = json.loads((self.layout.annex / "annex.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["claim_boundary"]["paper_headline"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
