#!/usr/bin/env python3
"""Static/no-real-prefix tests for M972."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE.parent / "scripts/execute_m972_m971_decoder_d2d3_10k_evidence_safe_r1.py"
RUNNER = HERE.parent / "scripts/run_m972_m971_decoder_d2d3_10k_evidence_safe_r1_one_shot.sh"
CONTRACT = HW / "contracts/m972_m971_decoder_d2d3_10k_evidence_safe_source_contract_r1_20260829.json"
SPEC = importlib.util.spec_from_file_location("m972_test_driver", DRIVER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import M972 driver")
M972 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M972
SPEC.loader.exec_module(M972)


def fake_row(layer="D2", transactions=157, commits=9):
    return {"row_identity": {"layer": layer,
                             "numerical_route": "EXACT_BINARY_SUPPORT"},
            "prefix": 10000, "elapsed_seconds": 0.2,
            "process_max_rss_kib": 2,
            "exact_miter": {
                "status": "PASS_M768_M861_M890_M896_EXACT_MITER",
                "expanded_request_count": 10000,
                "compressed_transaction_count": transactions,
                "commit_requests_in_prefix": commits,
                "combined_live_event_state_bytes": 2048}}


class M972SourceTest(unittest.TestCase):
    def test_generated_source_fetch_distinguishes_bytes_requests(self):
        self.assertEqual(M972.source_fetch_geometry("D2")["source_bytes"], 231600)
        self.assertEqual(M972.source_fetch_geometry("D2")["source_fetch_requests"], 1207)
        self.assertEqual(M972.source_fetch_geometry("D3")["source_bytes"], 465600)
        self.assertEqual(M972.source_fetch_geometry("D3")["source_fetch_requests"], 2425)

    def test_multi_transaction_and_commit_are_observations_not_rejections(self):
        value = M972.summarize_row(fake_row())
        self.assertEqual(value["observed_compressed_transaction_count"], 157)
        self.assertEqual(value["observed_commit_requests_in_prefix"], 9)
        self.assertFalse(value["prefix_stays_inside_first_source_fetch"])
        self.assertEqual(value["requests_beyond_first_source_fetch"], 8793)

    def test_per_row_success_and_exception_are_persisted_and_sealed(self):
        with tempfile.TemporaryDirectory(prefix="m972_test_") as temporary:
            work = Path(temporary) / (M972.RESULT.name + ".work.test")
            work.mkdir()
            ok = M972.execute_row_to_stage(
                "D2", work / "D2", lambda *args: fake_row("D2"))
            self.assertEqual(ok["payload"]["status"],
                             "PASS_M972_ROW_EXACT__EVIDENCE_SAFE")
            M972._verify_recursive(work / "D2")
            with self.assertRaisesRegex(RuntimeError, "injected"):
                M972.execute_row_to_stage(
                    "D3", work / "D3",
                    lambda *args: (_ for _ in ()).throw(RuntimeError("injected")))
            self.assertTrue((work / "D3/traceback.log").is_file())
            self.assertTrue((work / "D3/failure.json").is_file())
            M972._verify_recursive(work / "D3")

    def test_source_contract_is_inert_and_fresh(self):
        value = M972.validate_source_contract(CONTRACT, RUNNER)
        self.assertEqual(value["status"],
                         "PASS_M972_SOURCE_CONTRACT__NO_10K_EXECUTED")
        self.assertFalse(M972.FUTURE_RELEASE.exists())
        self.assertFalse(M972.RESULT.exists())
        self.assertFalse(M972.ATTEMPT.exists())

    def test_self_test_executes_no_real_prefix(self):
        value = M972.source_self_test()
        self.assertEqual(value["status"],
                         "PASS_M972_SOURCE_SELF_TEST__NO_REAL_PREFIX")
        self.assertFalse(value["real_prefix_executed"])
        self.assertFalse(value["full_row_authorized"])


if __name__ == "__main__":
    unittest.main()
