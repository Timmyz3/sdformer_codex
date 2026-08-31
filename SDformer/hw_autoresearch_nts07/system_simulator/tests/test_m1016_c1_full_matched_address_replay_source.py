#!/usr/bin/env python3
"""Small source tests only; production M410 replay is never invoked."""
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "scripts/run_m1016_c1_full_matched_address_replay.py"
RUNNER = HERE.parent / "scripts/run_m1016_c1_full_matched_address_replay_one_shot.sh"
SPEC = importlib.util.spec_from_file_location("m1016_engine_tested", ENGINE)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1016Test(unittest.TestCase):
    def test_frozen_geometry_is_exact(self):
        self.assertEqual(M.RAW_ROWS, 51_840_000)
        self.assertEqual(M.PHASES, 17_280)
        self.assertEqual(M.TASKS, 812_160)
        self.assertEqual(M.BLOCK_TASKS, 6_497_280)

    def test_empty_and_tiny_coverage_fail_closed(self):
        value = M.small_oracle()
        self.assertEqual(value["status"], "PASS_M1016_SMALL_ORACLE__NO_FULL_REPLAY")
        self.assertTrue(value["empty_coverage_rejected"])
        self.assertTrue(value["tiny_coverage_rejected"])

    def test_no_coverage_cli_or_environment_override(self):
        engine = ENGINE.read_text()
        runner = RUNNER.read_text()
        self.assertNotIn("--coverage-complete", engine)
        self.assertNotIn("COVERAGE_COMPLETE", runner)
        self.assertIn("caller_supplied_coverage\": False", engine)

    def test_exact_source_address_mapping_avoids_last_chunk_gap(self):
        last_chunk = M.task_index(0, 0, M.CHUNKS - 1, 0)
        next_phase = M.task_index(0, 0, 0, 1)
        self.assertEqual(M.source_row_base(last_chunk), (M.CHUNKS - 1) * M.ROW_TILE)
        self.assertEqual(M.source_row_base(next_phase), M.ROWS_PER_PHASE)

    def test_quota_telescopes_to_frozen_totals(self):
        weight = sum(M.quota(M.EXPECTED_SERVICE_COUNTS["weight"], i)
                     for i in range(M.TASKS))
        dma = sum(M.quota(M.EXPECTED_SERVICE_COUNTS["dma"], i)
                  for i in range(M.TASKS))
        self.assertEqual(weight, M.EXPECTED_SERVICE_COUNTS["weight"])
        self.assertEqual(dma, M.EXPECTED_SERVICE_COUNTS["dma"])

    def test_common_receipt_has_five_resources(self):
        receipt = M.common_receipt(0, 64)
        self.assertEqual(set(receipt["counts"]), set(M.RESOURCES))
        self.assertEqual(receipt["counts"]["psum"], 16)
        self.assertEqual(receipt["counts"]["source"], 64)

    def test_parent_stream_is_lazy_and_address_timed(self):
        self.assertTrue(inspect.isgeneratorfunction(M.iter_parent_address_events))
        events = list(M.iter_parent_address_events([1, 3, 5], 1, 50))
        self.assertTrue(events)
        self.assertTrue(all(event["cycle"] >= 50 for event in events))
        self.assertTrue(all("op" in event and "address" in event for event in events))

    def test_baselines_have_no_parent_macro_access(self):
        for design in ("strongest_zero", "same_coordinate_bit"):
            _, summary = M.parent_for_design(design, [1, 3, 5])
            self.assertEqual(summary, {"reads": 0, "writes": 0, "forwards": 0})

    def test_capacity_never_admitted_by_raw_source(self):
        proof = M.DerivedCoverage().proof()
        audit = M.PackingAudit().summary(proof)
        self.assertFalse(audit["capacity_only_214912B_raw_gate_pass"])
        self.assertFalse(audit["capacity_only_214912B_admitted"])
        self.assertTrue(audit["pending_independent_result_hammer"])

    def test_runner_is_future_release_gated_and_cpu_only(self):
        text = RUNNER.read_text()
        for token in ("M1016_RELEASE_JSON", "M1016_RELEASE_HAMMER_DIR",
                      "PASS_M1016_FULL_REPLAY_RELEASE_HAMMER", "max_attempts"):
            self.assertIn(token, text)
        for forbidden in ("/opt/synopsys", "dc_shell", "pt_shell", "pt_shell", "ssh ", "nvidia-smi"):
            self.assertNotIn(forbidden, text.lower())


if __name__ == "__main__":
    unittest.main()
