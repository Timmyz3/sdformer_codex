from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import local5_joint_candidate_reference_v3 as ref


def empty_head() -> ref.HeadTrace:
    return ref.HeadTrace(
        sources=tuple(ref.SourceTerms(source_id=i, terms=()) for i in range(ref.TOKENS))
    )


def one_term_head() -> ref.HeadTrace:
    term = ref.Term(source_id=0, lane=0, gate=7, destinations=((0, 0),))
    sources = [ref.SourceTerms(source_id=i, terms=()) for i in range(ref.TOKENS)]
    sources[0] = ref.SourceTerms(source_id=0, terms=(term,))
    return ref.HeadTrace(sources=tuple(sources))


class Local5JointCandidateReferenceV3Test(unittest.TestCase):
    def test_ordered_work_uses_capture_term_and_stall(self) -> None:
        row = ref.BackendHeadResult(
            descriptor_captures=11, term_issue_cycles=23, stall_cycles=5
        )
        self.assertEqual(ref.ordered_work_cycles(row), 39)
        self.assertEqual(ref.recompute_path_cycles(row, fixed_cycles=459), 498)
        self.assertEqual(ref.recompute_path_cycles(row, fixed_cycles=475), 514)

    def test_replay_retains_read_and_builder_capture(self) -> None:
        row = ref.BackendHeadResult(
            descriptor_captures=3, term_issue_cycles=5, stall_cycles=2
        )
        self.assertEqual(
            ref.replay_path_cycles(row, 3, fixed_cycles=459),
            ref.REPLAY_CONTROL_CYCLES + 3 + 3 + 5 + 2 + 1,
        )

    def test_forced_miss_is_more_expensive_than_recompute(self) -> None:
        row = ref.BackendHeadResult(
            descriptor_captures=1, term_issue_cycles=1, stall_cycles=0
        )
        recompute = ref.recompute_path_cycles(row, fixed_cycles=459)
        miss = ref.replay_path_cycles(
            row, 1, forced_replay_miss=True, fixed_cycles=459
        )
        self.assertGreater(miss, recompute)

    def test_empty_window_has_no_candidate_bias(self) -> None:
        scenarios = ref.candidate_window_cycle_scenarios((empty_head(),), 1)
        self.assertEqual(set(scenarios), set(ref.FIXED_SCENARIOS))
        for candidates in scenarios.values():
            self.assertEqual(set(candidates), set(ref.CANDIDATES))
            self.assertEqual(len(set(candidates.values())), 1)

    def test_conservative_fixed_is_never_faster(self) -> None:
        scenarios = ref.candidate_window_cycle_scenarios((one_term_head(),), 1)
        median = scenarios["calibrated_median_459"]
        conservative = scenarios["calibration_max_475"]
        for candidate in ref.CANDIDATES:
            self.assertGreaterEqual(conservative[candidate], median[candidate])

    def test_memo_admission_is_backend_independent(self) -> None:
        heads = (one_term_head(), empty_head())
        self.assertEqual(ref.memo_admission(heads), [True, True])


if __name__ == "__main__":
    unittest.main()
