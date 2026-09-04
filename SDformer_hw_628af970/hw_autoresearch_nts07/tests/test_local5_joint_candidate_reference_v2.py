from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import local5_joint_candidate_reference_v2 as ref


def empty_head() -> ref.HeadTrace:
    return ref.HeadTrace(
        sources=tuple(ref.SourceTerms(source_id=i, terms=()) for i in range(ref.TOKENS))
    )


class Local5JointCandidateReferenceTest(unittest.TestCase):
    def test_term_order_matches_rtl(self) -> None:
        source = ref.build_source_terms(
            source_id=0,
            plane=0,
            source_y=1,
            source_x=1,
            k_bitmap=(1 << 1) | (1 << 3),
            gates=(7, 9, 7, 0, 9),
            valid_mask=0b11111,
        )
        self.assertEqual(
            [(term.lane, term.gate) for term in source.terms],
            [(1, 7), (1, 9), (3, 7), (3, 9)],
        )
        self.assertEqual([len(term.destinations) for term in source.terms], [2, 2, 2, 2])

    def test_direct_first_touch_then_rmw(self) -> None:
        first = ref.Term(0, 0, 1, ((0, 0),))
        second = ref.Term(0, 1, 1, ((0, 0),))
        sources = [ref.SourceTerms(i, ()) for i in range(ref.TOKENS)]
        sources[0] = ref.SourceTerms(0, (first, second))
        result = ref.simulate_direct_window((ref.HeadTrace(tuple(sources)),))
        head = result.heads[0]
        self.assertEqual(head.term_issue_cycles, 3)
        self.assertEqual(head.first_touch_writes, 1)
        self.assertEqual(head.rmw_terms, 1)
        self.assertEqual(head.sram_reads, 1)
        self.assertEqual(head.sram_writes, 2)
        self.assertEqual(head.descriptor_captures, 1)
        self.assertEqual(
            head.service_cycles,
            1 + ref.ACTIVE_BITMAP_SCAN_CYCLES + 1 + 3 + 1,
        )

    def test_direct_preserves_validity_across_heads(self) -> None:
        term = ref.Term(0, 0, 1, ((0, 0),))
        sources = [ref.SourceTerms(i, ()) for i in range(ref.TOKENS)]
        sources[0] = ref.SourceTerms(0, (term,))
        trace = ref.HeadTrace(tuple(sources))
        result = ref.simulate_direct_window((trace, trace))
        self.assertEqual(result.heads[0].term_issue_cycles, 1)
        self.assertEqual(result.heads[1].term_issue_cycles, 2)
        self.assertEqual(result.final_valid_addresses, 1)
        self.assertEqual(result.final_readout_cycles, 451)

    def test_build_head_trace_rejects_source_misalignment(self) -> None:
        ids = np.arange(ref.TOKENS, dtype=np.uint16)
        ids[2] = 9
        spatial = np.arange(ref.TOKENS) % 225
        with self.assertRaises(ValueError):
            ref.build_head_trace(
                ids,
                np.arange(ref.TOKENS) // 225,
                spatial // 15,
                spatial % 15,
                np.zeros(ref.TOKENS, dtype=np.uint64),
                np.zeros((ref.TOKENS, 5), dtype=np.uint16),
                np.ones(ref.TOKENS, dtype=np.uint8),
            )

    def test_four_candidate_cycles_share_serializer(self) -> None:
        cycles = ref.candidate_window_cycles((empty_head(),), output_tiles=1)
        self.assertEqual(set(cycles), set(ref.CANDIDATES))
        self.assertEqual(len(set(cycles.values())), 1)
        self.assertGreater(cycles["c0_direct_recompute"], ref.SCALAR_SERIALIZER_CYCLES)

    def test_common_descriptor_capture_and_valid_address_contract(self) -> None:
        term = ref.Term(0, 0, 1, ((0, 0), (1, 0)))
        sources = [ref.SourceTerms(i, ()) for i in range(ref.TOKENS)]
        sources[0] = ref.SourceTerms(0, (term,))
        trace = ref.HeadTrace(tuple(sources))
        direct = ref.simulate_direct_window((trace, trace))
        gasr = ref.simulate_srac2_window((trace, trace))
        self.assertEqual([row.descriptor_captures for row in direct.heads], [1, 1])
        self.assertEqual([row.descriptor_captures for row in gasr.heads], [1, 1])
        self.assertEqual(direct.final_valid_sets, gasr.final_valid_sets)

    def test_vault_admission_is_backend_independent(self) -> None:
        term = ref.Term(0, 0, 1, ((0, 0),))
        sources = [ref.SourceTerms(i, ()) for i in range(ref.TOKENS)]
        sources[0] = ref.SourceTerms(0, (term,))
        trace = ref.HeadTrace(tuple(sources))
        self.assertEqual(ref._memo_admission((trace, empty_head())), [True, True])

    def test_replay_and_forced_miss_control_costs_are_explicit(self) -> None:
        backend = 100
        hit = ref._path_cycles(backend, 10, replay=True)
        recompute = ref._path_cycles(backend, 10, replay=False)
        miss = ref._path_cycles(backend, 10, replay=True, forced_replay_miss=True)
        self.assertLess(hit, recompute)
        self.assertGreater(miss, recompute)


if __name__ == "__main__":
    unittest.main()
