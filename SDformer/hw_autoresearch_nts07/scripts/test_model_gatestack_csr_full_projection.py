from __future__ import annotations

import unittest

from model_gatestack_csr_full_projection import (
    csr_commit_cycles,
    csr_replay_frontend_cycles,
    resident_replay_frontend_cycles,
    two_scratch_prepare_cycles,
)


class GateStackCsrFullProjectionModelTest(unittest.TestCase):
    def test_csr_commit_counts_scan_terms_and_events(self):
        self.assertEqual(
            csr_commit_cycles(
                mode="TERM_CSR",
                active_lanes=60,
                class_terms=11,
                tokens=162,
                event_compactor_width=4,
            ),
            188,
        )

    def test_exact_event_cycles_override_cross_token_packing(self):
        self.assertEqual(
            csr_commit_cycles(
                mode="TERM_CSR",
                active_lanes=60,
                class_terms=11,
                tokens=162,
                event_compactor_width=4,
                exact_event_cycles=24,
            ),
            197,
        )

    def test_active_token_scan_replaces_full_token_scan(self):
        self.assertEqual(
            csr_commit_cycles(
                mode="TERM_CSR",
                active_lanes=60,
                class_terms=11,
                tokens=162,
                event_compactor_width=4,
                exact_event_cycles=24,
                scan_cycles=18,
            ),
            53,
        )

    def test_raw_commit_is_one_token_per_cycle(self):
        self.assertEqual(
            csr_commit_cycles(
                mode="RAW_CAPACITY_OVERFLOW",
                active_lanes=1000,
                class_terms=70,
                tokens=162,
                event_compactor_width=4,
            ),
            162,
        )

    def test_two_scratch_keeps_head_pipeline_fill_and_drain(self):
        self.assertEqual(two_scratch_prepare_cycles(10, [15, 8, 20]), 55)
        self.assertEqual(two_scratch_prepare_cycles(10, [3]), 13)

    def test_invalid_compactor_width_fails(self):
        with self.assertRaises(ValueError):
            csr_commit_cycles(
                mode="TERM_CSR",
                active_lanes=1,
                class_terms=1,
                tokens=162,
                event_compactor_width=0,
            )

    def test_ipd32w_replay_frontend_counts_header_and_pairs(self):
        self.assertEqual(
            csr_replay_frontend_cycles(mode="TERM_CSR", class_terms=0), 2
        )

    def test_descriptor_residency_removes_only_bounded_csr_frontend(self):
        self.assertEqual(
            resident_replay_frontend_cycles(
                mode="TERM_CSR", class_terms=80, descriptor_cache_terms=80
            ),
            0,
        )
        self.assertEqual(
            resident_replay_frontend_cycles(
                mode="TERM_CSR", class_terms=81, descriptor_cache_terms=80
            ),
            43,
        )
        self.assertEqual(
            resident_replay_frontend_cycles(
                mode="RAW_CAPACITY_OVERFLOW",
                class_terms=1,
                descriptor_cache_terms=80,
            ),
            0,
        )
        self.assertEqual(
            csr_replay_frontend_cycles(mode="TERM_CSR", class_terms=1), 3
        )
        self.assertEqual(
            csr_replay_frontend_cycles(mode="TERM_CSR", class_terms=3), 4
        )
        self.assertEqual(
            csr_replay_frontend_cycles(
                mode="RAW_CAPACITY_OVERFLOW", class_terms=99
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
