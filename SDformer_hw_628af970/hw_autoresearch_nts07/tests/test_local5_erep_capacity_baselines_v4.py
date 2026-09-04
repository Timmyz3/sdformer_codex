from __future__ import annotations

import itertools
import unittest
from dataclasses import replace

from scripts.local5_erep_capacity_baselines_v4 import (
    C4_PAYLOAD_CAPACITY_BITS,
    C4_PAYLOAD_CAPACITY_RECORDS,
    C4_PAYLOAD_STATIC_UNUSED_BITS,
    C5_STAGE3_WORST_CASE_PAYLOAD_BITS,
    evaluate_c4_first_fit,
    evaluate_c4_oracle,
    evaluate_c5_full,
    head_order_first_fit_admission,
    offline_optimal_admission,
    trace_head_costs,
)
from tests.test_local5_erep_command_schedule_v4 import fixture


class Local5ErepCapacityBaselinesV4Test(unittest.TestCase):
    def test_first_fit_is_unconditional_and_does_not_use_saved_cycles(self) -> None:
        window = fixture()
        costs = trace_head_costs(window)
        self.assertTrue(all(cost.saved_cycles <= 0 for cost in costs))
        self.assertEqual(
            head_order_first_fit_admission(window),
            (True, True, True),
        )

    def test_payload_geometry_is_frozen(self) -> None:
        self.assertEqual(C4_PAYLOAD_CAPACITY_RECORDS, 5014)
        self.assertEqual(C4_PAYLOAD_STATIC_UNUSED_BITS, 32)
        self.assertEqual(
            C4_PAYLOAD_CAPACITY_RECORDS * 112 + C4_PAYLOAD_STATIC_UNUSED_BITS,
            C4_PAYLOAD_CAPACITY_BITS,
        )

    def test_costs_are_derived_only_from_window_traces(self) -> None:
        window = fixture()
        costs = trace_head_costs(window)
        self.assertEqual([cost.input_head for cost in costs], [0, 1, 2])
        for cost, head in zip(costs, window.heads, strict=True):
            self.assertEqual(cost.records, head.epoch_records)
            self.assertEqual(cost.fill_cycles, head.fill.duration)
            self.assertEqual(
                cost.direct_cycles_by_tile,
                tuple(trace.duration for trace in head.direct_by_tile),
            )
            self.assertEqual(
                cost.execute_cycles_by_tile,
                tuple(trace.duration for trace in head.execute_by_tile),
            )

    def test_oracle_matches_exhaustive_trace_derived_search(self) -> None:
        window = fixture()
        costs = trace_head_costs(window)
        capacity = 3 * 112
        observed = offline_optimal_admission(window, capacity_bits=capacity)
        candidates = []
        for bits in itertools.product((False, True), repeat=len(costs)):
            records = sum(cost.records for cost, keep in zip(costs, bits, strict=True) if keep)
            if records <= 3:
                saved = sum(cost.saved_cycles for cost, keep in zip(costs, bits, strict=True) if keep)
                heads = tuple(cost.input_head for cost, keep in zip(costs, bits, strict=True) if keep)
                candidates.append((saved, -records, tuple(-head for head in heads), bits))
        expected = max(candidates)[3]
        self.assertEqual(observed, expected)

    def test_cycle_ledger_has_common_prepare_and_drain_once(self) -> None:
        window = fixture()
        result = evaluate_c4_oracle(window)
        self.assertEqual(
            result.cycles,
            result.common_prepare_cycles
            + result.common_drain_cycles
            + result.direct_path_cycles
            + result.fill_path_cycles
            + result.execute_path_cycles,
        )
        self.assertEqual(
            result.common_prepare_cycles,
            sum(trace.duration for trace in window.prepare_by_tile),
        )
        self.assertEqual(
            result.common_drain_cycles,
            sum(trace.duration for trace in window.drain_by_tile),
        )

    def test_first_fit_has_fixed_metadata_and_exact_record_traffic(self) -> None:
        window = fixture()
        result = evaluate_c4_first_fit(window)
        admitted_records = sum(
            head.epoch_records
            for head, admitted in zip(window.heads, result.admission, strict=True)
            if admitted
        )
        self.assertEqual(result.epoch_record_writes, admitted_records)
        self.assertEqual(
            result.epoch_record_reads,
            admitted_records * window.output_tile_count,
        )
        self.assertEqual(result.tag_bits, 32 * len(result.admitted_heads))
        self.assertEqual(result.valid_bits, len(result.admitted_heads))
        self.assertFalse(result.metadata_in_payload_capacity)

    def test_c5_full_uses_fixed_non_area_matched_payload(self) -> None:
        result = evaluate_c5_full(fixture())
        self.assertTrue(all(result.admission))
        self.assertEqual(result.payload_capacity_bits, C5_STAGE3_WORST_CASE_PAYLOAD_BITS)
        self.assertEqual(result.payload_used_records, 5)

    def test_caller_cannot_supply_cycle_costs_or_non_bool_admission(self) -> None:
        window = fixture()
        head = window.heads[0]
        with self.assertRaises(TypeError):
            replace(head, miss_cycles=1)
        from scripts.local5_erep_capacity_baselines_v4 import evaluate_admission

        with self.assertRaises(ValueError):
            evaluate_admission(
                window,
                (1, False, False),
                baseline="c4_first_fit",
                capacity_bits=C4_PAYLOAD_CAPACITY_BITS,
            )


if __name__ == "__main__":
    unittest.main()
