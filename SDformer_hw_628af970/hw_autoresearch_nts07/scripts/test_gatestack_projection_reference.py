from __future__ import annotations

import unittest

import numpy as np

from gatestack_projection_reference import (
    build_head_representation,
    dense_full_projection,
    gatestack_full_projection,
    requantize_signed,
    run_trials,
)


class GateStackProjectionReferenceTest(unittest.TestCase):
    def test_two_heads_and_all_output_tiles_are_exact(self):
        k_event = np.asarray(
            [
                [[1, 0], [0, 1], [1, 1]],
                [[0, 1], [1, 0], [1, 0]],
            ],
            dtype=bool,
        )
        gate = np.asarray([[3, 5, 3], [7, 7, 9]], dtype=np.int16)
        weight = np.arange(-16, 16, dtype=np.int16).reshape(2, 2, 8)
        bias = np.arange(8, dtype=np.int32) - 4
        expected = dense_full_projection(
            k_event, gate, weight, bias, output_tile=4
        )
        actual, counters = gatestack_full_projection(
            k_event,
            gate,
            weight,
            bias,
            class_slots=4,
            output_tile=4,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(counters["heads"], 2)
        self.assertEqual(counters["output_tiles"], 2)
        self.assertEqual(counters["direct_heads"], 0)

    def test_overflow_switches_whole_head_to_exact_direct(self):
        k_event = np.ones((1, 6, 2), dtype=bool)
        gate = np.arange(1, 7, dtype=np.int16)[None, :]
        weight = np.arange(-8, 8, dtype=np.int16).reshape(1, 2, 8)
        bias = np.zeros(8, dtype=np.int32)
        representation = build_head_representation(
            k_event[0], gate[0], class_slots=4
        )
        self.assertEqual(representation["mode"], "DIRECT")
        expected = dense_full_projection(
            k_event, gate, weight, bias, output_tile=4
        )
        actual, counters = gatestack_full_projection(
            k_event,
            gate,
            weight,
            bias,
            class_slots=4,
            output_tile=4,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(counters["direct_heads"], 1)

    def test_global_head_channel_offset_changes_result(self):
        k_event = np.zeros((2, 1, 1), dtype=bool)
        k_event[1, 0, 0] = True
        gate = np.asarray([[1], [2]], dtype=np.int16)
        weight = np.asarray([[[3, 4]], [[11, 13]]], dtype=np.int16)
        bias = np.zeros(2, dtype=np.int32)
        actual, _ = gatestack_full_projection(
            k_event,
            gate,
            weight,
            bias,
            class_slots=4,
            output_tile=2,
        )
        np.testing.assert_array_equal(actual, np.asarray([[22, 26]]))

    def test_requant_rounds_away_from_zero_and_saturates(self):
        values = np.asarray([-1000, -5, -4, 4, 5, 1000], dtype=np.int64)
        actual = requantize_signed(values, right_shift=2, output_bits=8)
        np.testing.assert_array_equal(actual, [-128, -1, -1, 1, 1, 127])

    def test_random_all_stage_head_counts(self):
        result = run_trials(trials=20)
        self.assertEqual(result["mismatches"], 0)
        self.assertGreater(result["trials_with_direct_fallback"], 0)


if __name__ == "__main__":
    unittest.main()
