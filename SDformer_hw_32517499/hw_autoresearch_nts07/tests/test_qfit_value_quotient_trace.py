import unittest

from scripts.analyze_qfit_value_quotient_trace import (
    row_owned_segmented_frontier,
    segmented_frontier,
)


def term(seq, plane, y, lane, gate):
    return {
        "seq": seq,
        "plane": plane,
        "y": y,
        "x": seq,
        "lane": lane,
        "gate": gate,
        "mask": 1,
    }


class QfitValueQuotientTraceTest(unittest.TestCase):
    def test_capacity_seal_is_exact_performance_fallback(self):
        rows = [
            term(0, 0, 0, 0, 7),
            term(1, 0, 0, 0, 7),
            term(2, 0, 0, 0, 7),
        ]
        stats = segmented_frontier(
            rows,
            term_capacity=2,
            lane_ways=2,
        )
        self.assertEqual(stats["segments"], 2)
        self.assertEqual(stats["capacity_seals"], 1)
        self.assertEqual(stats["product_computes"], 2)

    def test_row_owned_context_preserves_interleaved_reuse(self):
        rows = [
            term(0, 0, 0, 0, 7),
            term(1, 0, 1, 0, 9),
            term(2, 0, 0, 0, 7),
            term(3, 0, 1, 0, 9),
        ]
        strict = segmented_frontier(
            rows,
            term_capacity=8,
            lane_ways=2,
        )
        owned = row_owned_segmented_frontier(
            rows,
            term_capacity=8,
            lane_ways=2,
        )
        self.assertEqual(strict["product_computes"], 4)
        self.assertEqual(owned["product_computes"], 2)
        self.assertEqual(owned["max_live_rows"], 2)

    def test_way_overflow_seals_without_dropping_future_terms(self):
        rows = [
            term(0, 0, 0, 0, 1),
            term(1, 0, 0, 0, 2),
            term(2, 0, 0, 0, 3),
            term(3, 0, 0, 0, 1),
        ]
        stats = segmented_frontier(
            rows,
            term_capacity=8,
            lane_ways=2,
        )
        self.assertEqual(stats["directory_seals"], 1)
        self.assertEqual(stats["segments"], 2)
        self.assertEqual(stats["product_computes"], 4)


if __name__ == "__main__":
    unittest.main()
