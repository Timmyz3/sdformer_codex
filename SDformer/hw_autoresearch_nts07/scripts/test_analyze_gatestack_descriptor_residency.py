#!/usr/bin/env python3

import unittest

from analyze_gatestack_descriptor_residency import summarize_depth


class GateStackDescriptorResidencyTest(unittest.TestCase):
    def test_depth_routes_only_bounded_csr_heads(self) -> None:
        rows = {
            0: [(10, 8, 2), (100, 33, 3), (1, 1, 5)],
            1: [(10, 4, 1)],
            2: [(10, 4, 1)],
            3: [(10, 4, 1)],
        }
        result = summarize_depth(rows, 32)
        self.assertEqual(result["csr_rows"], 5)
        self.assertEqual(result["cached_rows"], 4)
        self.assertGreater(result["weighted_frontend_cycle_reduction"], 0.0)

    def test_cache_storage_scales_with_depth(self) -> None:
        rows = {stage: [(10, 4, 1)] for stage in range(4)}
        shallow = summarize_depth(rows, 32)
        deep = summarize_depth(rows, 80)
        self.assertLess(
            shallow["stages"]["3"]["dual_context_cache_kib"],
            deep["stages"]["3"]["dual_context_cache_kib"],
        )


if __name__ == "__main__":
    unittest.main()
