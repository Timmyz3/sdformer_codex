#!/usr/bin/env python3
"""Local5源驻留缓存profile模型的最小单元测试。"""

import unittest

from profile_local5_source_stationary_acc_cache import (
    geometry_targets,
    simulate_dual_context_prefetch,
    source_targets,
)


class SourceStationaryCacheProfileTest(unittest.TestCase):
    def test_one_source_has_at_most_one_address_per_bank(self) -> None:
        source = [[(0, 3), (1, 7)], [(0, 3)], [(1, 7)]]
        self.assertEqual(source_targets(source), [3, 7, None, None, None])

    def test_geometry_targets_cover_five_distinct_banks(self) -> None:
        targets = geometry_targets(0, 7, 7)
        self.assertEqual(sum(target is not None for target in targets), 5)

    def test_descriptor_latency_is_explicit(self) -> None:
        source0 = [[(0, 1)], [(0, 1)], [(0, 1)]]
        source1 = [[(0, 2)], [(0, 2)]]
        no_latency = simulate_dual_context_prefetch([source0, source1], 0)
        long_latency = simulate_dual_context_prefetch([source0, source1], 4)
        self.assertEqual(no_latency.updates, 5)
        self.assertEqual(no_latency.reads, 0)
        self.assertEqual(no_latency.writes, 2)
        self.assertEqual(no_latency.stall_cycles, 0)
        self.assertEqual(long_latency.stall_cycles, 1)


if __name__ == "__main__":
    unittest.main()
