import unittest

import numpy as np

from scripts.local5_disep_reference import (
    destination_for_source,
    run_reference,
    source_for_edge,
    topology_mask,
)


class Local5DisepReferenceTest(unittest.TestCase):
    def test_inverse_direction_mapping(self):
        height = 4
        width = 5
        valid = topology_mask(2, height, width)
        for destination in range(valid.shape[0]):
            for direction in range(5):
                source = source_for_edge(
                    destination,
                    direction,
                    height=height,
                    width=width,
                )
                self.assertEqual(source is not None, bool(valid[destination, direction]))
                if source is not None:
                    self.assertEqual(
                        destination_for_source(
                            source,
                            direction,
                            height=height,
                            width=width,
                        ),
                        destination,
                    )

    def test_reference_zero_mismatch(self):
        result = run_reference(20, 0xD15E9)
        self.assertEqual(result["mismatches"], 0)
        self.assertGreater(result["compared_accumulators"], 0)
        self.assertEqual(
            result["deliveries"],
            result["gather_edge_lane_products"],
        )

    def test_topology_has_no_cross_time_edges(self):
        mask = topology_mask(2, 3, 3)
        self.assertEqual(mask.shape, (18, 5))
        self.assertTrue(np.all(mask[:, 0]))


if __name__ == "__main__":
    unittest.main()
