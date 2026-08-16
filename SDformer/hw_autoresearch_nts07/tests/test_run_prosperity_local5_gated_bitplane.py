import unittest

import numpy as np

from scripts.run_prosperity_local5_gated_bitplane import (
    reconstruct_group_activation,
)


class Local5GatedBitplaneTest(unittest.TestCase):
    def test_multiset_projection_activation(self):
        actual = reconstruct_group_activation(
            tokens=4,
            lanes=3,
            gate=np.asarray([7, 7, 9], dtype=np.uint16),
            lane=np.asarray([1, 1, 2], dtype=np.uint16),
            multiplicity=np.asarray([2, 1, 3], dtype=np.uint8),
            destination=np.asarray([0, 0, 3], dtype=np.uint16),
        )
        expected = np.zeros((4, 3), dtype=np.uint16)
        expected[0, 1] = 21
        expected[3, 2] = 27
        np.testing.assert_array_equal(actual, expected)

    def test_rejects_out_of_range_term(self):
        with self.assertRaises(ValueError):
            reconstruct_group_activation(
                tokens=2,
                lanes=2,
                gate=np.asarray([1], dtype=np.uint16),
                lane=np.asarray([2], dtype=np.uint16),
                multiplicity=np.asarray([1], dtype=np.uint8),
                destination=np.asarray([0], dtype=np.uint16),
            )


if __name__ == "__main__":
    unittest.main()
