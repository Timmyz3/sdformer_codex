import unittest

import numpy as np

from scripts.run_prosperity_motion_gated_bitplane import split_active_bitplanes


class GatedBitplaneTest(unittest.TestCase):
    def test_exact_reconstruction_and_density_order(self):
        activation = np.asarray(
            [[[0, 1, 2, 3, 4, 7, 64, 81]]], dtype=np.uint16
        )
        planes = split_active_bitplanes(activation)
        self.assertEqual([bit for bit, _ in planes], [0, 1, 2, 6, 4])
        reconstructed = sum(
            plane.astype(np.uint16) << bit for bit, plane in planes
        )
        np.testing.assert_array_equal(reconstructed, activation)

    def test_zero_input_has_no_active_plane(self):
        activation = np.zeros((2, 3, 4), dtype=np.uint16)
        self.assertEqual(split_active_bitplanes(activation), [])


if __name__ == "__main__":
    unittest.main()
