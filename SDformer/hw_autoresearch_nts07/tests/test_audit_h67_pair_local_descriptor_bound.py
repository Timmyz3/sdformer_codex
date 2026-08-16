import unittest

import numpy as np


class PairLocalDescriptorBoundTest(unittest.TestCase):
    def test_bound_and_membership(self):
        s0 = np.array([[1, 2, 3, 4]])
        s1 = np.array([[1, 9, 3, 8]])
        equal = s0 == s1
        pairs = s0.shape[-1]
        lower_bound = 2 * pairs - np.count_nonzero(equal, axis=-1)
        actual = np.where(equal, 1, 2).sum(axis=-1)
        membership = np.where(equal, 2, 2).sum(axis=-1)
        np.testing.assert_array_equal(actual, lower_bound)
        np.testing.assert_array_equal(membership, np.array([2 * pairs]))


if __name__ == "__main__":
    unittest.main()
