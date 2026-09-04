#!/usr/bin/env python3

import unittest

from scripts import generate_h67_real_weight_projection_all_vectors as gen


class ProjectionAllVectorTests(unittest.TestCase):
    def test_batch_and_scalar_contract(self) -> None:
        self.assertEqual(gen.BATCH_CHANNELS, 16)
        self.assertEqual(gen.BATCHES, 48)
        self.assertEqual(gen.expected_valid_scalars(), 67392)


if __name__ == "__main__":
    unittest.main()
