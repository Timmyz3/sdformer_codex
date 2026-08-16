#!/usr/bin/env python3
"""Focused tests for H67 real-weight projection2 vector helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import generate_h67_real_weight_projection2_vectors as gen


class Projection2VectorTests(unittest.TestCase):
    def test_expected_all12_row_count(self) -> None:
        self.assertEqual(len(gen.expected_names()), 12)
        self.assertEqual(
            sum(gen.EXPECTED_BLOCKS[s] * gen.EXPECTED_HEADS[s] for s in range(4)),
            138,
        )

    def test_parse_base_vectors_rejects_non_t450_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.txt"
            path.write_text("138 162\n", encoding="ascii")
            with self.assertRaises(ValueError):
                gen.parse_base_vectors(path)

    def test_bitmap32(self) -> None:
        import numpy as np

        bits = np.zeros(32, dtype=bool)
        bits[[0, 7, 31]] = True
        self.assertEqual(gen.bitmap32(bits), (1 << 0) | (1 << 7) | (1 << 31))


if __name__ == "__main__":
    unittest.main()
