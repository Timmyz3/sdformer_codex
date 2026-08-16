#!/usr/bin/env python3
"""Unit tests for Local5 source-owned Q-silent profiling helpers."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_local5_source_owned_qsilent import (
    PLANE_TOKENS,
    rne_div16_nonnegative,
    source_id,
)


class SourceOwnedQSilentHelpersTest(unittest.TestCase):
    def test_topology_center_and_edges(self) -> None:
        center = 7 * 15 + 7
        self.assertEqual(source_id(center, 0), center)
        self.assertEqual(source_id(center, 1), center - 15)
        self.assertEqual(source_id(center, 2), center + 15)
        self.assertEqual(source_id(center, 3), center - 1)
        self.assertEqual(source_id(center, 4), center + 1)
        self.assertIsNone(source_id(0, 1))
        self.assertIsNone(source_id(0, 3))
        self.assertEqual(source_id(PLANE_TOKENS, 0), PLANE_TOKENS)
        self.assertIsNone(source_id(PLANE_TOKENS, 1))

    def test_rne_div16_ties_to_even(self) -> None:
        self.assertEqual(rne_div16_nonnegative(7), 0)
        self.assertEqual(rne_div16_nonnegative(8), 0)
        self.assertEqual(rne_div16_nonnegative(9), 1)
        self.assertEqual(rne_div16_nonnegative(24), 2)
        self.assertEqual(rne_div16_nonnegative(32), 2)


if __name__ == "__main__":
    unittest.main()
