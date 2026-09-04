#!/usr/bin/env python3
from __future__ import annotations

import unittest

from scripts.h82_class_file_cycle_model import compare


class ClassFileCycleModelTests(unittest.TestCase):
    def test_schedule_length_matches_but_exp2_drops(self) -> None:
        result = compare(450, 10, 450)
        self.assertTrue(result["control_cycles_same"])
        self.assertEqual(result["exp2_h82"], 10)
        self.assertEqual(result["exp2_old"], 460)
        self.assertGreater(result["exp2_ratio"], 10.0)


if __name__ == "__main__":
    unittest.main()
