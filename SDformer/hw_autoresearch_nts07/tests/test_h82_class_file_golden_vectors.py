#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from scripts.generate_h82_class_file_golden_vectors import pack, rows
from scripts.h82_class_file_reference import integer_class_major_gates, q7_codes


class GoldenVectorTests(unittest.TestCase):
    def test_unequal_int_gates_differ_and_singletons_occupy_450(self) -> None:
        packed = {name: pack(name, scores) for name, scores in rows().items()}
        self.assertGreater(packed["unequal_mult"]["n_occupied"], 1)
        self.assertNotEqual(
            packed["unequal_mult"]["h82_gate_q17_int"][0],
            packed["unequal_mult"]["c7_gate_q17_int"][0],
        )
        self.assertEqual(packed["singletons"]["n_occupied"], 450)
        mixed = packed["mixed_pair_mask"]["class_file"]["records"]
        pair0 = next(
            record for record in mixed if any(m["pair_id"] == 0 for m in record["members"])
        )
        self.assertIn(pair0["temporal_mask"], (1, 3))

    def test_integer_path_is_class_broadcast(self) -> None:
        scores = rows()["mixed_pair_mask"]
        codes = q7_codes(scores)
        gates = integer_class_major_gates(codes, preserve_mean=True)
        self.assertTrue(np.all(gates[0:20] == gates[0]))
        self.assertTrue(np.all(gates[20:50] == gates[20]))


if __name__ == "__main__":
    unittest.main()
