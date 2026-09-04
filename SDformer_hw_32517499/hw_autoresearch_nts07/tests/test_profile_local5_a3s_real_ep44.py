#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_local5_a3s_real_ep44.py"
SPEC = importlib.util.spec_from_file_location("profile_local5_a3s_real_ep44", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BitmapTests(unittest.TestCase):
    def test_bitmap_is_lsb_first(self) -> None:
        got = MODULE.bitmap_to_bits(np.asarray([0b1001], dtype=np.uint64), lanes=4)
        np.testing.assert_array_equal(got, [[1.0, 0.0, 0.0, 1.0]])


class GeometryTests(unittest.TestCase):
    def test_source_transpose_preserves_role_and_boundary(self) -> None:
        source, valid = MODULE.build_local5_geometry(time_planes=2, spatial_side=3)
        tokens = 18
        gates = torch.arange(tokens * 5, dtype=torch.int64).reshape(1, tokens, 5)
        incoming, incoming_valid = MODULE.destination_to_source(gates, source, valid)
        self.assertEqual(tuple(incoming.shape), (1, tokens, 5))
        self.assertTrue(torch.equal(incoming_valid[0, :, 0], torch.ones(tokens, dtype=torch.bool)))
        for role in range(5):
            role_valid = valid[:, role]
            selected = source[role_valid, role]
            self.assertTrue(
                torch.equal(incoming[0, selected, role], gates[0, role_valid, role])
            )
            self.assertEqual(int(incoming_valid[0, :, role].sum()), int(role_valid.sum()))

    def test_valid_mask_codes(self) -> None:
        incoming_valid = torch.tensor([[[1, 1, 0, 1, 0], [1, 0, 1, 0, 1]]]).bool()
        self.assertEqual(MODULE.valid_mask_codes(incoming_valid).tolist(), [[11, 21]])


class SourceWorkTests(unittest.TestCase):
    def test_unique_gate_products_and_updates(self) -> None:
        gates = torch.tensor(
            [[[16, 16, 0, 32, 32], [0, 8, 8, 8, 0]]], dtype=torch.int64
        )
        kpop = torch.tensor([[3, 2]], dtype=torch.int64)
        unique, products, updates = MODULE.source_work(gates, kpop)
        self.assertEqual(unique.tolist(), [[2, 1]])
        self.assertEqual(products.tolist(), [[6, 2]])
        self.assertEqual(updates.tolist(), [[12, 6]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
