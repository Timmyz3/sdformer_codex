#!/usr/bin/env python3

import unittest

import numpy as np

from gatestack_ipd_format_reference import (
    parse_ipd24,
    parse_ipd32w,
    reconstruct,
    serialize_ipd24,
    serialize_ipd32w,
)


class GateStackIpdFormatReferenceTest(unittest.TestCase):
    def test_sparse_roundtrip(self) -> None:
        k_head = np.zeros((162, 32), dtype=bool)
        k_head[1, [0, 7]] = True
        k_head[63, [7, 31]] = True
        k_head[161, [0, 31]] = True
        gate = np.zeros(162, dtype=np.int16)
        gate[[1, 63, 161]] = [256, 3, 256]
        payload = serialize_ipd24(k_head, gate, tag=0x1234_5678)
        self.assertIsNotNone(payload)
        parsed = parse_ipd24(payload, tokens=162, lanes=32)
        actual_k, actual_gated = reconstruct(parsed, tokens=162, lanes=32)
        np.testing.assert_array_equal(actual_k, k_head)
        np.testing.assert_array_equal(
            actual_gated, k_head.astype(np.int16) * gate[:, None]
        )
        self.assertEqual(parsed["tag"], 0x1234_5678)

    def test_class_overflow_returns_raw(self) -> None:
        k_head = np.zeros((162, 32), dtype=bool)
        k_head[:5, 0] = True
        gate = np.zeros(162, dtype=np.int16)
        gate[:5] = np.arange(5)
        self.assertIsNone(serialize_ipd24(k_head, gate, tag=0))
        self.assertIsNone(serialize_ipd32w(k_head, gate, tag=0))

    def test_ipd32w_odd_term_roundtrip(self) -> None:
        k_head = np.zeros((162, 32), dtype=bool)
        k_head[[0, 17, 161], 7] = True
        gate = np.zeros(162, dtype=np.int16)
        gate[[0, 17, 161]] = 256
        payload = serialize_ipd32w(k_head, gate, tag=0x55AA)
        self.assertIsNotNone(payload)
        parsed = parse_ipd32w(payload or b"", tokens=162, lanes=32)
        actual_k, actual_gated = reconstruct(parsed, tokens=162, lanes=32)
        np.testing.assert_array_equal(actual_k, k_head)
        np.testing.assert_array_equal(
            actual_gated, k_head.astype(np.int16) * gate[:, None]
        )

    def test_corrupt_descriptor_reserved_is_rejected(self) -> None:
        k_head = np.zeros((162, 32), dtype=bool)
        k_head[0, 0] = True
        gate = np.zeros(162, dtype=np.int16)
        payload = bytearray(serialize_ipd24(k_head, gate, tag=0) or b"")
        payload[18] |= 0x80
        with self.assertRaises(ValueError):
            parse_ipd24(bytes(payload), tokens=162, lanes=32)


if __name__ == "__main__":
    unittest.main()
