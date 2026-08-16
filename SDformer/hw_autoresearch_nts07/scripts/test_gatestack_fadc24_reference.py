#!/usr/bin/env python3
"""GateStack FADC24金参考单元测试。"""

from __future__ import annotations

import unittest

from gatestack_fadc24_reference import deserialize_fadc24, serialize_fadc24


class Fadc24ReferenceTest(unittest.TestCase):
    def test_list_and_bitmap_round_trip(self) -> None:
        terms = [
            {"gate": 64, "lane": 3, "tokens": [1, 4, 7]},
            {"gate": 128, "lane": 9, "tokens": list(range(0, 162, 3))},
        ]
        payload = serialize_fadc24(terms, tag=0x1234_5678)
        decoded = deserialize_fadc24(payload)
        self.assertEqual(decoded["tag"], 0x1234_5678)
        self.assertEqual(decoded["terms"], terms)
        self.assertEqual(decoded["bitmap_terms"], 1)

    def test_bitmap_padding_is_checked(self) -> None:
        terms = [{"gate": 64, "lane": 0, "tokens": list(range(22))}]
        payload = bytearray(serialize_fadc24(terms, tag=0))
        payload[-1] |= 0x80
        with self.assertRaisesRegex(ValueError, "padding"):
            deserialize_fadc24(bytes(payload))

    def test_duplicate_term_is_rejected(self) -> None:
        terms = [
            {"gate": 64, "lane": 0, "tokens": [1]},
            {"gate": 64, "lane": 0, "tokens": [2]},
        ]
        with self.assertRaisesRegex(ValueError, "重复"):
            serialize_fadc24(terms, tag=0)


if __name__ == "__main__":
    unittest.main()
