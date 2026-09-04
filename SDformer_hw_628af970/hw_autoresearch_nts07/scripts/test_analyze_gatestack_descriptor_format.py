#!/usr/bin/env python3

import unittest

from analyze_gatestack_descriptor_format import FORMATS, classify


class GateStackDescriptorFormatTest(unittest.TestCase):
    def test_implicit_prefix_is_never_larger_than_legacy(self) -> None:
        legacy = FORMATS[0]
        ipd24 = FORMATS[2]
        for terms in range(129):
            for events in (0, 1, 31, 162, 1024, 5184):
                legacy_mode, legacy_bits = classify(events, terms, 4, legacy)
                ipd_mode, ipd_bits = classify(events, terms, 4, ipd24)
                if legacy_mode == "CSR":
                    self.assertEqual(ipd_mode, "CSR")
                    self.assertLessEqual(ipd_bits, legacy_bits)

    def test_class_overflow_has_priority(self) -> None:
        mode, bits = classify(0, 0, 5, FORMATS[2])
        self.assertEqual(mode, "RAW_CLASS")
        self.assertEqual(bits, 6642)

    def test_byte_aligned_capacity_boundary(self) -> None:
        fmt = FORMATS[2]
        mode, _ = classify(796, 6, 4, fmt)
        self.assertEqual(128 + 24 * 6 + 8 * 796, 6640)
        self.assertEqual(mode, "CSR")
        mode, _ = classify(797, 6, 4, fmt)
        self.assertEqual(mode, "RAW_CAPACITY")


if __name__ == "__main__":
    unittest.main()
