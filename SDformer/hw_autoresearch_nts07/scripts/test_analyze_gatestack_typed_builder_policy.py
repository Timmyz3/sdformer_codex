#!/usr/bin/env python3
"""GateStack三格式策略公式单元测试。"""

import unittest

from analyze_gatestack_typed_builder_policy import decide_format


class TypedBuilderPolicyTest(unittest.TestCase):
    def test_ipd_exact_slot_boundary(self) -> None:
        result = decide_format(
            active_classes=4, term_count=6, event_count=792,
            fadc_destination_bytes=126,
        )
        self.assertEqual(result["format"], "IPD32W")
        self.assertEqual(result["ipd_bytes"], 832)

    def test_ipd_overflow_uses_fadc(self) -> None:
        result = decide_format(
            active_classes=4, term_count=6, event_count=798,
            fadc_destination_bytes=126,
        )
        self.assertEqual(result["format"], "FADC24")
        self.assertEqual(result["reason"], "ipd_capacity")

    def test_class_overflow_uses_fadc(self) -> None:
        result = decide_format(
            active_classes=5, term_count=1, event_count=1,
            fadc_destination_bytes=1,
        )
        self.assertEqual(result["format"], "FADC24")
        self.assertEqual(result["reason"], "ipd_class")

    def test_both_overflow_use_raw(self) -> None:
        result = decide_format(
            active_classes=4, term_count=255, event_count=255,
            fadc_destination_bytes=255,
        )
        self.assertEqual(result["format"], "RAW41")
        self.assertEqual(result["word_count"], 104)

    def test_metadata_overflow_is_fail_safe(self) -> None:
        result = decide_format(
            active_classes=1, term_count=1, event_count=1,
            fadc_destination_bytes=1, metadata_overflow=True,
        )
        self.assertEqual(result["format"], "RAW41")
        self.assertEqual(result["reason"], "metadata_overflow")


if __name__ == "__main__":
    unittest.main()
