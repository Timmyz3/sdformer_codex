import unittest

from generate_gatestack_h67_stage3_trace import (
    RAW_HEAD_BITS,
    WORDS_PER_HEAD,
    build_terms,
    distribute_counts,
    serialize_ipd,
    serialize_raw,
)
from gatestack_ipd_format_reference import parse_ipd32w


class GateStackTraceGeneratorTests(unittest.TestCase):
    def test_distribution_preserves_sum_and_max(self):
        counts = distribute_counts(7, 31, 8)
        self.assertEqual(sum(counts), 31)
        self.assertEqual(max(counts), 8)
        self.assertTrue(all(value > 0 for value in counts))

    def test_ipd_roundtrip(self):
        terms = build_terms(7, 31, 2, 8)
        payload, bits = serialize_ipd(terms, 0x68000002)
        parsed = parse_ipd32w(payload, tokens=162, lanes=32)
        self.assertEqual(bits, len(payload) * 8)
        self.assertEqual(parsed["tag"], 0x68000002)
        self.assertEqual(parsed["term_count"], 7)
        self.assertEqual(parsed["event_count"], 31)
        self.assertEqual(len({term["gate_code"] for term in parsed["terms"]}), 2)

    def test_raw_has_fixed_capacity(self):
        terms = build_terms(61, 814, 3, 52)
        payload, bits = serialize_raw(terms)
        self.assertEqual(bits, RAW_HEAD_BITS)
        self.assertEqual(len(payload), WORDS_PER_HEAD * 8)

    def test_empty_head(self):
        self.assertEqual(build_terms(0, 0, 0, 0), [])
        payload, bits = serialize_ipd([], 7)
        parsed = parse_ipd32w(payload, tokens=162, lanes=32)
        self.assertEqual(bits, 128)
        self.assertEqual(parsed["term_count"], 0)


if __name__ == "__main__":
    unittest.main()
