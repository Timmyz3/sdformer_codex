#!/usr/bin/env python3

import unittest

from analyze_hatf_supertile_traffic import parse_result_line, summarize


class HatfTrafficTest(unittest.TestCase):
    def test_parse_result(self) -> None:
        fields = parse_result_line(
            "noise\nRESULT status=PASS projection_terms=62 bias=162 total_cycles=1843 mismatches=0\n"
        )
        self.assertEqual(fields["projection_terms"], "62")
        self.assertEqual(fields["status"], "PASS")

    def test_96_preserves_physical_accesses(self) -> None:
        rows = []
        terms = {32: 3, 64: 2, 96: 1, 128: 1}
        biases = {32: 3, 64: 2, 96: 1, 128: 1}
        for width in (32, 64, 96, 128):
            for stage in range(4):
                rows.append(
                    {
                        "width": width,
                        "stage": stage,
                        "terms": terms[width],
                        "bias_commits": biases[width],
                        "total_cycles": 1,
                        "mismatches": 0,
                    }
                )
        summary = {int(row["width"]): row for row in summarize(rows)}
        self.assertEqual(
            summary[32]["physical_weight_bank_accesses"],
            summary[96]["physical_weight_bank_accesses"],
        )
        self.assertEqual(summary[96]["weight_padding_overhead_pct"], 0.0)
        self.assertGreater(summary[64]["weight_padding_overhead_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
