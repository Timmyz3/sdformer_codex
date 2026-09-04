#!/usr/bin/env python3

import unittest

from summarize_gatestack_equal96_dctf2c import build_report, render_markdown


class Equal96Dctf2cSummaryTest(unittest.TestCase):
    def test_four_way_metrics(self) -> None:
        stage_rows = []
        for stage, terms, destinations in (
            (0, 2, 4), (1, 0, 0), (2, 3, 6), (3, 10, 50)
        ):
            stage_rows.append({
                "stage": stage,
                "logical_terms": terms,
                "destinations": destinations,
                "destinations_per_term": destinations / terms if terms else 0,
                "cycles": {
                    "central96": 100,
                    "independent32x3": 110,
                    "dctf96": 120,
                },
            })
        equal_cycles = {
            "rows": stage_rows,
            "summary": {
                "cycles": {
                    "central96": 400,
                    "independent32x3": 440,
                    "dctf96": 480,
                },
                "total_logical_terms": 15,
            },
        }
        equal_mapping = {"rows": [
            {"name": "central96_term", "logic_area": 100.0,
             "cells": 1000, "mem_v2": 2},
            {"name": "independent32x3_term", "logic_area": 120.0,
             "cells": 1200, "mem_v2": 6},
            {"name": "dctf96_term", "logic_area": 90.0,
             "cells": 900, "mem_v2": 11},
        ]}
        two_trace = {
            "Icarus": [
                {"stage": stage, "cycles": 90} for stage in range(4)
            ],
            "总周期": 360,
        }
        two_mapping = {"mapping": {
            "logic_area": 95.0,
            "cells": 950,
            "mem_v2": 20,
            "total_architectural_state_bits": 3138,
        }}
        report = build_report(
            equal_cycles, equal_mapping, two_trace, two_mapping
        )
        self.assertEqual(len(report["rows"]), 4)
        self.assertAlmostEqual(report["dctf2c"]["speedup_vs_dctf1c"],
                               480 / 360)
        self.assertAlmostEqual(
            report["rows"][-1]["area_normalized_throughput_vs_central"],
            400 * 100 / (360 * 95),
        )
        self.assertIn("3138 bit", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
