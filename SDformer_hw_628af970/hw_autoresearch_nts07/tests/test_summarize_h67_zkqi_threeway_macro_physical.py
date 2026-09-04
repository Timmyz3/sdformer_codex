from __future__ import annotations

import unittest

from scripts.summarize_h67_zkqi_threeway_macro_physical import throughput_metrics


class SummarizeH67ZkqiThreewayMacroPhysicalTest(unittest.TestCase):
    def test_throughput_includes_cycles_frequency_and_area(self) -> None:
        baseline = {"critical_path_delay_ns": 5.0, "instance_area_um2": 100.0}
        candidate = {"critical_path_delay_ns": 4.0, "instance_area_um2": 125.0}
        result = throughput_metrics(baseline, candidate, 200, 100)
        self.assertAlmostEqual(result["cycle_speedup"], 2.0)
        self.assertAlmostEqual(result["frequency_ratio_proxy"], 1.25)
        self.assertAlmostEqual(result["frequency_adjusted_throughput_proxy"], 2.5)
        self.assertAlmostEqual(result["area_ratio"], 1.25)
        self.assertAlmostEqual(result["area_normalized_throughput_proxy"], 2.0)


if __name__ == "__main__":
    unittest.main()
