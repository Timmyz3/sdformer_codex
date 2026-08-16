import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_h67_zkqi_threeway_production_physical.py"
SPEC = importlib.util.spec_from_file_location("production_physical", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProductionPhysicalParserTest(unittest.TestCase):
    def test_required_metrics_and_zero_unconstrained(self):
        text = """
finish check_setup
finish setup_violation_count
setup violation count 0
finish critical path delay
---------------------------
4.8123
"""
        self.assertEqual(
            MODULE.require_last_int(r"setup violation count\s+(\d+)", text, "setup"), 0
        )
        self.assertAlmostEqual(
            MODULE.require_last_float(
                r"finish critical path delay\s*-+\s*([0-9.]+)", text, "delay"
            ),
            4.8123,
        )
        matches = MODULE.re.findall(
            r"Warning: There (?:is|are) (\d+) unconstrained endpoint", text
        )
        self.assertEqual(max((int(value) for value in matches), default=0), 0)

    def test_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "证据.txt"
            path.write_text("abc", encoding="utf-8")
            receipt = MODULE.receipts([path])[0]
            self.assertEqual(receipt["bytes"], 3)
            self.assertEqual(len(receipt["sha256"]), 64)

    def test_fixed_clock_comparison_does_not_infer_fmax(self):
        baseline = {"instance_area_um2": 100.0, "critical_path_delay_ns": 5.2}
        candidate = {"instance_area_um2": 110.0, "critical_path_delay_ns": 4.0}
        result = MODULE.comparison(baseline, candidate, 120, 100)
        self.assertAlmostEqual(result["fixed_5ns_throughput_ratio"], 1.2)
        self.assertAlmostEqual(result["fixed_5ns_area_normalized_throughput"], 1.2 / 1.1)
        self.assertNotIn("frequency_ratio_proxy", result)


if __name__ == "__main__":
    unittest.main()
