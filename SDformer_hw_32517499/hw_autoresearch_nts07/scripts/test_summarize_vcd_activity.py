import tempfile
import unittest
from pathlib import Path

try:
    from summarize_vcd_activity import parse_vcd
except ModuleNotFoundError:
    from scripts.summarize_vcd_activity import parse_vcd


class SummarizeVcdActivityTest(unittest.TestCase):
    def test_known_bit_toggles_exclude_unknown_transitions(self) -> None:
        content = """$timescale 1ns $end
$scope module top $end
$scope module dut $end
$var wire 1 ! scalar $end
$var wire 2 # vector [1:0] $end
$upscope $end
$upscope $end
$enddefinitions $end
#0
x!
bxx #
#5
0!
b00 #
#10
1!
b11 #
#15
0!
b10 #
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.vcd"
            path.write_text(content, encoding="utf-8")
            result = parse_vcd(path)
        self.assertEqual(result["declared_variables"], 2)
        self.assertEqual(result["variables_with_updates"], 2)
        self.assertEqual(result["total_known_bit_toggles"], 5)
        self.assertEqual(result["timescale_ticks"]["span"], 15)


if __name__ == "__main__":
    unittest.main()
