from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.analyze_rqtb_vcd_activity import analyze


class AnalyzeRqtbVcdActivityTest(unittest.TestCase):
    def test_bit_level_toggle_count(self) -> None:
        vcd = """$scope module tb $end
$scope module u_fixed $end
$var wire 2 ! data $end
$upscope $end
$scope module u_rqtb $end
$var wire 2 \" data $end
$upscope $end
$upscope $end
$enddefinitions $end
#0
b00 !
b00 \"
#1
b11 !
b01 \"
#2
b10 !
b11 \"
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "activity.vcd"
            path.write_text(vcd, encoding="ascii")
            result = analyze(path)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["bit_toggles"], {"fixed": 3, "rqtb": 2})
        self.assertAlmostEqual(result["rqtb_reduction_ratio"], 1.0 / 3.0)

    def test_cross_design_alias_is_excluded(self) -> None:
        vcd = """$scope module tb $end
$scope module u_fixed $end
$var wire 1 ! shared $end
$var wire 1 # private $end
$upscope $end
$scope module u_rqtb $end
$var wire 1 ! shared $end
$var wire 1 $ private $end
$upscope $end
$upscope $end
$enddefinitions $end
#0
0!
0#
0$
#1
1!
1#
1$
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "alias.vcd"
            path.write_text(vcd, encoding="ascii")
            result = analyze(path)
        self.assertEqual(result["shared_alias_codes_excluded"], 1)
        self.assertEqual(result["bit_toggles"], {"fixed": 1, "rqtb": 1})


if __name__ == "__main__":
    unittest.main()
