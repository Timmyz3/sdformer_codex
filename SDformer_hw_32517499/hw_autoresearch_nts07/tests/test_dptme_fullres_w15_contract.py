from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_dptme_port_contract import analyze, packet_drain_cycles  # noqa: E402


class DptmeFullresW15ContractTest(unittest.TestCase):
    def test_fullres_w15_cycles(self):
        result = analyze(positions=225)
        self.assertEqual(result["T10"]["compute_cycles"], 2250)
        self.assertEqual(result["T2_candidates"][4]["compute_cycles"], 90)
        self.assertEqual(result["T2_candidates"][3]["compute_cycles"], 114)
        self.assertEqual(result["T2_candidates"][2]["compute_cycles"], 150)

    def test_fullres_w15_output_lower_bounds(self):
        result = analyze(positions=225)
        five_way = result["T2_candidates"][4]
        self.assertEqual(packet_drain_cycles(5, 32, positions=225), 450)
        self.assertEqual(five_way["system_cycle_lower_bound"]["32"], 450)
        self.assertEqual(five_way["system_cycle_lower_bound"]["256"], 90)


if __name__ == "__main__":
    unittest.main()
