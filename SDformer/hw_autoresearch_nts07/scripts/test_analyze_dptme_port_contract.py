from __future__ import annotations

import unittest

from analyze_dptme_port_contract import analyze, packet_drain_cycles


class DptmePortContractTest(unittest.TestCase):
    def test_five_way_32bit_output_erases_compute_gain(self):
        result = analyze()
        five_way = result["T2_candidates"][4]
        self.assertEqual(five_way["compute_cycles"], 34)
        self.assertEqual(five_way["system_cycle_lower_bound"]["32"], 162)
        self.assertEqual(five_way["system_cycle_lower_bound"]["256"], 34)

    def test_four_way_balances_128bit_output(self):
        result = analyze()
        four_way = result["T2_candidates"][3]
        self.assertEqual(four_way["compute_cycles"], 42)
        self.assertEqual(four_way["system_cycle_lower_bound"]["128"], 42)
        self.assertEqual(four_way["active_t2_macs"], 256)
        self.assertEqual(four_way["full_array_physical_macs"], 320)
        self.assertEqual(four_way["trimmed_array_t10_cycles"], 1620)

    def test_packet_tail_is_not_charged_as_full_packet(self):
        self.assertEqual(packet_drain_cycles(5, 32), 162)
        self.assertEqual(packet_drain_cycles(5, 256), 33)


if __name__ == "__main__":
    unittest.main()
