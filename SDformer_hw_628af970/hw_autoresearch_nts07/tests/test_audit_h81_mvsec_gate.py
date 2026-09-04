import unittest

from scripts.audit_h81_mvsec_gate import compare_sequences


class H81MvsecGateTest(unittest.TestCase):
    def test_all_sequence_gate(self):
        h81 = [
            {"sequence": "outdoor_day1", "samples": 10, "AEE": 0.8},
            {"sequence": "indoor_flying1", "samples": 10, "AEE": 1.7},
            {"sequence": "indoor_flying2", "samples": 10, "AEE": 2.6},
            {"sequence": "indoor_flying3", "samples": 10, "AEE": 2.0},
        ]
        nb0 = [
            {"sequence": "outdoor_day1", "samples": 10, "AEE": 0.9},
            {"sequence": "indoor_flying1", "samples": 10, "AEE": 1.6},
            {"sequence": "indoor_flying2", "samples": 10, "AEE": 2.7},
            {"sequence": "indoor_flying3", "samples": 10, "AEE": 2.1},
        ]
        result = compare_sequences(h81, nb0)
        self.assertFalse(result["all_sequence_better_than_NB0"])
        self.assertEqual(result["failing_sequences"], ["indoor_flying1"])


if __name__ == "__main__":
    unittest.main()
