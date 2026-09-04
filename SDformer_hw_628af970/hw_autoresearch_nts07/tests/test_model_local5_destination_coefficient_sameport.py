import unittest


class DestinationCoefficientSamePortTest(unittest.TestCase):
    def test_frozen_ep44_arithmetic(self):
        source_terms = 48_769
        coefficient_terms = 108_726
        memory_wait = 17_218
        production_cycles = 283_664
        net_tax = coefficient_terms - source_terms - memory_wait
        self.assertEqual(net_tax, 42_739)
        self.assertEqual(production_cycles + net_tax, 326_403)
        self.assertLess(production_cycles / (production_cycles + net_tax), 1.0)


if __name__ == "__main__":
    unittest.main()
