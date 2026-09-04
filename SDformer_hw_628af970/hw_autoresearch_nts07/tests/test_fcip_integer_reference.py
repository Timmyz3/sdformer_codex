import unittest

from scripts.fcip_integer_reference import run_reference


class FcipIntegerReferenceTest(unittest.TestCase):
    def test_fast_and_fallback_paths_are_exact(self):
        result = run_reference(trials=20, seed=0xFC1F)
        self.assertEqual(result["mismatches"], 0)
        self.assertEqual(result["class_count_mismatches"], 0)
        self.assertGreaterEqual(result["fallback_cases"], 2)


if __name__ == "__main__":
    unittest.main()
