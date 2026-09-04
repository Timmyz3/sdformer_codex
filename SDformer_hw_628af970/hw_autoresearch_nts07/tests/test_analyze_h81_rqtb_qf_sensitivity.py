import unittest

import numpy as np

from scripts.analyze_h81_rqtb_qf_sensitivity import rne_div_pow2, summarize


class H81QfSensitivityTest(unittest.TestCase):
    def test_round_nearest_even(self):
        values = np.asarray([7, 8, 9, 23, 24, 25], dtype=np.int32)
        self.assertEqual(
            rne_div_pow2(values, 16).tolist(),
            [0, 0, 1, 1, 2, 2],
        )

    def test_summary(self):
        result = summarize({"pairs": 100, "empty": 60, "equal": 90})
        self.assertEqual(result["nonempty_equal_pairs"], 30)
        self.assertAlmostEqual(result["nonempty_equal_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()
