import unittest

from scripts.profile_local5_quantization_phase_residual import (
    RAW_MAX,
    rne_div16,
    shiftmax5_gate,
)


class QuantizationPhaseResidualTest(unittest.TestCase):
    def test_signed_rne_ties_to_even(self):
        self.assertEqual(rne_div16(8), 0)
        self.assertEqual(rne_div16(24), 2)
        self.assertEqual(rne_div16(-8), 0)
        self.assertEqual(rne_div16(-24), -2)

    def test_phase_translation_edges(self):
        for anchor in (0, 1, 15, 16, 31, 32, 1024, RAW_MAX):
            phase = anchor & 31
            for candidate in (0, 1, 15, 16, 31, 32, 1024, RAW_MAX):
                common = rne_div16(anchor) - rne_div16(phase)
                self.assertEqual(
                    rne_div16(candidate) - common,
                    rne_div16(phase + candidate - anchor),
                )

    def test_shiftmax_is_translation_invariant(self):
        scores = [12, -3, 18, -256, 7]
        valid = [True, True, True, False, True]
        shifted = [value - 20 if keep else value for value, keep in zip(scores, valid)]
        self.assertEqual(shiftmax5_gate(scores, valid), shiftmax5_gate(shifted, valid))


if __name__ == "__main__":
    unittest.main()
