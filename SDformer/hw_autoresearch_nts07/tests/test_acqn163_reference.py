import unittest

from scripts.acqn163_reference import (
    Candidate,
    acqn_reference,
    expanded_reference,
    run_reference,
    verify_row,
)


class Acqn163ReferenceTest(unittest.TestCase):
    def test_equal_score_multiplicity_and_zero_k(self):
        row = [
            Candidate(0, 10, False, True, 4, k_bits=0),
            Candidate(1, 10, True, True, 1, k_bits=1),
            Candidate(2, 10, True, True, 1, k_bits=3),
            Candidate(3, 9, True, True, 1, k_bits=5),
        ]
        for preserve_mean in (False, True):
            verify_row(row, preserve_mean)

    def test_invalid_member_does_not_change_denominator(self):
        base = [Candidate(0, 20, True, True, 1, k_bits=1)]
        with_invalid = base + [
            Candidate(1, 162, False, False, 450, k_bits=0)
        ]
        for preserve_mean in (False, True):
            self.assertEqual(
                expanded_reference(base, preserve_mean),
                expanded_reference(with_invalid, preserve_mean),
            )
            self.assertEqual(
                acqn_reference(base, preserve_mean),
                acqn_reference(with_invalid, preserve_mean),
            )

    def test_full_score_domain(self):
        row = [
            Candidate(
                index,
                index,
                bool(index & 1),
                k_bits=int(bool(index & 1)),
            )
            for index in range(163)
        ]
        verify_row(row, False)
        verify_row(row, True)

    def test_rejects_mixed_context_and_k_flag_mismatch(self):
        with self.assertRaises(ValueError):
            verify_row(
                [
                    Candidate(0, 1, True, context=0, k_bits=1),
                    Candidate(1, 1, True, context=1, k_bits=1),
                ],
                False,
            )
        with self.assertRaises(ValueError):
            verify_row([Candidate(0, 1, False, k_bits=1)], False)

    def test_random_reference(self):
        result = run_reference(1000, 0xAC0A163)
        self.assertEqual(result["mismatches"], 0)
        self.assertGreater(result["mode_row_checks"], 2000)


if __name__ == "__main__":
    unittest.main()
