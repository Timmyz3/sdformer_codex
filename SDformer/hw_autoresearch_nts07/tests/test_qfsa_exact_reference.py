import unittest

from scripts.qfsa_exact_reference import (
    alpha_xnor_raw16,
    direct_scores,
    finite_width_contract,
    qfsa_scores,
    residual_raw16,
    rne_div16,
    run_random,
)


class QfsaExactReferenceTest(unittest.TestCase):
    def test_anchor_and_direct_match(self):
        q = 0x12345678
        k = 0xA5A5A5A5
        self.assertEqual(
            residual_raw16(q, k, k),
            alpha_xnor_raw16(q, k),
        )

    def test_changed_lane_residual_matches_direct(self):
        q = 0xFFFF0000
        anchor = 0x00FF00FF
        target = anchor ^ 0x80010008
        self.assertEqual(
            residual_raw16(q, anchor, target),
            alpha_xnor_raw16(q, target),
        )

    def test_rne_ties_to_even(self):
        self.assertEqual(rne_div16(8), 0)
        self.assertEqual(rne_div16(24), 2)
        self.assertEqual(rne_div16(25), 2)

    def test_five_scores_match(self):
        q = 0xCAFEBABE
        candidates = [
            0x12345678,
            0x12345679,
            0x12345670,
            0x02345678,
            0x92345678,
        ]
        self.assertEqual(
            qfsa_scores(q, candidates),
            direct_scores(q, candidates),
        )

    def test_random_reference(self):
        report = run_random(1000, 7)
        self.assertEqual(report["mismatches"], 0)
        self.assertEqual(report["compared_scores"], 5000)

    def test_finite_width_contract(self):
        contract = finite_width_contract()
        self.assertEqual(contract["accumulator_max_raw16"], 2048)
        self.assertEqual(contract["accumulator_signed_width"], 13)
        self.assertEqual(contract["wave_reducer_abs_max"], 256)
        self.assertEqual(contract["wave_reducer_signed_width"], 10)


if __name__ == "__main__":
    unittest.main()
