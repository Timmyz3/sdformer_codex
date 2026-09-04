import unittest

from scripts.model_architecture_innovation_round12 import (
    local5_frontier_storage,
    storage_ledger,
)


class ArchitectureInnovationRound12Test(unittest.TestCase):
    def test_factorized_storage_is_smaller_at_both_token_counts(self):
        for tokens in (162, 450):
            row = storage_ledger(tokens)
            self.assertLess(
                row["factorized_total_bits"],
                row["current_total_bits"],
            )

    def test_three_row_frontier_is_bounded(self):
        row = local5_frontier_storage(times=2, height=15, width=15)
        self.assertEqual(row["full_gate_plane_bits"], 20250)
        self.assertEqual(row["three_row_frontier_bits"], 2025)
        self.assertAlmostEqual(row["frontier_vs_full_ratio"], 0.1)


if __name__ == "__main__":
    unittest.main()
