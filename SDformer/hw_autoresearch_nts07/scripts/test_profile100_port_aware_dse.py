#!/usr/bin/env python3

from __future__ import annotations

import unittest

from analyze_h67_h68_profile100_arch_features import (
    commit_cycles,
    pair_category_row_means,
    three_stage_flowshop_cycles,
)


class PortAwareDseTest(unittest.TestCase):
    def setUp(self) -> None:
        # Two rows, each with 50 both-zero, 20 one-zero and 11 both-active pairs.
        self.record = {
            "batch_windows": 1,
            "num_heads": 2,
            "ttb_tok1_total": 162,
            "ttb_tok1_kzero": 100,
            "zaf_kzero_token_ratio": 120.0 / 162.0,
        }

    def test_pair_categories_reconstruct_token_marginal(self) -> None:
        pair = pair_category_row_means(self.record)
        self.assertAlmostEqual(pair["pairs"], 81.0)
        self.assertAlmostEqual(pair["both_kzero"], 50.0)
        self.assertAlmostEqual(pair["one_kzero"], 20.0)
        self.assertAlmostEqual(pair["both_active"], 11.0)
        zero_tokens = 2.0 * pair["both_kzero"] + pair["one_kzero"]
        self.assertAlmostEqual(zero_tokens, 120.0)

    def test_commit_port_bounds(self) -> None:
        self.assertAlmostEqual(commit_cycles(self.record, "dual_write_ideal"), 81.0)
        self.assertAlmostEqual(commit_cycles(self.record, "split_1w_no_merge"), 142.0)
        self.assertAlmostEqual(commit_cycles(self.record, "split_1w_perfect_pccc"), 92.0)
        self.assertAlmostEqual(commit_cycles(self.record, "unified_1w_no_merge"), 162.0)
        self.assertAlmostEqual(commit_cycles(self.record, "unified_1w_perfect_pccc"), 112.0)

    def test_context_limit_holds_until_backend_release(self) -> None:
        fetch = [1.0, 1.0]
        commit = [2.0, 2.0]
        backend = [3.0, 3.0]
        self.assertAlmostEqual(three_stage_flowshop_cycles(fetch, commit, backend, 1), 12.0)
        self.assertAlmostEqual(three_stage_flowshop_cycles(fetch, commit, backend, 2), 9.0)


if __name__ == "__main__":
    unittest.main()
