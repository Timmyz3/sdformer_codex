from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_local5_joint_hardware_contract as contract


class JointHardwareContractTest(unittest.TestCase):
    def test_weighted_summary_uses_weights(self) -> None:
        values = np.asarray([1, 10, 100], dtype=np.float64)
        weights = np.asarray([8, 1, 1], dtype=np.float64)
        result = contract.weighted_summary(values, weights)
        self.assertAlmostEqual(result["mean"], 11.8)
        self.assertEqual(result["p50"], 1.0)
        self.assertEqual(result["p95"], 100.0)

    def test_vector_traffic_common_boundary(self) -> None:
        result = contract.vector_traffic_words(3, 3)
        self.assertEqual(result["b0v_1rw_vector_accesses"], 12150)
        self.assertEqual(result["b2v_1rw_vector_accesses"], 1350)
        self.assertEqual(result["shared_scalar_results"], 43200)

    def test_relation_pair_metrics(self) -> None:
        k = np.arange(contract.TOKENS, dtype=np.uint64)
        gates = np.ones((contract.TOKENS, 5), dtype=np.uint16)
        valid = np.full(contract.TOKENS, 31, dtype=np.uint8)
        terms = np.ones(contract.TOKENS, dtype=np.uint16)
        same = (k, gates, valid, terms)
        exact, gate_equal, jaccard = contract.relation_pair_metrics(same, same)
        self.assertEqual((exact, gate_equal, jaccard), (1.0, 1.0, 1.0))

        changed_k = k.copy()
        changed_k[0] += 1
        changed_terms = terms.copy()
        changed_terms[0] = 0
        other = (changed_k, gates, valid, changed_terms)
        exact, gate_equal, jaccard = contract.relation_pair_metrics(same, other)
        self.assertAlmostEqual(exact, 449 / 450)
        self.assertEqual(gate_equal, 1.0)
        self.assertAlmostEqual(jaccard, 449 / 450)

        decomposition = contract.relation_pair_decomposition(same, other)
        self.assertEqual(decomposition["both_empty_source_fraction"], 0.0)
        self.assertAlmostEqual(decomposition["both_active_source_fraction"], 449 / 450)
        self.assertAlmostEqual(
            decomposition["exact_nonempty_source_fraction"], 449 / 450
        )

    def test_relation_pair_decomposition_all_empty_is_explicit(self) -> None:
        k = np.zeros(contract.TOKENS, dtype=np.uint64)
        gates = np.zeros((contract.TOKENS, 5), dtype=np.uint16)
        valid = np.ones(contract.TOKENS, dtype=np.uint8)
        terms = np.zeros(contract.TOKENS, dtype=np.uint16)
        descriptor = (k, gates, valid, terms)
        result = contract.relation_pair_decomposition(descriptor, descriptor)
        self.assertEqual(result["both_empty_source_fraction"], 1.0)
        self.assertEqual(result["both_active_source_fraction"], 0.0)
        self.assertEqual(result["exact_nonempty_source_fraction"], 1.0)

    def test_relation_equivalence_metrics(self) -> None:
        k = np.zeros((3, contract.TOKENS), dtype=np.uint64)
        gates = np.zeros((3, contract.TOKENS, 5), dtype=np.uint16)
        valid = np.ones((3, contract.TOKENS), dtype=np.uint8)
        terms = np.zeros((3, contract.TOKENS), dtype=np.uint16)
        k[2, 0] = 1
        terms[2, 0] = 1
        result = contract.relation_equivalence_metrics(k, gates, valid, terms)
        expected_classes = (2 + (contract.TOKENS - 1)) / contract.TOKENS
        self.assertAlmostEqual(result["equivalence_classes_per_source"], expected_classes)
        self.assertAlmostEqual(
            result["all_head_identical_source_fraction"],
            (contract.TOKENS - 1) / contract.TOKENS,
        )
        self.assertAlmostEqual(
            result["all_head_empty_source_fraction"],
            (contract.TOKENS - 1) / contract.TOKENS,
        )


if __name__ == "__main__":
    unittest.main()
