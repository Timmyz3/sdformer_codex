from __future__ import annotations

import unittest

import numpy as np

from scripts.report_local5_source_owned_gate_quotient import analyze_arrays


def fixture() -> dict[str, np.ndarray]:
    return {
        "group_offsets": np.asarray([0, 3, 6], dtype=np.int64),
        "item_mode_multiset": np.ones(6, dtype=np.uint8),
        "item_multiplicity": np.asarray([2, 1, 1, 1, 1, 5], dtype=np.uint8),
        "descriptor_group_offsets": np.asarray([0, 2, 3], dtype=np.int64),
        "descriptor_incoming_gates": np.asarray(
            [[16, 16, 32, 0, 0], [16, 16, 16, 16, 16], [32, 0, 0, 0, 0]],
            dtype=np.uint16,
        ),
        "descriptor_valid_mask": np.asarray([0b00111, 0b11111, 0b00001], dtype=np.uint8),
        "source_k_popcount": np.asarray([2, 1, 0], dtype=np.uint8),
        "source_gate_count": np.asarray([2, 1, 0], dtype=np.uint8),
        "source_term_count": np.asarray([4, 1, 0], dtype=np.uint16),
        "source_delivery_count": np.asarray([6, 5, 0], dtype=np.uint16),
    }


class SourceOwnedGateQuotientTests(unittest.TestCase):
    def test_exact_conservation_and_strong_baseline(self) -> None:
        result = analyze_arrays(fixture())
        self.assertEqual(result["relation_lane_delivery"], 11)
        self.assertEqual(result["destination_local_mfep_terms"], 6)
        self.assertEqual(result["source_owned_gate_lane_terms"], 5)
        self.assertTrue(
            result["checks"]["destination_source_delivery_conserved"]
        )

    def test_rejects_delivery_or_gate_contract_corruption(self) -> None:
        arrays = fixture()
        arrays["source_delivery_count"] = np.asarray([5, 5, 0], dtype=np.uint16)
        with self.assertRaises(ValueError):
            analyze_arrays(arrays)
        arrays = fixture()
        arrays["descriptor_incoming_gates"][0, 4] = 16
        with self.assertRaises(ValueError):
            analyze_arrays(arrays)


if __name__ == "__main__":
    unittest.main()
