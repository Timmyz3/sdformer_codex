from __future__ import annotations

import base64
import unittest
import zlib

import numpy as np

from analyze_gatestack_compactor_profile import (
    compactor_cycles_by_row,
    reconstruct_row_k_counts,
)


def encode(values: np.ndarray) -> dict:
    data = values.astype("<i2", copy=False)
    return {
        "shape": list(data.shape),
        "dtype": "int16_le",
        "codec": "zlib_base64",
        "data": base64.b64encode(zlib.compress(data.tobytes())).decode("ascii"),
    }


class GateStackCompactorProfileTest(unittest.TestCase):
    def test_temporal_layout_reconstructs_162_token_row_order(self):
        temporal = np.asarray(
            [
                [[ [1, 2], [3, 4] ]],
                [[ [5, 6], [7, 8] ]],
            ],
            dtype=np.int16,
        )
        record = {
            "tokens": 4,
            "pair_k_count_ordered_trace": encode(temporal),
        }
        rows = reconstruct_row_k_counts(record)
        np.testing.assert_array_equal(rows[0, 0], [1, 2, 5, 6])
        np.testing.assert_array_equal(rows[0, 1], [3, 4, 7, 8])

    def test_compactor_does_not_pack_across_tokens(self):
        temporal = np.asarray([[[[3, 1]]], [[[0, 4]]]], dtype=np.int16)
        record = {
            "tokens": 4,
            "pair_k_count_ordered_trace": encode(temporal),
        }
        self.assertEqual(compactor_cycles_by_row(record, 4), [3])


if __name__ == "__main__":
    unittest.main()
