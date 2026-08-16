from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_local5_acc32_numeric_diversity_v1 import (
    OUT_DIM,
    TOKENS,
    distribution,
    read_inputs,
    read_weights,
)


class NumericDiversityAuditorTest(unittest.TestCase):
    def test_homogeneous_distribution_keeps_real_extrema(self) -> None:
        positive = distribution(np.asarray([3, 7], dtype=np.int32))
        negative = distribution(np.asarray([-9, -2], dtype=np.int32))
        self.assertEqual((positive["minimum"], positive["maximum"]), (3, 7))
        self.assertEqual((negative["minimum"], negative["maximum"]), (-9, -2))

    def test_plane_y_x_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.txt"
            rows = []
            for source in range(TOKENS):
                plane, spatial = divmod(source, 225)
                y, x = divmod(spatial, 15)
                rows.append(
                    f"0 {plane} {y} {x} 00000000 "
                    "00000000 00000000 00000000 00000000 00000000 1f"
                )
            rows[0] = rows[0].replace("0 0 0 0 ", "0 0 15 0 ", 1)
            path.write_text("\n".join(rows) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "outside contract"):
                read_inputs(path, 1)

    def test_weight_hex_width_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.txt"
            rows = [
                f"0 0 {lane} {out} 01"
                for lane in range(32) for out in range(OUT_DIM)
            ]
            rows[0] = "0 0 0 0 100"
            path.write_text("\n".join(rows) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "8-bit hex"):
                read_weights(path, 1)


if __name__ == "__main__":
    unittest.main()
