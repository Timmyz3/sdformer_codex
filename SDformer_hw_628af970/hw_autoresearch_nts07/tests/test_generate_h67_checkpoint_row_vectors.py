from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_h67_checkpoint_row_vectors import (  # noqa: E402
    exp2_q8,
    row_gate_codes,
    score_q7,
)


class H67CheckpointRowVectorTest(unittest.TestCase):
    def test_score_reference_covers_motion_and_silence(self) -> None:
        self.assertEqual(score_q7(0, 0, 0), 2)
        self.assertEqual(score_q7(1, 1, 0), 7)
        self.assertEqual(score_q7(0, 0, (1 << 32) - 1), 34)

    def test_exp2_lut_boundaries(self) -> None:
        self.assertEqual(exp2_q8(0), 256)
        self.assertEqual(exp2_q8(-8), 245)
        self.assertEqual(exp2_q8(-128), 128)
        self.assertEqual(exp2_q8(-4096), 1)

    def test_row_gate_preserves_mean_with_power_of_two_denominator(self) -> None:
        q = [0, 1, 0, 1]
        current = [0, 1, 1, 0]
        peer = [0, 0, 1, 1]
        gates = row_gate_codes(q, current, peer)
        self.assertEqual(len(gates), 4)
        self.assertTrue(all(0 <= value <= 256 for value in gates))


if __name__ == "__main__":
    unittest.main()
