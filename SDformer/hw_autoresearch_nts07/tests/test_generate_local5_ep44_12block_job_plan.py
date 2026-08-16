from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_local5_ep44_12block_job_plan import (
    BLOCKS,
    decode_output_pair,
    select_rows,
)


class Local5Ep44TwelveBlockPlanTest(unittest.TestCase):
    def test_selects_first_nonempty_then_first_fallback(self) -> None:
        rows = []
        expected = []
        for stage, block in BLOCKS:
            base = len(rows)
            rows.extend(
                [
                    {"stage": stage, "block": block, "empty": True},
                    {
                        "stage": stage,
                        "block": block,
                        "empty": (stage, block) == (1, 0),
                    },
                ]
            )
            expected.append(base if (stage, block) == (1, 0) else base + 1)
        actual = [index for index, _ in select_rows(rows)]
        self.assertEqual(actual, expected)

    def test_rejects_missing_block(self) -> None:
        rows = [
            {"stage": stage, "block": block, "empty": False}
            for stage, block in BLOCKS[:-1]
        ]
        with self.assertRaisesRegex(ValueError, "exact 12-block"):
            select_rows(rows)

    def test_output_pair_uses_global_32_channel_tile(self) -> None:
        channels, tile, offset = decode_output_pair(
            {"projection_output_channels": [110, 111]}
        )
        self.assertEqual(channels, [110, 111])
        self.assertEqual(tile, 3)
        self.assertEqual(offset, 14)

    def test_output_pair_rejects_tile_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "OUT_DIM=2 pair"):
            decode_output_pair({"projection_output_channels": [31, 32]})


if __name__ == "__main__":
    unittest.main()
