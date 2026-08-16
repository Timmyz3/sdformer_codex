from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.local5_erep_formal_canary_expected import select_window_groups
from scripts.local5_erep_formal_canary_merge import merge_actual, read_actual


class Local5ErepFormalCanaryTest(unittest.TestCase):
    def test_select_same_window_all_heads(self) -> None:
        manifest = {
            "groups": [
                {"sample": 0, "stage": 0, "block": 0, "window": 7,
                 "head": head, "heads": 3}
                for head in range(3)
            ]
        }
        window, groups = select_window_groups(manifest, 0, 0, 0)
        self.assertEqual(window, 7)
        self.assertEqual([row[0] for row in groups], [0, 1, 2])

    def test_select_is_independent_of_manifest_row_order(self) -> None:
        manifest = {
            "groups": [
                {"sample": 0, "stage": 0, "block": 0, "window": 7,
                 "head": head, "heads": 3}
                for head in (2, 0, 1)
            ]
        }
        window, groups = select_window_groups(manifest, 0, 0, 0)
        self.assertEqual(window, 7)
        self.assertEqual([row[1]["head"] for row in groups], [0, 1, 2])
        self.assertEqual([row[0] for row in groups], [1, 2, 0])

    def test_select_rejects_duplicate_head(self) -> None:
        manifest = {
            "groups": [
                {"sample": 0, "stage": 0, "block": 0, "window": 7,
                 "head": head, "heads": 3}
                for head in (0, 0, 2)
            ]
        }
        with self.assertRaisesRegex(ValueError, "重复"):
            select_window_groups(manifest, 0, 0, 0)

    def test_merge_hxh(self) -> None:
        tasks = [
            {"input_group_index": head, "output_tile": tile}
            for tile in range(3)
            for head in range(3)
        ]
        actual = np.zeros((9, 450, 32), dtype=np.int32)
        for index, task in enumerate(tasks):
            actual[index].fill(100 * task["output_tile"] + task["input_group_index"])
        merged = merge_actual(tasks, actual, 3)
        for tile in range(3):
            self.assertTrue(np.all(merged[tile] == 300 * tile + 3))

    def test_duplicate_task_rejected(self) -> None:
        tasks = [
            {"input_group_index": 0, "output_tile": 0}
            for _ in range(9)
        ]
        with self.assertRaisesRegex(ValueError, "重复"):
            merge_actual(tasks, np.zeros((9, 450, 32), dtype=np.int32), 3)

    def test_actual_hex_signed_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actual.memh"
            values = ["ffffffff", "80000000", "00000001"]
            path.write_text(
                "\n".join(values * (450 * 32)) + "\n", encoding="ascii"
            )
            parsed = read_actual(path, 3)
            self.assertEqual(parsed.shape, (3, 450, 32))
            self.assertEqual(parsed[0, 0, :3].tolist(), [-1, -2147483648, 1])


if __name__ == "__main__":
    unittest.main()
