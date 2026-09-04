from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_local5_active_projection_postg0_vectors import (
    select_sample_disjoint_stage_groups,
    select_sample_population_weighted_groups,
)


class Local5SampleDisjointSelectionTest(unittest.TestCase):
    def test_covers_each_sample_once_and_each_stage_equally(self) -> None:
        groups = []
        for sample in range(100):
            for stage in range(4):
                for head in range(2):
                    groups.append(
                        {
                            "sample": sample,
                            "stage": stage,
                            "head": head,
                            "active_sources": 9999 - sample - head,
                        }
                    )
        selected = select_sample_disjoint_stage_groups(groups, 25)
        rows = [groups[index] for index in selected]
        self.assertEqual(len(rows), 100)
        self.assertEqual({int(row["sample"]) for row in rows}, set(range(100)))
        self.assertEqual(Counter(int(row["stage"]) for row in rows), Counter({0: 25, 1: 25, 2: 25, 3: 25}))
        self.assertTrue(all(int(row["stage"]) == int(row["sample"]) % 4 for row in rows))

    def test_rejects_incompatible_sample_count(self) -> None:
        groups = [{"sample": sample, "stage": sample % 4} for sample in range(8)]
        with self.assertRaisesRegex(ValueError, "4\\*per-stage"):
            select_sample_disjoint_stage_groups(groups, 25)

    def test_population_weighted_selection_matches_source_stage_mix(self) -> None:
        stage_groups = (6, 12, 72, 48)
        groups = []
        for sample in range(100):
            for stage, count in enumerate(stage_groups):
                for index in range(count):
                    groups.append(
                        {
                            "sample": sample,
                            "stage": stage,
                            "head": index,
                            "active_sources": 999999 - index,
                        }
                    )
        selected = select_sample_population_weighted_groups(groups, 100)
        rows = [groups[index] for index in selected]
        self.assertEqual({int(row["sample"]) for row in rows}, set(range(100)))
        self.assertEqual(
            Counter(int(row["stage"]) for row in rows),
            Counter({0: 4, 1: 9, 2: 52, 3: 35}),
        )


if __name__ == "__main__":
    unittest.main()
