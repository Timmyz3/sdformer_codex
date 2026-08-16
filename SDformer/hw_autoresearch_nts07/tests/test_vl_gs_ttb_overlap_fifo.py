from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_vl_gs_ttb_overlap_fifo import (
    classify_first_bind,
    motion_context_work,
    simulate_local_fifo,
    two_bank_flowshop,
)


class VlGsTtbOverlapFifoTest(unittest.TestCase):
    def test_motion_two_bank_is_bounded(self) -> None:
        work = motion_context_work([2, 1, 3], [5, 4, 6], 2)
        self.assertEqual(work, [(3, 5, False), (2, 4, False), (1, 6, True)])
        value = two_bank_flowshop(work)
        self.assertEqual(value["serialized_cycles"], 21)
        self.assertEqual(value["dual_bank_cycles"], 18)
        self.assertEqual(value["hidden_header_cycles"], 3)

    def test_local_classification_and_exact_retirement(self) -> None:
        rows = [
            {"lane": 0, "gate": 5},
            {"lane": 0, "gate": 7},
            {"lane": 0, "gate": 5},
            {"lane": 0, "gate": 9},
        ]
        events = classify_first_bind(rows, 2)
        self.assertEqual(
            [event["kind"] for event in events],
            ["fill", "fill", "hit", "bypass"],
        )
        value = simulate_local_fifo(
            events,
            2,
            commit_forward=True,
            elastic_output=True,
        )
        self.assertEqual(value["terms"], 4)
        self.assertEqual(value["fills"], 2)
        self.assertEqual(value["bypasses"], 1)
        self.assertEqual(value["cycles"], 6)
        self.assertEqual(value["cycle_overhead_vs_elastic_raw"], 0)

    def test_registered_decoder_is_slower(self) -> None:
        rows = [{"lane": 0, "gate": 5}] * 8
        events = classify_first_bind(rows, 2)
        registered = simulate_local_fifo(
            events,
            2,
            commit_forward=False,
            elastic_output=False,
        )
        atomic = simulate_local_fifo(
            events,
            2,
            commit_forward=True,
            elastic_output=True,
        )
        self.assertGreater(registered["cycles"], atomic["cycles"])


if __name__ == "__main__":
    unittest.main()
