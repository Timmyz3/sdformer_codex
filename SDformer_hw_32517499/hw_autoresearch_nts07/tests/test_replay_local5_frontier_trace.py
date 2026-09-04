import unittest

import numpy as np

from scripts.replay_local5_frontier_trace import (
    replay,
    trace_group_source_work,
)


def fixture():
    manifest = {
        "schema": "et3_ordered_term_trace_v2",
        "evidence_level": "synthetic",
        "groups": [
            {"tokens": 8},
            {"tokens": 8},
        ],
    }
    arrays = {
        "source_group_offsets": np.asarray([0, 8, 16], dtype=np.int64),
        "source_term_count": np.asarray(
            [
                1, 2, 0, 1, 0, 1, 0, 0,
                0, 1, 1, 0, 1, 0, 1, 0,
            ],
            dtype=np.uint16,
        ),
        "source_delivery_count": np.asarray(
            [
                2, 3, 0, 1, 0, 2, 0, 0,
                0, 2, 2, 0, 1, 0, 2, 0,
            ],
            dtype=np.uint16,
        ),
        "source_service_cycles_pipelined": np.asarray(
            [
                2, 3, 0, 1, 0, 2, 0, 0,
                0, 2, 2, 0, 1, 0, 2, 0,
            ],
            dtype=np.uint16,
        ),
        "source_retire_destination": np.asarray(
            [
                2, 3, 3, 3, 6, 7, 7, 7,
                2, 3, 3, 3, 6, 7, 7, 7,
            ],
            dtype=np.uint16,
        ),
        "destination_direct_score_cycles": np.asarray(
            [3, 5, 5, 3, 3, 5, 5, 3] * 2,
            dtype=np.uint8,
        ),
        "destination_qfsa_w4_score_cycles": np.asarray(
            [1, 2, 2, 1, 1, 2, 2, 1] * 2,
            dtype=np.uint8,
        ),
        "destination_qfsa_xb4_score_cycles": np.asarray(
            [1, 2, 2, 1, 1, 2, 2, 1] * 2,
            dtype=np.uint8,
        ),
        "destination_qfsa_xb4_t8_score_cycles": np.asarray(
            [1, 2, 2, 1, 1, 2, 2, 1] * 2,
            dtype=np.uint8,
        ),
        "destination_qfsa_xb4_t8b2_score_cycles": np.asarray(
            [1, 2, 2, 1, 1, 2, 2, 1] * 2,
            dtype=np.uint8,
        ),
        "destination_independent_w1x4_score_cycles": np.asarray(
            [1, 3, 3, 1, 1, 3, 3, 1] * 2,
            dtype=np.uint8,
        ),
    }
    return manifest, arrays


class ReplayLocal5FrontierTraceTest(unittest.TestCase):
    def test_group_decode_conserves_sources(self):
        manifest, arrays = fixture()
        events, work = trace_group_source_work(manifest, arrays, 0)
        self.assertEqual(sum(len(event) for event in events), 8)
        self.assertEqual(sum(work), 8)
        self.assertEqual(events[3], [1, 2, 3])

    def test_replay_reports_all_fifo_ready_points(self):
        manifest, arrays = fixture()
        report = replay(
            manifest,
            arrays,
            fifo_depths=(3, 8),
            ready_percents=(100, 75),
        )
        self.assertEqual(report["groups"], 2)
        self.assertEqual(len(report["configs"]), 4)
        for row in report["configs"].values():
            self.assertEqual(row["qfsa_frontier"]["groups"], 2)
            self.assertEqual(row["qfsa_xb4_frontier"]["groups"], 2)
            self.assertGreaterEqual(
                row["cdrp_vs_independent_speedup_mean"],
                1.0,
            )

    def test_missing_v2_arrays_rejected(self):
        manifest, arrays = fixture()
        del arrays["source_service_cycles_pipelined"]
        with self.assertRaisesRegex(ValueError, "v2"):
            trace_group_source_work(manifest, arrays, 0)


if __name__ == "__main__":
    unittest.main()
