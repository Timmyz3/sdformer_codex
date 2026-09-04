import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.profile_exact_promoted_accumulator import (
    PromotionTracker,
    motion_t450_profile,
)


class PromotionTrackerTest(unittest.TestCase):
    def test_dynamic_demote_and_sticky_promotion(self):
        tracker = PromotionTracker(width=4, entries=2)
        index = np.asarray([0], dtype=np.int64)
        tracker.update(index, np.asarray([0]), np.asarray([9]))
        tracker.update(index, np.asarray([9]), np.asarray([1]))
        row = tracker.row(entries=2)
        self.assertEqual(row["dynamic_peak_entries"], 1)
        self.assertEqual(row["sticky_entries"], 1)
        self.assertEqual(row["scalar_updates"], 2)
        self.assertEqual(row["dynamic_high_access_fraction"], 1.0)
        self.assertEqual(row["sticky_high_access_fraction"], 1.0)

    def test_in_range_updates_do_not_allocate(self):
        tracker = PromotionTracker(width=8, entries=4)
        index = np.asarray([0, 1], dtype=np.int64)
        tracker.update(index, np.asarray([0, 1]), np.asarray([2, -3]))
        row = tracker.row(entries=4)
        self.assertEqual(row["dynamic_peak_entries"], 0)
        self.assertEqual(row["sticky_entries"], 0)
        self.assertEqual(row["dynamic_high_access_fraction"], 0.0)

    def test_current_motion_term_stream_replays_hardware_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vector_dir = root / "vectors" / "s0_b0"
            vector_dir.mkdir(parents=True)

            def write_memh(name, values, width):
                digits = (width + 3) // 4
                mask = (1 << width) - 1
                (vector_dir / name).write_text(
                    "".join(f"{int(value) & mask:0{digits}x}\n" for value in values)
                )

            dim = 96
            tokens = 450
            weights = np.zeros((dim, dim), dtype=np.int64)
            weights[:, 0] = np.arange(dim, dtype=np.int64) - 48
            bias = np.arange(dim, dtype=np.int64) - 48
            expected = np.broadcast_to(bias, (tokens, dim)).copy()
            expected[7] += 3 * weights[:, 0]
            write_memh("head_term_offsets.memh", [0, 1, 1, 1], 32)
            write_memh("term_token_offsets.memh", [0, 1], 32)
            write_memh("term_gate_codes.memh", [3], 9)
            write_memh("term_lane_ids.memh", [0], 5)
            write_memh("term_tokens.memh", [7], 9)
            write_memh("projection_weights_int8.memh", weights.reshape(-1), 8)
            write_memh("projection_bias_acc32.memh", bias, 32)
            write_memh("expected_output_acc32.memh", expected.reshape(-1), 32)
            record = {
                "name": "S0.B0.attn",
                "stage": 0,
                "heads": 3,
                "dim": dim,
                "tokens": tokens,
                "vector_dir": str(vector_dir),
            }
            manifest = {
                "records": [record] * 12,
                "temporal_tokens": tokens,
                "source_manifest": "synthetic",
                "source_manifest_sha256": "synthetic",
            }
            manifest_path = root / "vectors_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            result = motion_t450_profile(manifest_path)
            self.assertEqual(result["records"], 12)
            self.assertEqual(result["mismatch"], 0)
            self.assertEqual(result["results"][0]["events"], 1)


if __name__ == "__main__":
    unittest.main()
