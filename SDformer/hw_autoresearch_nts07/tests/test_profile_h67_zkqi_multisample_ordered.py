import base64
import zlib
import unittest

import numpy as np

from scripts.profile_h67_zkqi_multisample_ordered import (
    compute_row_metrics,
    decode_trace,
    h67_score_from_counts,
    ttb_depth1_front_cycles,
)


def encode(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype="<i2")
    return {
        "codec": "zlib_base64",
        "dtype": "int16_le",
        "shape": list(array.shape),
        "data": base64.b64encode(zlib.compress(array.tobytes())).decode("ascii"),
    }


def scalar_front(counts: list[int]) -> int:
    producer = 0
    fifo = 0
    cycles = 0
    while producer < len(counts) or fifo:
        pop = fifo == 1
        accept = False
        if producer < len(counts):
            accept = counts[producer] == 0 or fifo == 0 or pop
        if fifo:
            fifo -= 1
        if accept:
            if counts[producer]:
                if fifo != 0:
                    raise AssertionError("reference FIFO overwrite")
                fifo = counts[producer]
            producer += 1
        cycles += 1
    return cycles


class H67ZkqiMultisampleTest(unittest.TestCase):
    def test_ordered_trace_decode(self) -> None:
        values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        self.assertTrue(np.array_equal(decode_trace(encode(values)), values))

    def test_depth1_front_matches_scalar_protocol(self) -> None:
        rows = np.asarray(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [4, 4, 4, 4],
                [2, 0, 3, 0],
                [0, 3, 0, 2],
            ],
            dtype=np.int64,
        )
        expected = np.asarray([scalar_front(row.tolist()) for row in rows])
        self.assertTrue(np.array_equal(ttb_depth1_front_cycles(rows), expected))

    def test_all_zero_exact_cycle_formula(self) -> None:
        pairs = 8
        q = np.zeros((2, 1, 1, pairs), dtype=np.int32)
        k = np.zeros_like(q)
        overlap = np.zeros_like(q)
        motion = np.zeros((1, 1, pairs), dtype=np.int32)
        metrics = compute_row_metrics(
            q, k, overlap, motion, bundle_size=4
        )
        self.assertEqual(int(metrics["occupied_classes"][0, 0]), 1)
        self.assertEqual(int(metrics["active_descriptors"][0, 0]), 0)
        self.assertEqual(int(metrics["backend_cycles"][0, 0]), 4)
        self.assertEqual(int(metrics["baseline_cycles"][0, 0]), 12)
        self.assertEqual(int(metrics["ttb_front_cycles"][0, 0]), 2)
        self.assertEqual(int(metrics["ttb_cycles"][0, 0]), 6)

    def test_score_and_active_descriptor_split(self) -> None:
        pairs = 4
        q = np.zeros((2, 1, 1, pairs), dtype=np.int32)
        k = np.zeros_like(q)
        overlap = np.zeros_like(q)
        motion = np.zeros((1, 1, pairs), dtype=np.int32)
        # pair0两时间片均active但score不同，必须产生两个active descriptor。
        q[0, 0, 0, 0] = 1
        k[0, 0, 0, 0] = 1
        overlap[0, 0, 0, 0] = 1
        k[1, 0, 0, 0] = 2
        overlap[1, 0, 0, 0] = 0
        motion[0, 0, 0] = 1
        score = h67_score_from_counts(q, k, overlap, motion)
        self.assertNotEqual(int(score[0, 0, 0, 0]), int(score[1, 0, 0, 0]))
        metrics = compute_row_metrics(q, k, overlap, motion, bundle_size=2)
        self.assertEqual(int(metrics["active_pairs"][0, 0]), 1)
        self.assertEqual(int(metrics["outputs"][0, 0]), 2)
        self.assertEqual(int(metrics["candidate_descriptors"][0, 0]), 2)
        self.assertEqual(int(metrics["active_descriptors"][0, 0]), 2)
        self.assertEqual(int(metrics["candidate_read_bits"][0, 0]), 192)

    def test_impossible_overlap_and_motion_fail_closed(self) -> None:
        q = np.zeros((2, 1, 1, 1), dtype=np.int32)
        k = np.zeros_like(q)
        overlap = np.zeros_like(q)
        motion = np.zeros((1, 1, 1), dtype=np.int32)
        overlap[0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "非法计数"):
            h67_score_from_counts(q, k, overlap, motion)

        overlap.fill(0)
        k[0, 0, 0, 0] = 1
        k[1, 0, 0, 0] = 1
        motion[0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "不可由真实bit向量实现"):
            h67_score_from_counts(q, k, overlap, motion)


if __name__ == "__main__":
    unittest.main()
