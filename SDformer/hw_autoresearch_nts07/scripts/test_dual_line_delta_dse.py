#!/usr/bin/env python3

import importlib.util
import random
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DSE = load_module("dual_line_dse", "model_dual_line_delta_dse.py")
REF = load_module("dual_line_ref", "dual_line_delta_reference.py")


class QueueModelTest(unittest.TestCase):
    def test_reflected_queue(self) -> None:
        cases = (
            ([0], (0, 0)),
            ([2], (1, 1)),
            ([0, 2], (1, 1)),
            ([2, 0], (0, 1)),
            ([3, 0, 0], (0, 2)),
        )
        for service, expected in cases:
            with self.subTest(service=service):
                self.assertEqual(
                    DSE.queue_backlog(np.asarray(service, dtype=np.int32)),
                    expected,
                )

    def test_empty_queue(self) -> None:
        self.assertEqual(
            DSE.queue_backlog(np.asarray([], dtype=np.int32)), (0, 0)
        )

    def test_fallback_extra_drain(self) -> None:
        self.assertEqual(
            DSE.queue_backlog(
                np.asarray([2], dtype=np.int32),
                np.asarray([1], dtype=np.int32),
            ),
            (0, 0),
        )
        self.assertEqual(
            DSE.queue_backlog(
                np.asarray([3, 0], dtype=np.int32),
                np.asarray([1, 0], dtype=np.int32),
            ),
            (0, 1),
        )

    def test_reflected_queue_matches_cycle_simulation(self) -> None:
        rng = random.Random(675004)
        for _ in range(1000):
            length = rng.randrange(1, 128)
            service = np.asarray(
                [rng.randrange(0, 9) for _ in range(length)],
                dtype=np.int32,
            )
            extra = np.asarray(
                [rng.randrange(0, 3) for _ in range(length)],
                dtype=np.int32,
            )
            queue = 0
            maximum = 0
            for work, drains in zip(service, extra):
                queue = max(0, queue + int(work) - 1 - int(drains))
                maximum = max(maximum, queue)
            self.assertEqual(
                DSE.queue_backlog(service, extra),
                (queue, maximum),
            )

    def test_width_equals_threshold_is_queue_free(self) -> None:
        rng = random.Random(674004)
        for width in (1, 2, 4, 8, 16, 32):
            counts = np.asarray(
                [rng.randrange(0, 33) for _ in range(10_000)],
                dtype=np.int32,
            )
            fallback = counts > width
            sparse_service = np.where(
                (counts > 0) & ~fallback,
                (counts + width - 1) // width,
                0,
            ).astype(np.int32)
            self.assertEqual(
                DSE.queue_backlog(
                    sparse_service,
                    fallback.astype(np.int32),
                ),
                (0, 0),
            )


class DeltaReferenceTest(unittest.TestCase):
    def test_random_equivalence(self) -> None:
        result = REF.run_random(seed=123, vectors=10_000)
        self.assertTrue(result["pass"])
        self.assertEqual(result["local5"]["raw_mismatches"], 0)
        self.assertEqual(result["local5"]["mismatches"], 0)
        self.assertEqual(result["h67_motion"]["raw_mismatches"], 0)
        self.assertEqual(result["h67_motion"]["mismatches"], 0)

    def test_single_active_lane_exhaustive(self) -> None:
        REF.exhaustive_single_active_lane()


if __name__ == "__main__":
    unittest.main()
