#!/usr/bin/env python3
"""CPU-only synthetic tests for the M463 analyzer; reads no M40 payload."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "system_simulator/scripts/analyze_m463_beta16_destination_stationary_dse.py"
SPEC = importlib.util.spec_from_file_location("m463_under_test", str(ANALYZER))
M463 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M463)


class M463SyntheticTest(unittest.TestCase):
    def test_selection_tie_and_dense_keep(self):
        originals = np.asarray([0, 3, 5, 15], dtype=np.uint16)
        centers = np.asarray([0, 3] + [7] * 30, dtype=np.uint16)
        selected = M463.select_rows(originals, centers)
        self.assertEqual(selected["best_index"].tolist(), [0, 1, 2, 2])
        self.assertEqual(selected["use_pwp"].tolist(),
                         [False, True, False, True])
        keep = np.full((8, 96), 0xffff, dtype=np.uint16)
        costs = M463.destination_cost(selected["correction"], keep)
        expected = M463.POPCOUNT[selected["correction"]][:, None]
        self.assertTrue(np.all(costs == expected))

    def test_pruned_pwp_direct_miter_and_counts(self):
        originals = np.asarray([0, 3, 5, 15], dtype=np.uint16)
        counts = np.asarray([2, 3, 5, 7], dtype=np.int64)
        centers = np.asarray([0, 3] + [7] * 30, dtype=np.uint16)
        weights = (np.arange(16 * 8 * 96, dtype=np.int16).reshape(
            16, 8, 96) % 31 - 15).astype(np.int8)
        all_keep = np.full((8, 96), 0xffff, dtype=np.uint16)
        no_keep = np.zeros((8, 96), dtype=np.uint16)
        phase = M463.phase_metrics(
            originals, counts, centers, {0: all_keep, 16: no_keep}, weights)
        self.assertEqual(phase["source_rows"], 17)
        self.assertEqual(phase["dense_keep_correction_work_by_block"][0], 17)
        self.assertEqual(phase["beta16_correction_work_by_block"], [0] * 8)

    def test_replay_fill_setup_charge_and_overlap(self):
        phase = {
            "operator": 0, "partition": 0, "active_rows": 2,
            "pwp_rows": 1, "nonzero_correction_rows": 1,
            "early_matcher": 5, "used_pwp_patterns": 1,
            "used_center_runs": 1,
            "work": [2] * 8,
        }
        phases = {sample: [dict(phase) for _ in range(1)]
                  for sample in range(10)}
        model = {
            "elastic_config_bytes": 96, "dram_bytes_per_cycle": 32,
            "dma_command_setup_cycles": 2, "weight_bytes_per_tile": 32,
            "elastic_center_stride_bytes": 32, "tile_slot_bytes": 32768,
            "descriptor_sram_latency_cycles": 3, "tail_cycles": 2,
            "commit_cycles_per_sample": 7,
        }
        free = M463.replay(phases, model, 0, 0, "none", "work")
        charged = M463.replay(phases, model, 1, 2, "block_local", "work")
        self.assertGreater(charged["cycles"], free["cycles"])
        self.assertEqual(charged["components"]["selector_setup"], 80)
        self.assertEqual(charged["components"]["pipeline_fill_drain"], 40)


if __name__ == "__main__":
    unittest.main()
