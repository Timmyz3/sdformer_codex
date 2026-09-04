from __future__ import absolute_import

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m25_resource_bounded_tiled_cycles.py"
CONTRACT = Path(__file__).resolve().parents[2] / "contracts/m25_tiled_cycle_input_contract_r1_20260822.json"


def load_module():
    spec = importlib.util.spec_from_file_location("m25_tiled", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M25TiledCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_discrete_atlif_rounds_each_invocation(self):
        rows = [
            {"name": "a", "deployment_dead_result": "False", "temporal_steps": "2",
             "elements_per_frame": "12", "dense_macs_per_frame": "24"},
            {"name": "b", "deployment_dead_result": "False", "temporal_steps": "2",
             "elements_per_frame": "12", "dense_macs_per_frame": "24"},
        ]
        result = self.module.packed_atlif_service(rows, lanes=1, slots=10)
        self.assertEqual(result["service_cycles"], 8)
        self.assertEqual(result["multipliers"], 10)

    def test_exact_row_aligned_plan_never_crosses_barrier(self):
        cohorts = [{
            "cohort_id": 0, "sample_id": "0", "sequence_key": "s",
            "name": "op", "operator": "Linear", "operator_call_index": "0",
            "weight_group": "0", "row_ids": [0, 1, 2, 3], "temporal_steps": 10,
            "contexts": 4, "source_widths": [96, 96, 96, 96], "fanout": 384,
        }]
        summary, plan = self.module.plan_exact_cohorts(
            "H67", 96, cohorts, {"total_bytes": 52032}, 10
        )
        self.assertEqual(summary["maximum_tiles_per_cohort"], 2)
        self.assertEqual([row["lane_slice_count"] for row in plan], [3, 1])
        self.assertTrue(all(row["barrier_key"] == plan[0]["barrier_key"] for row in plan))
        self.assertTrue(all(row["barrier_crossing"] == 0 for row in plan))
        self.assertTrue(all(row["maximum_simultaneous_bytes"] <= 96 * 1024 for row in plan))
        self.assertTrue(all(row["tile_state_bytes"] % 96 == 0 for row in plan))
        self.assertEqual(plan[0]["activation_payload_bytes"], 48)
        self.assertEqual(plan[0]["activation_state_bytes"], 96)

    def test_exact_plan_rejects_one_lane_slice_that_cannot_fit(self):
        cohorts = [{
            "cohort_id": 0, "sample_id": "0", "sequence_key": "s",
            "name": "op", "operator": "Linear", "operator_call_index": "0",
            "weight_group": "0", "row_ids": [0, 1, 2, 3], "temporal_steps": 10,
            "contexts": 4, "source_widths": [96, 96, 96, 96], "fanout": 96,
        }]
        with self.assertRaises(ValueError):
            self.module.plan_exact_cohorts(
                "H67", 63, cohorts, {"total_bytes": 52032}, 10
            )

    def test_exact_cohort_rejects_incomplete_chunk_identity(self):
        rows = []
        for row_id in range(4):
            rows.append({
                "sample_id": "0", "sequence_key": "seq", "name": "op",
                "operator": "Linear", "operator_call_index": "0",
                "row_id": str(row_id), "weight_group": "0", "source_width": "96",
                "chunks_per_row": "2", "output_channel_fanout": "96",
                "temporal_step": "0", "chunk_index": "0",
            })
        with self.assertRaises(ValueError):
            self.module.exact_cohorts(rows, 4)

    def test_fixed_reserve_is_charged_in_canonical_build(self):
        with tempfile.TemporaryDirectory(prefix="m25_fixed_") as directory:
            payload = self.module.build(CONTRACT, Path(directory) / "artifact")
            fixed = payload["fixed_resident_footprint"]
            self.assertEqual(fixed["total_bytes"], 52032)
            self.assertEqual(fixed["dynamic_bn_scale_offset_q32_bytes"], 12288)
            self.assertGreater(fixed["m21_fifo_sideband_and_snapshot_reserve_bytes"], 0)

    def test_memory_cycles_uses_frequency_and_decimal_bandwidth(self):
        # 64 GB/s / 320 MHz = 200 bytes/cycle.
        self.assertEqual(self.module.memory_cycles(401, 64, 320), 3)

    def test_flat_multiplier_point_is_explicit_lower_bound(self):
        rows = [
            {"deployment_dead_result": "False", "dense_macs_per_frame": "193"},
            {"deployment_dead_result": "False", "dense_macs_per_frame": "95"},
        ]
        result = self.module.flat_multiplier_service_lower_bound(rows, 96)
        self.assertEqual(result["service_cycles_lower_bound"], 4)
        self.assertIn("LOWER_BOUND", result["status"])

    def test_two_x_requirement_can_be_impossible(self):
        m7 = {"system_envelope": {
            "fixed_baseline_cycles": 1000,
            "noneligible_operator_cycles": 300,
            "qk_projection_cycles_frozen_unprofiled": 100,
            "rqtb_attention_cycles": 100,
            "m4_profiled_eligible_cycles": 500,
            "variants": {"local": {"effective_m4_speedup_vs_local_p1": 5.0}},
        }}
        result = self.module.two_x_requirements(
            m7, "local", {"service_cycles": 1}, 0, 0
        )
        self.assertEqual(
            result["required_m4_speed_at_this_discrete_atlif_point"],
            "IMPOSSIBLE_EVEN_INFINITE_M4",
        )

    def test_canonical_build_closes_shas_and_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="m25_test_") as directory:
            output = Path(directory) / "artifact"
            payload = self.module.build(CONTRACT, output)
            self.assertEqual(
                payload["status"],
                "PASS_FROZEN_C4_TILING_AND_CYCLE_ENVELOPE_HEADLINE_NO_GO",
            )
            self.assertEqual(payload["attention_completeness"]["Local5"]["speedup"], "UNKNOWN")
            self.assertFalse(
                payload["conclusions"]
                ["crosses_2x_at_discrete_l8_under_96_atlif_multiplier_budget"]
            )
            self.assertFalse(
                payload["conclusions"]
                ["crosses_2x_at_ideal_flat_exact96_arithmetic_lower_bound"]
            )
            self.assertEqual(
                payload["m21_registered_result_correction"]["registered_result_bubble_cycles"],
                738,
            )
            self.assertEqual(
                payload["m21_registered_result_correction"]
                ["local_fifo4_phase1_incremental_cycles"],
                6098531,
            )
            operator_schedule = payload["m21_registered_result_correction"][
                "per_operator_barrier_schedule"
            ]
            self.assertEqual(len(operator_schedule), 13)
            self.assertEqual(sum(row["lane_tile_count"] for row in operator_schedule), 123)
            self.assertTrue(all(
                row["lane_tile_count"] == (row["exact_frozen_fanout"] + 95) // 96
                for row in operator_schedule
            ))
            self.assertEqual(
                payload["exact_row_aligned_plan"]["dynamic_bn_barrier_crossings"], 0
            )
            manifest = json.loads(
                (output / "m25_output_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["files"]), 6)


if __name__ == "__main__":
    unittest.main()
