import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from profile_local5_hardware_features import (
    independent_direction_min_extra_cycles,
    joint_stencil_delta_stats,
    joint_stencil_route,
    source_gate_lane_stats,
    xorbank_stencil_route,
    xorbank_stencil_residual_bank_loads,
)


class Local5SourceGateLaneStatsTest(unittest.TestCase):
    def test_same_source_gate_lane_reuses_product(self):
        active = torch.zeros((1, 1, 3, 5, 2), dtype=torch.bool)
        gate = torch.zeros((1, 1, 3, 5), dtype=torch.long)
        neighbor = torch.tensor(
            [
                [0, 0, 1, 1, 0],
                [1, 0, 2, 2, 0],
                [2, 1, 2, 2, 1],
            ],
            dtype=torch.long,
        )
        active[0, 0, 0, 0, 0] = True
        active[0, 0, 1, 1, 0] = True
        gate[0, 0, 0, 0] = 7
        gate[0, 0, 1, 1] = 7
        result = source_gate_lane_stats(active, gate, neighbor)
        self.assertEqual(result["source_gate_lane_delivery"], 2)
        self.assertEqual(result["source_gate_lane_terms"], 1)
        self.assertEqual(result["source_gate_lane_max_fanout"], 2)
        self.assertEqual(result["source_instances"], 3)
        self.assertEqual(result["source_active_instances"], 1)
        self.assertEqual(result["source_gate_cardinality_histogram"][1], 1)
        self.assertEqual(
            result["source_gate_cardinality_all_histogram"][0],
            2,
        )

    def test_different_gate_does_not_merge(self):
        active = torch.zeros((1, 1, 2, 5, 1), dtype=torch.bool)
        gate = torch.zeros((1, 1, 2, 5), dtype=torch.long)
        neighbor = torch.tensor(
            [[0, 0, 1, 1, 0], [1, 0, 1, 1, 0]],
            dtype=torch.long,
        )
        active[0, 0, 0, 0, 0] = True
        active[0, 0, 1, 1, 0] = True
        gate[0, 0, 0, 0] = 7
        gate[0, 0, 1, 1] = 9
        result = source_gate_lane_stats(active, gate, neighbor)
        self.assertEqual(result["source_gate_lane_delivery"], 2)
        self.assertEqual(result["source_gate_lane_terms"], 2)
        self.assertEqual(result["source_active_instances"], 1)
        self.assertEqual(result["source_gate_cardinality_histogram"][2], 1)

    def test_dqfs_row_value_quotient_ignores_source(self):
        active = torch.zeros((1, 1, 4, 5, 2), dtype=torch.bool)
        gate = torch.zeros((1, 1, 4, 5), dtype=torch.long)
        neighbor = torch.zeros((4, 5), dtype=torch.long)
        for destination in range(4):
            neighbor[destination, 0] = destination
        active[0, 0, 0, 0, 0] = True
        active[0, 0, 1, 0, 0] = True
        gate[0, 0, 0, 0] = 7
        gate[0, 0, 1, 0] = 7
        result = source_gate_lane_stats(active, gate, neighbor)
        self.assertEqual(result["source_gate_lane_terms"], 2)
        self.assertEqual(result["dqfs_layout_supported"], 1)
        self.assertEqual(result["dqfs_row_groups"], 2)
        self.assertEqual(result["dqfs_row_value_product_computes"], 1)
        self.assertEqual(result["dqfs_row_value_key_histogram"][1], 1)
        self.assertEqual(result["dqfs_value_chain_length_histogram"][2], 1)
        self.assertEqual(result["dqfs_lane_way_overflow_groups_w2"], 0)

    def test_joint_delta_cross_direction_packing(self):
        # 四方向各改变一个不同lane，W4可在一个residual周期联合处理。
        k = torch.zeros((1, 1, 1, 5, 8), dtype=torch.bool)
        for direction in range(1, 5):
            k[0, 0, 0, direction, direction - 1] = True
        valid = torch.ones((1, 5), dtype=torch.bool)
        result = joint_stencil_delta_stats(k, valid)
        self.assertEqual(result["joint_delta_event_sum"], 4)
        self.assertEqual(result["direct_serial_score_cycle_sum"], 5)
        self.assertEqual(result["qfsa_w4_score_cycle_sum"], 2)
        self.assertEqual(result["qfsa_w2_score_cycle_sum"], 3)
        self.assertEqual(
            int(
                independent_direction_min_extra_cycles(
                    torch.tensor([[1, 1, 1, 1]])
                )[0].item()
            ),
            1,
        )

    def test_cross_direction_packing_beats_imbalanced_w1x4(self):
        counts = torch.tensor([[4, 0, 0, 0]], dtype=torch.long)
        independent = independent_direction_min_extra_cycles(counts)
        # Shared direct engine wins this dense single-direction case in one拍。
        self.assertEqual(int(independent[0].item()), 1)
        counts = torch.tensor([[2, 2, 0, 0]], dtype=torch.long)
        independent = independent_direction_min_extra_cycles(counts)
        self.assertEqual(int(independent[0].item()), 2)
        pooled, _, direct, residual = joint_stencil_route(counts, 4)
        self.assertEqual(int(pooled[0].item()), 1)
        self.assertEqual(int(direct[0].item()), 0)
        self.assertEqual(int(residual[0].item()), 1)

    def test_joint_delta_router_can_choose_direct_fallback(self):
        k = torch.zeros((1, 1, 1, 5, 8), dtype=torch.bool)
        k[0, 0, 0, 1, :] = True
        valid = torch.ones((1, 5), dtype=torch.bool)
        result = joint_stencil_delta_stats(k, valid)
        # 对8-lane dense方向，W4两波不如共享direct popcount一拍。
        self.assertEqual(result["qfsa_w4_score_cycle_sum"], 2)

    def test_xorbank_spreads_one_direction_across_four_banks(self):
        k = torch.zeros((1, 1, 1, 5, 32), dtype=torch.bool)
        k[0, 0, 0, 1, 0:4] = True
        valid = torch.ones((1, 5), dtype=torch.bool)
        cycles, mask, direct, residual = xorbank_stencil_route(k, valid)
        self.assertEqual(int(cycles[0, 0, 0].item()), 1)
        self.assertEqual(int(mask[0, 0, 0].item()), 0)
        self.assertEqual(int(direct[0, 0, 0].item()), 0)
        self.assertEqual(int(residual[0, 0, 0].item()), 1)
        loads = xorbank_stencil_residual_bank_loads(k, valid, mask)
        self.assertEqual(loads[0, 0, 0].tolist(), [1, 1, 1, 1])

    def test_xorbank_bank_load_excludes_direct_direction(self):
        k = torch.zeros((1, 1, 1, 5, 32), dtype=torch.bool)
        k[0, 0, 0, 1, 0:12] = True
        k[0, 0, 0, 2, 0:4] = True
        valid = torch.ones((1, 5), dtype=torch.bool)
        _, mask, _, residual = xorbank_stencil_route(
            k,
            valid,
            threshold=8,
        )
        self.assertEqual(int(mask[0, 0, 0].item()), 1)
        loads = xorbank_stencil_residual_bank_loads(k, valid, mask)
        self.assertEqual(loads[0, 0, 0].tolist(), [1, 1, 1, 1])
        self.assertEqual(int(residual[0, 0, 0].item()), 1)

    def test_dual_bound_route_removes_adversarial_hot_bank(self):
        k = torch.zeros((1, 1, 1, 5, 32), dtype=torch.bool)
        for direction in range(4):
            for lane in range(32):
                if (lane % 4) == direction:
                    k[0, 0, 0, direction + 1, lane] = True
        valid = torch.ones((1, 5), dtype=torch.bool)
        cycles_t8, mask_t8, _, residual_t8 = xorbank_stencil_route(
            k,
            valid,
            threshold=8,
        )
        self.assertEqual(int(mask_t8[0, 0, 0].item()), 0)
        self.assertEqual(int(residual_t8[0, 0, 0].item()), 32)
        cycles_b2, mask_b2, direct_b2, residual_b2 = (
            xorbank_stencil_route(
                k,
                valid,
                threshold=8,
                bank_pressure_threshold=2,
            )
        )
        self.assertEqual(int(mask_b2[0, 0, 0].item()), 15)
        self.assertEqual(int(direct_b2[0, 0, 0].item()), 4)
        self.assertEqual(int(residual_b2[0, 0, 0].item()), 0)
        self.assertEqual(int(cycles_b2[0, 0, 0].item()), 4)
        self.assertGreater(
            int(cycles_t8[0, 0, 0].item()),
            int(cycles_b2[0, 0, 0].item()),
        )
        cycles_t8_pipe, _, _, _ = xorbank_stencil_route(
            k,
            valid,
            threshold=8,
            pipeline_drain=True,
        )
        self.assertEqual(int(cycles_t8_pipe[0, 0, 0].item()), 33)

    def test_pipeline_drain_is_inside_direct_residual_max(self):
        k = torch.zeros((1, 1, 1, 5, 32), dtype=torch.bool)
        k[0, 0, 0, 1, 0:12] = True
        k[0, 0, 0, 2, 0:4] = True
        valid = torch.ones((1, 5), dtype=torch.bool)
        cycles, mask, direct, residual = xorbank_stencil_route(
            k,
            valid,
            threshold=8,
            pipeline_drain=True,
        )
        self.assertEqual(int(mask[0, 0, 0].item()), 1)
        self.assertEqual(int(direct[0, 0, 0].item()), 1)
        self.assertEqual(int(residual[0, 0, 0].item()), 1)
        self.assertEqual(int(cycles[0, 0, 0].item()), 2)


if __name__ == "__main__":
    unittest.main()
