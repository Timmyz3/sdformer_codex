#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from et3_ordered_trace_replay import load_trace
from profile_local5_hardware_features import (
    OrderedTermTraceSink,
    POST_G0_LANES,
    POST_G0_TOKENS,
    mfep_destination_stream_stats,
    post_g0_qualification,
    rotating_flat_indices,
    source_descriptor_trace,
    source_frontier_work,
    stratified_dataset_indices,
    string_list_sha256,
    validate_binary_k_contract,
    validate_threshold_k_contract,
)


class Local5OrderedTraceSinkTest(unittest.TestCase):
    def test_sequence_stratified_temporal_midpoint_indices(self) -> None:
        files = [
            [f"seq_a_{index:04d}.npy"] for index in range(4)
        ] + [
            [f"seq_b_{index:04d}.npy"] for index in range(6)
        ]
        selected = stratified_dataset_indices(files, 5)
        self.assertEqual(selected, [1, 3, 5, 7, 9])
        self.assertEqual(selected, stratified_dataset_indices(files, 5))
        self.assertEqual(len(set(selected)), 5)

    def test_post_g0_threshold_k_contract(self) -> None:
        event = torch.tensor([0.0, 1.0, 1.0])
        self.assertEqual(validate_binary_k_contract(event.clone(), event), 1.0)
        self.assertEqual(
            validate_threshold_k_contract(
                torch.tensor([0.0, 0.5, 0.5]), event
            ),
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "多个非零幅值"):
            validate_threshold_k_contract(
                torch.tensor([0.0, 0.5, 1.0]), event
            )
        with self.assertRaisesRegex(ValueError, "含负值"):
            validate_threshold_k_contract(
                torch.tensor([0.0, -0.5, 0.0]),
                torch.tensor([0.0, 0.0, 0.0]),
            )
        with self.assertRaisesRegex(ValueError, "支持集不等价"):
            validate_binary_k_contract(
                torch.tensor([0.0, 1.0, 0.0]), event
            )

    def test_mfep_destination_stream_stats_preserve_term_and_parity(self) -> None:
        stats = mfep_destination_stream_stats(
            torch.tensor([10, 10, 20, 20, 20, 30, 30]),
            torch.tensor([0, 2, 1, 4, 9, 0, 20]),
            tokens=32,
        )
        self.assertEqual(stats["mfep_scalar_delivery"], 7)
        self.assertEqual(stats["mfep_ppdi_delivery_exact"], 6)
        self.assertAlmostEqual(
            stats["mfep_ppdi_command_reduction"],
            1.0 / 7.0,
        )
        self.assertEqual(stats["mfep_destination_continuations"], 4)
        self.assertEqual(stats["mfep_destination_delta_histogram"][2], 1)
        self.assertEqual(stats["mfep_destination_delta_histogram"][3], 1)
        self.assertEqual(stats["mfep_destination_delta_histogram"][5], 1)
        self.assertEqual(stats["mfep_destination_delta_histogram"][20], 1)
        self.assertEqual(stats["mfep_destination_delta_escape_b4"], 1)
        self.assertEqual(stats["mfep_destination_delta_escape_b6"], 0)

    def test_uniform_group_capture_and_roundtrip(self) -> None:
        k_candidates = torch.zeros((2, 2, 3, 5, 2), dtype=torch.bool)
        gate = torch.zeros((2, 2, 3, 5), dtype=torch.long)
        valid = torch.ones((3, 5), dtype=torch.bool)
        neighbor = torch.arange(3, dtype=torch.long).view(3, 1).expand(3, 5)
        neighbor = neighbor.clone()
        neighbor[:, 1] = torch.tensor([1, 0, 2])
        neighbor[:, 2] = torch.tensor([0, 2, 1])
        neighbor[:, 4] = torch.tensor([0, 2, 1])
        source_bits = torch.tensor(
            [[1, 0], [0, 1], [1, 1]], dtype=torch.bool
        )
        for window in range(2):
            for head in range(2):
                k_candidates[window, head] = source_bits[neighbor]

        # Uniform selection with two groups chooses flat group 0 and 3.
        # Hardware order is destination, lane, then gate first-occurrence.
        # Gate 128 first appears at candidate 0 but is active on lane 0 only
        # at candidate 2. This distinguishes the contract from candidate-major
        # torch.nonzero traversal.
        gate[0, 0, 1, 0] = 128
        gate[0, 0, 1, 1] = 64
        gate[0, 0, 1, 2] = 128
        gate[1, 1, 2, 4] = 9

        sink = OrderedTermTraceSink(
            groups_per_block_sample=2,
            evidence_level="synthetic",
        )
        sink.capture(
            name="layers.0.swin_blocks.0.attn",
            stage=0,
            block=0,
            sample_id=4,
            k_candidates=k_candidates,
            valid=valid,
            gate_code=gate,
            neighbor_index=neighbor,
        )
        self.assertEqual(sink.group_offsets, [0, 3, 4])
        self.assertEqual(sink.item_gate, [128, 64, 128, 9])
        self.assertEqual(sink.item_lane, [0, 0, 1, 1])
        self.assertEqual(sink.item_mult, [1, 1, 2, 1])
        self.assertEqual(sink.item_dest, [1, 1, 1, 2])
        self.assertEqual(sink.source_group_offsets, [0, 3, 6])
        self.assertEqual(sum(sink.source_term_count), 5)
        self.assertEqual(sink.descriptor_group_offsets, [0, 3, 6])
        self.assertEqual(len(sink.descriptor_k_bitmap), 6)
        self.assertEqual(len(sink.descriptor_incoming_gates), 6)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.yml"
            checkpoint = root / "checkpoint.pth"
            config.write_text(
                """
bsa_attention:
  mode: binary_axnor_local5_shiftmax
  hardware_quant_enabled: true
  hardware_rtl_shiftmax_enabled: true
  hardware_mask_invalid_candidates: true
  hardware_score_step: 0.0078125
  hardware_gate_step: 0.0078125
loader:
  crop: null
  resolution: [480, 640]
test:
  scale_factor: 1
  bn_policy: no_running
swin_transformer:
  window_size: [2, 15, 15]
""".lstrip(),
                encoding="utf-8",
            )
            checkpoint.write_bytes(b"checkpoint")
            sample_keys = ["sample-a"]
            sequence_keys = ["sequence-a"]
            manifest_path, _ = sink.write(
                output_dir=root,
                config=config,
                checkpoint=checkpoint,
                cohort={
                    "sample_key_sha256": string_list_sha256(sample_keys)
                },
                sample_keys=sample_keys,
                sequence_keys=sequence_keys,
                full_resolution=True,
                software_contract={
                    "attention_mode": "binary_axnor_local5_shiftmax",
                    "hardware_quant_enabled": True,
                    "hardware_rtl_shiftmax_enabled": True,
                    "hardware_mask_invalid_candidates": True,
                    "hardware_score_step": 1.0 / 128.0,
                    "hardware_gate_step": 1.0 / 128.0,
                    "crop": None,
                    "resolution": [480, 640],
                    "scale_factor": 1.0,
                    "bn_policy": "no_running",
                    "window_size": [2, 15, 15],
                },
                threshold_semantics={
                    "threshold_modes": ["official_atlif"],
                    "homeostatic_freeze_after_step": 1224,
                    "optimizer_gradient_freeze_enabled": False,
                    "optimizer_threshold_lr": 5.0e-6,
                    "inference_threshold_source": "checkpoint_static_parameter",
                },
            )
            manifest, arrays = load_trace(manifest_path)
            self.assertEqual(manifest["evidence_level"], "synthetic")
            self.assertTrue(manifest["resolution"]["full_resolution"])
            self.assertEqual(len(manifest["groups"]), 2)
            self.assertEqual(int(arrays["group_offsets"][-1]), 4)
            self.assertEqual(int(arrays["source_group_offsets"][-1]), 6)
            self.assertEqual(
                int(arrays["descriptor_group_offsets"][-1]), 6
            )
            self.assertEqual(
                arrays["descriptor_incoming_gates"].shape, (6, 5)
            )
            self.assertEqual(len(arrays["descriptor_valid_mask"]), 6)
            self.assertEqual(int(arrays["source_term_count"].sum()), 5)
            self.assertEqual(
                len(arrays["source_delivery_count"]),
                6,
            )
            self.assertTrue(
                np.all(
                    arrays["source_service_cycles_pipelined"]
                    >= arrays["source_term_count"]
                )
            )
            self.assertEqual(
                arrays["source_retire_destination"].tolist(),
                [1, 2, 2, 1, 2, 2],
            )
            self.assertEqual(len(arrays["destination_delta_total"]), 6)
            self.assertEqual(
                arrays["destination_direction_delta_counts"].shape,
                (6, 4),
            )
            self.assertEqual(
                len(arrays["destination_qfsa_w4_score_cycles"]),
                6,
            )
            self.assertEqual(
                len(arrays["destination_qfsa_w4_direct_mask"]),
                6,
            )
            self.assertEqual(
                len(arrays["destination_qfsa_w4_residual_waves"]),
                6,
            )
            self.assertEqual(
                len(arrays["destination_qfsa_xb4_score_cycles"]),
                6,
            )
            self.assertEqual(
                len(arrays["destination_qfsa_xb4_t8_score_cycles"]),
                6,
            )
            self.assertFalse(manifest["qualification"]["qualified"])

    def test_source_frontier_work_uses_latest_consumer(self) -> None:
        # 1x3 self/E/W拓扑；N/S位置使用self占位但由valid屏蔽。
        neighbor = torch.tensor(
            [
                [0, 0, 0, 0, 1],
                [1, 1, 1, 0, 2],
                [2, 2, 2, 1, 2],
            ],
            dtype=torch.long,
        )
        valid = torch.tensor(
            [
                [1, 0, 0, 0, 1],
                [1, 0, 0, 1, 1],
                [1, 0, 0, 1, 0],
            ],
            dtype=torch.bool,
        )
        gate = torch.zeros((3, 5), dtype=torch.long)
        gate[valid] = 64
        k = torch.zeros((3, 5, 2), dtype=torch.bool)
        source_bits = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool)
        for destination in range(3):
            for candidate in range(5):
                k[destination, candidate] = source_bits[
                    neighbor[destination, candidate]
                ]
        result = source_frontier_work(k, gate, valid, neighbor)
        self.assertEqual(result["source_retire_destination"], [1, 2, 2])
        self.assertEqual(result["source_k_popcount"], [1, 2, 1])
        self.assertEqual(sum(result["source_term_count"]), 4)

    def test_source_descriptor_trace_matches_relation_transpose(self) -> None:
        neighbor = torch.tensor(
            [
                [0, 0, 0, 0, 1],
                [1, 1, 1, 0, 2],
                [2, 2, 2, 1, 2],
            ],
            dtype=torch.long,
        )
        valid = torch.tensor(
            [
                [1, 0, 0, 0, 1],
                [1, 0, 0, 1, 1],
                [1, 0, 0, 1, 0],
            ],
            dtype=torch.bool,
        )
        gate = torch.tensor(
            [
                [10, 0, 0, 0, 11],
                [20, 0, 0, 21, 22],
                [30, 0, 0, 31, 0],
            ],
            dtype=torch.long,
        )
        source_bits = torch.tensor(
            [[1, 0], [1, 1], [0, 1]], dtype=torch.bool
        )
        k = torch.zeros((3, 5, 2), dtype=torch.bool)
        for destination in range(3):
            for role in range(5):
                k[destination, role] = source_bits[
                    neighbor[destination, role]
                ]
        result = source_descriptor_trace(k, gate, valid, neighbor)
        self.assertEqual(result["source_id"], [0, 1, 2])
        self.assertEqual(result["source_k_bitmap"], [1, 3, 2])
        self.assertEqual(result["source_y"], [0, 0, 0])
        self.assertEqual(result["source_x"], [0, 1, 2])
        self.assertEqual(
            result["incoming_gates"],
            [
                [10, 0, 0, 21, 0],
                [20, 0, 0, 31, 11],
                [30, 0, 0, 0, 22],
            ],
        )
        self.assertEqual(
            result["incoming_valid_mask"],
            [0b01001, 0b11001, 0b10001],
        )

    def test_rotating_sampling_covers_all_heads(self) -> None:
        observed = set()
        for sample in range(12):
            for flat in rotating_flat_indices(
                total_groups=60,
                selected_groups=4,
                sample_id=sample,
                stage=2,
                block=1,
            ):
                observed.add(flat % 12)
        self.assertEqual(observed, set(range(12)))

    def test_formal_qualification_accepts_rotating_coverage(self) -> None:
        groups = []
        block_pairs = (
            (0, 0), (0, 1), (1, 0), (1, 1),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (2, 4), (2, 5), (3, 0), (3, 1),
        )
        for module_id, (stage, block) in enumerate(block_pairs):
            for sample in range(100):
                indices = rotating_flat_indices(
                    total_groups=60,
                    selected_groups=4,
                    sample_id=sample,
                    stage=stage,
                    block=block,
                )
                for flat in indices:
                    groups.append(
                        {
                            "module": (
                                f"layers.{stage}.swin_blocks.{block}.attn"
                            ),
                            "stage": stage,
                            "block": block,
                            "sample": sample,
                            "heads": 12,
                            "batch_windows": 5,
                            "window": flat // 12,
                            "head": flat % 12,
                            "flat_group": flat,
                            "tokens": 450,
                            "lanes": 32,
                            "time_planes": 2,
                            "plane_tokens": 225,
                            "spatial_side": 15,
                            "selection": (
                                "coprime_rotating_flat_window_head_v1"
                            ),
                        }
                    )
        value = post_g0_qualification(
            groups,
            processed_samples=100,
            attached_blocks=12,
            groups_per_block_sample=4,
            run_identity_bound=True,
        )
        self.assertTrue(value["qualified"])
        self.assertTrue(value["checks"]["all_head_coverage"])
        self.assertTrue(
            value["checks"]["rotating_flat_group_coverage"]
        )
        tampered = [dict(group) for group in groups]
        changed_flat = (int(tampered[0]["flat_group"]) + 1) % 60
        tampered[0]["flat_group"] = changed_flat
        tampered[0]["window"] = changed_flat // 12
        tampered[0]["head"] = changed_flat % 12
        rejected = post_g0_qualification(
            tampered,
            processed_samples=100,
            attached_blocks=12,
            groups_per_block_sample=4,
            run_identity_bound=True,
        )
        self.assertFalse(rejected["qualified"])
        self.assertFalse(rejected["checks"]["exact_rotating_indices"])

    def test_source_descriptor_covers_nswe_and_two_planes(self) -> None:
        side = 3
        plane_tokens = side * side
        tokens = 2 * plane_tokens
        neighbor = torch.zeros((tokens, 5), dtype=torch.long)
        valid = torch.zeros((tokens, 5), dtype=torch.bool)
        for source in range(tokens):
            plane = source // plane_tokens
            spatial = source % plane_tokens
            y, x = divmod(spatial, side)
            candidates = [
                (y, x),
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ]
            for role, (candidate_y, candidate_x) in enumerate(
                candidates
            ):
                if (
                    0 <= candidate_y < side
                    and 0 <= candidate_x < side
                ):
                    valid[source, role] = True
                    neighbor[source, role] = (
                        plane * plane_tokens
                        + candidate_y * side
                        + candidate_x
                    )
                else:
                    neighbor[source, role] = source
        source_k = torch.stack(
            [
                torch.tensor(
                    [bool(source & 1), bool(source & 2)]
                )
                for source in range(tokens)
            ]
        )
        k = source_k[neighbor]
        gate = torch.zeros((tokens, 5), dtype=torch.long)
        for destination in range(tokens):
            for role in range(5):
                if valid[destination, role]:
                    gate[destination, role] = (
                        1 + destination * 5 + role
                    )
        result = source_descriptor_trace(k, gate, valid, neighbor)
        for plane in range(2):
            center = plane * plane_tokens + 4
            self.assertEqual(
                result["incoming_gates"][center],
                [
                    int(gate[center, 0]),
                    int(gate[center + side, 1]),
                    int(gate[center - side, 2]),
                    int(gate[center + 1, 3]),
                    int(gate[center - 1, 4]),
                ],
            )
            self.assertEqual(
                result["incoming_valid_mask"][center],
                0b11111,
            )
            self.assertEqual(result["source_plane"][center], plane)
            self.assertEqual(result["source_y"][center], 1)
            self.assertEqual(result["source_x"][center], 1)

    def test_post_g0_rejects_non_t450_shape(self) -> None:
        sink = OrderedTermTraceSink(
            groups_per_block_sample=1,
            evidence_level="post_g0",
        )
        with self.assertRaisesRegex(ValueError, "T450"):
            sink.capture(
                name="layers.0.swin_blocks.0.attn",
                stage=0,
                block=0,
                sample_id=0,
                k_candidates=torch.zeros(
                    (1, 1, 3, 5, 2), dtype=torch.bool
                ),
                valid=torch.ones((3, 5), dtype=torch.bool),
                gate_code=torch.zeros(
                    (1, 1, 3, 5), dtype=torch.long
                ),
                neighbor_index=torch.arange(3)
                .view(3, 1)
                .expand(3, 5),
            )

    def test_post_g0_requires_query_trace(self) -> None:
        sink = OrderedTermTraceSink(
            groups_per_block_sample=1,
            evidence_level="post_g0",
        )
        with self.assertRaisesRegex(ValueError, "q_event"):
            sink.capture(
                name="layers.0.swin_blocks.0.attn",
                stage=0,
                block=0,
                sample_id=0,
                k_candidates=torch.zeros(
                    (1, 1, POST_G0_TOKENS, 5, POST_G0_LANES),
                    dtype=torch.bool,
                ),
                valid=torch.ones(
                    (POST_G0_TOKENS, 5), dtype=torch.bool
                ),
                gate_code=torch.zeros(
                    (1, 1, POST_G0_TOKENS, 5), dtype=torch.long
                ),
                neighbor_index=torch.arange(POST_G0_TOKENS)
                .view(POST_G0_TOKENS, 1)
                .expand(POST_G0_TOKENS, 5),
            )

    def test_strict_w15_geometry_rejects_left_right_swap(self) -> None:
        side = 15
        tokens = 2 * side * side
        grid = torch.arange(tokens).reshape(2, side, side)
        neighbors = [grid]
        masks = [torch.ones_like(grid, dtype=torch.bool)]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy = torch.arange(side).view(1, side, 1) + dy
            xx = torch.arange(side).view(1, 1, side) + dx
            valid = (
                (yy >= 0) & (yy < side)
                & (xx >= 0) & (xx < side)
            )
            yy = yy.clamp(0, side - 1).expand(2, side, side)
            xx = xx.clamp(0, side - 1).expand(2, side, side)
            tt = torch.arange(2).view(2, 1, 1).expand_as(yy)
            neighbors.append(grid[tt, yy, xx])
            masks.append(valid.expand(2, side, side))
        neighbor = torch.stack(neighbors, dim=-1).reshape(tokens, 5)
        valid = torch.stack(masks, dim=-1).reshape(tokens, 5)
        source_k = torch.arange(tokens).remainder(3).ne(0).view(
            tokens, 1
        )
        k_candidates = source_k[neighbor]
        gate = torch.zeros((tokens, 5), dtype=torch.long)
        source_descriptor_trace(
            k_candidates,
            gate,
            valid,
            neighbor,
            strict_local5_geometry=True,
        )
        swapped = neighbor.clone()
        swapped[:, [3, 4]] = swapped[:, [4, 3]]
        with self.assertRaisesRegex(ValueError, "精确W15"):
            source_descriptor_trace(
                k_candidates,
                gate,
                valid,
                swapped,
                strict_local5_geometry=True,
            )


if __name__ == "__main__":
    unittest.main()
