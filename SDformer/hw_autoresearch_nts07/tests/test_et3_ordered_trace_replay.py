#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from et3_ordered_trace_replay import (
    ReplayConfig,
    causal_dual_context_cycles,
    canonical_item_hash,
    file_sha256,
    int_list_sha256,
    load_trace,
    replay_group,
    string_list_sha256,
    validate_ordered_trace_cohort,
)


class Et3OrderedTraceReplayTest(unittest.TestCase):
    def make_trace(self, root: Path, duplicate: bool = False) -> Path:
        # Group 0 forces two drains with cap=1, segment=2, fallback=1.
        rows = [
            (1, 4, 2, 2, 0),
            (1, 4, 2, 2, 2),
            (1, 1, 3, 3, 1),
            (1, 2, 1, 4, 4),
            (1, 3, 0, 2, 6),
        ]
        if duplicate:
            rows[-1] = rows[0]
        arrays = {
            "group_offsets": np.asarray([0, len(rows), len(rows)], dtype=np.int64),
            "group_tags": np.asarray([0x22, 0x33], dtype=np.uint64),
            "item_mode_multiset": np.asarray([row[0] for row in rows], dtype=np.uint8),
            "item_gate_code": np.asarray([row[1] for row in rows], dtype=np.uint16),
            "item_lane_id": np.asarray([row[2] for row in rows], dtype=np.uint16),
            "item_multiplicity": np.asarray([row[3] for row in rows], dtype=np.uint8),
            "item_destination": np.asarray([row[4] for row in rows], dtype=np.uint16),
        }
        payload = root / "ordered_items.npz"
        np.savez_compressed(payload, **arrays)
        groups = [
            {
                "tag": 0x22,
                "empty": False,
                "sample": 0,
                "stage": 0,
                "block": 0,
                "window": 0,
                "head": 0,
                "ordered_item_sha256": canonical_item_hash(
                    arrays, 0, len(rows)
                ),
            },
            {
                "tag": 0x33,
                "empty": True,
                "sample": 0,
                "stage": 0,
                "block": 0,
                "window": 1,
                "head": 0,
                "ordered_item_sha256": hashlib.sha256(b"").hexdigest(),
            },
        ]
        manifest = {
            "schema": "et3_ordered_term_trace_v1",
            "evidence_level": "synthetic",
            "payload_file": payload.name,
            "payload_sha256": file_sha256(payload),
            "config_sha256": "synthetic",
            "checkpoint_sha256": "synthetic",
            "cohort_sha256": "synthetic",
            "resolution": {"tokens": 16, "full_resolution": False},
            "groups": groups,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_replay_drain_empty_and_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, arrays = load_trace(self.make_trace(Path(temp)))
            self.assertEqual(manifest["evidence_level"], "synthetic")
            config = ReplayConfig(
                key_cap=1,
                segment_depth=2,
                fallback_depth=1,
                weight_read_latency=2,
            )
            stats = replay_group(arrays, 0, 5, 0x22, config)
            self.assertEqual(stats.items, 5)
            self.assertEqual(stats.ideal_terms, 4)
            self.assertEqual(stats.online_terms, 4)
            self.assertEqual(stats.partial_drains, 1)
            self.assertEqual(stats.fallback_items, 2)
            self.assertEqual(stats.native_product_computes, 5)
            self.assertEqual(stats.et3_product_computes, 4)
            self.assertEqual(stats.native_queue_cycles, 7)
            self.assertEqual(stats.et3_single_context_cycles, 14)
            self.assertEqual(
                stats.et3_dual_context_causal_cycles,
                12,
            )
            empty = replay_group(arrays, 5, 5, 0x33, config)
            self.assertTrue(empty.empty)
            self.assertEqual(empty.native_queue_cycles, 1)

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_trace(Path(temp))
            manifest = json.loads(path.read_text())
            manifest["payload_sha256"] = "bad"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_trace(path)

    def test_duplicate_upstream_item_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _, arrays = load_trace(
                self.make_trace(Path(temp), duplicate=True)
            )
            config = ReplayConfig(1, 2, 1)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                replay_group(arrays, 0, 5, 0x22, config)

    def test_invalid_motion_multiplicity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_trace(Path(temp))
            manifest, arrays = load_trace(path)
            arrays["item_mode_multiset"][0] = 0
            manifest["groups"][0]["ordered_item_sha256"] = canonical_item_hash(
                arrays, 0, 5
            )
            with self.assertRaisesRegex(ValueError, "Motion SET"):
                from et3_ordered_trace_replay import validate_trace

                validate_trace(manifest, arrays)

    def test_causal_dual_context_fill_and_drain(self) -> None:
        self.assertEqual(
            causal_dual_context_cycles(
                [
                    (3, 4, 1),
                    (2, 3, 0),
                    (5, 2, 0),
                ]
            ),
            15,
        )

    def test_post_g0_requires_complete_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_trace(Path(temp))
            manifest = json.loads(path.read_text())
            manifest["evidence_level"] = "post_g0"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_trace(path)

    def test_post_g0_rejects_label_and_fake_hash_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_trace(Path(temp))
            manifest = json.loads(path.read_text())
            manifest.update(
                {
                    "evidence_level": "post_g0",
                    "config": "missing-config.yml",
                    "config_sha256": "a" * 64,
                    "checkpoint": "missing-checkpoint.pth",
                    "checkpoint_sha256": "b" * 64,
                    "cohort_file": "missing-cohort.json",
                    "cohort_file_sha256": "c" * 64,
                    "cohort_sha256": "d" * 64,
                }
            )
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "artifact is missing"):
                load_trace(path)

    def test_v2_cohort_sampling_contract(self) -> None:
        sample_keys = [f"sample-{index}" for index in range(100)]
        sequence_keys = [f"sequence-{index // 10}" for index in range(100)]
        dataset_indices = list(range(100))
        cohort = {
            "schema": "ordered_trace_cohort_v2",
            "count": 100,
            "sample_keys": sample_keys,
            "sequence_keys": sequence_keys,
            "sample_key_sha256": string_list_sha256(sample_keys),
            "sequence_key_sha256": string_list_sha256(sequence_keys),
            "dataset_sampling_id": (
                "sequence_proportional_temporal_midpoint_v1"
            ),
            "dataset_size": 825,
            "dataset_indices": dataset_indices,
            "dataset_indices_sha256": int_list_sha256(dataset_indices),
            "sequence_counts": {
                f"sequence-{index}": 10 for index in range(10)
            },
        }
        validate_ordered_trace_cohort(cohort)
        cohort["dataset_indices_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "v2 sampling contract"):
            validate_ordered_trace_cohort(cohort)


if __name__ == "__main__":
    unittest.main()
