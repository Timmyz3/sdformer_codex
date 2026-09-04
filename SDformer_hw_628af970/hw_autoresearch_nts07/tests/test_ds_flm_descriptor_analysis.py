from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ds_flm_descriptor_manifest import (
    analyze,
    expected_source_binding_paths,
    validate_formal_identity_and_coverage,
)
from et3_ordered_trace_replay import file_sha256
from profile_local5_hardware_features import (
    OrderedTermTraceSink,
    rotating_flat_indices,
    string_list_sha256,
)


class DsFlmDescriptorAnalysisTest(unittest.TestCase):
    def test_post_g0_descriptor_analysis(self) -> None:
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
            [[[[10, 0, 0, 0, 11], [20, 0, 0, 21, 22],
               [30, 0, 0, 31, 0]]]],
            dtype=torch.long,
        )
        source_bits = torch.tensor(
            [[1, 0], [1, 1], [0, 1]], dtype=torch.bool
        )
        k = torch.zeros((1, 1, 3, 5, 2), dtype=torch.bool)
        for destination in range(3):
            for role in range(5):
                k[0, 0, destination, role] = source_bits[
                    neighbor[destination, role]
                ]
        sink = OrderedTermTraceSink(
            groups_per_block_sample=1,
            evidence_level="synthetic",
        )
        sink.capture(
            name="layers.0.swin_blocks.0.attn",
            stage=0,
            block=0,
            sample_id=0,
            k_candidates=k,
            valid=valid,
            gate_code=gate,
            neighbor_index=neighbor,
        )
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
            keys = ["sample-a"]
            manifest, _ = sink.write(
                output_dir=root,
                config=config,
                checkpoint=checkpoint,
                cohort={"sample_key_sha256": string_list_sha256(keys)},
                sample_keys=keys,
                sequence_keys=["sequence-a"],
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
            value = analyze(manifest, require_formal=False)
            self.assertEqual(value["evidence_level"], "synthetic")
            self.assertEqual(value["descriptors"], 3)
            self.assertEqual(value["nonempty_descriptors"], 3)
            self.assertEqual(value["active_lanes"]["max"], 2)
            self.assertGreaterEqual(value["unique_gates"]["max"], 1)
            self.assertEqual(
                value["state_invariant_nonempty_descriptors"], 3
            )
            self.assertIn("lane", value["lane_major_hamming"])
            self.assertIn("gate", value["gate_major_hamming"])
            self.assertEqual(
                value["lane_major_within_descriptor_lane_runs"],
                sum(
                    int(bitmap).bit_count()
                    for bitmap in [1, 3, 2]
                ),
            )
            self.assertEqual(
                value["gate_major_within_descriptor_lane_runs"],
                8,
            )
            with self.assertRaisesRegex(ValueError, "只接受post_g0"):
                analyze(manifest)

            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            payload = root / manifest_value["payload_file"]
            with np.load(payload, allow_pickle=False) as loaded:
                arrays = {name: loaded[name].copy() for name in loaded.files}
            gates = arrays["descriptor_incoming_gates"]
            nonzero = np.argwhere(gates > 0)
            self.assertGreater(len(nonzero), 0)
            row, role = nonzero[0]
            gates[row, role] = int(gates[row, role]) + 1
            np.savez_compressed(payload, **arrays)
            manifest_value["payload_sha256"] = file_sha256(payload)
            manifest.write_text(
                json.dumps(manifest_value), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "更新多重集不等价"):
                analyze(manifest, require_formal=False)

    def test_rejects_pre_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(
                """
{
  "evidence_level": "post_g0",
  "source_descriptor_contract": {
    "id": "qfit_relation_transpose_source_descriptor_v1"
  }
}
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "v3 source descriptor"):
                analyze(manifest)

    def test_fail_closed_identity_and_full_coverage(self) -> None:
        groups = []
        block_pairs = (
            (0, 0), (0, 1), (1, 0), (1, 1),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (2, 4), (2, 5), (3, 0), (3, 1),
        )
        for module_id, (stage, block) in enumerate(block_pairs):
            module = f"layers.{stage}.swin_blocks.{block}.attn"
            for sample in range(100):
                for flat in rotating_flat_indices(
                    total_groups=60,
                    selected_groups=4,
                    sample_id=sample,
                    stage=stage,
                    block=block,
                ):
                    groups.append(
                        {
                            "module": module,
                            "stage": stage,
                            "block": block,
                            "sample": sample,
                            "heads": 12,
                            "batch_windows": 5,
                            "window": flat // 12,
                            "head": flat % 12,
                            "flat_group": flat,
                            "selection": (
                                "coprime_rotating_flat_window_head_v1"
                            ),
                        }
                    )
        relation_rtl = ROOT / "rtl_qfit/qfit_relation_transpose_leaf.sv"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            status = root / "deploy.log"
            ranking = root / "ranking.md"
            config = root / "config.yml"
            checkpoint = root / "checkpoint_epoch7.pth"
            identity_path = root / "identity.json"
            manifest_path = root / "manifest.json"
            receipt_path = root / "release_receipt.json"
            cohort_path = root / "cohort.json"
            prefix = b"[watcher] WAIT\n"
            marker_line = (
                "[watcher] ALL COMPLETE fullres deploy followup H67 H66d\n"
            )
            status.write_bytes(prefix + marker_line.encode("utf-8"))
            ranking.write_text("ranked\n", encoding="utf-8")
            config.write_text("config\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            receipt = {
                "schema": "local5_release_receipt_v2",
                "watcher_session_uuid": "unit-test-watcher",
                "release_marker": (
                    "ALL COMPLETE fullres deploy followup"
                ),
                "marker_line": marker_line.rstrip("\n"),
                "status_path": str(status),
                "status_prefix_bytes": len(prefix),
                "status_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "marker_start_offset": len(prefix),
                "marker_end_offset": len(prefix) + len(
                    marker_line.encode("utf-8")
                ),
                "ranking_path": str(ranking.resolve()),
                "ranking_sha256": file_sha256(ranking),
                "best_epoch": 7,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "config_path": str(config.resolve()),
                "config_sha256": file_sha256(config),
            }
            receipt_path.write_text(
                json.dumps(receipt),
                encoding="utf-8",
            )
            dataset_indices = list(range(100))
            cohort = {
                "schema": "ordered_trace_cohort_v2",
                "count": 100,
                "dataset_sampling_id": (
                    "sequence_proportional_temporal_midpoint_v1"
                ),
                "dataset_size": 100,
                "dataset_indices": dataset_indices,
                "dataset_indices_sha256": hashlib.sha256(
                    ("\n".join(str(value) for value in dataset_indices) + "\n")
                    .encode("utf-8")
                ).hexdigest(),
                "sequence_counts": {"sequence-a": 100},
            }
            cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
            identity = {
                "schema": "local5_post_g0_run_identity_v3",
                "release_marker": (
                    "ALL COMPLETE fullres deploy followup"
                ),
                "deploy_status": str(status),
                "release_receipt": str(receipt_path),
                "release_receipt_sha256": file_sha256(receipt_path),
                "watcher_session_uuid": "unit-test-watcher",
                "ranking": str(ranking),
                "ranking_sha256": file_sha256(ranking),
                "config": str(config.resolve()),
                "config_sha256": file_sha256(config),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "best_epoch": 7,
                "relation_rtl_sha256": file_sha256(relation_rtl),
                "samples": 100,
                "groups_per_block_sample": 4,
                "sampling_id": (
                    "coprime_rotating_flat_window_head_v1"
                ),
                "dataset_sampling_id": (
                    "sequence_proportional_temporal_midpoint_v1"
                ),
                "source_bindings": {
                    name: {
                        "path": str(path.resolve()),
                        "sha256": file_sha256(path),
                    }
                    for name, path in expected_source_binding_paths().items()
                },
            }
            identity_path.write_text(
                json.dumps(identity),
                encoding="utf-8",
            )
            manifest = {
                "config_sha256": file_sha256(config),
                "checkpoint_sha256": file_sha256(checkpoint),
                "run_identity_file": str(identity_path),
                "run_identity_file_sha256": file_sha256(identity_path),
                "source_descriptor_contract": {
                    "rtl_reference_sha256": file_sha256(relation_rtl)
                },
                "cohort_file": cohort_path.name,
                "cohort_file_sha256": file_sha256(cohort_path),
                "groups": groups,
            }
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            value = validate_formal_identity_and_coverage(
                manifest_path,
                manifest,
            )
            self.assertEqual(value["samples"], 100)
            self.assertEqual(value["blocks"], 12)
            self.assertEqual(value["groups"], 4800)

            status.write_bytes(b"[watcher] EDIT\n" + marker_line.encode("utf-8"))
            with self.assertRaisesRegex(ValueError, "release receipt"):
                validate_formal_identity_and_coverage(
                    manifest_path,
                    manifest,
                )
            status.write_bytes(prefix + marker_line.encode("utf-8"))

            manifest["groups"] = groups[:-1]
            with self.assertRaisesRegex(ValueError, "100 sample"):
                validate_formal_identity_and_coverage(
                    manifest_path,
                    manifest,
                )


if __name__ == "__main__":
    unittest.main()
