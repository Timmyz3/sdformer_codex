from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neuron_experiments.H9_bipolar_self_attention.entrypoints.build_mvsec_cicc_manifests import (
    build_manifest,
)
from neuron_experiments.H9_bipolar_self_attention.entrypoints.run_h9_standard_mvsec_eval import (
    audit_eval_load,
)
from third_party.SDformerFlow.MDR_dataloader.mvsec_protocol import (
    MVSECDirectAugmentor,
    apply_mvsec_source_valid_region,
    event_activity_mask,
    load_mvsec_split_manifest,
)


class MVSECDirectProtocolTest(unittest.TestCase):
    def test_manifest_has_isolated_train_validation_and_fixed_test_sets(self) -> None:
        manifest = build_manifest()
        splits = manifest["splits"]
        train = splits["train"]["indices"]
        validation = splits["validation"]["indices"]
        self.assertEqual(len(train), 2363)
        self.assertEqual(len(validation), 263)
        self.assertGreaterEqual(min(validation) - max(train), 2)
        self.assertFalse(set(train) & set(validation))
        for sequence in (
            "outdoor_day1",
            "indoor_flying1",
            "indoor_flying2",
            "indoor_flying3",
        ):
            indices = splits[f"test_fixed800_{sequence}"]["indices"]
            self.assertEqual(len(indices), 800)
            self.assertEqual(len(set(indices)), 800)

    def test_manifest_loader_rejects_wrong_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(build_manifest()) + "\n", encoding="utf-8")
            indices, digest = load_mvsec_split_manifest(path, "train", "outdoor_day2")
            self.assertEqual(len(indices), 2363)
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(RuntimeError, "expected 'outdoor_day1'"):
                load_mvsec_split_manifest(path, "train", "outdoor_day1")

    def test_direct_augmentor_transforms_flow_and_valid_mask_together(self) -> None:
        event = np.arange(3 * 4, dtype=np.float32).reshape(3, 4, 1)
        flow = np.ones((3, 4, 2), dtype=np.float32)
        valid = np.zeros((3, 4), dtype=bool)
        valid[0, 0] = True
        augmentor = MVSECDirectAugmentor(
            (3, 4),
            horizontal_flip_probability=1.0,
            vertical_flip_probability=1.0,
        )
        event1, event2, d_event1, d_event2, transformed_flow, transformed_valid = augmentor(
            event, event, event, event, flow, valid
        )
        for transformed_event in (event1, event2, d_event1, d_event2):
            np.testing.assert_array_equal(transformed_event, event[::-1, ::-1])
        np.testing.assert_array_equal(transformed_flow[..., 0], -np.ones((3, 4)))
        np.testing.assert_array_equal(transformed_flow[..., 1], -np.ones((3, 4)))
        self.assertTrue(transformed_valid[-1, -1])
        self.assertEqual(int(transformed_valid.sum()), 1)

    def test_event_activity_mask_collapses_time_and_polarity(self) -> None:
        event_volume = torch.zeros((2, 10, 2, 3, 4))
        event_volume[0, 2, 0, 1, 2] = 1
        event_volume[0, 2, 1, 1, 2] = -1
        event_volume[1, 9, 1, 2, 3] = -1
        mask = event_activity_mask(event_volume)
        self.assertEqual(tuple(mask.shape), (2, 1, 3, 4))
        self.assertTrue(mask[0, 0, 1, 2])
        self.assertTrue(mask[1, 0, 2, 3])
        self.assertEqual(int(mask.sum()), 2)

        four_dimensional = event_volume[:, :, 0]
        self.assertEqual(
            tuple(event_activity_mask(four_dimensional).shape),
            (2, 1, 3, 4),
        )

    def test_source_valid_region_is_applied_before_center_crop(self) -> None:
        valid = torch.ones((260, 346), dtype=torch.bool)
        outdoor = apply_mvsec_source_valid_region(valid, "outdoor_day2")
        self.assertTrue(outdoor[192].all())
        self.assertFalse(outdoor[193:].any())
        center256 = outdoor[2:258, 45:301]
        self.assertEqual(tuple(center256.shape), (256, 256))
        self.assertTrue(center256[190].all())
        self.assertFalse(center256[191:].any())
        self.assertTrue(apply_mvsec_source_valid_region(valid, "indoor_flying1").all())

    def test_eval_load_audit_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "eval.log"
            log_path.write_text(
                "[H9] eval installed ATLIFTernaryPSN: 105 modules\n"
                "[H9] eval installed Shiftmax attention: 12 modules\n"
                "[H9] load audit: checkpoint_overlay_keys=210, model_overlay_keys=210, "
                "missing=0, unexpected=0\n",
                encoding="utf-8",
            )
            config = {
                "atlif_ternary_psn": {"enabled": True},
                "bsa_attention": {"enabled": True},
            }
            audit_eval_load(log_path, config)
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace("missing=0", "missing=2"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "load audit failed"):
                audit_eval_load(log_path, config)


if __name__ == "__main__":
    unittest.main()
