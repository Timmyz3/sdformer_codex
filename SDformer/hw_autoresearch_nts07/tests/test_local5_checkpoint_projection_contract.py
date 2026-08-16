from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from generate_local5_active_projection_postg0_vectors import (  # noqa: E402
    load_checkpoint_projection_contract,
)
from profile_local5_hardware_features import (  # noqa: E402
    POST_G0_BLOCK_PAIRS,
    file_sha256,
    write_checkpoint_projection_contract,
)


class FakeAttention(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self._h9_shiftmax_cfg = SimpleNamespace(
            mode="binary_axnor_local5_shiftmax"
        )
        self.num_heads = dim // 32
        self.proj = torch.nn.Linear(dim, dim)
        self.sn_k = SimpleNamespace(
            spiking_neuron=SimpleNamespace(thresh=torch.nn.Parameter(torch.tensor(0.75)))
        )


class FakeModel:
    def __init__(self) -> None:
        dimensions = {0: 96, 1: 192, 2: 384, 3: 768}
        self.rows = [
            (
                "sttmultires_unet.encoders.swin3d.layers."
                f"{stage}.swin_blocks.{block}.attn",
                FakeAttention(dimensions[stage]),
            )
            for stage, block in POST_G0_BLOCK_PAIRS
        ]

    def named_modules(self):
        yield "", self
        yield from self.rows


class Local5CheckpointProjectionContractTest(unittest.TestCase):
    def test_exports_and_reloads_all_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint-fixture")
            manifest_path, payload_path, manifest = (
                write_checkpoint_projection_contract(
                    FakeModel(),
                    output_dir=root,
                    checkpoint=checkpoint,
                    bn_policy="no_running",
                )
            )

            self.assertEqual(
                manifest["schema"],
                "local5_checkpoint_projection_contract_v2",
            )
            self.assertEqual(manifest["status"], "THETA_FOLDED_WEIGHT_CONTRACT")
            self.assertEqual(
                manifest["topology_contract"],
                "local5_swin_2_2_6_2_c96_192_384_768_h3_6_12_24_v1",
            )
            self.assertEqual(len(manifest["blocks"]), 12)
            self.assertEqual(
                manifest["blocks"][0]["bias_name"],
                "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0."
                "attn.proj.bias",
            )
            self.assertIn("not_permitted", manifest["bn_folding"])
            with np.load(payload_path) as payload:
                self.assertEqual(
                    payload["s0_b0_weight_int8"].shape, (96, 96)
                )
                self.assertEqual(
                    payload["s3_b1_weight_int8"].shape, (768, 768)
                )
                np.testing.assert_allclose(
                    payload["s0_b0_effective_weight_float32"],
                    payload["s0_b0_weight_float32"] * 0.75,
                )

            trace = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "projection_contract_file": manifest_path.name,
                "projection_contract_file_sha256": file_sha256(manifest_path),
                "projection_contract_payload": payload_path.name,
                "projection_contract_payload_sha256": file_sha256(payload_path),
            }
            rows, payload, binding = load_checkpoint_projection_contract(
                root, trace
            )
            try:
                self.assertEqual(set(rows), set(POST_G0_BLOCK_PAIRS))
                self.assertEqual(
                    payload["s2_b5_weight_int8"].shape, (384, 384)
                )
                self.assertEqual(
                    binding["payload_sha256"], file_sha256(payload_path)
                )
                self.assertEqual(
                    binding["schema"], "local5_checkpoint_projection_contract_v2"
                )
            finally:
                payload.close()

            stale = json.loads(json.dumps(trace))
            stale["projection_contract_payload_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA"):
                load_checkpoint_projection_contract(root, stale)


if __name__ == "__main__":
    unittest.main()
