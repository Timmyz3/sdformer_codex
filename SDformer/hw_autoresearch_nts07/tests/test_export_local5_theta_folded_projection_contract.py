from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from export_local5_theta_folded_projection_contract import (  # noqa: E402
    POST_G0_BLOCK_PAIRS,
    build_theta_folded_contract,
    file_sha256,
)


def fixture_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for stage, block in POST_G0_BLOCK_PAIRS:
        base = (
            "sttmultires_unet.encoders.swin3d.layers."
            f"{stage}.swin_blocks.{block}"
        )
        values = torch.linspace(-0.25, 0.25, 32 * 32).reshape(32, 32)
        state[f"{base}.attn.proj.weight"] = values + (stage + block) * 1e-5
        state[f"{base}.attn.proj.bias"] = torch.arange(32).float() / 128
        state[f"{base}.attn.sn_k.spiking_neuron.thresh"] = torch.tensor(
            1.0 - (stage * 3 + block + 1) * 1e-6
        )
    return state


class ThetaFoldedProjectionContractTest(unittest.TestCase):
    def test_exports_twelve_block_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"theta-folded-fixture")
            manifest_path, payload_path, manifest = build_theta_folded_contract(
                fixture_state(),
                output_dir=root,
                checkpoint=checkpoint,
            )

            self.assertEqual(
                manifest["schema"], "local5_checkpoint_projection_contract_v2"
            )
            self.assertEqual(len(manifest["blocks"]), 12)
            self.assertEqual(manifest["checkpoint_sha256"], file_sha256(checkpoint))
            self.assertEqual(manifest["payload_sha256"], file_sha256(payload_path))
            self.assertIn("theta_K*W_float", manifest["quantization_order"])
            self.assertTrue(manifest_path.is_file())
            with np.load(payload_path) as payload:
                theta = float(payload["s0_b0_theta_float32"][0])
                raw = payload["s0_b0_weight_float32"]
                effective = payload["s0_b0_effective_weight_float32"]
                np.testing.assert_allclose(effective, raw * theta, rtol=0, atol=0)
                self.assertEqual(payload["s3_b1_weight_int8"].shape, (32, 32))
                self.assertEqual(payload["s2_b5_weight_scale_exp2"].shape, (32,))

            decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(decoded["payload_sha256"], file_sha256(payload_path))

    def test_missing_theta_fails_closed(self) -> None:
        state = fixture_state()
        del state[
            "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0."
            "attn.sn_k.spiking_neuron.thresh"
        ]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pth"
            checkpoint.write_bytes(b"missing-theta")
            with self.assertRaisesRegex(ValueError, "缺少 K-ATLIF theta"):
                build_theta_folded_contract(
                    state,
                    output_dir=Path(temporary) / "out",
                    checkpoint=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
