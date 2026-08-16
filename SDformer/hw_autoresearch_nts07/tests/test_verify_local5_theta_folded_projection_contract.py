from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_local5_theta_folded_projection_contract as verifier  # noqa: E402


class FakeCheckpoint:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self._state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state


class Local5ThetaFoldedVerifierTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, FakeCheckpoint]:
        checkpoint = root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint-fixture")
        arrays: dict[str, np.ndarray] = {}
        state: dict[str, torch.Tensor] = {}
        blocks = []
        index = 0
        for stage, depth in enumerate(verifier.EXPECTED_STAGE_DEPTHS):
            for block in range(depth):
                channels = verifier.EXPECTED_STAGE_CHANNELS[stage]
                module = (
                    "sttmultires_unet.encoders.swin3d.layers."
                    f"{stage}.swin_blocks.{block}.attn"
                )
                prefix = f"s{stage}_b{block}"
                weight_name = f"{module}.proj.weight"
                theta_name = f"{module}.sn_k.spiking_neuron.thresh"
                bias_name = f"{module}.proj.bias"
                weight = (
                    torch.arange(channels * channels, dtype=torch.float32)
                    .reshape(channels, channels)
                    / float(max(4096, channels * channels))
                    - 0.125
                )
                theta = torch.tensor(0.75 + index / 100.0, dtype=torch.float32)
                bias = torch.arange(channels, dtype=torch.float32) / float(channels)
                state[weight_name] = weight
                state[theta_name] = theta.reshape(1)
                state[bias_name] = bias
                effective = weight * float(theta.item())
                weight_int8, scale = verifier.quantize_projection_weight_dyadic(
                    effective
                )
                arrays[f"{prefix}_theta_float32"] = np.asarray(
                    [float(theta.item())], dtype=np.float32
                )
                arrays[f"{prefix}_weight_float32"] = weight.numpy()
                arrays[f"{prefix}_effective_weight_float32"] = effective.numpy()
                arrays[f"{prefix}_weight_int8"] = weight_int8
                arrays[f"{prefix}_weight_scale_exp2"] = scale
                arrays[f"{prefix}_bias_float32"] = bias.numpy()
                blocks.append(
                    {
                        "stage": stage,
                        "block": block,
                        "module": module,
                        "prefix": prefix,
                        "weight_name": weight_name,
                        "theta_name": theta_name,
                        "bias_name": bias_name,
                        "theta": float(theta.item()),
                        "weight_shape": [channels, channels],
                        "heads": verifier.EXPECTED_STAGE_HEADS[stage],
                        "head_dim": verifier.EXPECTED_HEAD_DIM,
                        "bias_present": True,
                    }
                )
                index += 1
        payload = root / "projection.npz"
        np.savez_compressed(payload, **arrays)
        manifest = root / "projection.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "local5_checkpoint_projection_contract_v2",
                    "status": "THETA_FOLDED_WEIGHT_CONTRACT",
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": verifier.file_sha256(checkpoint),
                    "payload_file": payload.name,
                    "payload_sha256": verifier.file_sha256(payload),
                    "topology_contract": verifier.TOPOLOGY_CONTRACT,
                    "blocks": blocks,
                }
            ),
            encoding="utf-8",
        )
        return manifest, payload, FakeCheckpoint(state)

    def test_independent_recompute_accepts_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, payload, checkpoint = self._fixture(Path(temporary))
            with patch.object(verifier, "load_checkpoint", return_value=checkpoint):
                report = verifier.verify_contract(manifest, payload)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["blocks"], 12)
            self.assertEqual(report["arrays"], 72)

    def test_independent_recompute_rejects_self_consistent_tampered_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, payload, checkpoint = self._fixture(Path(temporary))
            with np.load(payload) as source:
                arrays = {key: source[key].copy() for key in source.files}
            arrays["s0_b0_weight_int8"][0, 0] ^= np.int8(1)
            np.savez_compressed(payload, **arrays)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["payload_sha256"] = verifier.file_sha256(payload)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with (
                patch.object(verifier, "load_checkpoint", return_value=checkpoint),
                self.assertRaisesRegex(ValueError, "checkpoint独立重算"),
            ):
                verifier.verify_contract(manifest, payload)

    def test_independent_recompute_rejects_self_consistent_block_remap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, payload, checkpoint = self._fixture(Path(temporary))
            value = json.loads(manifest.read_text(encoding="utf-8"))
            first = value["blocks"][0]
            second = value["blocks"][1]
            for field in ("module", "weight_name", "theta_name", "bias_name", "theta"):
                first[field], second[field] = second[field], first[field]

            suffixes = (
                "theta_float32",
                "weight_float32",
                "effective_weight_float32",
                "weight_int8",
                "weight_scale_exp2",
                "bias_float32",
            )
            with np.load(payload) as source:
                arrays = {key: source[key].copy() for key in source.files}
            for suffix in suffixes:
                first_key = f"s0_b0_{suffix}"
                second_key = f"s0_b1_{suffix}"
                arrays[first_key], arrays[second_key] = (
                    arrays[second_key].copy(),
                    arrays[first_key].copy(),
                )
            np.savez_compressed(payload, **arrays)
            value["payload_sha256"] = verifier.file_sha256(payload)
            manifest.write_text(json.dumps(value), encoding="utf-8")

            with (
                patch.object(verifier, "load_checkpoint", return_value=checkpoint),
                self.assertRaisesRegex(ValueError, "拓扑映射错误"),
            ):
                verifier.verify_contract(manifest, payload)


if __name__ == "__main__":
    unittest.main()
