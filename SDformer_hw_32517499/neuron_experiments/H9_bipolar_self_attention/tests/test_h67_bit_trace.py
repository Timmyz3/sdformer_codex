from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

from h67_bit_trace import (  # noqa: E402
    AttentionBitTraceWriter,
    estimate_record_bytes,
    quantize_projection_weight_dyadic,
)


class FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=True)


class H67BitTraceTest(unittest.TestCase):
    def test_dyadic_projection_quantization_is_bounded(self) -> None:
        weight = torch.tensor(
            [[-0.25, -0.125, 0.0, 0.125], [0.0, 0.5, -0.5, 0.25]],
            dtype=torch.float32,
        )
        code, exponent = quantize_projection_weight_dyadic(weight)
        self.assertEqual(code.dtype, np.int8)
        self.assertEqual(exponent.dtype, np.int16)
        self.assertLessEqual(int(code.max()), 127)
        self.assertGreaterEqual(int(code.min()), -127)
        restored = code.astype(np.float32) * np.exp2(exponent.astype(np.float32))[:, None]
        error = np.abs(restored - weight.numpy())
        step = np.exp2(exponent.astype(np.float32))[:, None]
        self.assertTrue(np.all(error <= step / 2 + 1e-7))

    def test_real_tensor_trace_round_trip_and_manifest(self) -> None:
        module = FakeAttention()
        q = torch.zeros(2, 2, 2, 3, 8)
        k = torch.zeros(2, 2, 6, 8)
        gate = torch.zeros(2, 2, 6, 1)
        q[0, 0, 0, 1, 2] = 1
        k[0, 1, 4, 7] = 1
        gate[0, 0, 3, 0] = 0.5
        with tempfile.TemporaryDirectory() as tmp:
            writer = AttentionBitTraceWriter(
                Path(tmp), sample_limit=1, windows_per_call=1
            )
            writer.bind_run_context(
                {"artifact_identity": {"checkpoint_sha256": "fixture"}}
            )
            writer.capture(
                name="S0.B0.attn",
                sample_id=0,
                sample_key="fixture",
                module=module,
                q_orig=q,
                k_orig=k,
                gate=gate,
            )
            manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["run_context"]["artifact_identity"]["checkpoint_sha256"],
                "fixture",
            )
            self.assertEqual(manifest["coverage"]["stages"], [0])
            self.assertFalse(manifest["coverage"]["four_stage_complete"])
            payload = np.load(manifest["records"][0]["file"])
            q_shape = tuple(int(value) for value in payload["q_shape"])
            k_shape = tuple(int(value) for value in payload["k_shape"])
            q_bits = np.unpackbits(
                payload["q_bits_packed"], bitorder="little"
            )[: np.prod(q_shape)].reshape(q_shape)
            k_bits = np.unpackbits(
                payload["k_bits_packed"], bitorder="little"
            )[: np.prod(k_shape)].reshape(k_shape)
            self.assertEqual(int(q_bits.sum()), 1)
            self.assertEqual(int(k_bits.sum()), 1)
            self.assertEqual(int(payload["gate_q17"][0, 0, 3]), 64)

    def test_first_block_and_sample_filters(self) -> None:
        module = FakeAttention()
        q = torch.zeros(2, 1, 2, 3, 8)
        k = torch.zeros(1, 2, 6, 8)
        gate = torch.zeros(1, 2, 6, 1)
        with tempfile.TemporaryDirectory() as tmp:
            writer = AttentionBitTraceWriter(Path(tmp), sample_limit=1)
            for name, sample_id in (("S0.B1.attn", 0), ("S0.B0.attn", 1)):
                writer.capture(
                    name=name,
                    sample_id=sample_id,
                    sample_key="filtered",
                    module=module,
                    q_orig=q,
                    k_orig=k,
                    gate=gate,
                )
            self.assertEqual(writer.records, [])

    def test_size_estimate_is_positive(self) -> None:
        self.assertGreater(
            estimate_record_bytes(
                heads=24, spatial_tokens=81, lanes=32, dim=768
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
