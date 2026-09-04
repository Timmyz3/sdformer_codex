from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ENTRYPOINTS = HERE.parents[1] / "neuron_experiments/H9_bipolar_self_attention/entrypoints"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENTRYPOINTS))

from audit_h67_bit_trace import audit  # noqa: E402
from h67_bit_trace import AttentionBitTraceWriter  # noqa: E402


class FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)


class AuditH67BitTraceTest(unittest.TestCase):
    def test_audit_recomputes_real_bit_work(self) -> None:
        q = torch.zeros(2, 1, 1, 3, 8)
        k = torch.zeros(1, 1, 6, 8)
        gate = torch.zeros(1, 1, 6, 1)
        k[0, 0, 0, 2] = 1
        k[0, 0, 1, 2] = 1
        k[0, 0, 4, 7] = 1
        gate[0, 0, 0, 0] = 0.5
        gate[0, 0, 1, 0] = 0.5
        gate[0, 0, 4, 0] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            writer = AttentionBitTraceWriter(Path(tmp), sample_limit=1)
            writer.capture(
                name="S0.B0.attn",
                sample_id=0,
                sample_key="fixture",
                module=FakeAttention(),
                q_orig=q,
                k_orig=k,
                gate=gate,
            )
            result = audit(writer.manifest_path)
            row = result["records"][0]
            self.assertEqual(row["direct_active_lane_work"], 3)
            self.assertEqual(row["gatestack_equivalent_terms"], 2)
            self.assertAlmostEqual(row["equivalent_term_reduction_ratio"], 1 / 3)
            self.assertTrue(row["sha256_ok"])

    def test_four_stage_requirement_rejects_partial_trace(self) -> None:
        q = torch.zeros(2, 1, 1, 1, 8)
        k = torch.zeros(1, 1, 2, 8)
        gate = torch.zeros(1, 1, 2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            writer = AttentionBitTraceWriter(Path(tmp), sample_limit=1)
            writer.capture(
                name="S0.B0.attn",
                sample_id=0,
                sample_key="fixture",
                module=FakeAttention(),
                q_orig=q,
                k_orig=k,
                gate=gate,
            )
            with self.assertRaises(ValueError):
                audit(writer.manifest_path, require_four_stages=True)


if __name__ == "__main__":
    unittest.main()
