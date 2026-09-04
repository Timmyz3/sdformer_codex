from __future__ import annotations

import unittest

import torch

from analyze_gcmp_projection_quant_contract import (
    fold_linear_batch_norm,
    quantization_audit,
    quantized_projection,
    symmetric_int8_quantize,
)


class GcmpProjectionQuantContractTest(unittest.TestCase):
    def test_batch_norm_fold_matches_eval_formula(self):
        weight = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        bias = torch.tensor([0.25, -0.5])
        gamma = torch.tensor([2.0, 0.5])
        beta = torch.tensor([-1.0, 1.0])
        mean = torch.tensor([0.1, -0.2])
        var = torch.tensor([0.25, 4.0])
        eps = 1.0e-5
        folded_weight, folded_bias = fold_linear_batch_norm(
            weight, bias, gamma, beta, mean, var, eps
        )
        x = torch.tensor([[1.0, 2.0], [-1.0, 0.25]])
        linear = x @ weight.T + bias
        expected = gamma * (linear - mean) / torch.sqrt(var + eps) + beta
        actual = x @ folded_weight.T + folded_bias
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_per_channel_int8_uses_each_output_range(self):
        weight = torch.tensor([[1.0, -1.0], [0.01, -0.02]])
        code, scale = symmetric_int8_quantize(weight, per_output_channel=True)
        self.assertEqual(tuple(code.shape), (2, 2))
        self.assertEqual(tuple(scale.shape), (2,))
        self.assertEqual(int(code[0, 0]), 127)
        self.assertEqual(int(code[1, 1]), -127)

    def test_integer_projection_is_exact_for_representable_weights(self):
        weight_code = torch.tensor([[64, -32], [16, 8]], dtype=torch.int8)
        scale = torch.tensor(1.0 / 64.0)
        bias = torch.tensor([0.5, -0.25])
        gate_code = torch.tensor([[128, 0], [256, 128]], dtype=torch.int64)
        actual, bounds = quantized_projection(gate_code, weight_code, scale, bias)
        weight = weight_code.float() * scale
        expected = gate_code.float().div(128.0) @ weight.T + bias
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(bounds["bias_int32_clip_count"], 0)
        self.assertGreater(bounds["accumulator_int32_margin"], 1.0)

    def test_theoretical_bound_dominates_synthetic_accumulator(self):
        weight = torch.tensor([[1.0, -0.5], [0.25, 0.125]])
        bias = torch.tensor([0.5, -0.25])
        result = quantization_audit(weight, bias, seed=7, samples=8)
        for row in result.values():
            self.assertGreaterEqual(
                row["theoretical_accumulator_abs_bound"],
                row["synthetic_accumulator_absmax"],
            )
            self.assertGreater(row["theoretical_accumulator_int32_margin"], 1.0)


if __name__ == "__main__":
    unittest.main()
