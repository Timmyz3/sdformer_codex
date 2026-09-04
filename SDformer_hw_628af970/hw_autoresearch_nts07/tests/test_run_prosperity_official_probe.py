import unittest

import torch

from scripts.run_prosperity_official_probe import (
    load_official_api,
    run_official_fc,
    validate_sdformer_matrix_contract,
)


class ProsperityOfficialProbeTest(unittest.TestCase):
    def test_sdformer_matrix_contract_accepts_binary_tensor(self):
        activation = torch.tensor(
            [[[1, 0, 1, 0], [1, 0, 0, 0]]],
            dtype=torch.uint8,
        )
        metadata = {
            "schema": "sdformer_binary_matrix_v1",
            "line": "Motion",
            "sample_id": 0,
            "block": 0,
            "head": 0,
            "time_steps": 1,
            "sequence_length": 2,
            "input_dim": 4,
            "output_dim": 8,
            "semantic": "projection_activation",
        }
        validate_sdformer_matrix_contract(activation, metadata)

    def test_sdformer_matrix_contract_rejects_nonbinary_tensor(self):
        activation = torch.tensor([[[0, 2]]], dtype=torch.int8)
        metadata = {
            "schema": "sdformer_binary_matrix_v1",
            "line": "Local5",
            "sample_id": 0,
            "block": 0,
            "head": 0,
            "time_steps": 1,
            "sequence_length": 1,
            "input_dim": 2,
            "output_dim": 8,
            "semantic": "attention_k",
        }
        with self.assertRaisesRegex(ValueError, "只能包含 0/1"):
            validate_sdformer_matrix_contract(activation, metadata)

    def test_official_cpu_run_fc_path(self):
        _, FC, _, _ = load_official_api()
        operator = FC(
            "unit_fc",
            input_dim=8,
            output_dim=16,
            sequence_length=4,
            batch_size=1,
            time_steps=1,
        )
        operator.activation_tensor.sparse_map = torch.tensor(
            [
                [
                    [1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                ]
            ],
            dtype=torch.uint8,
        )
        product = run_official_fc(operator, True)
        bit = run_official_fc(operator, False)
        self.assertGreater(product.total_cycles, 0)
        self.assertGreater(bit.total_cycles, 0)
        self.assertEqual(product.layer, "unit_fc")
        self.assertLessEqual(product.g_wgt_reads, bit.g_wgt_reads)


if __name__ == "__main__":
    unittest.main()
