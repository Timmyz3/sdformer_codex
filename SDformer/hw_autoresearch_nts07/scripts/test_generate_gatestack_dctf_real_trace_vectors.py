import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_gatestack_dctf_real_trace_vectors import (
    TOKENS,
    flatten_term_stream,
    generate_record,
    projection_reference,
    record_vector_name,
    split_terms_by_destination_limit,
)


class DctfRealTraceVectorTests(unittest.TestCase):
    def test_all12_record_names_are_disambiguated_by_block(self):
        record = {"name": "S2.B5.attn"}
        self.assertEqual(record_vector_name(record, disambiguate_block=True), "s2_b5")
        self.assertEqual(record_vector_name(record, disambiguate_block=False), "s2")

    def test_destination_fanout_is_split_without_losing_token_449(self):
        terms = [{"gate": 3, "lane": 7, "tokens": list(range(450))}]
        split = split_terms_by_destination_limit(terms)
        self.assertEqual([len(term["tokens"]) for term in split], [255, 195])
        self.assertEqual(split[1]["tokens"][-1], 449)

    def test_flatten_stream_preserves_head_term_and_token_boundaries(self):
        rows = [
            [
                {"gate": 7, "lane": 2, "tokens": [1, 5]},
                {"gate": 9, "lane": 3, "tokens": [8]},
            ],
            [],
            [{"gate": 11, "lane": 4, "tokens": [0, 2, 4]}],
        ]
        stream = flatten_term_stream(rows)
        self.assertEqual(stream["head_offsets"], [0, 2, 2, 3])
        self.assertEqual(stream["token_offsets"], [0, 2, 3, 6])
        self.assertEqual(stream["gates"], [7, 9, 11])
        self.assertEqual(stream["lanes"], [2, 3, 4])
        self.assertEqual(stream["counts"], [2, 1, 3])
        self.assertEqual(stream["tokens"], [1, 5, 8, 0, 2, 4])

    def test_projection_reference_uses_output_major_int8_weights_and_acc32_bias(self):
        heads = 3
        dim = heads * 32
        k_rows = np.zeros((heads, TOKENS, 32), dtype=bool)
        gate_rows = np.zeros((heads, TOKENS), dtype=np.int64)
        k_rows[0, 0, 1] = True
        k_rows[2, 0, 3] = True
        gate_rows[0, 0] = 5
        gate_rows[2, 0] = 7
        weight = np.zeros((dim, dim), dtype=np.int64)
        weight[:, 1] = np.arange(dim) % 5 - 2
        weight[:, 67] = np.arange(dim) % 7 - 3
        bias = np.arange(dim, dtype=np.int64) - 48

        expected = projection_reference(k_rows, gate_rows, weight, bias)
        np.testing.assert_array_equal(
            expected[0], bias + 5 * weight[:, 1] + 7 * weight[:, 67]
        )
        np.testing.assert_array_equal(expected[1], bias)

    def test_generate_record_writes_auditable_s0_vectors(self):
        heads = 3
        dim = heads * 32
        shape = (2, 1, heads, TOKENS // 2, 32)
        k_bits = np.zeros(shape, dtype=np.uint8)
        k_bits[0, 0, 0, 0, 1] = 1
        k_bits[1, 0, 0, 0, 1] = 1
        gate = np.full((1, heads, TOKENS), 4, dtype=np.int16)
        weight = np.zeros((dim, dim), dtype=np.int8)
        weight[:, 1] = 2
        bias = np.arange(dim, dtype=np.int64)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "trace.npz"
            np.savez(
                source,
                k_shape=np.asarray(shape),
                k_bits_packed=np.packbits(k_bits.reshape(-1), bitorder="little"),
                gate_q17=gate,
                projection_weight_int8=weight,
                projection_weight_scale_exp2=np.zeros(dim, dtype=np.int16),
                projection_bias_acc_int64=bias,
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            record = {
                "name": "S0.B0.attn",
                "sample_id": 0,
                "file": str(source),
                "sha256": digest,
                "quantization_contract": "测试量化合同",
            }
            result = generate_record(record, root / "vectors")
            vector_dir = root / "vectors" / "s0"

            self.assertEqual(result["heads"], 3)
            self.assertEqual(result["terms_per_full_input_replay"], 1)
            self.assertEqual(result["events_per_full_input_replay"], 2)
            self.assertEqual(result["expected_physical_weight_requests"], 3)
            self.assertEqual(result["expected_final_checks"], TOKENS * dim)
            self.assertEqual(
                (vector_dir / "head_term_offsets.memh").read_text().splitlines(),
                ["00000000", "00000001", "00000001", "00000001"],
            )
            self.assertEqual(
                (vector_dir / "term_tokens.memh").read_text().splitlines(),
                ["00", "51"],
            )
            self.assertTrue((vector_dir / "manifest.json").is_file())

    def test_generate_record_supports_t450_and_nine_bit_token_ids(self):
        heads = 3
        dim = heads * 32
        tokens = 450
        shape = (2, 1, heads, tokens // 2, 32)
        k_bits = np.zeros(shape, dtype=np.uint8)
        k_bits[:, 0, 0, :, 0] = 1
        gate = np.full((1, heads, tokens), 4, dtype=np.int16)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "trace_t450.npz"
            np.savez(
                source,
                k_shape=np.asarray(shape),
                k_bits_packed=np.packbits(k_bits.reshape(-1), bitorder="little"),
                gate_q17=gate,
                projection_weight_int8=np.zeros((dim, dim), dtype=np.int8),
                projection_weight_scale_exp2=np.zeros(dim, dtype=np.int16),
                projection_bias_acc_int64=np.zeros(dim, dtype=np.int64),
            )
            record = {
                "name": "S0.B0.attn",
                "sample_id": 0,
                "file": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "quantization_contract": "T450 fixture",
            }
            result = generate_record(record, root / "vectors")
            vector_dir = root / "vectors" / "s0"
            self.assertEqual(result["tokens"], 450)
            self.assertEqual(result["token_id_width"], 9)
            self.assertEqual(result["terms_per_full_input_replay"], 2)
            self.assertEqual(result["events_per_full_input_replay"], 450)
            self.assertEqual(
                (vector_dir / "term_destination_counts.memh").read_text().splitlines(),
                ["ff", "c3"],
            )
            self.assertEqual(
                (vector_dir / "term_tokens.memh").read_text().splitlines()[-1],
                "1c1",
            )


if __name__ == "__main__":
    unittest.main()
