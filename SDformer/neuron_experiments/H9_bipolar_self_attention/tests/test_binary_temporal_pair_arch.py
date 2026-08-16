from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "entrypoints"))


class BinaryTemporalPairArchitectureTest(unittest.TestCase):
    def test_tensor_record_reports_representation_suitability(self):
        try:
            from profile_nts11_hardware_p0 import tensor_record
        except ModuleNotFoundError as exc:
            self.skipTest(f"profiling environment dependency unavailable: {exc}")

        tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0, 1.5])
        record = tensor_record(tensor, include_value_stats=True)

        self.assertEqual(record["finite_count"], 5)
        self.assertEqual(record["value_min"], -1.0)
        self.assertEqual(record["value_max"], 2.0)
        self.assertAlmostEqual(record["near_integer_ratio"], 0.8)
        self.assertAlmostEqual(record["binary01_ratio"], 0.4)
        self.assertAlmostEqual(record["ternary_ratio"], 0.6)

    def test_atlif_hook_samples_temporal_quantization_sensitivity(self):
        try:
            from profile_nts11_hardware_p0 import HardwareProfiler
            from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN
        except ModuleNotFoundError as exc:
            self.skipTest(f"profiling environment dependency unavailable: {exc}")

        module = ATLIFTernaryPSN(T=2, thresh=1.0, output_mode="binary", threshold_mode="official_atlif")
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[0.5, 0.25], [0.0, 0.75]]))
            module.bias.copy_(torch.tensor([[0.0], [0.0]]))
        profiler = HardwareProfiler(torch.nn.Identity())
        handle = module.register_forward_hook(profiler._atlif_hook("unit"))
        module(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        handle.remove()

        record = profiler.atlif_records["unit"]
        self.assertEqual(record["temporal_steps"], 2)
        self.assertEqual(record["parameter_entries"], 7)
        self.assertEqual(record["recomputed_reference_mismatch"], 0)
        self.assertEqual(record["parameter_q8_event_mismatch"], 0)
        self.assertIn("input_first_binary01_ratio", record)

    def test_sample_workload_record_joins_flow_and_pair_features(self):
        try:
            from profile_nts11_hardware_p0 import HardwareProfiler
        except ModuleNotFoundError as exc:
            self.skipTest(f"profiling environment dependency unavailable: {exc}")

        profiler = HardwareProfiler(torch.nn.Identity())
        profiler.begin_sample(7)
        profiler.h60_records.append({
            "sample_id": 7,
            "stage": 0,
            "pair_total": 2,
            "pair_empty": 1,
            "token_total": 4,
            "token_kzero": 2,
            "four_vector_event_histogram": [1, 0, 1],
            "four_vector_union_histogram": [1, 1],
        })
        chunk = torch.zeros(1, 2, 2, 2, 2)
        chunk[0, 0, 0, 0, 0] = 1
        label = torch.ones(1, 2, 2, 2)
        mask = torch.ones(1, 1, 2, 2)
        prediction = torch.zeros_like(label)
        profiler.record_sample(
            chunk=chunk,
            label=label,
            mask=mask,
            prediction=prediction,
            flow_scaling=1.0,
        )

        record = profiler.sample_records[0]
        self.assertEqual(record["sample_id"], 7)
        self.assertEqual(record["input_events"], 1)
        self.assertAlmostEqual(record["pair_empty_ratio"], 0.5)
        self.assertAlmostEqual(record["mean_events_per_pair"], 1.0)
        self.assertAlmostEqual(record["mean_union_lanes_per_pair"], 0.5)
        self.assertAlmostEqual(record["token_kzero_ratio"], 0.5)
        self.assertAlmostEqual(record["label_flow_mag_mean"], 2**0.5, places=6)
        self.assertAlmostEqual(record["sample_aee"], 2**0.5, places=6)

    def test_stage_hook_records_only_same_sequence_cross_sample_delta(self):
        try:
            from profile_nts11_hardware_p0 import HardwareProfiler
        except ModuleNotFoundError as exc:
            self.skipTest(f"profiling environment dependency unavailable: {exc}")

        profiler = HardwareProfiler(torch.nn.Identity())
        hook = profiler._stage_hook(0)
        first = torch.tensor([0.0, 1.0, -1.0, 2.0])
        second = torch.tensor([0.0, 0.0, -1.0, 3.0])

        profiler.begin_sample(0, sample_key="seq_a_0001", sequence_key="seq_a")
        hook(torch.nn.Identity(), (), (first, first))
        profiler.begin_sample(1, sample_key="seq_a_0002", sequence_key="seq_a")
        hook(torch.nn.Identity(), (), (second, second))

        second_skip = profiler.activation_records[2]
        self.assertTrue(second_skip["cross_sample_comparable"])
        self.assertAlmostEqual(second_skip["cross_sample_exact_equal_ratio"], 0.5)
        self.assertAlmostEqual(second_skip["cross_sample_active_xor_ratio"], 0.25)
        self.assertAlmostEqual(second_skip["cross_sample_sign_class_change_ratio"], 0.25)

        profiler.begin_sample(2, sample_key="seq_b_0001", sequence_key="seq_b")
        hook(torch.nn.Identity(), (), (second, second))
        third_skip = profiler.activation_records[4]
        self.assertFalse(third_skip["cross_sample_comparable"])

    def test_operator_hook_counts_linear_runtime_macs_and_activity(self):
        try:
            from profile_nts11_hardware_p0 import HardwareProfiler
        except ModuleNotFoundError as exc:
            self.skipTest(f"profiling environment dependency unavailable: {exc}")

        profiler = HardwareProfiler(torch.nn.Identity())
        module = torch.nn.Linear(4, 3, bias=False)
        hook = profiler._operator_hook("sttmultires_unet.encoders.unit")
        inputs = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        outputs = module(inputs)
        hook(module, (inputs,), outputs)
        row = profiler.operator_records["sttmultires_unet.encoders.unit"]

        self.assertEqual(row["scope"], "encoder")
        self.assertEqual(row["dense_macs"], 24)
        self.assertEqual(row["input_elements"], 8)
        self.assertEqual(row["input_active"], 3)
        self.assertAlmostEqual(row["activity_weighted_macs_proxy"], 9.0)
        self.assertAlmostEqual(row["input_sample_binary01_ratio"], 1.0)

    def test_analyzer_reconstructs_exact_pair_payload(self):
        from analyze_binary_temporal_pair_arch import analyze
        from models.STSwinNet_SNN.bsa_attention import (
            _binary_temporal_pair_stats,
            _delta_locality_stats,
            _token_time_bundle_stats,
        )

        q = torch.zeros(2, 1, 1, 2, 4, dtype=torch.bool)
        k = torch.zeros_like(q)
        q[:, 0, 0, 0, :2] = True
        k[0, 0, 0, 0, 1:3] = True
        k[1, 0, 0, 0, 2] = True
        record = {
            "name": "S0.B0.attn",
            "stage": 0,
            "block": 0,
            "head_dim": 4,
            **_binary_temporal_pair_stats(q, k, include_ordered_trace=True),
            **_token_time_bundle_stats(q, k, include_ordered_trace=True),
            **_delta_locality_stats(q[0] ^ q[1], k[0] ^ k[1], include_ordered_trace=True),
        }
        profile = {
            "experiment": "synthetic",
            "checkpoint": "none",
            "samples": 1,
            "ordered_trace": True,
            "summary": {"h60_records": [record], "binary_temporal_pairs": {}},
        }
        result = analyze(profile)["model_summary"]

        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["pair_empty"], 1)
        self.assertEqual(result["events"], 7)
        self.assertEqual(result["union_lanes"], 3)
        self.assertEqual(result["traffic_bits"]["dense_bitmap"], 32)
        self.assertEqual(result["traffic_bits"]["separate_index"], 38)
        self.assertEqual(result["traffic_bits"]["fused_union"], 24)
        self.assertEqual(result["adaptive_traffic_bits"], 19)
        self.assertEqual(result["cycles_l4"]["dense_bitmap"], 2)
        self.assertEqual(result["cycles_l4"]["oracle_adaptive"], 2)


if __name__ == "__main__":
    unittest.main()
