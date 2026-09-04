from __future__ import annotations

import unittest

from model_hit_flow_full_encoder_budget import build_model


class HitFlowBudgetTest(unittest.TestCase):
    def setUp(self):
        self.storage = {
            "models": {"H67": {
                "activation_evidence": {"long_skip_elements_s0_s2": 1000},
                "atlif_execution_graph": {
                    "live_temporal_macs_per_frame": 6400,
                    "live_output_elements_per_frame": 3200,
                },
            }}
        }
        port = {}
        for contexts in (2, 4):
            port[f"fetch128_split_1w_no_merge_contexts{contexts}"] = {"mean": 100}
            port[f"fetch128_split_1w_perfect_pccc_contexts{contexts}"] = {"mean": 80}
        self.profile = {"results": [{
            "model": "H67",
            "whole": {"port_aware_pipeline_dse": port},
        }]}
        self.sops = {"estimated_total_sops": 1024, "dense_ops": 4096}

    def test_event_bypass_reduces_only_event_bank_cycles(self):
        result = build_model(self.storage, self.profile, self.sops)
        rows = [row for row in result["configurations"] if (
            row["temporal_arrays"] == 2
            and row["spatial_lanes"] == 256
            and row["contexts"] == 2
            and row["pccc_mode"] == "关闭"
            and row["memory_bus_bits"] == 256
        )]
        no_bypass = next(row for row in rows if row["event_bypass_ratio"] == 0.0)
        full_bypass = next(row for row in rows if row["event_bypass_ratio"] == 1.0)
        self.assertEqual(no_bypass["event_bank_rw_cycles"], 25)
        self.assertEqual(full_bypass["event_bank_rw_cycles"], 0)
        self.assertEqual(no_bypass["serial_cycles"] - full_bypass["serial_cycles"], 25)

    def test_pccc_is_only_an_attention_upper_bound(self):
        result = build_model(self.storage, self.profile, self.sops)
        rows = [row for row in result["configurations"] if (
            row["temporal_arrays"] == 2
            and row["spatial_lanes"] == 256
            and row["contexts"] == 2
            and row["memory_bus_bits"] == 256
            and row["event_bypass_ratio"] == 0.0
        )]
        disabled = next(row for row in rows if row["pccc_mode"] == "关闭")
        perfect = next(row for row in rows if row["pccc_mode"] == "理想全合并上界")
        self.assertEqual(disabled["attention_cycles"] - perfect["attention_cycles"], 20)
        self.assertEqual(disabled["serial_cycles"] - perfect["serial_cycles"], 20)

    def test_runtime_encoder_operator_profile_replaces_global_proxy(self):
        runtime = {
            "samples": 10,
            "summary": {"operator_by_scope": [{
                "scope": "encoder",
                "dense_macs": 5000,
                "activity_weighted_macs_proxy": 1250,
            }]},
        }
        result = build_model(self.storage, self.profile, self.sops, runtime)
        self.assertEqual(result["inputs"]["legacy_dense_ops_proxy"], 500)
        self.assertEqual(result["inputs"]["legacy_whole_network_event_ops_proxy"], 125)
        self.assertIn("逐算子encoder", result["inputs"]["spatial_proxy_source"])


if __name__ == "__main__":
    unittest.main()
