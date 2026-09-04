#!/usr/bin/env python3
"""unit tests for projection_scs_cycle_ledger (CPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from projection_scs_cycle_ledger import (
    approx_projection_cycles,
    build_ledger,
    projection_phase_overhead,
    scs_row_cycles,
    scs_ledger,
    projection_work_items,
)


class TestScs(unittest.TestCase):
    def test_row_cycles_fixed_vs_sparse(self) -> None:
        fixed = scs_row_cycles(18.0, 35.0)
        sparse = scs_row_cycles(18.0, 2.0 * 2.3)
        self.assertGreater(fixed, sparse)
        # 162 + max(active,1) + class + active + 3
        self.assertEqual(fixed, 162 + 18 + 35 + 18 + 3)

    def test_scs_ledger_reduces(self) -> None:
        model = {
            "stages": [
                {
                    "stage": 0,
                    "rows_per_frame": 100,
                    "zaf_active_entries_mean": 20.0,
                    "zaf_fold_classes_mean": 2.5,
                    "zaf_kzero_token_ratio": 0.8,
                    "ttb2_empty_ratio": 0.3,
                }
            ],
            "weighted": {"kzero": 0.8},
        }
        out = scs_ledger(model)
        self.assertEqual(out["rows_per_frame"], 100)
        self.assertGreater(out["cycle_reduction"], 0.0)
        self.assertLess(out["scs_cycles_per_frame"], out["fixed_cycles_per_frame"])


class TestProjection(unittest.TestCase):
    def test_work_items(self) -> None:
        bt = {
            "projection_baseline_active_lanes": 1000,
            "projection_gate_class_channel_terms_deploy": 200,
            "projection_class_channel_terms_h67": 400,
            "projection_gate_group_terms_g1": 200,
            "projection_gate_multicast_delivery_m1": 1000,
            "projection_gate_multicast_delivery_m4": 400,
            "row_active_projection_gate_classes_mean_deploy": 0.9,
            "row_active_projection_classes_mean_h67": 2.0,
            "pair_empty_ratio": 0.74,
            "token_kzero_ratio": 0.88,
        }
        w = projection_work_items(bt)
        self.assertAlmostEqual(w["product_reduction_vs_direct"], 0.8)
        self.assertAlmostEqual(w["gate_vs_score_extra_merge"], 0.5)

    def test_approx_bottleneck_delivery(self) -> None:
        a = approx_projection_cycles(
            baseline_lanes=1000,
            gate_terms=200,
            delivery=500,
            output_channels=192,
            output_lanes=32,
            product_engines=1,
        )
        self.assertEqual(a["chunks"], 6)
        self.assertEqual(a["bottleneck"], "delivery")

    def test_build_ledger_from_fixture(self) -> None:
        compact = {
            "schema_version": 1,
            "models": {
                "H67": {
                    "samples": 1,
                    "ordered_trace": True,
                    "binary_temporal_pairs": {
                        "projection_baseline_active_lanes": 1000,
                        "projection_gate_class_channel_terms_deploy": 200,
                        "projection_class_channel_terms_h67": 400,
                        "projection_gate_group_terms_g1": 200,
                        "projection_gate_multicast_delivery_m1": 1000,
                        "projection_gate_multicast_delivery_m4": 400,
                        "projection_gate_multicast_delivery_m8": 250,
                        "row_active_projection_gate_classes_mean_deploy": 0.9,
                        "row_active_projection_classes_mean_h67": 2.0,
                        "pair_empty_ratio": 0.74,
                        "token_kzero_ratio": 0.88,
                    },
                    "stages": [
                        {
                            "stage": 0,
                            "rows_per_frame": 6720,
                            "zaf_active_entries_mean": 18.0,
                            "zaf_fold_classes_mean": 2.3,
                            "zaf_kzero_token_ratio": 0.88,
                            "ttb2_empty_ratio": 0.5,
                        }
                    ],
                    "weighted": {"kzero": 0.88, "active_entries": 18.0, "fold_classes": 2.3},
                }
            },
        }
        dse = {
            "schema_version": 1,
            "configurations": [
                {
                    "class_slots": 4,
                    "multicast_width": 4,
                    "output_lanes": 32,
                    "product_engines": 1,
                    "rows": 10,
                    "overflow_rows": 0,
                    "overflow_ratio": 0.0,
                    "direct_cycles": 1000,
                    "candidate_cycles": 400,
                    "product_cycles": 300,
                    "delivery_cycles": 400,
                    "ideal_speedup": 2.5,
                    "candidate_p50": 1.0,
                    "candidate_p95": 2.0,
                    "candidate_p99": 3.0,
                    "candidate_max": 4.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpath = root / "compact.json"
            dpath = root / "dse.json"
            cpath.write_text(json.dumps(compact), encoding="utf-8")
            dpath.write_text(json.dumps(dse), encoding="utf-8")
            result = build_ledger(
                compact_path=cpath,
                dse_path=dpath,
                variant="H67",
                output_channels=192,
            )
            self.assertFalse(result["gpu_required"])
            self.assertAlmostEqual(
                result["projection_work_items"]["product_reduction_vs_direct"], 0.8
            )
            self.assertEqual(len(result["projection_dse_lock"]), 1)
            self.assertEqual(
                result["recommendation"]["preferred_gcmp_config"]["multicast_width"], 4
            )
            row = result["projection_dse_lock"][0]
            self.assertIn("phase_overhead", row)
            self.assertGreater(row["total_serial_with_overhead"], row["candidate_cycles"])
            # effective speedup should be closer to 1 than pure DSE ideal
            self.assertLess(
                row["effective_speedup_vs_direct_serial"], row["ideal_speedup"] + 1e-9
            )


class TestPhaseOverhead(unittest.TestCase):
    def test_bias_and_build_additive(self) -> None:
        ph = projection_phase_overhead(
            rows=10,
            tokens_per_row=162,
            nmf_slots=4,
            nmf_k_lanes=32,
            acc_banks=2,
            mean_terms_per_row=5.0,
            dse_candidate_total=1000,
            dse_product_total=800,
            dse_delivery_total=1000,
            dse_direct_total=2000,
        )
        self.assertEqual(ph["phases"]["nmf_build_token_stream"], 10 * 162)
        # ceil(162/2)*2 = 162
        self.assertEqual(ph["phases"]["bias_commit"], 10 * 162)
        self.assertEqual(ph["phases"]["finish_done"], 20)
        self.assertEqual(
            ph["total_serial_with_overhead"],
            1000 + 1620 + 1620 + 20,
        )
        self.assertLess(ph["effective_speedup_vs_direct_serial"], ph["dse_only_speedup"])


if __name__ == "__main__":
    unittest.main()
