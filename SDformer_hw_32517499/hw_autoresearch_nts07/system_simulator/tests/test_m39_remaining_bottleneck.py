import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "hw_autoresearch_nts07/system_simulator/scripts/"
    "analyze_m39_remaining_bottleneck.py"
)
RESULT = (
    ROOT
    / "hw_autoresearch_nts07/results/m39_remaining_bottleneck_r1_20260822/"
    "m39_remaining_bottleneck.json"
)
SPEC = importlib.util.spec_from_file_location("m39", str(SCRIPT))
M39 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M39)


class M39RemainingBottleneckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = M39.build()

    def test_shared_non_atlif_and_162m_reconcile_exactly(self):
        ledger = self.result["remaining_cycle_ledger"]
        lines = {
            row["line"]: row
            for row in ledger["shared_non_atlif_by_line"]
        }
        self.assertEqual(lines["Local"]["shared_non_atlif_cycles"], 225815624)
        self.assertEqual(lines["Motion"]["shared_non_atlif_cycles"], 224145350)
        for row in lines.values():
            self.assertEqual(
                sum(row["parts"].values()), row["shared_non_atlif_cycles"]
            )
        split = ledger["noneligible_plus_qk_decomposition"]
        self.assertEqual(split["noneligible_operator_cycles"], 132987740)
        self.assertEqual(split["qk_cycles"], 29072080)
        self.assertEqual(split["total_cycles"], 162059820)
        self.assertEqual(
            sum(
                row["remaining_noneligible_cycles"]
                for row in split["noneligible_categories"]
            ),
            132987740,
        )

    def test_only_bottleneck_single_bucket_can_save_50m(self):
        rows = self.result["remaining_cycle_ledger"][
            "independent_cycle_reduction_ceilings"
        ]
        passing = [row["scope"] for row in rows if row["can_save_50m_alone"]]
        self.assertEqual(passing, ["four_bottleneck_conv3x3"])
        qk = next(row for row in rows if row["scope"] == "qk_plus_rqtb_attention")
        self.assertEqual(qk["cycles"], 32162811)

    def test_local5_attention_is_missing_not_zero(self):
        attention = self.result["attention_and_trace_completeness"]
        self.assertIn("MISSING UNKNOWN NONZERO", attention["local5_ep44"])
        self.assertIn("not Local5 ep44", attention["local_motion_name_boundary"])
        admission = self.result["admission"]
        self.assertFalse(admission["local5_full_system_admitted"])
        self.assertFalse(admission["system_speedup_admitted"])

    def test_best_m30_anchor_and_m38_ideal_use_dual256(self):
        dse = self.result["conditional_dse"]
        anchor = dse["selected_m30_anchor"]
        self.assertEqual(anchor["name"], "dual256b_independent_output_packed24")
        self.assertEqual(anchor["local_cycles"], 305047198)
        self.assertEqual(anchor["motion_cycles"], 303376924)
        ideal = dse["m38_conditional_ideal"]
        self.assertEqual(ideal["local_cycles"], 268455448)
        self.assertEqual(ideal["motion_cycles"], 266785174)
        self.assertFalse(self.result["admission"]["executable_integrated_cycles_admitted"])

    def test_four_bottleneck_rows_are_conserved_and_serial(self):
        rows = {
            (row["line"], row["late_scale_implementation"]): row
            for row in self.result["conditional_dse"]["four_bottleneck_rows"]
        }
        expected = {
            ("Local", "M33_shared96"): (13282495, 2304000, 1484515, 17071010, 205895501),
            ("Motion", "M33_shared96"): (12836419, 2304000, 1524011, 16664430, 203818647),
            ("Local", "M35_zero_mul_sidecar"): (13282495, 1152000, 1484515, 15919010, 204743501),
            ("Motion", "M35_zero_mul_sidecar"): (12836419, 1152000, 1524011, 15512430, 202666647),
        }
        for key, values in expected.items():
            row = rows[key]
            event, late, control, replacement, after = values
            self.assertEqual(row["replacement"]["event_accumulation_cycles"], event)
            self.assertEqual(row["replacement"]["late_scale_cycles"], late)
            self.assertEqual(row["replacement"]["frontend_control_cycles"], control)
            self.assertEqual(row["replacement"]["overlap_credit_cycles"], 0)
            self.assertEqual(row["replacement"]["total_cycles"], replacement)
            self.assertEqual(row["conditional_cycles_after_substitution"], after)
            self.assertEqual(
                after + row["before_cycles"],
                row["m38_ideal_before_scope_substitution_cycles"] + replacement,
            )
            self.assertTrue(row["minimum_50m_saving_pass"])
            self.assertIn("M38 changes only", row["bucket_disjointness"])

    def test_ten_consumer_rows_are_alternatives_not_additive(self):
        dse = self.result["conditional_dse"]
        self.assertTrue(dse["scope_alternatives_not_additive"])
        rows = {
            (row["line"], row["late_scale_implementation"]): row
            for row in dse["ten_consumer_rows"]
        }
        self.assertEqual(rows[("Local", "M33_shared96")]["replacement"]["total_cycles"], 27250233)
        self.assertEqual(rows[("Motion", "M33_shared96")]["replacement"]["total_cycles"], 26709587)
        self.assertEqual(rows[("Local", "M35_zero_mul_sidecar")]["replacement"]["total_cycles"], 23443233)
        self.assertEqual(rows[("Motion", "M35_zero_mul_sidecar")]["replacement"]["total_cycles"], 22902587)
        self.assertAlmostEqual(
            rows[("Local", "M33_shared96")]["conditional_speedup_vs_fixed"],
            3.2708696265302937,
        )
        self.assertAlmostEqual(
            rows[("Motion", "M35_zero_mul_sidecar")]["conditional_speedup_vs_fixed"],
            3.3779636332543204,
        )

    def test_m33_m35_resource_tradeoff_and_formality_boundary(self):
        rows = {
            row["name"]: row
            for row in self.result["late_scale_architecture_alternatives"]
        }
        m33 = rows["M33_shared96_generic_UQ0p24"]
        m35 = rows["M35_parallel_complement_CSD_sidecar"]
        self.assertEqual(m33["outputs_per_cycle"], 4)
        self.assertEqual(m33["additional_int8_multipliers"], 0)
        self.assertIn("80/96", m33["pool_contention"])
        self.assertEqual(m35["outputs_per_cycle"], 8)
        self.assertEqual(m35["additional_int8_multipliers"], 0)
        self.assertEqual(m35["maximum_csd_terms"], 4)
        self.assertEqual(m35["latest_r7_formality"], "PENDING")
        self.assertAlmostEqual(
            m35["standalone_throughput_density_vs_flat_m33"],
            1.3239978888247064,
        )
        self.assertFalse(self.result["admission"]["m35_r7_formality_admitted"])

    def test_prosperity_phi_are_gated_not_claimed(self):
        adapters = self.result["prosperity_phi_adapter_assessment"]
        self.assertIn("product density", adapters["Prosperity"]["blocking_observability"])
        self.assertIn("<=29,630,957", adapters["Prosperity"]["go_gate"])
        self.assertIn("fine-tuning", adapters["Phi"]["risk"])
        self.assertIn("32,162,811", adapters["qk_attention_conclusion"])
        probe = self.result["resource_bandwidth_sram_contract"][
            "prosperity_probe_tile_m256_k16_n96"
        ]
        self.assertEqual(probe["with_fixed_resident_bytes"], 158912)
        self.assertTrue(probe["fits_240kib"])

    def test_cycle_no_go_thresholds_are_pre_registered(self):
        gates = self.result["go_no_go_matrix"]["cycle_gates"]
        joined = " ".join(gates)
        self.assertIn("67,383,950/69,054,224", joined)
        self.assertIn("44,388,830/46,059,104", joined)
        for row in self.result["conditional_dse"]["ten_consumer_rows"]:
            targets = {item["target_speedup"]: item for item in row["target_gates"]}
            self.assertTrue(targets[2.7]["crosses_target_in_conditional_dse"])
            self.assertTrue(targets[3.0]["crosses_target_in_conditional_dse"])

    def test_contract_hash_rule_and_attention_drift_fail_closed(self):
        contract = json.loads(M39.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bad_hash = copy.deepcopy(contract)
            bad_hash["inputs"]["m32_threshold_carry"]["sha256"] = "0" * 64
            bad_hash_path = directory / "bad_hash.json"
            bad_hash_path.write_text(json.dumps(bad_hash), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input hash drift"):
                M39.build(bad_hash_path)

            bad_rule = copy.deepcopy(contract)
            bad_rule["frozen_dse_rules"]["selected_m30_local_cycles"] += 1
            bad_rule_path = directory / "bad_rule.json"
            bad_rule_path.write_text(json.dumps(bad_rule), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen DSE rule drift"):
                M39.build(bad_rule_path)

            source = json.loads(
                M39.resolve(contract["inputs"]["m22_summary"]["path"])
                .read_text(encoding="utf-8")
            )
            source["identities"]["local_ep44"]["attention_coverage_status"] = "ZERO"
            bad_source = directory / "bad_m22.json"
            bad_source.write_text(json.dumps(source), encoding="utf-8")
            bad_attention = copy.deepcopy(contract)
            bad_attention["inputs"]["m22_summary"] = {
                "path": str(bad_source),
                "sha256": M39.sha256(bad_source),
            }
            bad_attention_path = directory / "bad_attention.json"
            bad_attention_path.write_text(json.dumps(bad_attention), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing-attention"):
                M39.build(bad_attention_path)

    def test_frozen_result_rebuilds_exactly_and_claims_stay_closed(self):
        frozen = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(frozen, self.result)
        admission = frozen["admission"]
        self.assertTrue(admission["remaining_cycle_decomposition_admitted"])
        self.assertTrue(admission["conditional_h67_compute_dse_admitted"])
        self.assertFalse(admission["integrated_rtl_admitted"])
        self.assertFalse(admission["address_timed_memory_admitted"])
        self.assertFalse(admission["accuracy_admitted"])
        self.assertFalse(admission["power_energy_admitted"])
        self.assertFalse(admission["headline_admitted"])

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m39.json"
            output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                M39.write_output(output, {"must_not": "overwrite"})
            self.assertEqual(output.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
