#!/usr/bin/env python3
"""Regression and adversarial tests for M39-r3 conditional bottleneck DSE."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
SCRIPT = HW_ROOT / "system_simulator/scripts/analyze_m39_remaining_bottleneck_r3.py"
CONTRACT = HW_ROOT / "contracts/m39_remaining_bottleneck_input_contract_r3_20260822.json"
SPEC = importlib.util.spec_from_file_location("m39_r3", str(SCRIPT))
M39 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M39)


class M39R3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = M39.build(CONTRACT)
        cls.contract = M39.read_json(CONTRACT)

    def write_contract(self, root, payload, name="contract.json"):
        path = Path(root) / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_status_and_current_anchor_chain(self):
        self.assertEqual(
            self.result["status"],
            "PASS_M39_R3_CURRENT_ANCHORS_CONDITIONAL_BOTTLENECK_DSE_ONLY")
        audit = self.result["recursive_evidence_audit"]
        self.assertEqual(audit["m38_r5_model_only"]["admission_sha256"],
                         "2d231c4a88d616158bcac0e867ec166a109fe8df55f10fc81182fc8ec01f08fe")
        self.assertEqual(audit["m38_r5_model_only"]["review_sha256"],
                         "36bb10294a209bd32ad4131d8b0171749aa50535083166dc38b5de5b28d2d529")
        self.assertEqual(audit["m38_r5_model_only"]["validator_sha256"],
                         "ce34da7dd759c0b43efc147a9b8f22f700414e17f7a8a9f1a3336c4afb64b445")
        self.assertEqual(audit["m35_r3_standalone"]["receipt_sha256"],
                         "d088daa8e51a40eb26ee07624f2c6a3b06f95bd0d1395c4bb91bdd1532195b84")

    def test_scope_boundaries_are_separate(self):
        scopes = self.result["scope_boundaries"]
        self.assertEqual(set(scopes), {
            "Local", "Motion", "four_bottleneck_conv3x3", "Local5_ep44"})
        self.assertIn("not_Local5", scopes["Local"]["definition"])
        self.assertIn("not_a_separate_trained_model", scopes["Motion"]["definition"])
        self.assertFalse(scopes["Local"]["system_speedup_admitted"])
        self.assertFalse(scopes["Motion"]["system_speedup_admitted"])
        self.assertFalse(scopes["Local5_ep44"]["full_system_admitted"])

    def test_four_bottleneck_exact_aggregate(self):
        row = self.result["four_bottleneck_event_late_scale_model"]["aggregate"]
        self.assertEqual(row["operators"], 4)
        self.assertEqual(row["dense_product_terms"], 63700992000)
        self.assertEqual(row["active_product_terms"], 7644571775)
        self.assertEqual(row["baseline_activity_cycles_96"], 79630957)
        self.assertEqual(row["outputs"], 9216000)
        self.assertEqual(row["observed_product_density_exact"],
                         {"numerator": 305782871, "denominator": 2548039680})

    def test_per_operator_work_and_m35_thresholds(self):
        rows = self.result["four_bottleneck_event_late_scale_model"][
            "operator_census"]
        self.assertEqual([row["active_product_terms"] for row in rows],
                         [2630357176, 947018995, 2898921692, 1168273912])
        self.assertEqual([row["baseline_activity_cycles_96"] for row in rows],
                         [27399554, 9864782, 30197101, 12169520])
        self.assertEqual([row["m35_csd_nonzero_terms"] for row in rows],
                         [3, 2, 3, 4])
        self.assertEqual(sum(row["nominal_conditional_m4_projection_cycles_by_line"][
            "Local"] for row in rows), 13282496)
        self.assertEqual(sum(row["nominal_conditional_m4_projection_cycles_by_line"][
            "Motion"] for row in rows), 12836420)
        self.assertTrue(all(row["fixed_trace_source_row_admitted"] for row in rows))
        self.assertTrue(all(not row["integrated_cycle_admitted"] for row in rows))

    def test_nominal_four_bottleneck_dse_reconciles(self):
        rows = {(row["line"], row["late_scale_implementation"]): row for row in
                self.result["conditional_dse"]["four_bottleneck_rows"]}
        local = rows[("Local", "M35_parallel_complement_CSD_sidecar")]
        motion = rows[("Motion", "M35_parallel_complement_CSD_sidecar")]
        self.assertEqual(local["replacement"]["total_cycles"], 15919011)
        self.assertEqual(local["conditional_cycles_after_scope_substitution"],
                         204743502)
        self.assertEqual(motion["replacement"]["total_cycles"], 15512431)
        self.assertEqual(motion["conditional_cycles_after_scope_substitution"],
                         202666648)
        self.assertEqual(local["replacement"]["overlap_credit_cycles"], 0)
        self.assertFalse(local["system_speedup_admitted"])

    def test_nominal_hard_bound_exposes_required_six_x_coalescing(self):
        rows = [row for row in self.result[
            "four_bottleneck_event_late_scale_model"]["resource_cycle_sensitivity"]
                if row["density_name"] == "observed_exact"
                and row["lanes"] == 96 and row["banks"] == 24]
        self.assertEqual(len(rows), 2)
        local = next(row for row in rows if row["line"] == "Local")
        motion = next(row for row in rows if row["line"] == "Motion")
        self.assertEqual(local["uncoalesced_event_cycle_lower_bound"], 79630957)
        self.assertEqual(local["conditional_m4_projected_event_cycles"], 13282496)
        self.assertEqual(motion["conditional_m4_projected_event_cycles"], 12836420)
        self.assertGreater(
            local["minimum_effective_event_work_reduction_required_exact"]["numerator"],
            5 * local["minimum_effective_event_work_reduction_required_exact"][
                "denominator"])
        self.assertTrue(local["conditional_projection_only"])

    def test_resource_sensitivity_population_and_monotonicity(self):
        rows = self.result["four_bottleneck_event_late_scale_model"][
            "resource_cycle_sensitivity"]
        self.assertEqual(len(rows), 90)
        observed_local = [row for row in rows
                          if row["line"] == "Local"
                          and row["density_name"] == "observed_exact"
                          and row["banks"] == 24]
        by_lane = {row["lanes"]: row for row in observed_local}
        self.assertGreater(by_lane[48]["conditional_m4_projected_event_cycles"],
                           by_lane[96]["conditional_m4_projected_event_cycles"])
        self.assertEqual(by_lane[192]["effective_event_service_width"], 96)
        self.assertEqual(by_lane[192]["conditional_m4_projected_event_cycles"],
                         by_lane[96]["conditional_m4_projected_event_cycles"])

    def test_density_sensitivity_is_not_system_speedup(self):
        rows = self.result["four_bottleneck_event_late_scale_model"][
            "resource_cycle_sensitivity"]
        selected = {row["density_name"]: row for row in rows
                    if row["line"] == "Motion" and row["lanes"] == 96
                    and row["banks"] == 24}
        self.assertLess(selected["density_5pct"]["active_product_terms"],
                        selected["density_10pct"]["active_product_terms"])
        self.assertLess(selected["density_10pct"]["active_product_terms"],
                        selected["density_20pct"]["active_product_terms"])
        self.assertLess(selected["density_20pct"]["active_product_terms"],
                        selected["density_40pct"]["active_product_terms"])
        self.assertTrue(all(not row["system_speedup_admitted"] for row in rows))

    def test_bandwidth_traffic_lower_bounds(self):
        row = self.result["four_bottleneck_event_late_scale_model"][
            "bandwidth_traffic_lower_bounds"]
        self.assertEqual(row["unique_int8_weight_bytes"], 21233664)
        self.assertEqual(row["q24_intermediate_bytes"], 27648000)
        self.assertEqual(row["packed_output_bytes"], 1152000)
        self.assertEqual(row["fused_compulsory_bytes_lower_bound"], 31601664)
        self.assertEqual(row["materialized_compulsory_bytes_lower_bound"], 86897664)
        self.assertEqual(row["uncoalesced_event_weight_bank_cycles_nominal"],
                         79630957)
        self.assertFalse(row["address_timed_memory_admitted"])

    def test_sram_capacity_requires_tiling(self):
        row = self.result["four_bottleneck_event_late_scale_model"][
            "sram_capacity_lower_bounds"]
        self.assertEqual(row["preferred_available_bytes_after_fixed"], 193728)
        self.assertEqual(row["hard_available_bytes_after_fixed"], 365760)
        self.assertEqual(row["minimum_weight_tiles_preferred"], 110)
        self.assertEqual(row["minimum_weight_tiles_hard_cap"], 59)
        self.assertFalse(row["all_weights_fit_hard_cap"])
        self.assertFalse(row["full_q24_intermediate_fits_hard_cap"])

    def test_exact_target_gates_and_no_float_thresholds(self):
        row = next(row for row in self.result["conditional_dse"][
            "four_bottleneck_rows"] if row["line"] == "Local" and
            row["late_scale_implementation"].startswith("M35"))
        gate27, gate3 = row["target_gates"]
        self.assertEqual(gate27["target_conditional_compute_speedup"],
                         {"numerator": 27, "denominator": 10})
        self.assertEqual(gate27["target_cycle_ceiling"],
                         {"numerator": 2069560810, "denominator": 9})
        self.assertEqual(gate3["target_cycle_ceiling"],
                         {"numerator": 206956081, "denominator": 1})
        self.assertTrue(gate27["crosses_in_conditional_dse"])
        self.assertTrue(gate3["crosses_in_conditional_dse"])
        self.assertFalse(gate3["system_speedup_admitted"])

    def test_ten_consumer_legacy_formula_is_reconciled_not_additive(self):
        dse = self.result["conditional_dse"]
        self.assertTrue(dse["scope_alternatives_not_additive"])
        rows = {(row["line"], row["late_scale_implementation"]): row for row in
                dse["ten_consumer_legacy_reconciled_rows"]}
        self.assertEqual(rows[("Local", "M33_shared96_generic_UQ0p24")][
            "conditional_cycles_after_scope_substitution"], 189817484)
        self.assertEqual(rows[("Motion", "M35_parallel_complement_CSD_sidecar")][
            "conditional_cycles_after_scope_substitution"], 183799564)

    def test_recursive_type_strict_contract_forgeries_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            payload = copy.deepcopy(self.contract)
            payload["frozen_dse_rules"]["bottleneck_operator_count"] = True
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["sensitivity_rules"]["lane_points"][1] = 96.0
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["external_comparison_boundary"][
                "external_accelerator_normalized_comparison_admitted"] = 0
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["sensitivity_rules"]["density_points"][0]["ratio"][
                "numerator"] = False
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["inputs"]["m38_model_only_admission"]["sha256"] = "0" * 64
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["claim_boundary"] += " FORGED"
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["extra"] = True
            cases.append(payload)
            for index, forged in enumerate(cases):
                with self.assertRaisesRegex(ValueError, "recursive type-strict drift"):
                    M39.build(self.write_contract(
                        root, forged, "forged_{}.json".format(index)))

    def test_duplicate_keys_and_nonstandard_numbers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = CONTRACT.read_text(encoding="utf-8")
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema":"FORGED",' + canonical.lstrip()[1:],
                                 encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                M39.build(duplicate)
            for index, token in enumerate(("NaN", "Infinity", "-Infinity")):
                forged = root / "constant_{}.json".format(index)
                forged.write_text(canonical.replace(
                    '"bottleneck_operator_count": 4',
                    '"bottleneck_operator_count": {}'.format(token), 1),
                    encoding="utf-8")
                with self.assertRaisesRegex(
                        ValueError, "non-standard JSON numeric constant"):
                    M39.build(forged)

    def test_admission_never_opens_system_or_headline_claims(self):
        admission = self.result["admission"]
        self.assertTrue(admission["conditional_dse_math_admitted"])
        self.assertTrue(admission["h67_four_bottleneck_resource_lower_bounds_admitted"])
        self.assertFalse(admission[
            "conditional_m4_projection_is_executable_cycle_evidence"])
        for key in M39.FORBIDDEN_ADMISSION_KEYS:
            self.assertFalse(admission[key], key)

    def test_repeat_build_is_byte_identical(self):
        a = json.dumps(M39.build(CONTRACT), indent=2, sort_keys=True) + "\n"
        b = json.dumps(M39.build(CONTRACT), indent=2, sort_keys=True) + "\n"
        self.assertEqual(a.encode("utf-8"), b.encode("utf-8"))

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occupied.json"
            path.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                M39.write_output(path, {})
            self.assertEqual(path.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
