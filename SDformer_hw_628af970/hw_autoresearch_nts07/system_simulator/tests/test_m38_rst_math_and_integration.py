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
    "analyze_m38_rst_math_and_integration.py"
)
RESULT = (
    ROOT
    / "hw_autoresearch_nts07/results/m38_rst_math_and_integration_r1_20260822/"
    "m38_rst_math_and_integration.json"
)
SPEC = importlib.util.spec_from_file_location("m38", str(SCRIPT))
M38 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M38)


class M38RSTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = M38.build()

    def test_complete_q8_ternary_scalar_domain_and_neg128(self):
        scalar = self.result["scalar_ternary_audit"]
        self.assertEqual(scalar["pairs_checked"], 256 * 3)
        self.assertTrue(scalar["all_products_exact"])
        self.assertEqual(scalar["product_range"], [-128, 128])
        self.assertEqual(scalar["minimum_signed_product_bits"], 9)
        self.assertEqual(
            scalar["negative_minimum_negation_witness"]["result"], 128
        )
        for row in scalar["rows"]:
            self.assertEqual(
                row["product"], row["input_q8"] * row["coefficient"]
            )

    def test_illegal_code_and_out_of_range_input_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "illegal M38 ternary code"):
            M38.ternary_product(0, 3)
        with self.assertRaisesRegex(ValueError, "not signed q8"):
            M38.ternary_product(128, 1)
        with self.assertRaisesRegex(ValueError, "not signed q8"):
            M38.ternary_product(-129, 2)

    def test_rank3_width_saturation_and_threshold_semantics(self):
        audit = self.result["rank3_q24_threshold_audit"]
        self.assertEqual(audit["rank3_sum_range"], [-384, 384])
        self.assertEqual(audit["minimum_signed_rank3_sum_bits"], 10)
        self.assertEqual(audit["mathematical_minimum_bias_plus_rank_sum_bits"], 25)
        self.assertEqual(audit["implemented_pre_saturation_bits_target"], 26)
        self.assertGreater(audit["positive_saturation_witnesses"], 0)
        self.assertGreater(audit["negative_saturation_witnesses"], 0)
        self.assertEqual(audit["threshold_equality_event"], 1)
        self.assertEqual(audit["threshold_just_below_event"], 0)
        self.assertEqual(
            audit["threshold_equality_checks"], audit["saturation_vectors_checked"]
        )

    def test_integrated_candidate_ledger_is_normalized_and_conditional(self):
        ledger = self.result["integrated_theory_ledger"]
        rows = {row["name"]: row for row in ledger["candidates"]}
        self.assertEqual(rows["m31_serialized_shared96"]["conditional_t10_steady_ii_cycles"], 10)
        for name in (
                "direct_second96_parallel",
                "m37_csd4_parallel_normalized_integration_target",
                "m38_rst_parallel"):
            self.assertEqual(rows[name]["conditional_t10_steady_ii_cycles"], 5)
            self.assertEqual(
                rows[name]["conditional_t10_steady_throughput_ratio_vs_m31"], 2.0
            )
            self.assertFalse(rows[name]["system_speedup_admitted"])
            self.assertFalse(rows[name]["area_admitted"])
            self.assertFalse(rows[name]["energy_admitted"])
        self.assertEqual(rows["direct_second96_parallel"]["additional_int8_multiplier_lanes"], 96)
        self.assertEqual(rows["m37_csd4_parallel_normalized_integration_target"]["added_programmable_shift_term_sites_per_cycle"], 384)
        self.assertEqual(rows["m38_rst_parallel"]["added_programmable_shift_term_sites_per_cycle"], 0)
        self.assertEqual(rows["m38_rst_parallel"]["added_ternary_select_sites_per_cycle"], 96)

    def test_configuration_bit_ledger(self):
        ledger = self.result["configuration_bit_ledger"]
        rows = {row["name"]: row for row in ledger["rows"]}
        self.assertEqual(ledger["common_payload_bits_excluding_left_factor"], 509)
        self.assertEqual(rows["m31_serialized_shared96"]["t10_parameter_payload_bits"], 749)
        self.assertEqual(rows["direct_second96_parallel"]["t10_parameter_payload_bits"], 749)
        self.assertEqual(rows["m37_csd4_parallel_normalized_integration_target"]["t10_parameter_payload_bits"], 1349)
        self.assertEqual(rows["m38_rst_parallel"]["t10_parameter_payload_bits"], 569)
        self.assertEqual(rows["m38_rst_parallel"]["t10_context_bits_with_integrity"], 617)
        self.assertFalse(rows["m38_rst_parallel"]["parameter_load_cycles_included_in_throughput"])

    def test_contract_hash_and_architecture_drift_fail_closed(self):
        contract = json.loads(M38.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bad_hash = copy.deepcopy(contract)
            bad_hash["inputs"]["m31_vcs_contract"]["sha256"] = "0" * 64
            bad_hash_path = directory / "bad_hash.json"
            bad_hash_path.write_text(json.dumps(bad_hash), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input hash drift"):
                M38.build(bad_hash_path)

            bad_arch = copy.deepcopy(contract)
            bad_arch["frozen_architecture"]["rank"] = 2
            bad_arch_path = directory / "bad_arch.json"
            bad_arch_path.write_text(json.dumps(bad_arch), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "architecture drift"):
                M38.build(bad_arch_path)

            bad_theory = copy.deepcopy(contract)
            bad_theory["theory_rules"]["system_speedup_admitted"] = True
            bad_theory_path = directory / "bad_theory.json"
            bad_theory_path.write_text(json.dumps(bad_theory), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "theory rule drift"):
                M38.build(bad_theory_path)

    def test_frozen_result_rebuilds_exactly_and_claims_stay_closed(self):
        frozen = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(frozen, self.result)
        admission = frozen["admission"]
        self.assertTrue(admission["q8_times_ternary_scalar_math_admitted"])
        self.assertFalse(admission["trained_codebook_admitted"])
        self.assertFalse(admission["integrated_rtl_admitted"])
        self.assertFalse(admission["area_timing_power_energy_admitted"])
        self.assertFalse(admission["system_speedup_admitted"])
        self.assertFalse(admission["headline_admitted"])

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m38.json"
            output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                M38.write_output(output, {"must_not": "overwrite"})
            self.assertEqual(output.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
