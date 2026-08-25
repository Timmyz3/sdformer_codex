from __future__ import print_function

import importlib.util
import json
import os
import tempfile
import unittest


REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
AUDITOR_PATH = os.path.join(
    REPO_ROOT,
    "hw_autoresearch_nts07/dc_handoff/scripts/"
    "audit_m37_r10_area_recovery_arch.py",
)
CONTRACT_PATH = os.path.join(
    REPO_ROOT,
    "hw_autoresearch_nts07/contracts/"
    "m37_r10_area_recovery_arch_contract_r1_20260822.json",
)
RTL_PATH = os.path.join(
    REPO_ROOT,
    "hw_autoresearch_nts07/rtl_m37_r10/"
    "qfit_atlif_csd_reconstruct_t10.sv",
)


def load_auditor():
    spec = importlib.util.spec_from_file_location("m37_r10_auditor", AUDITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M37R10AreaRecoveryArchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.auditor = load_auditor()
        with open(RTL_PATH, "r") as handle:
            cls.rtl = handle.read()

    def audit_rtl_mutation(self, old, new):
        self.assertIn(old, self.rtl)
        mutated = self.rtl.replace(old, new, 1)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sv", delete=False) as handle:
            handle.write(mutated)
            temporary = handle.name
        try:
            return self.auditor.audit(
                REPO_ROOT, CONTRACT_PATH, temporary, False
            )
        finally:
            os.unlink(temporary)

    def test_00_canonical_candidate_passes_static_scope(self):
        result = self.auditor.audit(
            REPO_ROOT, CONTRACT_PATH, RTL_PATH, True
        )
        self.assertEqual(result["failures"], [])
        self.assertEqual(
            result["status"],
            "PASS_M37_R10_STATIC_AREA_RECOVERY_ARCHITECTURE_ONLY",
        )

    def test_01_missing_phase_arm_fails_closed(self):
        result = self.audit_rtl_mutation(
            "3'd4: phase_bundle_comb = phase_table_q[672 +: PHASE_BUNDLE_W];",
            "3'd4: phase_bundle_comb = '0;",
        )
        self.assertTrue(any("phase" in item for item in result["failures"]))

    def test_02_r9_coefficient_equality_scan_fails_closed(self):
        result = self.audit_rtl_mutation(
            "assign uses_integer_multiplier = 1'b0;",
            "logic selected_coefficient;\n"
            "    logic coefficient_index;\n"
            "    assign uses_integer_multiplier = selected_coefficient "
            "== coefficient_index;",
        )
        self.assertTrue(any("equality-expanded" in item
                            for item in result["failures"]))

    def test_03_nonzero_multiplier_flag_fails_closed(self):
        result = self.audit_rtl_mutation(
            "assign uses_integer_multiplier = 1'b0;",
            "assign uses_integer_multiplier = 1'b1;",
        )
        self.assertTrue(any("multiplier" in item
                            for item in result["failures"]))

    def test_04_missing_bias_pair_pipeline_fails_closed(self):
        result = self.audit_rtl_mutation(
            "product_bias_pair_q <= phase_bias_pair_comb;",
            "product_bias_pair_q <= '0;",
        )
        self.assertTrue(any("bias_pair" in item
                            for item in result["failures"]))

    def test_05_unpacked_descriptor_storage_fails_closed(self):
        result = self.audit_rtl_mutation(
            "logic config_loaded_q;",
            "logic config_loaded_q;\n"
            "    logic term_valid_q [0:COEFFICIENTS-1][0:TERMS-1];",
        )
        self.assertTrue(any("unpacked stored valid" in item
                            for item in result["failures"]))

    def test_06_resource_equation_type_or_value_drift_fails_closed(self):
        with open(CONTRACT_PATH, "r") as handle:
            contract = json.load(handle)
        contract["resource_equation"][
            "candidate_architectural_state_bits_before_optimization"
        ] = True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(contract, handle)
            temporary = handle.name
        try:
            result = self.auditor.audit(
                REPO_ROOT, temporary, RTL_PATH, False
            )
        finally:
            os.unlink(temporary)
        self.assertTrue(any("candidate_architectural_state_bits" in item
                            for item in result["failures"]))


if __name__ == "__main__":
    unittest.main()
