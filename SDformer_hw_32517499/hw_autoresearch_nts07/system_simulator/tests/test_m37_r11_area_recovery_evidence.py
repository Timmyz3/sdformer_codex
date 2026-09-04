from __future__ import print_function

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
AUDITOR = os.path.join(
    ROOT,
    "hw_autoresearch_nts07/dc_handoff/scripts/"
    "audit_m37_r11_area_recovery_evidence.py",
)
CONTRACT = os.path.join(
    ROOT,
    "hw_autoresearch_nts07/contracts/"
    "m37_r11_area_recovery_evidence_contract_r1_20260822.json",
)
RTL = os.path.join(
    ROOT,
    "hw_autoresearch_nts07/rtl_m37_r10/"
    "qfit_atlif_csd_reconstruct_t10.sv",
)
PIN = os.path.join(
    ROOT,
    "hw_autoresearch_nts07/contracts/"
    "m37_r11_evidence_pin_r1_20260822.json",
)


def load_auditor():
    spec = importlib.util.spec_from_file_location("m37_r11_auditor", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M37R11AreaRecoveryEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.auditor = load_auditor()
        with open(RTL, "r") as handle:
            cls.rtl = handle.read()
        with open(CONTRACT, "r") as handle:
            cls.contract = json.load(handle)

    def mutate(self, old, new):
        self.assertIn(old, self.rtl)
        return self.rtl.replace(old, new, 1)

    def assert_semantic_rejection(self, old, new, expected_code):
        failures = self.auditor.semantic_failures(self.mutate(old, new))
        self.assertIn(expected_code, failures)

    def test_00_canonical_pinned_audit_passes(self):
        result = self.auditor.audit()
        self.assertEqual(result["failures"], [])
        self.assertEqual(
            result["status"],
            "PASS_M37_R11_CANONICAL_PINNED_STATIC_SOURCE_EVIDENCE_ONLY",
        )
        self.assertIs(result["physical_shared_mux_proven"], False)

    def test_01_bias_row_swap_is_rejected(self):
        self.assert_semantic_rejection(
            "(result_row_group*ACC_W) +: ACC_W",
            "((1-result_row_group)*ACC_W) +: ACC_W",
            "SEM_BIAS_ROW_MAPPING",
        )

    def test_02_product_beat_off_by_one_is_rejected(self):
        self.assert_semantic_rejection(
            "product_beat_q <= phase_cycle_q;",
            "product_beat_q <= phase_cycle_q + 1'b1;",
            "SEM_PRODUCT_BEAT_CAPTURE",
        )

    def test_03_phase4_replacement_clear_is_rejected(self):
        self.assert_semantic_rejection(
            "bank_valid_q[input_bank] <= 1'b1;",
            "bank_valid_q[input_bank] <= 1'b0;",
            "SEM_PHASE4_REPLACEMENT_ASSIGNMENTS",
        )

    def test_04_reversed_phase_table_load_is_rejected(self):
        self.assert_semantic_rejection(
            "phase_table_q[(config_phase*PHASE_BUNDLE_W)",
            "phase_table_q[((PHASES-1-config_phase)*PHASE_BUNDLE_W)",
            "SEM_PHASE_TABLE_LOAD_PACKING",
        )

    def test_05_valid_negative_miswire_is_rejected(self):
        self.assert_semantic_rejection(
            "config_term_valid[(config_phase*PHASE_VALID_W)",
            "config_term_negative[(config_phase*PHASE_VALID_W)",
            "SEM_PHASE_TABLE_LOAD_PACKING",
        )

    def test_06_extra_issue_product_driver_is_rejected(self):
        self.assert_semantic_rejection(
            "endmodule\n\n`default_nettype wire",
            "always_comb begin\n"
            "        issue_product_comb[0] = 18'sd0;\n"
            "    end\n"
            "endmodule\n\n`default_nettype wire",
            "SEM_ISSUE_PRODUCT_DRIVER_SET",
        )

    def test_07_bool_int_nested_contract_confusion_is_rejected(self):
        mutated = json.loads(json.dumps(self.contract))
        mutated["semantic_invariants"]["phase_bundle_offsets"][0] = False
        failures = self.auditor.validate_contract(mutated, ROOT)
        self.assertTrue(any("phase_bundle_offsets" in item for item in failures))

    def test_08_bool_int_scalar_contract_confusion_is_rejected(self):
        mutated = json.loads(json.dumps(self.contract))
        mutated["resource_equation"]["state_delta_bits"] = True
        failures = self.auditor.validate_contract(mutated, ROOT)
        self.assertTrue(any("state_delta_bits" in item for item in failures))

    def test_09_alternate_candidate_path_is_rejected(self):
        mutated = json.loads(json.dumps(self.contract))
        mutated["candidate"]["path"] = "../rtl_m37_r10/qfit.sv"
        failures = self.auditor.validate_contract(mutated, ROOT)
        self.assertTrue(any("candidate" in item for item in failures))

    def test_10_duplicate_json_key_is_rejected(self):
        content = '{"schema":"a","schema":"b"}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                self.auditor.load_json_strict(path)
        finally:
            os.unlink(path)

    def test_11_nonfinite_json_constant_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            handle.write('{"area": NaN}')
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                self.auditor.load_json_strict(path)
        finally:
            os.unlink(path)

    def test_12_realpath_escape_is_rejected(self):
        failures = []
        path = self.auditor.canonical_contained_path(
            ROOT, "../outside", "../outside", failures, "attack"
        )
        self.assertIsNone(path)
        self.assertTrue(any("non-normal" in item or "escapes" in item
                            for item in failures))

    def test_13_cli_path_override_is_not_available(self):
        process = subprocess.Popen(
            ["/usr/bin/python3.6", AUDITOR, "--rtl", "/tmp/attack.sv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        self.assertEqual(process.returncode, 2)
        self.assertIn(b"accepts no path", stderr)
        self.assertEqual(stdout, b"")

    def test_14_pin_auditor_sha_drift_is_rejected(self):
        pin = self.auditor.load_json_strict(PIN)
        pin["artifacts"]["auditor"]["sha256"] = "0" * 64
        failures = self.auditor.validate_pin(pin, ROOT)
        self.assertTrue(any("auditor" in item and "SHA mismatch" in item
                            for item in failures))


if __name__ == "__main__":
    unittest.main()
