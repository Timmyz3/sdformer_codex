#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
import json
import os
from pathlib import Path
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1832_c2_fresh_mapped_formality_dual_corner_pt_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1832_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C = load(CHECKER, "m1832_formal_source_checker")
R = load(RUNNER, "m1832_formal_runner")


class M1832FormalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = C.source_map()

    def mutated(self, name, old, new):
        self.assertIn(old, self.texts[name])
        return {name: self.texts[name].replace(old, new, 1)}

    def assert_rejected(self, overrides):
        with self.assertRaises(C.CheckFailure):
            C.check(overrides)

    def test_01_actual_formal_source_passes(self):
        result = C.check()
        self.assertEqual(result["status"], "PASS_M1832_FORMAL_SOURCE_STATIC")
        self.assertFalse(result["launch_authorized"])
        self.assertFalse(result["eda_or_license_run"])

    def test_02_authoring_namespace_is_fresh(self):
        self.assertFalse(R.ATTEMPT.exists())
        self.assertFalse(R.RESULT.exists())
        self.assertFalse(R.WORK.exists())
        self.assertFalse(R.LAUNCH_LOCK.exists())

    def test_03_rejects_m1811_manifest_drift(self):
        self.assert_rejected(self.mutated("runner", C.M1811_MANIFEST, "0" * 64))

    def test_04_rejects_m1811_outer_drift(self):
        self.assert_rejected(self.mutated("runner", C.M1811_OUTER, "0" * 64))

    def test_05_rejects_m1830_review_drift(self):
        self.assert_rejected(self.mutated("runner", C.M1830_REVIEW, "0" * 64))

    def test_06_rejects_m1830_manifest_drift(self):
        self.assert_rejected(self.mutated("runner", C.M1830_MANIFEST, "0" * 64))

    def test_07_rejects_m1830_outer_drift(self):
        self.assert_rejected(self.mutated("runner", C.M1830_OUTER, "0" * 64))

    def test_08_rejects_k8_mapped_v_drift(self):
        digest = C.ARTIFACT_SHAS["K8"]["mapped_v_sha256"]
        self.assert_rejected(self.mutated("runner", digest, "0" * 64))

    def test_09_rejects_k8_sdc_drift(self):
        digest = C.ARTIFACT_SHAS["K8"]["mapped_sdc_sha256"]
        self.assert_rejected(self.mutated("runner", digest, "0" * 64))

    def test_10_rejects_k8_svf_drift(self):
        digest = C.ARTIFACT_SHAS["K8"]["svf_sha256"]
        self.assert_rejected(self.mutated("runner", digest, "0" * 64))

    def test_11_rejects_k1x8_mapped_v_drift(self):
        digest = C.ARTIFACT_SHAS["K1X8"]["mapped_v_sha256"]
        self.assert_rejected(self.mutated("runner", digest, "0" * 64))

    def test_12_rejects_k1x8_sdc_drift(self):
        digest = C.ARTIFACT_SHAS["K1X8"]["mapped_sdc_sha256"]
        self.assert_rejected(self.mutated("runner", digest, "0" * 64))

    def test_13_rejects_k1x8_svf_drift(self):
        digest = C.ARTIFACT_SHAS["K1X8"]["svf_sha256"]
        self.assert_rejected(self.mutated("runner", digest, "0" * 64))

    def test_14_rejects_axis_mode_swap(self):
        self.assert_rejected(self.mutated(
            "runner", '"elab_parameters": "ARCH_MODE=1"',
            '"elab_parameters": "ARCH_MODE=0"'))

    def test_15_rejects_shared_axis_artifact(self):
        attack = self.texts["runner"].replace(
            'M1811 / "k1x8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v"',
            'M1811 / "k8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v"', 1)
        self.assert_rejected({"runner": attack})

    def test_16_rejects_formality_call_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "run_tool(FM_SHELL, FM_TCL, axis, fm_dir, \"formality.log\")",
            "# formality removed"))

    def test_17_rejects_pt_call_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "run_tool(PT_SHELL, PT_TCL, axis, pt_dir, \"pt.log\")",
            "# pt removed"))

    def test_18_rejects_retry_enable(self):
        self.assert_rejected(self.mutated(
            "runner", '"automatic_retry": False', '"automatic_retry": True'))

    def test_19_rejects_hold_failure_as_publication_block(self):
        self.assert_rejected(self.mutated(
            "runner", '"hold_failure_blocks_result_publication": False',
            '"hold_failure_blocks_result_publication": True'))

    def test_20_rejects_hidden_hold_slack(self):
        self.assert_rejected(self.mutated(
            "pt", 'puts $summary_fp "hold_wns_ns=$hold_slack"',
            'puts $summary_fp "hold_wns_ns=0.0"'))

    def test_21_rejects_false_path(self):
        self.assert_rejected({"pt": self.texts["pt"] + "\nset_false_path -from A -to B\n"})

    def test_22_rejects_multicycle(self):
        self.assert_rejected({"pt": self.texts["pt"] + "\nset_multicycle_path 2\n"})

    def test_23_rejects_pt_eco(self):
        self.assert_rejected({"pt": self.texts["pt"] + "\nfix_eco_timing -type hold\n"})

    def test_24_rejects_missing_min_library(self):
        self.assert_rejected(self.mutated(
            "pt", "set_min_library $std_slow_db -min_version $std_fast_db",
            "# min library removed"))

    def test_25_rejects_missing_ocv(self):
        self.assert_rejected(self.mutated(
            "pt", "set_operating_conditions -analysis_type on_chip_variation",
            "set_operating_conditions"))

    def test_26_rejects_missing_formality_verify(self):
        self.assert_rejected(self.mutated(
            "formality", "set verification_succeeded [verify]",
            "set verification_succeeded true"))

    def test_27_rejects_contract_author_execution(self):
        value = json.loads(self.texts["contract"])
        value["authorization_now"]["pt_runs"] = 1
        self.assert_rejected({"contract": json.dumps(value)})

    def test_28_rejects_contract_future_third_run(self):
        value = json.loads(self.texts["contract"])
        value["future_execution_budget"]["pt_runs_exact"] = 3
        self.assert_rejected({"contract": json.dumps(value)})

    def test_29_rejects_contract_hold_hiding(self):
        value = json.loads(self.texts["contract"])
        value["timing_violation_policy"]["negative_setup_or_hold_is_reported"] = False
        self.assert_rejected({"contract": json.dumps(value)})

    def test_30_rejects_contract_claim_promotion(self):
        value = json.loads(self.texts["contract"])
        value["claim_boundary"]["formality"] = True
        self.assert_rejected({"contract": json.dumps(value)})

    def test_31_rejects_duplicate_contract_key(self):
        attack = self.texts["contract"].replace(
            '"milestone": "M1832",',
            '"milestone": "M1832",\n  "milestone": "M1831",', 1)
        self.assert_rejected({"contract": attack})

    def test_32_rejects_stale_draft_identity(self):
        self.assert_rejected({"runner": self.texts["runner"] + "\n# UNSEALED_SOURCE_DRAFT\n"})

    def test_33_rejects_legacy_m1833_authority(self):
        self.assert_rejected({"runner": self.texts["runner"] + "\n# m1833_old_review\n"})

    def test_34_rejects_legacy_m1835_release(self):
        self.assert_rejected({"runner": self.texts["runner"] + "\n# m1835_old_release\n"})

    def test_35_contract_source_hashes_are_exact(self):
        data = json.loads(self.texts["contract"])
        for rel, digest in data["source_files"].items():
            self.assertEqual(digest, C.sha(HW / rel))


if __name__ == "__main__":
    unittest.main(verbosity=2)
