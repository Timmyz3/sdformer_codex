#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1877_c2_fresh_mapped_formality_dual_corner_pt_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1877_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py"
M1858_FAILURE = HW / "dc_handoff/runs/m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.2511659.quarantine"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C = load(CHECKER, "m1877_formal_source_checker")
R = load(RUNNER, "m1877_formal_runner")


class M1877FormalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = C.source_map()
        cls.black_boxes = (M1858_FAILURE /
                           "k8/formality/reports/formality_black_boxes.rpt").read_text()
        cls.formality_status = (M1858_FAILURE /
                                "k8/formality/reports/formality_status.rpt").read_text()

    def mutated(self, name, old, new):
        self.assertIn(old, self.texts[name])
        return {name: self.texts[name].replace(old, new, 1)}

    def assert_rejected(self, overrides):
        overrides = dict(overrides)
        changed_sources = [name for name in overrides
                           if name != "contract" and name in C.PATHS]
        if changed_sources:
            contract = json.loads(overrides.get("contract", self.texts["contract"]))
            for name in changed_sources:
                rel = C.PATHS[name].relative_to(HW).as_posix()
                contract["source_files"][rel] = hashlib.sha256(
                    overrides[name].encode()).hexdigest()
            overrides["contract"] = json.dumps(contract, sort_keys=True)
        with self.assertRaises((C.CheckFailure, SyntaxError)):
            C.check(overrides)

    def test_01_actual_formal_source_passes(self):
        result = C.check()
        self.assertEqual(result["status"], "PASS_M1877_FORMAL_SOURCE_STATIC")
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
            '"milestone": "M1877",',
            '"milestone": "M1877",\n  "milestone": "M1831",', 1)
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

    def test_36_rejects_live_rtl_exact_check_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "exact_regular(HW / rel, sources[rel])",
            "# live RTL exact check removed"))

    def test_37_rejects_filelist_order_check_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "rows != list(sources.keys())", "False"))

    def test_38_rejects_m1811_filelist_byte_check_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "M1811_INPUT_FILELIST.read_bytes() != REFERENCE_FILELIST.read_bytes()",
            "False"))

    def test_39_rejects_release_review_manifest_binding_removal(self):
        self.assert_rejected(self.mutated(
            "runner", '"m1878_source_review_manifest_sha256": review_manifest,',
            "# manifest binding removed"))

    def test_40_rejects_release_review_outer_binding_removal(self):
        self.assert_rejected(self.mutated(
            "runner", '"m1878_source_review_outer_seal_file_sha256": review_outer,',
            "# outer binding removed"))

    def test_41_rejects_pt_exception_report_gate_removal(self):
        self.assert_rejected(self.mutated(
            "runner", 'reports / "exceptions.rpt", reports / "design.rpt",',
            'reports / "design.rpt",'))

    def test_42_rejects_pt_design_report_gate_removal(self):
        self.assert_rejected(self.mutated(
            "runner", 'reports / "exceptions.rpt", reports / "design.rpt",',
            'reports / "exceptions.rpt",'))

    def test_43_rejects_pt_wire_load_report_gate_removal(self):
        self.assert_rejected(self.mutated(
            "runner", 'reports / "wire_load.rpt",', "# wire-load gate removed"))

    def test_44_rejects_pt_constraint_semantic_gate_removal(self):
        self.assert_rejected(self.mutated(
            "runner", 'reports / "constraint_semantics_machine.txt",',
            "# constraint semantics removed"))

    def test_45_rejects_pt_coverage_semantic_parser_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "summary.update(parse_pt_semantics(reports))",
            "# semantic parser removed"))

    def test_46_rejects_tcl_constraint_count_removal(self):
        self.assert_rejected(self.mutated(
            "pt", 'puts $constraint_fp "setup_violating_paths=[sizeof_collection $setup_violators]"',
            'puts $constraint_fp "setup_violating_paths=0"'))

    def test_47_rejects_contract_live_rtl_digest_drift(self):
        value = json.loads(self.texts["contract"])
        first = next(iter(value["reference"]["live_rtl_source_identity"]))
        value["reference"]["live_rtl_source_identity"][first] = "0" * 64
        self.assert_rejected({"contract": json.dumps(value)})

    def test_48_rejects_m1873_review_pin_drift(self):
        value = json.loads(self.texts["contract"])
        value["m1873_failure_review"]["review_sha256"] = "0" * 64
        self.assert_rejected({"contract": json.dumps(value)})

    def test_49_rejects_reserved_m1845_namespace(self):
        self.assert_rejected({"runner": self.texts["runner"] + "\n# m1845_reserved\n"})

    def test_50_rejects_raw_constraint_semantic_parser_removal(self):
        self.assert_rejected(self.mutated(
            "runner", 'constraint_report = (reports / "constraint_violators.rpt").read_text(',
            'constraint_report = "" # removed ('))

    def test_51_rejects_second_pre_attempt_authority_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", "current_release_sha, current_live_rtl_identity = verify_authority()",
            "current_release_sha, current_live_rtl_identity = (release_sha, live_rtl_identity)"))

    def test_52_rejects_unique_attempt_call_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "            write_attempt(release_sha)",
            "            pass  # attempt consumption removed"))

    def test_53_rejects_check_timing_uniqueness_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", 'check_text.count("check_timing succeeded.") != 1', "False"))

    def test_54_rejects_coverage_conservation_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", 'row["total"] != row["met"] + row["violated"] + row["untested"]',
            "False"))

    def test_55_rejects_exact_coverage_rows_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", 'set(coverage) != {"setup", "hold", "All Checks"}', "False"))

    def test_56_rejects_machine_constraint_count_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", 're.fullmatch(r"\\d+", constraint_values.get(key, "")) is None',
            "False"))

    def test_57_rejects_raw_constraint_visibility_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", "and raw_constraint_violation_marker_count == 0", "and False"))

    def test_58_rejects_verbatim_setup_wns_rewrite(self):
        self.assert_rejected(self.mutated(
            "pt", 'puts $summary_fp "setup_wns_ns=$setup_slack"',
            'puts $summary_fp "setup_wns_ns=0.0"'))

    def test_59_rejects_formality_result_verification_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", "passing = verify_formality(axis, fm_dir)", "passing = 1"))

    def test_60_rejects_pt_result_verification_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", "timing = verify_pt(axis, pt_dir)", "timing = {}"))

    def test_61_rejects_fmr147_filter_removal(self):
        self.assert_rejected(self.mutated(
            "formality", "set_mismatch_message_filter -warn FMR_ELAB-147",
            "# FMR_ELAB-147 filter removed"))

    def test_62_rejects_fmr147_filter_after_reference_set_top(self):
        pair = ("set_mismatch_message_filter -warn FMR_ELAB-147\n"
                "set_top r:/WORK/$reference_top -parameter $reference_elab_parameters")
        self.assert_rejected(self.mutated(
            "formality", pair,
            "set_top r:/WORK/$reference_top -parameter $reference_elab_parameters\n"
            "set_mismatch_message_filter -warn FMR_ELAB-147"))

    def test_63_rejects_duplicate_fmr147_filter(self):
        self.assert_rejected(self.mutated(
            "formality", "set_mismatch_message_filter -warn FMR_ELAB-147",
            "set_mismatch_message_filter -warn FMR_ELAB-147\n"
            "set_mismatch_message_filter -warn FMR_ELAB-147"))

    def test_64_rejects_other_mismatch_message_id(self):
        self.assert_rejected(self.mutated(
            "formality", "set_mismatch_message_filter -warn FMR_ELAB-147",
            "set_mismatch_message_filter -warn FMR_ELAB-999"))

    def test_65_rejects_ignore_filter(self):
        self.assert_rejected(self.mutated(
            "formality", "set_mismatch_message_filter -warn FMR_ELAB-147",
            "set_mismatch_message_filter -ignore FMR_ELAB-147"))

    def test_66_rejects_suppress_message(self):
        self.assert_rejected({
            "formality": self.texts["formality"] + "\nsuppress_message FM-999\n"})

    def test_67_rejects_set_message_info(self):
        self.assert_rejected({
            "formality": self.texts["formality"] +
            "\nset_message_info -id FM-999 -limit 0\n"})

    def test_68_rejects_warning_site_set_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", "warning_sites != EXPECTED_FMR_ELAB_147_SITES", "False"))

    def test_69_rejects_warning_count_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", "len(warning_pattern.findall(log_text)) != 8", "False"))

    def test_70_rejects_valid_design_pair_status_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", "status.count(token) != 1", "False"))

    def test_71_rejects_positive_passing_compare_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", 're.search(r"[1-9][0-9]*\\s+Passing compare points", status)',
            "True"))

    def test_72_rejects_failing_total_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", "failing_row is None or int(failing_row.group(1)) != 0",
            "False"))

    def test_73_rejects_blackbox_valid_pair_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", '"Reference and implementation designs are not set" in black_boxes',
            "False"))

    def test_74_rejects_m1858_attempt_pin_drift(self):
        value = json.loads(self.texts["contract"])
        value["supersedes_failed_execution"]["attempt_json_sha256"] = "0" * 64
        self.assert_rejected({"contract": json.dumps(value)})

    def test_75_rejects_m1858_failure_pin_drift(self):
        value = json.loads(self.texts["contract"])
        value["supersedes_failed_execution"]["failure_manifest_sha256"] = "0" * 64
        self.assert_rejected({"contract": json.dumps(value)})

    def test_76_rejects_m1873_review_pin_drift(self):
        value = json.loads(self.texts["contract"])
        value["future_authority"]["m1873_failure_review_sha256"] = "0" * 64
        self.assert_rejected({"contract": json.dumps(value)})

    def test_77_rejects_runner_m1873_semantic_bypass(self):
        self.assert_rejected(self.mutated(
            "runner", 'failure_review.get("audit_status") != "PASS"', "False"))

    def test_78_rejects_release_m1873_binding_removal(self):
        self.assert_rejected(self.mutated(
            "runner", '"m1873_failure_review_manifest_sha256": M1873_MANIFEST_SHA,',
            "# M1873 manifest binding removed"))

    def assert_black_box_rejected(self, black_boxes=None, status=None):
        with self.assertRaises(R.M1877Error):
            R.verify_formality_black_box_policy(
                self.black_boxes if black_boxes is None else black_boxes,
                self.formality_status if status is None else status)

    def test_79_actual_m1858_report_passes_section_aware_policy(self):
        result = R.verify_formality_black_box_policy(
            self.black_boxes, self.formality_status)
        self.assertEqual(result["exact_symmetric_snps_bushold_entries"], 2)
        self.assertEqual(result["passing_bbpin"], 0)
        self.assertEqual(result["failing_bbpin"], 0)

    def test_80_rejects_removed_reference_snps_bushold_side(self):
        block = ("e      SNPS_BUSHOLD\n\n"
                 "       Instances : 2 of 2\n"
                 "       ------------------------\n"
                 "       r:/TCBN28HPCPLUSBWP35P140SSG0P9V125C/BHDBWP35P140/C0\n"
                 "       r:/TCBN28HPCPLUSBWP35P140SSG0P9V125C/BHDBWP35P140#PWR/C2\n")
        self.assertEqual(self.black_boxes.count(block), 1)
        self.assert_black_box_rejected(self.black_boxes.replace(block, "", 1))

    def test_81_rejects_added_third_snps_bushold_side(self):
        block = ("e      SNPS_BUSHOLD\n\n"
                 "       Instances : 2 of 2\n"
                 "       ------------------------\n"
                 "       i:/TCBN28HPCPLUSBWP35P140SSG0P9V125C/BHDBWP35P140/C0\n"
                 "       i:/TCBN28HPCPLUSBWP35P140SSG0P9V125C/BHDBWP35P140#PWR/C2\n")
        self.assertEqual(self.black_boxes.count(block), 1)
        self.assert_black_box_rejected(self.black_boxes.replace(block, block + "\n" + block, 1))

    def test_82_rejects_snps_bushold_instance_count_change(self):
        self.assert_black_box_rejected(self.black_boxes.replace(
            "Instances : 2 of 2", "Instances : 1 of 2", 1))

    def test_83_rejects_snps_bushold_rename(self):
        self.assert_black_box_rejected(self.black_boxes.replace(
            "e      SNPS_BUSHOLD", "e      SNPS_OTHER", 1))

    def test_84_rejects_snps_bushold_path_change(self):
        self.assert_black_box_rejected(self.black_boxes.replace(
            "BHDBWP35P140/C0", "BHDBWP35P140/C1", 1))

    def test_85_rejects_nonzero_design_library_empty_module(self):
        marker = "####    DESIGN LIBRARY - r:/WORK\n"
        injection = (marker +
                     "##################################################################\n"
                     "Type  Design Name\n----  ----------\n"
                     "e      ATTACK_DESIGN\n\n"
                     "       Instances : 1 of 1\n"
                     "       ------------------------\n"
                     "       r:/WORK/TOP/U_ATTACK\n\n")
        self.assertEqual(self.black_boxes.count(marker), 1)
        self.assert_black_box_rejected(self.black_boxes.replace(marker, injection, 1))

    def test_86_rejects_nonzero_passing_bbpin(self):
        status = self.formality_status.replace(
            "Passing (equivalent)           0", "Passing (equivalent)           1", 1)
        self.assertNotEqual(status, self.formality_status)
        self.assert_black_box_rejected(status=status)

    def test_87_rejects_nonzero_failing_bbpin(self):
        status = self.formality_status.replace(
            "Failing (not equivalent)       0", "Failing (not equivalent)       1", 1)
        self.assertNotEqual(status, self.formality_status)
        self.assert_black_box_rejected(status=status)

    def test_88_rejects_generic_nonzero_technology_empty_module(self):
        marker = "e      SNPS_BUSHOLD\n"
        injection = ("e      GENERIC_TECH_EMPTY\n\n"
                     "       Instances : 1 of 1\n"
                     "       ------------------------\n"
                     "       i:/TCBN28HPCPLUSBWP35P140SSG0P9V125C/GENERIC/X\n\n"
                     + marker)
        self.assertEqual(self.black_boxes.count(marker), 2)
        self.assert_black_box_rejected(self.black_boxes.replace(marker, injection, 1))

    def test_89_rejects_black_box_policy_call_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "verify_formality_black_box_policy(black_boxes, status)",
            "{}  # black-box policy removed"))

    def test_90_rejects_contract_generic_tech_e_permission(self):
        value = json.loads(self.texts["contract"])
        value["formality_black_box_policy"][
            "generic_technology_u_e_star_nonzero_allowed"] = True
        self.assert_rejected({"contract": json.dumps(value)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
