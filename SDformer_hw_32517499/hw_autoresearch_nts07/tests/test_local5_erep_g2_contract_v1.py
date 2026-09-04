#!/usr/bin/env python3
"""Fail-closed tests for the Local5 EREP v4 G2 preimplementation contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "contracts/local5_erep_g2_preimplementation_contract_v1_20260810.json"
)
DOC_PATH = ROOT / "docs/291_Local5_EREP_G2物理评估预实现合同_20260810.md"
STATE_PATH = ROOT / "memory/architecture/local5_erep_g2_run_state.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_bytes(*args: str, cwd: Path) -> bytes:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


class Local5ErepG2ContractV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_top_level_state_and_evidence_boundaries(self) -> None:
        contract = self.contract
        self.assertEqual(contract["schema_version"], 1)
        self.assertEqual(contract["contract_state"], "FROZEN_PREIMPLEMENTATION_CONTRACT")
        self.assertEqual(contract["signoff_state"], "NOT_ACHIEVED")
        self.assertEqual(contract["evidence_labels"]["g2a"], "[开放物理代理]")
        self.assertEqual(contract["evidence_labels"]["g2b"], "[待验证]")
        self.assertFalse(contract["g2a_open_physical_proxy"]["asic_ppa_claim_allowed"])
        self.assertFalse(contract["constraint_validation"]["g2b_may_run_now"])

    def test_c0_c5_share_exact_functional_boundary(self) -> None:
        candidates = self.contract["candidates"]
        self.assertEqual([row["id"] for row in candidates], [f"C{i}" for i in range(6)])
        boundary = self.contract["scope"]["boundary_id"]
        self.assertTrue(all(row["boundary_id"] == boundary for row in candidates))
        self.assertTrue(all(row["physical_candidate"] for row in candidates))
        self.assertIn("bit-identical Acc32", self.contract["scope"]["equivalence_rule"])
        self.assertEqual(self.contract["scope"]["accumulator_width_bits"], 32)

    def test_candidate_definitions_and_c4_first_fit_are_frozen(self) -> None:
        by_id = {row["id"]: row for row in self.contract["candidates"]}
        self.assertEqual(
            (by_id["C0"]["stripe_width"], by_id["C0"]["epoch_slots"], by_id["C0"]["accumulator_contexts"]),
            (1, 0, 1),
        )
        self.assertEqual(
            (by_id["C1"]["stripe_width"], by_id["C1"]["epoch_slots"], by_id["C1"]["accumulator_contexts"]),
            (2, 1, 2),
        )
        self.assertEqual(
            (by_id["C2"]["stripe_width"], by_id["C2"]["epoch_slots"], by_id["C2"]["accumulator_contexts"]),
            (1, 2, 1),
        )
        self.assertEqual(
            (by_id["C3"]["stripe_width"], by_id["C3"]["epoch_slots"], by_id["C3"]["accumulator_contexts"]),
            (2, 2, 2),
        )
        c4 = by_id["C4"]
        self.assertEqual(c4["payload_capacity_bits"], 561_600)
        self.assertEqual(c4["payload_capacity_records"], 5_014)
        self.assertEqual(c4["record_bits"], 112)
        self.assertEqual(c4["static_unused_payload_bits"], 32)
        self.assertIn("input-head-order unconditional first-fit", c4["admission_policy"])
        self.assertIn("no replacement", c4["admission_policy"])
        self.assertIn("unconditional", c4["admission_policy"])
        self.assertNotIn("positive trace-derived", c4["admission_policy"])
        self.assertIn("no saved-cycle oracle", c4["admission_policy"])
        self.assertEqual(c4["metadata"]["fixed_directory_entries"], 24)
        self.assertEqual(c4["metadata"]["fixed_directory_bits"], 1320)
        self.assertTrue(
            c4["metadata"]["charged_outside_payload_capacity_but_inside_area_power_energy"]
        )
        charged = " ".join(c4["required_charged_blocks"])
        for term in ("first-fit", "metadata", "payload storage", "clock", "leakage"):
            self.assertIn(term, charged)

    def test_incremental_state_has_fixed_bits_ports_and_g2a_macro_mapping(self) -> None:
        memories = self.contract["logical_memory_classes"]
        self.assertEqual(memories["base_relation_and_acc"]["logical_bits"], 511_200)
        self.assertEqual(
            memories["base_relation_and_acc"]["logical_bits"],
            memories["additional_acc_context"]["logical_bits"]
            + memories["epoch_slot"]["logical_bits"],
        )
        self.assertEqual(memories["additional_acc_context"]["logical_bits"], 460_800)
        self.assertIn("five independently", memories["additional_acc_context"]["organization"])
        self.assertEqual(memories["additional_acc_context"]["g2a_macros_per_context"], {
            "fakeram45_128x256": 20
        })
        self.assertEqual(memories["epoch_slot"]["logical_bits"], 50_400)
        self.assertIn("single-port 1RW", memories["epoch_slot"]["organization"])
        self.assertEqual(sum(memories["epoch_slot"]["g2a_macros_per_slot"].values()), 12)
        self.assertEqual(memories["c4_payload"]["g2a_physical_records"], 5120)
        self.assertEqual(memories["c4_payload"]["fixed_standard_cell_metadata_bits"], 1320)
        expected = {
            "C0": (0, 32), "C1": (32, 64), "C2": (24, 56),
            "C3": (44, 76), "C4": (40, 72), "C5": (85, 117),
        }
        for candidate, (incremental, total) in expected.items():
            row = self.contract["candidate_incremental_state_map"][candidate]
            self.assertEqual(row["g2a_incremental_macros"], incremental)
            self.assertEqual(row["g2a_total_macros"], total)

    def test_g2a_common_physical_policy(self) -> None:
        proxy = self.contract["g2a_open_physical_proxy"]
        fairness = self.contract["common_fairness"]
        self.assertEqual(proxy["clock_period_ns"], 5.0)
        self.assertEqual(proxy["frequency_mhz"], 200.0)
        self.assertEqual(proxy["out_dim"], 32)
        self.assertEqual(proxy["fixed_floorplan_um"]["die"], [0, 0, 2000, 1600])
        self.assertEqual(proxy["fixed_floorplan_um"]["core"], [20, 20, 1980, 1580])
        self.assertEqual(proxy["pin_placement"]["random_seed"], 42)
        self.assertEqual(proxy["base_sram_macros"]["count"], 32)
        self.assertEqual(
            sum(proxy["base_sram_macros"]["instances_by_type"].values()), 32
        )
        self.assertTrue(fairness["candidate_specific_frequency_downbinning_forbidden"])
        self.assertTrue(fairness["candidate_specific_macro_substitution_forbidden"])
        self.assertTrue(fairness["timing_failure_eliminates_candidate"])
        self.assertTrue(
            fairness["same_external_logical_transactions_and_per_tile_numeric_order"]
        )
        admission = proxy["timing_and_route_admission"]
        self.assertEqual(admission["post_route_setup_violations_max"], 0)
        self.assertEqual(admission["post_route_hold_violations_max"], 0)
        self.assertEqual(admission["detailed_route_drc_max"], 0)
        self.assertEqual(admission["failure_action"], "ELIMINATE_CANDIDATE")

    def test_all_frozen_repo_file_hashes_match(self) -> None:
        proxy = self.contract["g2a_open_physical_proxy"]
        anchored = [
            proxy["canonical_anchor"][key]
            for key in ("config", "runner", "sdc", "orfs_lock", "macro_orientation_hook")
        ]
        anchored.extend(proxy["platform_files"])
        for row in anchored:
            path = ROOT / row["path"]
            self.assertTrue(path.is_file(), row["path"])
            self.assertEqual(sha256_file(path), row["sha256"], row["path"])

    def test_orfs_commit_dirty_patch_and_modified_files_match(self) -> None:
        identity = self.contract["g2a_open_physical_proxy"]["orfs_identity"]
        orfs = ROOT / identity["path"]
        head = command_bytes("git", "rev-parse", "HEAD", cwd=orfs).decode().strip()
        self.assertEqual(head, identity["commit"])
        patch = command_bytes(
            "git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", ".", cwd=orfs
        )
        status = command_bytes("git", "status", "--porcelain=v1", cwd=orfs)
        self.assertEqual(bool(status), identity["worktree_dirty"])
        self.assertEqual(hashlib.sha256(patch).hexdigest(), identity["dirty_patch_sha256"])
        self.assertEqual(
            hashlib.sha256(status).hexdigest(), identity["porcelain_status_sha256"]
        )
        observed_paths = [line[3:] for line in status.decode().splitlines()]
        expected_paths = [row["path"] for row in identity["modified_tracked_files"]]
        self.assertEqual(observed_paths, expected_paths)
        for row in identity["modified_tracked_files"]:
            self.assertEqual(sha256_file(orfs / row["path"]), row["sha256"])

    def test_tool_binary_hashes_and_versions_match(self) -> None:
        tools = self.contract["g2a_open_physical_proxy"]["tool_identity"]
        self.assertEqual(
            sha256_file(Path(tools["openroad_binary_path"])),
            tools["openroad_binary_sha256"],
        )
        self.assertEqual(
            sha256_file(Path(tools["yosys_binary_path"])), tools["yosys_binary_sha256"]
        )
        openroad_version = command_bytes(
            tools["openroad_binary_path"], "-version", cwd=ROOT
        ).decode()
        yosys_version = command_bytes(tools["yosys_binary_path"], "-V", cwd=ROOT).decode()
        self.assertIn(tools["openroad_commit"], openroad_version)
        self.assertIn(tools["yosys_version"], yosys_version)
        self.assertIn(tools["yosys_commit"], yosys_version)

    def test_memory_inclusive_edp_formula_and_components_are_closed(self) -> None:
        edp = self.contract["memory_inclusive_edp"]
        self.assertIn("E_clock_dynamic", edp["per_trace_energy_joules"])
        self.assertIn("E_memory_dynamic", edp["per_trace_energy_joules"])
        self.assertIn("P_logic_leakage", edp["per_trace_energy_joules"])
        self.assertIn("P_memory_leakage", edp["per_trace_energy_joules"])
        self.assertTrue(edp["ratio_numerator"].startswith("C0 "))
        self.assertTrue(edp["ratio_denominator"].startswith("Cx "))
        forbidden = " ".join(edp["forbidden_accounting"])
        for term in ("memory-excluded", "idle", "clock", "leakage", "C4 metadata"):
            self.assertIn(term, forbidden)

    def test_activity_idle_and_clock_gating_are_common_and_pending_hash_is_honest(self) -> None:
        policy = self.contract["g2a_open_physical_proxy"]["activity_and_idle_policy"]
        self.assertIsNone(policy["common_activity_stimulus_sha256"])
        self.assertEqual(policy["common_activity_stimulus_status"], "[待验证]")
        self.assertTrue(policy["candidate_saif_sha256_required"])
        self.assertEqual(policy["automatic_clock_gating"], "disabled for G2a; no candidate-specific ICG or ideal gating")
        self.assertTrue(policy["activity_window_trimming_forbidden"])
        self.assertEqual(policy["weighted_trace_count"], 1200)
        self.assertEqual(
            policy["weight_source_sha256"],
            "4e8732210a64cfcb553e7f4eee3657be70cc975a38839527e4792668d6deaf6b",
        )
        self.assertEqual(policy["activity_annotation_coverage_percent_min"], 95)
        self.assertEqual(policy["unknown_toggle_count_max"], 0)
        self.assertTrue(policy["sram_pin_activity_audit_required"])

    def test_edp_thresholds_and_unique_calculator_are_frozen(self) -> None:
        edp = self.contract["memory_inclusive_edp"]
        self.assertEqual(
            [gate["exact_test"] for gate in edp["frozen_gates"]],
            ["4*EDP_C0 >= 5*EDP_C3", "19*EDP_C4 >= 20*EDP_C3"],
        )
        calculator = edp["unique_calculator"]
        path = ROOT / calculator["path"]
        self.assertTrue(path.is_file())
        self.assertEqual(sha256_file(path), calculator["sha256"])

    def test_g2b_receipt_is_fail_closed_and_has_all_required_hash_classes(self) -> None:
        receipt = self.contract["g2b_target_asic_receipt"]
        self.assertTrue(receipt["receipt_required_before_any_candidate_run"])
        self.assertTrue(receipt["run_with_any_null_required_field_forbidden"])
        self.assertTrue(receipt["same_receipt_applies_to_all_candidates"])
        preflight = receipt["preflight"]
        preflight_path = ROOT / preflight["path"]
        self.assertTrue(preflight_path.is_file())
        self.assertEqual(sha256_file(preflight_path), preflight["sha256"])
        self.assertTrue(preflight["must_pass_before_any_candidate_run"])
        required = receipt["required_fields"]
        self.assertTrue(required)
        self.assertTrue(all(value is None for value in required.values()))
        required_names = set(required)
        for field in (
            "target_library_lib_sha256",
            "target_library_db_sha256",
            "pvt_corner",
            "dc_version",
            "dc_command_sha256",
            "sta_tool_version",
            "sta_command_sha256",
            "saif_tool_version",
            "saif_generation_command_sha256",
            "ptpx_version",
            "ptpx_command_sha256",
            "sdc_sha256",
            "sram_macro_lib_sha256",
            "sram_macro_db_sha256",
            "sram_macro_lef_or_area_model_sha256",
            "sram_macro_dynamic_power_model_sha256",
            "sram_macro_leakage_model_sha256",
            "rtl_manifest_sha256",
            "filelist_sha256",
            "parameter_manifest_sha256",
            "common_activity_stimulus_sha256",
            "operating_condition_sha256",
            "wireload_or_physical_aware_policy_sha256",
        ):
            self.assertIn(field, required_names)
        rules = receipt["shared_run_rules"]
        self.assertTrue(rules["candidate_specific_frequency_downbinning_forbidden"])
        self.assertTrue(rules["candidate_specific_macro_substitution_forbidden"])
        self.assertTrue(rules["timing_failure_eliminates_candidate"])

    def test_missing_target_inputs_block_signoff(self) -> None:
        validation = self.contract["constraint_validation"]
        self.assertTrue(validation["clock_constraint_present"])
        self.assertFalse(validation["absolute_area_budget_present"])
        self.assertFalse(validation["absolute_power_budget_present"])
        self.assertFalse(validation["target_library_pvt_present"])
        self.assertTrue(validation["architecture_signoff_entry"].startswith("HARD_FAIL"))
        missing = self.contract["g2b_target_asic_receipt"]["missing_now"]
        self.assertTrue(all(value == "[待验证]" for value in missing.values()))

    def test_chinese_document_and_run_state_record_hard_boundaries(self) -> None:
        doc = DOC_PATH.read_text(encoding="utf-8")
        state = STATE_PATH.read_text(encoding="utf-8")
        for phrase in (
            "开放物理代理",
            "不能写成 ASIC PPA",
            "C4 first-fit",
            "时序失败直接淘汰",
            "目标库与绝对功耗预算",
            "动态能量",
            "时钟能量",
            "漏电能量",
        ):
            self.assertIn(phrase, doc)
        self.assertIn("design_name: Local5_EREP_v4_G2_preimplementation_evaluation_contract", state)
        self.assertIn("overall:     COMPLETE_PREIMPLEMENTATION_CONTRACT_ONLY", state)


if __name__ == "__main__":
    unittest.main()
