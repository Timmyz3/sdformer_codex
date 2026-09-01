#!/usr/bin/env python3
"""Fail-closed M1772 source/runtime checker; this file never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1701 = HW / "dc_handoff/runs/m1701_m1695_c1_tool_entity_repair_dc_r1_20260901.failed_or_incomplete.2502881.quarantine"
M1722 = HW / "dc_handoff/runs/m1722_m1701_c1_salvage_formality_pt_r1_20260901.failed_or_incomplete.quarantine"
M1733 = HW / "dc_handoff/runs/m1733_m1722_m1701_c1_formality_reuse_pt_only_r1_20260901.failed_or_incomplete.quarantine"
M1590 = HW / "results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901"
M1590_LEDGER = M1590 / "ep34_c1_support16_rows.memh"
M1590_RESULT = M1590 / "m1579_ep34_c1_same_ledger_cycle_model_result_r1.json"
M1743 = HW / "contracts/m1743_m1742_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_release_r1_20260901.json"
TIMING_RESULT = HW / "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901"
TIMING_RECEIPT = TIMING_RESULT / "receipt.json"
M1745 = HW / "reviews/m1745_m1739_m1701_c1_public_port_mapped_production_energy_source_hammer_r1_20260901"
M1745_REVIEW = M1745 / "review.json"
M1772_FAILURE = HW / "reviews/m1766_m1759_m1757_c1_unit_delay_saif_failure_receipt_r1_20260902"
M1772_FAILURE_RECEIPT = M1772_FAILURE / "receipt.json"
M1771 = HW / "reviews/m1771_m1766_c1_two_bank_warmup_correction_overlay_r1_20260902"
M1771_CORRECTION = M1771 / "correction.json"
DESIGN = "m935_m912_three_stage_exact_parent_match_product_capture_island"
NET = M1701 / ("netlist/" + DESIGN + "_m1695_fastmin_hold_closed_mapped.v")
SDC = M1701 / ("netlist/" + DESIGN + "_m1695_fastmin_hold_closed_mapped.sdc")
TB = HW / "dc_handoff/tb/tb_m1772_c1_m1701_two_bank_public_warmup_energy.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1772_c1_m1701_two_bank_public_warmup_energy.f"
UCLI = HW / "dc_handoff/scripts/m1772_c1_m1701_two_bank_public_warmup_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1750_c1_m1701_public_port_mapped_whole_component_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1772_m1701_c1_two_bank_public_warmup_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1772_c1_m1701_two_bank_public_warmup_energy_source.py"
CONTRACT = HW / "contracts/m1772_m1701_c1_two_bank_public_warmup_energy_source_contract_r1_20260902.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
CELL_V = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
STD_TT = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
STD_SS = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
MACRO_ROOT = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821")
MACRO_V = MACRO_ROOT / "ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
MACRO_DB = MACRO_ROOT / "ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
TOP = "tb_m1772_c1_m1701_two_bank_public_warmup_energy"
SAIF_SCOPE = TOP + ".dut"

CLAIMS = dict((key, False) for key in (
    "launch_authorized", "launch_executed", "mapped_vcs", "production_saif",
    "ptpx", "logic_power", "component_energy", "total_c1_energy",
    "energy_per_frame", "performance", "system_speedup", "paper_ppa_ready",
    "headline"))

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    NET: "d990bb416370fd07a1c241849e2fa494b94a179b47687a1a3ff2b1ab92c255e8",
    SDC: "04cb67affcfd629cd9540d789110107888d9ae956168dae37c34aa44c15e2d62",
    M1701 / "SHA256SUMS": "f132ca694a747e2da51708fb03f2ba6c84360606b4d38d2cc2e97998f9f3a022",
    M1701 / "SHA256SUMS.seal.sha256": "a65f2901b4ab4339a94bb032b9412b652a77afc50d1c72b403c8bd44d15f55a6",
    M1722 / "SHA256SUMS": "415521aee26a6c4da176bd2d22d36e0bed8af458fc491dbf73c02b7d52a378fc",
    M1722 / "SHA256SUMS.seal.sha256": "8beb486ec43dd855548a76ae10103b4d84339e11b47f3d672860c18164b6deca",
    M1722 / "formality/FORMALITY_INTERNAL_COMPLETE.txt": "c3a1837201846cb13e9e45dce3ff36b33e1c319b7136c3479c62677fdcb41f6c",
    M1722 / "formality/reports/formality_status.rpt": "1ce8c6c17a4890be8a54c86b63b8e36c958398f6b7a92fb9881df2b3b73ba19d",
    M1733 / "SHA256SUMS": "9093eb197b4a837471f5edda6893e8b9806cb734c0995e299fee3aa4909aa614",
    M1733 / "SHA256SUMS.seal.sha256": "d7b93e1b15e96ca1b3d3a86064931c52a770a59ef62934b72834e8104f9bde3e",
    M1733 / "ptsta/PTSTA_INTERNAL_COMPLETE.txt": "fcdf45d03c8b1c6cee84bf627f27cca01847a0a6547d68f320d28ac2263d1a09",
    M1733 / "ptsta/reports/timing_summary_machine.txt": "9e5160a8381fb839aac7e8df409667a5d9486ee6a280d5a8e808d1b40b6d3947",
    M1590 / "SHA256SUMS": "50881cd508bec486e6527ec483e451a1f03b7aba1fea7a047d54f1c1f5f08707",
    M1590 / "SHA256SUMS.seal.sha256": "9e7de8638deb0875ba7e2bd27c20859905fdbf441e8cce9759b32bb06b8b3127",
    M1590_LEDGER: "daa6265115df9c0bae5d96e5a133a4b5fbc9786de75598e53ab2e5812bfdb835",
    M1590_RESULT: "facfecaf3b25a4c79299517de31283ed3815af26a5dd87c91a6985f6fc68516f",
    M1743: "3c623618115c4ecf2e4bfec6efe167c90296825428ce87e16e6d52bd79216921",
    Path(str(M1743) + ".sha256"): "7d481d605bffd1386b8926e709424a2c78b3f78eff340caf1727dbe7ec84cfe1",
    Path(str(M1743) + ".sha256.seal.sha256"): "7a52c2e7692b62857dfe1d2b1bd9e2825372a0fc839822abf086d4837bbcf112",
    TIMING_RESULT / "SHA256SUMS": "d3f2e14a6f6c0600abce2f5af2479402d41986736e3d9c32c6044e4225f64c75",
    TIMING_RESULT / "SHA256SUMS.seal.sha256": "6f2b17f7016665cd663b9694a1ccbd29fa551ecf75ba29aa52c4bb56c5769b38",
    TIMING_RECEIPT: "0b3ee22f9369a38eb83f674a4f1eb73fac39757ee85a3e1aeebe032bd0c76a1e",
    M1745_REVIEW: "44fca21fde5163ae39f249f5a485c5f2d4953910d8ff76e911aff6a543373359",
    M1745 / "independent_hammer.py": "f451d66fb1306db9639174505d4daa33c89fb5b40c5d319c4a78d27d616a185d",
    M1745 / "SHA256SUMS": "c5b1f83b618ab8aadff16dc9e2a8f6498a852c66559d7a55171f93831bf3595a",
    M1745 / "SHA256SUMS.seal.sha256": "f81c8c0166da2d2e6ce7a99aa469bad9d800193edab865c33fe64ab6753c0404",
    M1772_FAILURE_RECEIPT: "1f9d843b203cf020733ee3fb44c133920b6ddf14a459b6db3b27dc9c682f8946",
    M1772_FAILURE / "forensic_counts.json": "eee15cbe8a41f027ae928f6e3474652030782fc7b7cd26b1affd79bc5cd07091",
    M1772_FAILURE / "operator_attestation.md": "36e1ba969065885df5ba6b9603e96c817f9f9828de0a8c13c151a5d3bea86b42",
    M1772_FAILURE / "SHA256SUMS": "c7c782669ed910df1c61c8b757a59fdb6a9471fbbf4c7e22b5e71b596ff8169a",
    M1772_FAILURE / "SHA256SUMS.seal.sha256": "46aa4e2164c1ff84134418fa305b22c24588c54c5ea3404858d5f59777ea57c2",
    M1771_CORRECTION: "878a67435502c4b2e76c0964ea9d919f278f747385a4cfae5775b9d22d78b192",
    M1771 / "correction.md": "f2280410653c08065ec9da5bca5c4be02accc3847a942b1158c96ad7651a1d63",
    M1771 / "SHA256SUMS": "f931cc0cdf2818585d08b4d27fd1411b5e3b94e404522c1713da42496eeb68ac",
    M1771 / "SHA256SUMS.seal.sha256": "6b3bed433ef9884772337c468d4e9375d9c1e5a669c5d6aa3035feb55d28d024",
    CELL_V: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    STD_TT: "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
    STD_SS: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    MACRO_V: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    MACRO_DB: "cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "JSON not regular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def active_lines(path):
    return [raw.split("#", 1)[0].strip()
            for raw in Path(path).read_text().splitlines()
            if raw.split("#", 1)[0].strip()]


def verify_seal(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest")
        need(sha(root / rel) == digest, "manifest member drift " + name)
        listed.add(name)


def strip_sv_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def validate_sources():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    for root in (M1701, M1722, M1733, M1590, TIMING_RESULT, M1745,
                 M1772_FAILURE, M1771):
        verify_seal(root)
    failed_campaign = strict_json(M1772_FAILURE_RECEIPT)
    need(failed_campaign.get("status") ==
         "SEALED_M1757_FAILURE__ATTEMPT_CONSUMED__FUNCTIONAL_PASS__SAIF_TX_REJECT__PTPX_ZERO__NO_AUTORETRY"
         and failed_campaign.get("observed_execution", {}).get(
             "canonical_result_absent") is True
         and failed_campaign.get("observed_execution", {}).get(
             "ptpx_runs") == 0,
         "M1766 failed campaign disposition drift")
    correction = strict_json(M1771_CORRECTION)
    corrected_geometry = correction.get("corrected_successor_geometry", {})
    protocol_basis = corrected_geometry.get("protocol_basis", {})
    need(correction.get("status") ==
         "PASS_M1771_ADDITIVE_CORRECTION__M1766_FAILURE_STANDS__SINGLE_WARMUP_REVOKED__TWO_BANK_PING_PONG_REQUIRED"
         and corrected_geometry.get("warmup_bank0_epoch") == 5943
         and corrected_geometry.get("warmup_bank1_epoch") == 5944
         and corrected_geometry.get("measured_epoch") == 5945
         and corrected_geometry.get("same_64_masks_for_all_three_tasks") is True
         and corrected_geometry.get("public_only") is True
         and corrected_geometry.get("hierarchical_state_access") is False
         and protocol_basis.get("per_task_counters_clear_at_execution_start") is True,
         "M1771 correction disposition drift")
    failed_review = strict_json(M1745_REVIEW)
    need(failed_review.get("status") ==
         "FAIL_M1745_P0_DO_NOT_AUTHORIZE_M1746__ADDITIVE_PTPX_MACRO_POWER_REPAIR_REQUIRED"
         and failed_review.get("p0_count") == 1
         and failed_review.get("m1746_authorized") is False,
         "M1745 fail disposition drift")
    formal = (M1722 / "formality/reports/formality_status.rpt").read_text()
    need("Verification SUCCEEDED" in formal
         and "16549 Passing compare points" in formal
         and re.search(r"Failing \(not equivalent\)\s+0\s+0\s+0\s+0\s+0\s+0\s+0\s+0", formal),
         "M1722 Formality subproof drift")
    timing = (M1733 / "ptsta/reports/timing_summary_machine.txt").read_text()
    for token in ("setup_wns_ns=0.027871", "hold_wns_ns=0.001827",
                  "setup_violating_paths=0", "hold_violating_paths=0"):
        need(token in timing, "M1733 timing subproof drift " + token)
    m1590 = strict_json(M1590_RESULT)
    need(m1590.get("ledger", {}).get("rows") == 51840000
         and m1590.get("conservation", {}).get("active_rows_per_output_block")
             == 25304213,
         "M1590 support denominator drift")
    timing = strict_json(TIMING_RECEIPT)
    need(timing.get("schema") ==
         "m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_receipt_r1_v1"
         and timing.get("status") ==
         "PASS_CANONICAL_C1_FORMALITY_AND_INDEPENDENT_PT_PRELAYOUT",
         "M1743 timing receipt schema/status drift")
    need(timing.get("prime_time") == {
        "clock_period_ns": "3.000", "hold_tns_ns": "0.0",
        "hold_uncertainty_ns": "0.050", "hold_violating_paths": "0",
        "hold_wns_ns": "0.001827", "macro_count": "9",
        "setup_tns_ns": "0.0", "setup_uncertainty_ns": "0.200",
        "setup_violating_paths": "0", "setup_wns_ns": "0.027871"},
        "M1743 PrimeTime values drift")
    formality = timing.get("formality", {})
    need(formality.get("verification_succeeded") is True
         and formality.get("passing_compare_points") == 16549
         and [formality.get(key) for key in
              ("failing", "aborted", "unverified", "unmatched")]
             == [0, 0, 0, 0]
         and formality.get("macro_instances_per_side") == 9,
         "M1743 Formality values drift")
    need(timing.get("scope") == {"ideal_clock": True, "macro_count": 9,
        "parasitics": False, "power_or_energy": False, "prelayout": True,
        "wireload": "ZeroWireload"}
        and timing.get("claim_boundary") == {
            "cycle_speedup": False, "dc": False, "energy": False,
            "formality": True, "headline": False, "independent_pt": True,
            "paper_citable": True, "paper_ppa_ready": False,
            "power": False, "system_speedup": False},
         "M1743 scope/claim boundary drift")

    expected_filelist = [str(CELL_V), str(MACRO_V), str(NET), str(TB)]
    need(active_lines(FILELIST) == expected_filelist, "filelist/order drift")
    tb_text = TB.read_text()
    active_tb = strip_sv_comments(tb_text).lower()
    need("force " not in active_tb and "release " not in active_tb,
         "TB drives internal state")
    need("dut." not in active_tb, "TB reads hierarchical DUT state")
    for token in ("load_public_task", "issue_request_source_valid",
                  "psum_write_data",
                  "PASS_M1772_C1_M1701_TWO_BANK_WARMUP_MAPPED_DIRECTED_COMPONENT_ACTIVITY",
                  "count_macro_reads", "count_macro_writes"):
        need(token in tb_text, "TB omits " + token)
    for token in ("WARMUP0_EPOCH = 16'd5943",
                  "WARMUP1_EPOCH = 16'd5944",
                  "TEST_EPOCH = 16'd5945",
                  "psum_write_ready = 1'b0",
                  "row_complete_ready = 1'b0",
                  "load_public_task(WARMUP1_EPOCH, 1'b0)",
                  "load_public_task(TEST_EPOCH, 1'b1)",
                  "COVERAGE_M1772_TWO_BANK_PUBLIC_WARMUP"):
        need(token in tb_text, "TB warmup geometry omits " + token)
    warmup_order = (
        "load_public_task(WARMUP0_EPOCH, 1'b0)", "wait (execute_busy)",
        "psum_write_ready = 1'b0", "row_complete_ready = 1'b0",
        "load_public_task(WARMUP1_EPOCH, 1'b0)",
        "psum_write_ready = 1'b1", "row_complete_ready = 1'b1",
        "load_public_task(TEST_EPOCH, 1'b1)")
    cursor = 0
    for token in warmup_order:
        cursor = tb_text.find(token, cursor)
        need(cursor >= 0, "TB public warmup order drift " + token)
        cursor += len(token)
    need("count_psum_commits != 64" in tb_text
         and "count_row_completions != 64" in tb_text,
         "TB per-task counter contract drift")
    need(active_lines(UCLI) == [
        "power -gate_level all mda sv", "power " + SAIF_SCOPE, "run",
        "power -enable", "run", "power -disable",
        "power -report $::env(M1772_SAIF_FILE) 1e-9 " + SAIF_SCOPE, "quit"],
        "UCLI exact scope/order drift")
    pt = PT_TCL.read_text()
    for token in ("expected_macro_count 9", "read_saif -strip_path",
                  "M1750_FAIL_EXACT_NET_ANNOTATION_GATE",
                  "M1750_FAIL_EXACT_LEAF_ANNOTATION_GATE",
                  "ptpx_whole_mapped_c1_including_9macro_liberty.rpt",
                  "primary_report=whole_mapped_c1_top_including_9macro_liberty",
                  "top_minus_macro=false",
                  "ptpx_plus_datasheet_sram_combined=false",
                  "parent_sram_datasheet_is_separate_alternative_sensitivity=true",
                  "corner_classification=mixed_corner_component_estimate",
                  "standard_cell_power_library=TT_0p9V_25C",
                  "parent_sram_macro_liberty=SSG_0p9V_125C",
                  "not_single_corner_signoff=true",
                  "not_energy_per_frame=true"):
        need(token in pt, "PTPX Tcl omits " + token)
    active_pt = strip_sv_comments(pt)
    need("report_power $macro_cells" not in active_pt,
         "selected macro summary is forbidden")
    need("ptpx_nine_parent_macros" not in active_pt,
         "macro subtraction report is forbidden")
    runner_text = RUNNER.read_text()
    for forbidden in ("CHECK." + "combine_power(", "--macro" + "-power",
                      "ptpx_nine_parent_macros_liberty_diagnostic.rpt",
                      "known_component_workload_energy_pj"):
        need(forbidden not in runner_text, "runner retains forbidden accounting " + forbidden)
    for token in ("whole_component_power(",
                  "ptpx_whole_mapped_c1_including_9macro_liberty.rpt",
                  "mixed_corner_component_estimate",
                  "ptpx_plus_datasheet_sram_combined\": False"):
        need(token in runner_text, "runner omits repaired accounting " + token)
    need(runner_text.count('"+define+UNIT_DELAY"') == 1,
         "runner must contain exactly one UNIT_DELAY compiler define")
    for forbidden in ("+notimingcheck", "+no_notifier", "+nospecify",
                      "+initreg", "+define+no_warning",
                      "+define+NO_INPUT_FLOATING_CHECK",
                      "m1750_c1_public_port_mapped_component_energy_r1_20260901.private_build"):
        need(forbidden not in runner_text,
             "runner contains forbidden bypass/reuse token " + forbidden)
    macro_text = MACRO_V.read_text()
    for token in ("provides UNIT_DELAY mode for the fast function",
                  "All timing values in the specification are not checked in the",
                  "`ifdef UNIT_DELAY", "specify"):
        need(token in macro_text, "foundry UNIT_DELAY contract drift " + token)
    for path in (TB, FILELIST, UCLI, PT_TCL, RUNNER):
        forbidden_initializer = "init" + "reg"
        need(forbidden_initializer not in path.read_text().lower(),
             "forbidden gate initializer " + str(path))
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1772_m1701_c1_two_bank_public_warmup_energy_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1766_FAILURE_AND_M1771_CORRECTION_BOUND__TWO_BANK_PUBLIC_WARMUP__UNIT_DELAY_FUNCTIONAL__M1743_TIMING_PINNED__M1773_REVIEW_AND_M1774_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIMS, "source claim promotion")
    accounting = contract.get("energy_accounting", {})
    need(accounting.get("primary") ==
         "whole mapped C1 top including nine SRAM macro Liberty models"
         and accounting.get("top_minus_macro") is False
         and accounting.get("ptpx_plus_datasheet_sram_combined") is False
         and accounting.get("corner_classification") ==
             "mixed_corner_component_estimate"
         and accounting.get("standard_cell_power_library") == "TT 0.9V 25C"
         and accounting.get("parent_sram_macro_liberty") == "SSG 0.9V 125C",
         "contract accounting/corner boundary drift")
    functional = contract.get("gate_simulation", {})
    need(functional.get("mode") == "UNIT_DELAY_functional"
         and functional.get("timing_simulation") is False
         and functional.get("independent_pt_timing") is True
         and functional.get("unit_delay_define_count") == 1,
         "functional gate simulation boundary drift")
    warmup = contract.get("two_bank_public_warmup", {})
    need(warmup.get("epochs") == [5943, 5944, 5945]
         and warmup.get("measurement_epoch") == 5945
         and warmup.get("warmup_inside_saif") is False
         and warmup.get("hierarchical_drive") is False,
         "contract warmup boundary drift")
    rows = contract.get("source_files")
    need(isinstance(rows, list), "source inventory")
    mapping = dict((row.get("path"), row.get("sha256")) for row in rows)
    expected = (TB, FILELIST, UCLI, PT_TCL, RUNNER, CHECKER, TEST)
    need(len(mapping) == len(rows) and set(mapping) == set(
        path.relative_to(HW).as_posix() for path in expected),
        "source inventory paths")
    for path in expected:
        need(mapping[path.relative_to(HW).as_posix()] == sha(path),
             "source inventory SHA " + str(path))
    return {"schema": "m1772_c1_energy_source_check_r1_v1",
            "status": "PASS_M1772_TWO_BANK_PUBLIC_WARMUP_SOURCE_ONLY_NO_EDA",
            "mapped_netlist_sha256": sha(NET), "mapped_sdc_sha256": sha(SDC),
            "public_port_only": True, "new_rtl_wrapper": False,
            "gate_simulation_mode": "UNIT_DELAY_functional",
            "timing_simulation": False,
            "independent_pt_timing": True,
            "warmup_epochs": [5943, 5944], "measurement_epoch": 5945,
            "claim_boundary": CLAIMS}


COUNTER_PATTERN = re.compile(
    r"M1772_PUBLIC_COUNTERS cycles=([1-9][0-9]*) issue_accepts=([1-9][0-9]*)"
    r" parent_edges=([1-9][0-9]*) macro_reads=([0-9]+) macro_writes=([0-9]+)"
    r" forwards=([0-9]+) dead_write_elisions=([0-9]+)"
    r" psum_commits=64 row_completions=64")


def validate_runtime(log_path):
    text = Path(log_path).read_text(errors="strict")
    need(text.count("PASS_M1772_C1_M1701_TWO_BANK_WARMUP_MAPPED_DIRECTED_COMPONENT_ACTIVITY") == 1,
         "runtime PASS absent/duplicated")
    need(text.count("COVERAGE_M1772_TWO_BANK_PUBLIC_WARMUP") == 1
         and "bank0_epoch=5943 bank1_epoch=5944" in text,
         "two-bank public warmup cover absent/duplicated")
    need(not any(token in text for token in ("$fatal", "Assertion failed", "Error-[")),
         "runtime failure token")
    hits = COUNTER_PATTERN.findall(text)
    need(len(hits) == 1, "public counters absent/duplicated")
    values = [int(item) for item in hits[0]]
    cycles, issue, parents, reads, writes, forwards, dead = values
    need(reads + forwards == parents, "parent conservation")
    need(writes + dead == 64, "write/elision conservation")
    return {"status": "PASS_M1772_PUBLIC_PORT_RUNTIME",
            "measurement_cycles": cycles, "issue_accepts": issue,
            "parent_edges": parents, "macro_reads": reads,
            "macro_writes": writes, "forwards": forwards,
            "dead_write_elisions": dead, "log_sha256": sha(log_path)}


def strip_saif_block_comments(text):
    """Strip C block comments only outside strings; reject truncation."""
    output = []
    index = 0
    count = 0
    while index < len(text):
        if text[index] == '"':
            start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            need(index <= len(text) and index > start + 1
                 and text[index - 1] == '"', "unterminated SAIF string")
            output.append(text[start:index])
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            need(end >= 0, "unterminated SAIF block comment")
            output.append(" ")
            index = end + 2
            count += 1
        else:
            output.append(text[index])
            index += 1
    return "".join(output), count


def sexpr_tokens(text):
    return re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text)


def parse_saif(text):
    cleaned, comment_count = strip_saif_block_comments(text)
    tokens = sexpr_tokens(cleaned)
    pos = [0]
    def parse_one():
        need(pos[0] < len(tokens) and tokens[pos[0]] == "(", "malformed SAIF")
        pos[0] += 1
        node = []
        while pos[0] < len(tokens) and tokens[pos[0]] != ")":
            if tokens[pos[0]] == "(":
                node.append(parse_one())
            else:
                node.append(tokens[pos[0]])
                pos[0] += 1
        need(pos[0] < len(tokens), "unterminated SAIF")
        pos[0] += 1
        return node
    root = parse_one()
    need(pos[0] == len(tokens) and root and root[0] == "SAIFILE", "SAIF root")
    return root, comment_count


def forms(node, tag):
    return [item for item in node[1:]
            if isinstance(item, list) and item and item[0] == tag]


def all_forms(node, tag):
    found = []
    if isinstance(node, list):
        if node and node[0] == tag:
            found.append(node)
        for item in node:
            if isinstance(item, list):
                found.extend(all_forms(item, tag))
    return found


def direct_instance(node, name):
    hits = [item for item in forms(node, "INSTANCE")
            if len(item) >= 2 and item[1].lstrip("\\") == name]
    need(len(hits) == 1, "SAIF instance absent/duplicated " + name)
    return hits[0]


def validate_saif(path, cycles, expected_activity_forms=117690):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and cycles > 0,
         "SAIF input invalid")
    root, comment_count = parse_saif(path.read_text(errors="strict"))
    duration = forms(root, "DURATION")
    need(len(duration) == 1 and len(duration[0]) == 2, "SAIF duration")
    duration_ns = float(duration[0][1])
    need(math.isfinite(duration_ns)
         and abs(duration_ns - cycles * 3.0) <= 1e-6,
         "SAIF duration/cycle mismatch")
    need(len(forms(root, "INSTANCE")) == 1, "SAIF top hierarchy count")
    top = direct_instance(root, TOP)
    need(len(forms(top, "INSTANCE")) == 1, "SAIF DUT hierarchy count")
    dut = direct_instance(top, "dut")
    scratch = direct_instance(dut, "u_parent_scratch")
    need(scratch[1].lstrip("\\") == "u_parent_scratch",
         "mapped scratch hierarchy absent")
    groups = dict((tag, all_forms(dut, tag))
                  for tag in ("T0", "T1", "TX", "TC", "IG"))
    need(type(expected_activity_forms) is int and expected_activity_forms > 0,
         "expected activity count domain")
    need(all(len(value) == expected_activity_forms
             for value in groups.values()), "mapped DUT activity-form count")
    tx = groups["TX"]
    need(all(len(item) == 2 and float(item[1]) == 0.0 for item in tx),
         "SAIF contains unknown activity")
    for t0, t1, unknown in zip(groups["T0"], groups["T1"], groups["TX"]):
        need(all(len(item) == 2 for item in (t0, t1, unknown))
             and abs(float(t0[1]) + float(t1[1]) + float(unknown[1])
                     - duration_ns) <= 1e-6,
             "SAIF activity duration conservation")
    tc = groups["TC"]
    need(tc and any(len(item) == 2 and float(item[1]) > 0.0 for item in tc),
         "mapped DUT SAIF has no toggles")
    return {"status": "PASS_M1772_EXACT_WINDOW_TWO_BANK_WARMED_DUT_ONLY_SAIF",
            "cycles": cycles, "duration_ns": duration_ns,
            "tx_nonzero": 0, "saif_scope": SAIF_SCOPE,
            "activity_forms_per_tag": expected_activity_forms,
            "block_comments_skipped_outside_strings": comment_count,
            "saif_sha256": sha(path)}


POWER_FIELDS = ("Net Switching Power", "Cell Internal Power",
                "Cell Leakage Power", "Total Power")


def parse_power(path):
    text = Path(path).read_text(errors="strict")
    need("Report : Averaged Power" in text and "-unit mW" in text,
         "power report mode/unit")
    values = {}
    for field in POWER_FIELDS:
        hits = re.findall(re.escape(field) + r"\s*=\s*([0-9.eE+-]+)", text)
        need(len(hits) == 1, "power field " + field)
        values[field] = float(hits[0])
        need(math.isfinite(values[field]) and values[field] >= 0.0,
             "invalid power field")
    return values


def whole_component_power(top_path, cycles, reads, writes):
    need(cycles > 0 and reads >= 0 and writes >= 0, "metric domain")
    top = parse_power(top_path)
    subtotal = (top["Net Switching Power"] + top["Cell Internal Power"]
                + top["Cell Leakage Power"])
    tolerance = max(1e-8, 1e-6 * max(1.0, top["Total Power"]))
    need(abs(subtotal - top["Total Power"]) <= tolerance,
         "whole-component power is not conserved")
    duration_ns = cycles * 3.0
    whole_energy_pj = top["Total Power"] * duration_ns
    read_energy_pj = reads * 94.57074
    write_energy_pj = writes * 90.65763
    leakage_energy_pj = 0.54009423 * duration_ns
    return {
        "status": "PASS_M1772_WHOLE_MAPPED_COMPONENT_ENERGY_PENDING_RESULT_HAMMER",
        "cycles": cycles, "duration_ns": duration_ns,
        "ptpx_whole_mapped_c1_including_9macro_liberty": {
            "net_switching_power_mw": top["Net Switching Power"],
            "cell_internal_power_mw": top["Cell Internal Power"],
            "cell_leakage_power_mw": top["Cell Leakage Power"],
            "total_power_mw": top["Total Power"],
            "directed_window_energy_pj": whole_energy_pj,
            "component_total_conserved": True,
            "macro_count": 9},
        "parent_sram_datasheet_alternative_sensitivity": {
            "macro_count": 9, "reads": reads, "writes": writes,
            "read_pj_per_1152b_access": 94.57074,
            "write_pj_per_1152b_access": 90.65763,
            "leakage_power_mw": 0.54009423,
            "dynamic_energy_pj": read_energy_pj + write_energy_pj,
            "leakage_energy_pj": leakage_energy_pj,
            "alternative_sram_energy_pj":
                read_energy_pj + write_energy_pj + leakage_energy_pj,
            "role": "separate_alternative_sensitivity_only",
            "added_to_ptpx_whole_component": False},
        "ptpx_plus_datasheet_sram_combined": False,
        "claim_boundary": {"directed_component_workload": True,
            "whole_mapped_component_ptpx_including_9macro_liberty": True,
            "standard_cell_logic_total": False,
            "parent_sram_datasheet_alternative_sensitivity": True,
            "top_minus_macro": False,
            "ptpx_plus_datasheet_sram_combined": False,
            "total_c1_schedule_energy": False, "energy_per_frame": False,
            "system_energy": False, "silicon_measurement": False}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "runtime", "saif", "power"), required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--top-power", type=Path)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--reads", type=int)
    parser.add_argument("--writes", type=int)
    args = parser.parse_args()
    if args.mode == "source":
        value = validate_sources()
    elif args.mode == "runtime":
        need(args.log is not None, "runtime log absent")
        value = validate_runtime(args.log)
    elif args.mode == "saif":
        need(args.log is not None and args.cycles is not None,
             "SAIF path/cycles absent")
        value = validate_saif(args.log, args.cycles)
    else:
        need(None not in (args.top_power, args.cycles, args.reads, args.writes),
             "power inputs absent")
        value = whole_component_power(args.top_power, args.cycles,
                                      args.reads, args.writes)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
