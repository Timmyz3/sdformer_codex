#!/usr/bin/env python3
"""Independently validate the M43+M62 hammer review and its source evidence."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REVIEW = HERE / "m43_m62_independent_hammer_review.json"
M43 = Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/"
    "m43_parent_delta_p8_l96_r3_exact_sha_synopsys_3p000ns_r2_20260823")
M62_VCS = HW / "dc_handoff/runs/m62_p48_directed_vcs_r1_20260823"
M62_DC = HW / "dc_handoff/runs/m62_p48_dc_3p000ns_r1b_20260823"


EXPECTED = {
    "m43_receipt": "0aff75f5e03fb96dbb353b88d6cffdcd077ceebba0d28519c93a0e470684cc75",
    "m43_seal": "af2e6ee17b7bfcbf5c2b2bfd04287a3a6978fc5ad32ce9d7da9fd038c4ef11e3",
    "m43_rtl": "e70239b1ec9a7d4541b0ae8d0a8f55e252fa6c804b364ab126d8201e108e0deb",
    "m43_contract": "27e3881239afa48fb5d7257ebc8401af819e56f10edc55498793c277ee242da8",
    "m43_vcs_receipt": "3e416d615829c9b82206547ef3ab23178bfe3e01eeb0b0ff5a789bec116fe51a",
    "m62_rtl": "4ba42f70e664d7fc30716a04678acc955612008a2be5a0dad693778bbd776f0f",
    "m62_sva": "16a7907340711ab722ce1f2ec978da776004befef791d03d8bc34893d128cd05",
    "m62_tb": "f6b9a4ad2967af302a093b16f0cef37a99b389486e1cfaa86568ca548a6392e8",
    "m62_contract": "cc70780bcd539eec5badf420f4b8c2e58e6c4bd6c402d9b74041cce836233b24",
    "m62_receipt": "e003d9efe60c46323f4a7bc69350d0ab1b083dad868d4dae76d70b16f9c71a6a",
    "m62_compile_log": "120e0715ea005ff90c0c4dc443a2fb23f643506376ecf2ecd280c24785a10f0a",
    "m62_sim_log": "7d03a20260ee6837878992a4fb4caad11a882dae1afac89681b9139a703fda49",
    "m62_dc_admission": "2b0000d9f48e0c9d9b3f4883b383c3a7a6db4f7ca077b840625aef790fdf49e7",
    "m62_dc_log": "f6d5e9318bc94af9aa9ec1ed8e07ccc41c1a4820dc0eb5c8265a540fd633b8c0",
    "m62_dc_area": "0eae6b6d23ff9816b01cffed3a1a70e33713d58243673c8ea597e514decb72c3",
    "m62_dc_setup": "750fc46caa9a0b68725619ef6d87276db908f47f0168087c25f946735fb67bc4",
    "m62_dc_hold": "66e0a6de2d26ed8d33e64ed6755effd65f8fbc99df421eaa28b71dcb84745df8",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(pattern, text, label, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    require(match is not None, "missing {}".format(label))
    return cast(match.group(1))


def check_hash(path, expected, label):
    observed = sha256_path(path)
    require(observed == expected,
            "{} SHA drift {} != {}".format(label, observed, expected))
    return observed


def validate_completion_seal():
    lines = (M43 / "completion_seal.sha256").read_text(
        encoding="utf-8").splitlines()
    require(len(lines) >= 5, "M43 completion seal too short")
    for line in lines:
        expected, relative = line.split(None, 1)
        relative = relative.lstrip(" *")
        check_hash(M43 / relative, expected, "M43 sealed {}".format(relative))
    return len(lines)


def validate_review_core(review):
    require(review["schema"] == "m43_m62_independent_hammer_review_v1",
            "review schema drift")
    require(review["status"] ==
            "SHARE_WITH_CAVEATS_NOT_DATE_HEADLINE_READY",
            "review status drift")
    require(review["issues"]["P0"] == [] and
            len(review["issues"]["P1"]) == 4 and
            len(review["issues"]["P2"]) == 5,
            "review severity inventory drift")
    expected_scores = {"m43": 74, "m62": 57, "combined_story": 64}
    for name, expected_score in expected_scores.items():
        row = review["scores"][name]
        require(row["date_prosperity_phi_completeness_score"] == expected_score,
                "{} score drift".format(name))
        require(sum(row["subscores"].values()) == expected_score,
                "{} subscores do not sum".format(name))
    require(review["m62_findings"]["ratio_boundary"] == {
        "m60_dense_over_bounded_signed_event_plus_commit":
            3.086389221379181,
        "is_head_kernel_opportunity_ratio": True,
        "is_rtl_measured_speedup": False,
        "is_full_network_or_system_speedup": False,
        "current_m60_m61_m62_artifacts_mislabel_it_as_system_speedup": False,
    }, "3.086x boundary drift")
    require(review["admission_gate_for_next_milestone"]["current_result"] ==
            "NO_GO_FOR_DATE_HEADLINE_GO_FOR_INTEGRATED_NEXT_MILESTONE",
            "next admission decision drift")


def validate_m43(review):
    check_hash(M43 / "m43_r3_synopsys_receipt.json",
               EXPECTED["m43_receipt"], "M43 receipt")
    check_hash(M43 / "completion_seal.sha256", EXPECTED["m43_seal"],
               "M43 completion seal")
    sealed_files = validate_completion_seal()
    snap = M43 / "snapshot/inputs/hw_autoresearch_nts07"
    check_hash(snap / "rtl_m43/qfit_parent_delta_p8_l96_multicontext.sv",
               EXPECTED["m43_rtl"], "M43 candidate RTL")
    check_hash(snap / "contracts/m43_r3_exact_sha_synopsys_contract_r2_20260823.json",
               EXPECTED["m43_contract"], "M43 Synopsys contract")
    check_hash(snap / "contracts/m43_r2_exact_sha_vcs_receipt_r1_20260823.json",
               EXPECTED["m43_vcs_receipt"], "M43 VCS receipt")
    receipt = load_json(M43 / "m43_r3_synopsys_receipt.json")
    require(receipt["status"] ==
            "PASS_EXACT_SHA_FRESH_M43_R3_DC_STA_FORMALITY",
            "M43 receipt not terminal PASS")
    require(receipt["gates"]["all_pass"] is True and
            all(receipt["gates"].values()), "M43 receipt gate failure")
    logic = receipt["logic_only_ppa"]
    require(logic["total_cell_area_um2"] == 88664.435688 and
            logic["cell_count"] == 120125 and
            logic["sequential_cell_count"] == 9910 and
            logic["setup_wns_ns_slow_ssg0p9v125c"] == 0.1776 and
            logic["hold_wns_ns_fast_ffg1p05vm40c"] == 0.01 and
            logic["macro_or_blackbox_cell_count"] == 0,
            "M43 receipt PPA drift")
    formality = receipt["formality"]
    require(formality["passing_compare_points"] == 11868 and
            formality["failing_compare_points"] == 0 and
            formality["unmatched_compare_points"] == 0 and
            formality["unverified_compare_points"] == 0 and
            formality["aborted_compare_points"] == 0,
            "M43 Formality totals drift")
    area = (M43 / "reports/area.rpt").read_text(encoding="utf-8")
    setup = (M43 / "reports/timing_setup.rpt").read_text(encoding="utf-8")
    hold = (M43 / "reports/timing_hold.rpt").read_text(encoding="utf-8")
    fm_status = (M43 / "reports/formality_status.rpt").read_text(
        encoding="utf-8")
    require(number(r"Total cell area:\s+([0-9.]+)", area,
                   "M43 area") == 88664.435688,
            "M43 area report mismatch")
    require(number(r"slack \(MET\)\s+([0-9.]+)", setup,
                   "M43 setup") == 0.1776,
            "M43 setup report mismatch")
    require(number(r"slack \(MET\)\s+([0-9.]+)", hold,
                   "M43 hold") == 0.01,
            "M43 hold report mismatch")
    require("Verification SUCCEEDED" in fm_status and
            "11868 Passing compare points" in fm_status,
            "M43 Formality status mismatch")
    require(receipt["claim_boundary"]["paper_ppa_ready"] is False and
            receipt["claim_boundary"]["system_speedup_admitted"] is False and
            receipt["claim_boundary"]["power_or_energy_admitted"] is False,
            "M43 claim boundary widened")
    require(review["m43_findings"]["identity"]["candidate_changed"] is False,
            "review claims changed candidate")
    return sealed_files


def validate_m62(review):
    paths = {
        "m62_rtl": HW / "rtl_m62/qfit_head_p48_signed_lane_fold.sv",
        "m62_sva": HW / "verif_m62/qfit_head_p48_signed_lane_fold_assertions.sv",
        "m62_tb": HW / "tb_m62/tb_qfit_head_p48_signed_lane_fold.sv",
        "m62_contract": HW / "contracts/m62_p48_signed_lane_fold_directed_vcs_contract_r1_20260823.json",
        "m62_receipt": HW / "results/m62_p48_directed_vcs_r1_20260823/m62_p48_directed_vcs_receipt_r1.json",
        "m62_compile_log": M62_VCS / "compile_r4/compile.log",
        "m62_sim_log": M62_VCS / "sim_r4/sim.raw.log",
        "m62_dc_admission": M62_DC / "admission.txt",
        "m62_dc_log": M62_DC / "dc.log",
        "m62_dc_area": M62_DC / "reports/area.rpt",
        "m62_dc_setup": M62_DC / "reports/timing_setup.rpt",
        "m62_dc_hold": M62_DC / "reports/timing_hold.rpt",
    }
    for label, path in paths.items():
        check_hash(path, EXPECTED[label], label)
    receipt = load_json(paths["m62_receipt"])
    contract = load_json(paths["m62_contract"])
    require(receipt["status"] == "PASS_DIRECTED_VCS_SVA_R4" and
            receipt["compile_rc"] == receipt["sim_rc"] == 0,
            "M62 directed VCS receipt not terminal")
    require(receipt["covers"] == {
        "cp_zero_group": 15,
        "cp_full_eight_source_event": 1,
        "cp_positive_and_negative": 787,
        "cp_output_stall": 322,
        "cp_protocol_fault": 1,
    }, "M62 cover receipt drift")
    require(receipt["admission"]["directed_vcs_sva_admitted"] is True and
            receipt["admission"]["headline_admitted"] is False and
            receipt["admission"]["system_speedup_admitted"] is False and
            receipt["admission"]["dc_sta_formality_admitted"] is False,
            "M62 VCS receipt boundary widened")
    require(any("DC/STA/Formality" in item
                for item in contract["claim_boundary"]["forbidden"]),
            "M62 directed contract no longer forbids DC expansion")
    sim = paths["m62_sim_log"].read_text(encoding="utf-8")
    require(sim.count(receipt["pass_line"]) == 1,
            "M62 PASS line missing/duplicated")
    for cover, matches in receipt["covers"].items():
        require(re.search(re.escape(cover) + r".*?([0-9]+) match", sim),
                "M62 cover report missing {}".format(cover))
        observed = number(re.escape(cover) + r".*?([0-9]+) match", sim,
                          cover, int)
        require(observed == matches, "M62 cover total drift {}".format(cover))
    tb = paths["m62_tb"].read_text(encoding="utf-8")
    require("event_source_valid = {SLOTS{1'b1}};" in tb and
            "event_negative_mask[slot*PIXELS+pixel] = 1'b1;" in tb and
            "event_positive_mask[slot*PIXELS+pixel] = 1'b1;" in tb and
            "for (int lane = 0; lane < LANES; lane++)" in tb and
            "accumulator mismatch" in tb,
            "M62 full-eight functional depth drift")
    require("$urandom" not in tb and "random" in
            contract["claim_boundary"]["admitted"][1],
            "M62 deterministic-vs-random terminology observation drift")
    dc_log = paths["m62_dc_log"].read_text(encoding="utf-8")
    require(len(re.findall(r"^Error:", dc_log, re.MULTILINE)) == 0 and
            len(re.findall(r"^Warning:", dc_log, re.MULTILINE)) == 18,
            "M62 DC error/warning inventory drift")
    dc_area = paths["m62_dc_area"].read_text(encoding="utf-8")
    dc_setup = paths["m62_dc_setup"].read_text(encoding="utf-8")
    dc_hold = paths["m62_dc_hold"].read_text(encoding="utf-8")
    require(number(r"Total cell area:\s+([0-9.]+)", dc_area,
                   "M62 DC area") == 35459.171730 and
            number(r"Number of cells:\s+([0-9]+)", dc_area,
                   "M62 cells", int) == 34359 and
            number(r"Number of sequential cells:\s+([0-9]+)", dc_area,
                   "M62 sequential cells", int) == 2659 and
            number(r"Number of macros/black boxes:\s+([0-9]+)", dc_area,
                   "M62 macros", int) == 0,
            "M62 DC area report mismatch")
    require(number(r"slack \(MET\)\s+([0-9.]+)", dc_setup,
                   "M62 setup") == 0.6523 and
            number(r"slack \(MET\)\s+([0-9.]+)", dc_hold,
                   "M62 hold") == 0.0101,
            "M62 DC timing mismatch")
    require(not (M62_DC / "formality.raw.log").exists() and
            not (M62_DC / "completion_seal.sha256").exists() and
            not (M62_DC / "dc.rc").exists(),
            "M62 DC evidence scope changed; re-review required")
    m60 = load_json(HW / "results/m60_prediction_head_bounded_signed_tile_dse_r1_20260823/m60_prediction_head_bounded_signed_tile_dse_result_r2.json")
    receipt60 = load_json(HW / "results/m60_prediction_head_bounded_signed_tile_dse_r1_20260823/m60_prediction_head_bounded_signed_tile_dse_validation_receipt_r2.json")
    ratio = receipt60["selected_h16_w48"][
        "dense_over_bounded_signed_event_plus_commit_not_system"]
    require(abs(ratio - 3.086389221379181) < 1e-12,
            "M60 head-kernel ratio drift")
    require("ratios_not_system_speedup" in json.dumps(m60) and
            m60["claim_boundary"]["forbidden"][0] ==
            "full-network or system speedup",
            "M60 system-speedup boundary widened")
    require(review["m62_findings"]["directed_vcs"][
                "full_eight_cover_valid"] is True,
            "review invalidated full-eight cover")
    return ratio


def negative_attacks(review, m43_receipt, m62_receipt):
    attacks = []
    cases = []
    bad = copy.deepcopy(review)
    bad["scores"]["combined_story"][
        "date_prosperity_phi_completeness_score"] = 65
    cases.append(("review_score_tamper", bad, "review"))
    bad = copy.deepcopy(review)
    bad["m62_findings"]["ratio_boundary"][
        "is_full_network_or_system_speedup"] = True
    cases.append(("system_ratio_promotion", bad, "review"))
    for name, payload, kind in cases:
        rejected = False
        try:
            if kind == "review":
                validate_review_core(payload)
        except Exception:
            rejected = True
        require(rejected, "negative attack accepted {}".format(name))
        attacks.append({"name": name, "rejected": True})
    require(m43_receipt["formality"]["passing_compare_points"] == 11868,
            "M43 source receipt unexpectedly changed")
    require(m62_receipt["admission"]["system_speedup_admitted"] is False,
            "M62 source receipt unexpectedly widened")
    return attacks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing validator output overwrite")
    review = load_json(REVIEW)
    validate_review_core(review)
    sealed_files = validate_m43(review)
    ratio = validate_m62(review)
    m43_receipt = load_json(M43 / "m43_r3_synopsys_receipt.json")
    m62_receipt = load_json(
        HW / "results/m62_p48_directed_vcs_r1_20260823/"
        "m62_p48_directed_vcs_receipt_r1.json")
    attacks = negative_attacks(review, m43_receipt, m62_receipt)
    payload = {
        "schema": "m43_m62_independent_hammer_validation_receipt_v1",
        "status": "PASS_M43_M62_INDEPENDENT_HAMMER_REVIEW",
        "review_sha256": sha256_path(REVIEW),
        "validator_sha256": sha256_path(Path(__file__)),
        "m43_receipt_sha256": EXPECTED["m43_receipt"],
        "m43_completion_seal_entry_count": sealed_files,
        "m62_vcs_receipt_sha256": EXPECTED["m62_receipt"],
        "m62_full_eight_cover_effective": True,
        "m62_premacro_dc_reports_sealed": False,
        "m60_ratio_not_system_speedup": ratio,
        "scores": {name: review["scores"][name][
            "date_prosperity_phi_completeness_score"]
            for name in ("m43", "m62", "combined_story")},
        "severity_counts": {name: len(review["issues"][name])
                            for name in ("P0", "P1", "P2")},
        "negative_attacks": attacks,
        "next_gate": review["admission_gate_for_next_milestone"][
            "current_result"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M43+M62 independent hammer review score={}/{}/{} P0/P1/P2={}/{}/{}".format(
        payload["scores"]["m43"], payload["scores"]["m62"],
        payload["scores"]["combined_story"],
        payload["severity_counts"]["P0"],
        payload["severity_counts"]["P1"],
        payload["severity_counts"]["P2"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M43+M62 independent hammer review: {}".format(error))
        raise SystemExit(1)
