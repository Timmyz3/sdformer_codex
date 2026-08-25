#!/usr/bin/env python3
"""Fail-closed machine audit for the independent M134 hammer review."""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
DC_RUN = ROOT / "dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_logic_only_dc_3p000ns_r1_sealed_20260824"


def read(path):
    return path.read_text(encoding="utf-8", errors="strict")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_sha(path, expected):
    require(path.is_file(), "missing frozen file: {}".format(path))
    observed = sha256(path)
    require(observed == expected, "SHA mismatch: {}".format(path))
    return observed


def verify_manifest(path, base):
    result = {}  # type: Dict[str, str]
    for line in read(path).splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed manifest line: {!r}".format(line))
        expected, relative = match.groups()
        target = Path(relative)
        if not target.is_absolute():
            target = base / target
        require(target.is_file(), "manifest target missing: {}".format(target))
        require(sha256(target) == expected, "manifest target SHA mismatch: {}".format(target))
        result[relative] = expected
    require(bool(result), "empty manifest: {}".format(path))
    return result


def pass_tokens(path, prefix):
    lines = [line for line in read(path).splitlines() if line.startswith(prefix)]
    require(len(lines) == 1, "PASS line count mismatch for {}".format(prefix))
    result = {}
    for token in lines[0][len(prefix):].strip().split():
        require("=" in token, "malformed PASS token: {}".format(token))
        key, value = token.split("=", 1)
        if value in ("true", "false"):
            result[key] = value == "true"
        elif re.fullmatch(r"-?[0-9]+", value):
            result[key] = int(value)
        else:
            result[key] = value
    return result


def cover_counts(path):
    result = {}
    for line in read(path).splitlines():
        match = re.search(r"\.sva\.(.+?), .*?, ([0-9]+) match$", line)
        if match:
            result[match.group(1)] = int(match.group(2))
    return result


FROZEN = {
    "contracts/m134_conflict_free_16bank_dualrow_mapper_vcs_contract_r1_20260824.json": "5536ddc291254f2daea2169aad6160e9be8b36299da00a0002cd671e1a64e6da",
    "contracts/m134_conflict_free_16bank_dualrow_mapper_logic_only_dc_contract_r1_20260824.json": "6ae3166be05e30a48eb4933d68610a1a18a5b128c25157c59afdefec90da9b95",
    "contracts/m132_r1_independent_review_correction_overlay_r1_20260824.json": "82ca925af73a7fecb55c4a47d6d95fbba5eb5c22698a2c27695b6a68fbda36a9",
    "rtl_m134/m134_conflict_free_16bank_dualrow_mapper.sv": "497eb7ac803d08692352ac0d77db54f585cfb597ddd081632d53ca0ff91fdbe3",
    "verif_m134/m134_conflict_free_16bank_dualrow_mapper_assertions.sv": "0d626b4ef1038d046b128e9a1d04fcb121ca2e0ccca2a978b5175c13884032c8",
    "tb_m134/tb_m134_conflict_free_16bank_dualrow_mapper.sv": "b274eae135db56492ebda13ff2a25e6a3f4bcf690d6d7bbafa299e8d2559d91b",
    "dc_handoff/filelists/date_m134_conflict_free_16bank_dualrow_mapper_directed_vcs.f": "11cc9888135e5226ffeded5e29290f5e0e8953e3f78d22a368339d040d132f4c",
    "dc_handoff/filelists/date_m134_conflict_free_16bank_dualrow_mapper_logic_only_dc.f": "76d4e88ef1b7bfd60c383ab8e18579742fbf7b60349f87a6fe34f648930da9f3",
    "dc_handoff/scripts/run_vcs_m134_conflict_free_16bank_dualrow_mapper.sh": "35e127051f3f973179df6055087b58dbb8b593125cfd106955cba9c7c75de3fb",
    "dc_handoff/scripts/run_dc_m134_conflict_free_16bank_dualrow_mapper_logic_only.sh": "ba1db5da9bb4e4ad276247179804b2ecee4bc8020e3e71177132cf6bf815681b",
    "dc_handoff/scripts/run_dc_m125_m127_logic_only_ab_exploratory.tcl": "7aa51dd4869b44bcf1a1d20693e9af337a04d85c7a595b143c5539d5e524ffb9",
    "dc_handoff/constraints/date_m134_comb_mapper_logic_only_3ns.sdc": "2621cd0dfa75d00627dfe86de74fb9f66606ca57946930ae6a79c772910815b7",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def main():
    frozen = {relative: verify_sha(ROOT / relative, expected) for relative, expected in FROZEN.items()}
    review = json.loads(read(REVIEW / "m134_conflict_free_16bank_dualrow_mapper_independent_hammer_review.json"))
    vcs_contract = json.loads(read(ROOT / "contracts/m134_conflict_free_16bank_dualrow_mapper_vcs_contract_r1_20260824.json"))
    dc_contract = json.loads(read(ROOT / "contracts/m134_conflict_free_16bank_dualrow_mapper_logic_only_dc_contract_r1_20260824.json"))
    dc_receipt = json.loads(read(DC_RUN / "m134_logic_only_dc_receipt_r1.json"))

    require(review["score"]["overall"] == 92, "score drift")
    require(review["severity_counts"] == {"P0": 0, "P1": 1, "P2": 4}, "severity drift")
    require(vcs_contract["mapping"]["logical_words"] == 3680, "WORDS contract drift")
    require(vcs_contract["mapping"]["banks"] == 16, "bank contract drift")
    require(vcs_contract["mapping"]["reads_per_bank_per_service"] == 1, "read contract drift")
    for key in ("foundry_macro", "macro_inclusive_ppa", "matched_frequency", "power_or_energy", "physical_speedup", "system_speedup", "headline"):
        require(vcs_contract["admission"][key] is False, "VCS contract improperly admits {}".format(key))
    for key in ("foundry_sram_macro", "macro_inclusive_ppa", "complete_512bit_frontend_timing", "matched_8bank_vs_16bank_physical_comparison", "power_or_energy", "physical_speedup", "system_speedup", "paper_ppa_ready", "headline"):
        require(dc_contract["admission"][key] is False, "DC contract improperly admits {}".format(key))

    preflight = [line for line in read(REVIEW / "preflight_sha_checks.txt").splitlines() if line]
    require(len(preflight) == 23, "review preflight count drift")
    require(all(re.search(r"expected=([0-9a-f]{64}) observed=\1$", line) for line in preflight), "review preflight mismatch")
    vcs_hashes = verify_manifest(REVIEW / "vcs_output.sha256", REVIEW)
    parameter_hashes = verify_manifest(REVIEW / "parameter_attack_output.sha256", REVIEW)
    dc_hashes = verify_manifest(DC_RUN / "evidence_manifest.sha256", DC_RUN)

    production = pass_tokens(REVIEW / "production_vcs_replay/sim.raw.log", "PASS M134 conflict-free 16-bank dualrow mapper VCS")
    independent = pass_tokens(REVIEW / "independent_vcs/sim.raw.log", "PASS M134 independent hammer")
    require(production == {
        "legal_windows": 3665, "logical_words": 58640,
        "physical_bank_reads": 58640, "row_crossings": 3435,
        "base_offsets": 16, "illegal_windows": 3, "words": 3680,
        "banks": 16, "word_bits": 32, "service_bits": 512,
        "reads_per_bank": 1, "macro": False, "physical_speedup": False,
        "system_speedup": False, "headline": False,
    }, "production PASS metric drift")
    require(independent == {
        "legal_windows": 3665, "illegal_windows": 431,
        "logical_words": 58640, "physical_addresses": 58640,
        "one_read_per_bank_checks": 58640, "row_crossings": 3435,
        "crossed_bank_addresses": 27480, "base_offset0": 230,
        "other_base_offsets": 229, "valid_low_payload_checks": 64,
        "stale_or_skewed_data_undetected": 1, "x_base_not_fail_closed": 1,
        "words": 3680, "rows_per_bank": 230, "banks": 16,
        "word_bits": 32, "service_bits": 512, "exposed_address_bits": 128,
        "exposed_bank_data_bits": 512, "macro": False, "macro_latency": False,
        "response_tag": False, "parameter_guard_synthesis_hard": False,
        "physical_speedup": False, "system_speedup": False, "headline": False,
    }, "independent PASS metric drift")
    for run in ("production_vcs_replay", "independent_vcs"):
        require(read(REVIEW / run / "compile.rc").strip() == "0", "compile RC failed: {}".format(run))
        require(read(REVIEW / run / "sim.rc").strip() == "0", "sim RC failed: {}".format(run))
        combined = read(REVIEW / run / "sim.raw.log") + read(REVIEW / run / "assert.report")
        require(re.search(r"failed at|Offending|^Error|^Fatal|watchdog timeout", combined, re.I | re.M) is None,
                "VCS failure marker: {}".format(run))

    covers = cover_counts(REVIEW / "independent_vcs/assert.report")
    require(covers["cp_crosses_physical_row"] == 3435, "crossing cover drift")
    require(covers["cp_last_legal_window"] == 1, "last legal cover drift")
    require(covers["cp_first_illegal_window"] == 1, "first illegal cover drift")
    require(covers["offset_covers[0].cp_every_base_bank"] == 230, "offset0 cover drift")
    for offset in range(1, 16):
        require(covers["offset_covers[{}].cp_every_base_bank".format(offset)] == 229,
                "offset cover drift: {}".format(offset))

    attacks = ("words_guard", "banks_guard", "word_w_guard", "base_w_guard", "row_w_guard")
    for attack in attacks:
        directory = REVIEW / "parameter_attacks" / attack
        require(read(directory / "compile.rc").strip() == "0", "parameter compile failed: {}".format(attack))
        require(read(directory / "sim.rc").strip() == "0", "unexpected VCS fatal RC mapping: {}".format(attack))
        require("M134 production geometry drift" in read(directory / "sim.raw.log"), "guard fatal missing: {}".format(attack))
    bypass = REVIEW / "parameter_attacks/banks8_synthesis_guard_bypass"
    require(read(bypass / "compile.rc").strip() == "0" and read(bypass / "sim.rc").strip() == "0", "guard bypass run failed")
    require("PASS M134 synthesis-define parameter guard bypass banks=8 guard_active=false hardcoded_modulo16_unknown=true production_geometry_only=true" in read(bypass / "sim.raw.log"), "guard bypass PASS missing")
    require("Warning-[SIOB] Select index out of bounds" in read(REVIEW / "parameter_attacks/base_w_guard/compile.raw.log"), "BASE_W attack did not expose SIOB")

    rtl = read(ROOT / "rtl_m134/m134_conflict_free_16bank_dualrow_mapper.sv")
    for token in ("`ifndef SYNTHESIS", "WORDS != 3680", "BANKS != 16", "WORD_W != 32", "BASE_W != 12", "ROW_W != 8", "13'd15", "logical_base_word[3:0]", "logical_base_word[11:4]", "16'hffff", "4'hf"):
        require(token in rtl, "static geometry token missing: {}".format(token))
    require("response_valid" not in rtl and "response_row" not in rtl and "response_bank" not in rtl,
            "unexpected response identity field found")

    require(read(DC_RUN / "dc.rc").strip() == "0", "sealed DC RC failed")
    require(dc_receipt["status"] == "PASS_EXACT_SHA_LOGIC_ONLY_PREMACRO_NOT_PAPER_PPA", "DC receipt status drift")
    expected_dc = {
        "cell_area_um2": 2054.555977,
        "cell_count": 3808,
        "logic_levels": 14,
        "setup_worst_slack_ns": 1.5947,
        "hold_worst_slack_ns": 0.5069,
        "critical_path_data_arrival_ns": 0.9553,
        "macro_count": 0,
    }
    for key, value in expected_dc.items():
        require(dc_receipt[key] == value, "DC receipt metric drift: {}".format(key))
    require(dc_receipt["admission"]["mapper_combinational_logic_area_timing"] is True, "mapper DC not admitted")
    for key in ("foundry_sram_macro", "macro_inclusive_ppa", "complete_512bit_frontend_timing", "power_or_energy", "physical_speedup", "system_speedup", "paper_ppa_ready", "headline"):
        require(dc_receipt["admission"][key] is False, "receipt improperly admits {}".format(key))
    dc_log = read(DC_RUN / "dc.log")
    require(len(re.findall(r"\(VER-318\)", dc_log)) == 2, "VER-318 count drift")
    precheck = read(DC_RUN / "reports/check_design_precompile.rpt")
    require(re.search(r"Shorted outputs \(LINT-31\)\s+16", precheck), "LINT-31 count drift")
    require(re.search(r"Cells do not drive \(LINT-1\)\s+1", precheck), "LINT-1 count drift")
    require(precheck.count("Warning:") == 17, "precompile warning total drift")
    require(read(DC_RUN / "reports/check_design_postcompile.rpt").count("Warning:") == 0, "postcompile warnings found")

    result = {
        "schema": "m134_independent_machine_audit_v1",
        "status": "PASS_CONDITIONAL_EXHAUSTIVE_LOGICAL_MAPPING_PHYSICAL_FRONTEND_OPEN",
        "review_score": review["score"],
        "severity_counts": review["severity_counts"],
        "production_exact_sha_vcs_replay": production,
        "independent_exhaustive_vcs": independent,
        "independent_sva_covers": covers,
        "parameter_boundary": {
            "simulation_guard_rejections": 5,
            "synthesis_define_bypass_observed": True,
            "hardcoded_modulo16_unknown_with_banks8": True,
            "production_geometry_only": True,
        },
        "port_cut_boundary": {
            "response_valid": False,
            "response_identity_tags": False,
            "stale_or_skewed_response_detected": False,
            "macro_latency_modeled": False,
        },
        "sealed_logic_only_dc": expected_dc,
        "dc_warning_boundary": {
            "VER-318": 2,
            "LINT-1": 1,
            "LINT-31": 16,
            "postcompile_warning_lines": 0,
        },
        "claim_boundary": {
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "matched_8bank_vs_16bank_physical_comparison": False,
            "power_or_energy": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
        "frozen_input_sha256": frozen,
        "review_vcs_output_sha256": vcs_hashes,
        "parameter_attack_output_sha256": parameter_hashes,
        "sealed_dc_evidence_sha256": dc_hashes,
        "sealed_dc_receipt_sha256": sha256(DC_RUN / "m134_logic_only_dc_receipt_r1.json"),
    }
    target = REVIEW / "m134_independent_machine_audit.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M134 machine audit score=92 P0=0 P1=1 P2=4 mapping=true macros=false physical_speedup=false headline=false")


if __name__ == "__main__":
    main()
