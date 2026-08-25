#!/usr/bin/env python3
"""Fail-closed validator for the independent M85 hammer evidence bundle."""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REVIEW = Path(__file__).resolve().parent

EXPECTED_SHA256 = {
    "rtl_m82/zero_bubble_elastic_pwp_stream.sv":
        "2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f",
    "rtl_m85/guarded_wordpacked_pwp_stream.sv":
        "ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0",
    "verif_m85/guarded_wordpacked_pwp_stream_assertions.sv":
        "7403ad62988b5b082788b415713cee74982c47be15878648ca1294897c7fe2f7",
    "tb_m85/tb_guarded_wordpacked_pwp_stream.sv":
        "6ee304eaacdf8d3881cb87a96c199b3fc89e01d6350eda4cb23bb07061ac4c21",
    "contracts/m85_guarded_wordpacked_pwp_stream_vcs_contract_r1_20260823.json":
        "2f1225acb79ceaf16df35bc477dcd05c54bf0d299675cec388bce66cb1e576af",
    "results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin":
        "52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0",
}

PASS_LINE = (
    "PASS M85 actual-record integration phases=1728 entries=221184 "
    "outputs=221184 escape=1 beats=835383 "
    "masked_nonzero_words=733459 ii_checks=219456 "
    "metadata_poison_attacks=3"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_value_file(path: Path) -> Dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> None:
    observed = {}
    for relative, expected in EXPECTED_SHA256.items():
        digest = sha256(HW / relative)
        require(digest == expected, f"identity drift: {relative}")
        observed[relative] = digest

    oracle_path = REVIEW / "independent_oracle_recheck.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    require(oracle["status"] ==
            "PASS_INDEPENDENT_BINARY_MASK_ADDRESS_AND_SIGNED_ORACLE",
            "independent oracle status mismatch")
    replay = oracle["full_replay"]
    require(replay["phases"] == 1728, "phase count mismatch")
    require(replay["entries"] == 221184, "entry count mismatch")
    require(replay["regular_vectors_checked"] == 221183,
            "regular vector count mismatch")
    require(replay["escape_controls"] == 1, "escape count mismatch")
    require(replay["beats_including_escape"] == 835383,
            "beat count mismatch")
    require(replay["within_phase_start_ii_checks_expected"] == 219456,
            "II extent mismatch")

    final_mask = oracle["final_mask"]
    require(final_mask["valid_words_on_last_beat"] == {
        "8": 8, "9": 3, "10": 6, "11": 1},
        "final mask geometry mismatch")
    require(final_mask["nonzero_successor_words_that_require_mask"] ==
            733459, "dirty successor word count mismatch")
    require(oracle["signed_output_oracle"]["actual_vectors_bit_exact"] ==
            221183, "bit-exact vector count mismatch")
    require(oracle["address_oracle"]["maximum_row_address"] == 459,
            "maximum row mismatch")
    require(oracle["metadata"]["all_actual_phases_accepted"],
            "actual metadata acceptance mismatch")
    require(oracle["metadata"]["maximum_fetch_end"] == 3680,
            "metadata fetch boundary mismatch")
    for attack in (
        "reserved_predecessor_code_5",
        "reserved_predecessor_code_6",
        "reserved_predecessor_code_7",
        "wrong_pattern4_base",
        "pattern15_base_8191",
        "fetch_and_terminal_over_460x8",
    ):
        require(not oracle["metadata"]["parser_attacks"][attack]["accepted"],
                f"metadata attack accepted: {attack}")

    rerun = REVIEW / "sealed_vcs_rerun"
    require((rerun / "compile.rc").read_text().strip() == "0",
            "VCS compile rc nonzero")
    require((rerun / "sim.rc").read_text().strip() == "0",
            "VCS simulation rc nonzero")
    log = (rerun / "sim.raw.log").read_text(encoding="utf-8")
    require(PASS_LINE in log, "sealed VCS PASS line missing")
    require(not re.search(
        r"failed at|Offending|^Error|^Fatal|watchdog timeout", log,
        flags=re.IGNORECASE | re.MULTILINE),
        "failure signature in sealed VCS log")
    stall_match = re.search(r"cp_lookup_stall,\s+\d+ attempts,\s+(\d+) match", log)
    require(stall_match is not None and int(stall_match.group(1)) == 0,
            "expected always-ready replay with zero stall cover hits")

    complete = key_value_file(rerun / "RUN_COMPLETE.txt")
    for field in ("rtl_cycle_speedup", "paper_ppa_ready",
                  "system_speedup", "headline"):
        require(complete.get(field) == "false",
                f"claim boundary drift: {field}")
    require(complete.get("synchronous_sram") == "false",
            "synchronous SRAM improperly admitted")
    require(complete.get("real_escape_fallback") == "false",
            "escape fallback improperly admitted")

    contract = json.loads((HW / (
        "contracts/m85_guarded_wordpacked_pwp_stream_vcs_contract_r1_20260823.json"
    )).read_text(encoding="utf-8"))
    boundary = contract["claim_boundary"]
    require(not boundary["m78_shared32_speedup_re_admitted_by_m85_alone"],
            "M78 1.409x improperly re-admitted")
    for field in ("paper_ppa_ready", "system_speedup", "headline"):
        require(not boundary[field], f"contract admits {field}")

    result = {
        "schema": "m85_independent_hammer_review_evidence_validation_v1",
        "status": "PASS_M85_SCOPED_EVIDENCE_FAIL_CLOSED",
        "exact_sha_identity": observed,
        "vcs": {
            "compile_rc": 0,
            "simulation_rc": 0,
            "pass_line": PASS_LINE,
            "lookup_stall_cover_matches": 0,
        },
        "full_replay": {
            "phases": 1728,
            "entries": 221184,
            "regular_vectors_bit_exact": 221183,
            "escape_controls": 1,
            "beats_including_escape": 835383,
            "final_mask_nonzero_successor_words_suppressed": 733459,
        },
        "metadata": {
            "bytes_per_phase": 74,
            "all_actual_phases_accepted": True,
            "maximum_fetch_end_words": 3680,
            "independent_attack_classes_rejected": 6,
        },
        "claim_boundary": {
            "combinational_bank_response_only": True,
            "synchronous_sram": False,
            "random_backpressure": False,
            "real_escape_fallback": False,
            "m78_shared32_1p409x_re_admitted": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    output = REVIEW / "review_evidence_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "phases": 1728,
        "entries": 221184,
        "masked_nonzero_words": 733459,
        "stall_cover_matches": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
