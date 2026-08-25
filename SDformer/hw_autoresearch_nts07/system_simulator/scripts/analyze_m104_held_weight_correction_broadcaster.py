#!/usr/bin/env python3
"""Seal M104 directed VCS evidence and its conditional service-token envelope."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m104_held_weight_correction_broadcaster_vcs_contract_r1_20260824.json"
M103 = HW / (
    "reviews/m103_correction_service_reuse_preflight_independent_hammer_r1_20260824/"
    "m103_correction_reuse_preflight_audit.json"
)
M102 = HW / (
    "results/m102_r2_fail_closed_matched_vector_service_islands_vcs_cycle_ledger_r1_20260824/"
    "m102_r2_fail_closed_matched_vector_service_islands.json"
)
VCS_RUN = HW / (
    "dc_handoff/runs/m104_held_weight_correction_broadcaster_vcs_r1_sealed_20260824"
)

EXPECTED_SHA256 = {
    "contract": "bbd086a36719f3682216d39450dfc86db46c9373fc508f65657cfac2277dbdd5",
    "m103": "935119fab809e15f49089926550f89b3c84c2b13c0be58c96b0ea8709ed683fe",
    "m102": "a5d465b7d3361ed2ff176b4230d9051c29137aee86211cec9c3eb9ee8131aad5",
    "run_complete": "f3d587e55a31d2ec5367f4fc0512a0046087a89ff784d0cf6faf3259bdc54071",
    "inputs": "c9ebd7df61f5166be171a2494b94d1296c4d5b2b027e10659d0f1d975b9fd92d",
    "outputs": "113de83686f9588bdfa6c2c1c02c8e85f671f1ec932cb609fe3329374e1f5925",
    "sim": "276de390c70347e313521ec6be72730133d7309f7688cf594c46f7435d3e984f",
    "assert": "e79fcc80dfa3a8dfa03b1c8224b8d3b5ba0035cf0038f6d46c846332b1a8be29",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject,
    )


def require_sha(label, path):
    observed = sha256(path)
    require(observed == EXPECTED_SHA256[label],
            "M104 identity drift {} {}".format(label, observed))
    return observed


def require_cover(report, name, matches):
    lines = [line for line in report.splitlines()
             if name in line and ", {} match".format(matches) in line]
    require(len(lines) == 1,
            "M104 cover mismatch {} expected {}".format(name, matches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M104 output overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source_sha = sha256(Path(__file__).resolve())
    identity = {
        "analyzer_start_end_sha256": source_sha,
        "contract_sha256": require_sha("contract", CONTRACT),
        "m103_audit_sha256": require_sha("m103", M103),
        "m102_r2_ledger_sha256": require_sha("m102", M102),
        "vcs_run_complete_sha256": require_sha(
            "run_complete", VCS_RUN / "RUN_COMPLETE.txt"),
        "vcs_input_manifest_sha256": require_sha(
            "inputs", VCS_RUN / "input_sha256.txt"),
        "vcs_output_manifest_sha256": require_sha(
            "outputs", VCS_RUN / "output_sha256.txt"),
        "vcs_sim_log_sha256": require_sha("sim", VCS_RUN / "sim.raw.log"),
        "vcs_assert_report_sha256": require_sha(
            "assert", VCS_RUN / "assert.report"),
    }
    contract = strict_json(CONTRACT)
    m103 = strict_json(M103)
    m102 = strict_json(M102)
    run_complete = (VCS_RUN / "RUN_COMPLETE.txt").read_text(encoding="utf-8")
    sim = (VCS_RUN / "sim.raw.log").read_text(encoding="utf-8")
    report = (VCS_RUN / "assert.report").read_text(encoding="utf-8")

    require(run_complete.splitlines()[0] ==
            "status=PASS_M104_HELD_WEIGHT_CORRECTION_BROADCASTER_DIRECTED_VCS_SVA",
            "M104 sealed run admission missing")
    directed = contract["directed_vcs"]
    require(directed["expected_pass_line"] in sim, "M104 PASS line missing")
    for name, matches in directed["required_cover_matches"].items():
        require_cover(report, name, matches)

    envelope = contract["frozen_heldout_envelope"]
    grouping = m103["order_independent_grouping"]["weight_groups"]
    baseline = envelope["fixed8_baseline_service_cycles"]
    events = grouping["events"]
    groups = grouping["groups"]
    correction_tokens = events + 2 * groups
    combined_tokens = correction_tokens + envelope["existing_pwp_service_cycles"]
    ratio = baseline / float(combined_tokens)
    require(events == 188148490 and groups == 1105920,
            "M104 group population drift")
    require(correction_tokens == envelope["held_key_conditional_tokens"] == 190360330,
            "M104 correction token envelope drift")
    require(combined_tokens == envelope["held_key_combined_candidate_tokens"] == 416582585,
            "M104 combined token envelope drift")
    require(math.isclose(ratio,
                         envelope["conditional_same_clock_service_slot_ratio"],
                         rel_tol=0.0, abs_tol=1e-15),
            "M104 ratio drift")
    require(m102["analytical_service_ledger"]["baseline_service_cycles"] == baseline,
            "M104 M102 denominator drift")
    require(sha256(Path(__file__).resolve()) == source_sha,
            "M104 analyzer changed during run")

    payload = {
        "schema": "m104_held_weight_correction_broadcaster_result_v1",
        "status": "PASS_M104_DIRECTED_VCS_AND_CONDITIONAL_TOKEN_ENVELOPE_PRE_DC",
        "identity": identity,
        "directed_vcs": {
            "groups_loaded": 6,
            "load_beats": 21,
            "events": 9,
            "consecutive_event_ii1_pairs": 5,
            "stalls": 3,
            "protocol_attacks": 10,
            "continuation_attacks": 3,
            "buffered_fault_attacks": 1,
            "lanes": 96,
            "macros": 0,
        },
        "conditional_service_token_envelope": {
            "correction_or_fallback_events": events,
            "phase_weight_groups": groups,
            "weight_load_tokens_per_group": 3,
            "destination_tokens_per_event": 1,
            "correction_tokens": correction_tokens,
            "existing_pwp_tokens": envelope["existing_pwp_service_cycles"],
            "combined_candidate_tokens": combined_tokens,
            "fixed8_baseline_tokens": baseline,
            "same_clock_ratio": ratio,
            "assumption": "perfect phase-local source/output-block batching",
        },
        "port_cuts": contract["module_scope"]["port_cuts"],
        "admission": {
            "directed_vcs_functional": True,
            "signed_int8_to_signed12": True,
            "same_cycle_fault_quarantine": True,
            "conditional_order_independent_token_envelope": True,
            "ordered_transpose_schedule": False,
            "actual_record_rtl_replay": False,
            "scheduled_cycle_speedup": False,
            "physical_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "paper_ppa_ready": False,
            "full_network_or_system_speedup": False,
            "accuracy": False,
            "headline": False,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PASS M104 conditional token ratio={:.12f} scheduled=false physical=false".format(ratio))


if __name__ == "__main__":
    main()
