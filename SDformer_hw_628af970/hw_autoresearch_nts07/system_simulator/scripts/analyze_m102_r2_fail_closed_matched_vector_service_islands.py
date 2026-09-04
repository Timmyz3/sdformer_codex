#!/usr/bin/env python3
"""Seal the fail-closed M102 r2 VCS and analytical service-slot ledger."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / (
    "contracts/m102_r2_fail_closed_matched_vector_service_islands_vcs_contract_r1_20260824.json"
)
VCS_RUN = HW / (
    "dc_handoff/runs/"
    "m102_r2_fail_closed_matched_vector_service_islands_vcs_r1_sealed_20260824"
)

EXPECTED_SHA256 = {
    "contract": "d104a8affd17ca8f456816db92ea7b4d81dc499846f2640faa3e085ef7745bff",
    "vcs_run_complete": "7e643d61830b8f67980101fafa53295cb2806396f75f8839bb04a9a7b11382c1",
    "vcs_inputs": "59069d31ba8d65e92bc15c64207ae14d66f14190ef1d4adec03db158eebc4841",
    "vcs_outputs": "39a0ef249494bc90b5d0d9f8f0086ac9f0f0c1e51427b425d3eaac0e0ffd5e37",
    "baseline_log": "9d79a66ffd5e6d22d13c6e2a6d950b16d28d4276ac096f827deb479ba50af52b",
    "candidate_log": "b0a84ad5b57a5ee1460f20ecc47886b047516b1acfad38403bc0493fbcd77949",
    "baseline_assert": "0948564efa62be0cb16a984183706d6268305f80459db390626f012aaa577c02",
    "candidate_assert": "d6c26532f60b73b31b031f0fc87395885492d6f72c3ddfbfad69da2017446602",
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
    require(
        observed == EXPECTED_SHA256[label],
        "M102 r2 identity drift for {}: {}".format(label, observed),
    )
    return observed


def require_cover(report, name, matches):
    lines = [line for line in report.splitlines()
             if name in line and ", {} match".format(matches) in line]
    require(len(lines) == 1,
            "M102 r2 cover mismatch {} expected {}".format(name, matches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M102 r2 output overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source_start_sha = sha256(Path(__file__).resolve())
    identities = {
        "contract_sha256": require_sha("contract", CONTRACT),
        "vcs_run_complete_sha256": require_sha(
            "vcs_run_complete", VCS_RUN / "RUN_COMPLETE.txt"),
        "vcs_input_manifest_sha256": require_sha(
            "vcs_inputs", VCS_RUN / "input_sha256.txt"),
        "vcs_output_manifest_sha256": require_sha(
            "vcs_outputs", VCS_RUN / "output_sha256.txt"),
        "baseline_sim_log_sha256": require_sha(
            "baseline_log", VCS_RUN / "sim_baseline.raw.log"),
        "candidate_sim_log_sha256": require_sha(
            "candidate_log", VCS_RUN / "sim_candidate.raw.log"),
        "baseline_assert_report_sha256": require_sha(
            "baseline_assert", VCS_RUN / "assert_baseline.report"),
        "candidate_assert_report_sha256": require_sha(
            "candidate_assert", VCS_RUN / "assert_candidate.report"),
    }

    contract = strict_json(CONTRACT)
    run_complete = (VCS_RUN / "RUN_COMPLETE.txt").read_text(encoding="utf-8")
    baseline_log = (VCS_RUN / "sim_baseline.raw.log").read_text(encoding="utf-8")
    candidate_log = (VCS_RUN / "sim_candidate.raw.log").read_text(encoding="utf-8")
    baseline_assert = (VCS_RUN / "assert_baseline.report").read_text(encoding="utf-8")
    candidate_assert = (VCS_RUN / "assert_candidate.report").read_text(encoding="utf-8")

    require(
        run_complete.splitlines()[0]
        == "status=PASS_M102_R2_FAIL_CLOSED_MATCHED_VECTOR_SERVICE_ISLANDS_DIRECTED_VCS_SVA",
        "M102 r2 sealed VCS admission missing",
    )
    directed = contract["directed_vcs"]
    require(directed["baseline_expected_pass_line"] in baseline_log,
            "M102 r2 baseline PASS line missing")
    require(directed["candidate_expected_pass_line"] in candidate_log,
            "M102 r2 candidate PASS line missing")
    for name, matches in directed["baseline_required_cover_matches"].items():
        require_cover(baseline_assert, name, matches)
    for name, matches in directed["candidate_required_cover_matches"].items():
        require_cover(candidate_assert, name, matches)

    ledger = contract["frozen_workload_ledger"]
    parser_scope = contract["parser_exposure"]
    baseline_cycles = ledger["baseline_service_cycles"]
    candidate_cycles = ledger["candidate_combined_service_cycles"]
    service_ratio = baseline_cycles / float(candidate_cycles)
    parser_ratio = baseline_cycles / float(
        parser_scope["one_load_edge_per_phase_candidate_cycles"])
    require(math.isclose(service_ratio,
                         ledger["same_clock_service_slot_work_ratio"],
                         rel_tol=0.0, abs_tol=1e-15),
            "M102 r2 service ratio mismatch")
    require(math.isclose(parser_ratio,
                         parser_scope["one_load_edge_per_phase_ratio_upper_bound"],
                         rel_tol=0.0, abs_tol=1e-15),
            "M102 r2 parser-inclusive ratio mismatch")
    require(sha256(Path(__file__).resolve()) == source_start_sha,
            "M102 r2 analyzer source changed during run")
    identities["analyzer_start_end_sha256"] = source_start_sha

    payload = {
        "schema": "m102_r2_fail_closed_matched_vector_service_islands_ledger_v1",
        "status": "PASS_M102_R2_FAIL_CLOSED_DIRECTED_VCS_AND_ANALYTICAL_LEDGER_PRE_DC",
        "identity": identities,
        "p0_repair": {
            "trigger": contract["independent_hammer_trigger"]["p0"],
            "repair": contract["independent_hammer_trigger"]["r2_fix"],
            "buffered_output_quarantine_cover_matches": 3,
            "sticky_until_reset": True,
        },
        "vcs_functional_evidence": {
            "baseline": {
                "vectors": 90,
                "accepted_beats": 274,
                "protocol_attacks": 6,
                "reset_recoveries": 7,
            },
            "candidate": {
                "vectors": 8,
                "accepted_beats": 28,
                "pwp_vectors": 4,
                "correction_vectors": 2,
                "fallback_vectors": 2,
                "protocol_attacks": 12,
                "continuation_identity_attacks": 6,
                "metadata_attacks": 1,
                "fault_stall_attacks": 1,
                "pwp_to_correction_seam_cover_matches": 2,
            },
        },
        "analytical_service_ledger": {
            "baseline_active_source_vector_ops":
                ledger["baseline_active_source_vector_ops"],
            "baseline_service_cycles": baseline_cycles,
            "candidate_correction_or_fallback_vector_ops":
                ledger["candidate_correction_or_fallback_vector_ops"],
            "candidate_correction_or_fallback_service_cycles":
                ledger["candidate_correction_or_fallback_service_cycles"],
            "candidate_pwp_uses_by_width": ledger["candidate_pwp_uses_by_width"],
            "candidate_pwp_service_cycles": ledger["candidate_pwp_service_cycles"],
            "candidate_combined_service_cycles": candidate_cycles,
            "same_clock_service_slot_work_ratio": service_ratio,
            "current_single_context_parser_and_one_load_edge_cycles":
                parser_scope["one_load_edge_per_phase_candidate_cycles"],
            "current_single_context_parser_and_load_ratio_upper_bound":
                parser_ratio,
        },
        "scope": {
            "matched_aggregate_service_slots": 1,
            "service_slot_bits": 256,
            "current_candidate_metadata_contexts": 1,
            "precompacted_active_source_input": True,
            "actual_record_rtl_replay": False,
            "memory_response_mux_sram_matcher_dma_accumulator": "port_cut",
            "logic_only_vcs_no_macros": True,
        },
        "admission": {
            "directed_vcs_functional": True,
            "fault_quarantine": True,
            "analytical_same_clock_service_slot_ratio": True,
            "current_single_context_parser_inclusive_upper_bound": True,
            "physical_frequency_normalized_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "full_network_or_system_speedup": False,
            "accuracy": False,
            "paper_ppa_ready": False,
            "date_or_best_paper_headline": False,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "PASS M102 r2 service_ratio={:.12f} parser_load_upper={:.12f} physical=pending".format(
            service_ratio, parser_ratio
        )
    )


if __name__ == "__main__":
    main()
