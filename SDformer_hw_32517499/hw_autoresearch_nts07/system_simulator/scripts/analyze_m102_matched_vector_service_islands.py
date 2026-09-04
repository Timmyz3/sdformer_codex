#!/usr/bin/env python3
"""Seal the M102 matched vector-service-island cycle ledger.

This script only admits the cycle ratio represented by the two directed VCS
tops.  Physical throughput remains pending until each corresponding top has a
matched DC/STA frequency; matcher, DMA, SRAM macros, and accumulation are port
cuts and therefore cannot be promoted to a full-module or system claim.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / (
    "contracts/m102_matched_vector_service_islands_vcs_contract_r1_20260824.json")
PREFLIGHT = HW / (
    "reviews/m102_bit_sparse_physical_baseline_preflight_independent_hammer_r1_20260824/"
    "m102_bit_sparse_baseline_preflight_audit.json")
M88_RESULT = HW / (
    "results/m88_bounded_sync_bank_double_buffer_valid825_internal_r1_20260823/"
    "m88_bounded_sync_bank_double_buffer.json")
VCS_RUN = HW / (
    "dc_handoff/runs/m102_matched_vector_service_islands_vcs_r1_sealed_20260824")

EXPECTED_SHA256 = {
    "contract": "24b49136151bde04cc809122101b928aa7a778c100b44aefed0db1732b1e7ac2",
    "preflight": "baae8778f8ebf3a20c5314b8d60cd47b5f84be15a75d5e3552fb723e89ee863d",
    "m88_result": "36e9b0603422ccff7afd23e6e5e2309bc5d53b3c7e9898538095d6baa23da483",
    "vcs_run_complete": "2a64a611618f1c8029763a4d44bacdd7888a9dc8692f2d71f46adb6a8cbb62be",
    "vcs_inputs": "94f101e08d39803ef081af9d2b1742ee665c5fe7f5a333b0ae010240833ea200",
    "vcs_outputs": "4078cd5a273dff942427d054f83e6c2f531da9d133e6daa2ce7ec9b149b1912c",
    "baseline_log": "98f693109dd0546c5b316915a511e2fcdcabaa3e73318db21b261a27ac81f3cd",
    "candidate_log": "66a41acb3a6018ef76db34498e05160b1628e995cfbbd696dea3c0f23a854ad7",
    "baseline_assert": "0948564efa62be0cb16a984183706d6268305f80459db390626f012aaa577c02",
    "candidate_assert": "1c1730b088e26e9ceaec639be359ed83e342d8c2f95a2b7e258fa4bb4faece14",
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

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def require_sha(label, path):
    observed = sha256(path)
    require(observed == EXPECTED_SHA256[label],
            "M102 identity drift for {}: {}".format(label, observed))
    return observed


def require_cover(report, name, matches):
    needle = "{}".format(name)
    matched = [line for line in report.splitlines()
               if needle in line and ", {} match".format(matches) in line]
    require(len(matched) == 1,
            "M102 cover mismatch {} expected {}".format(name, matches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M102 output overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source_start_sha = sha256(Path(__file__).resolve())
    identities = {
        "contract_sha256": require_sha("contract", CONTRACT),
        "preflight_audit_sha256": require_sha("preflight", PREFLIGHT),
        "m88_result_sha256": require_sha("m88_result", M88_RESULT),
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
    preflight = strict_json(PREFLIGHT)
    m88 = strict_json(M88_RESULT)
    run_complete = (VCS_RUN / "RUN_COMPLETE.txt").read_text(encoding="utf-8")
    base_log = (VCS_RUN / "sim_baseline.raw.log").read_text(encoding="utf-8")
    cand_log = (VCS_RUN / "sim_candidate.raw.log").read_text(encoding="utf-8")
    base_assert = (VCS_RUN / "assert_baseline.report").read_text(encoding="utf-8")
    cand_assert = (VCS_RUN / "assert_candidate.report").read_text(encoding="utf-8")

    require(run_complete.splitlines()[0] ==
            "status=PASS_M102_MATCHED_VECTOR_SERVICE_ISLANDS_DIRECTED_VCS_SVA",
            "M102 sealed VCS admission missing")
    directed = contract["directed_vcs"]
    require(directed["baseline_expected_pass_line"] in base_log,
            "M102 baseline PASS line missing")
    require(directed["candidate_expected_pass_line"] in cand_log,
            "M102 candidate PASS line missing")
    for name, matches in directed["baseline_required_cover_matches"].items():
        require_cover(base_assert, name, matches)
    for name, matches in directed["candidate_required_cover_matches"].items():
        require_cover(cand_assert, name, matches)

    baseline = preflight["m78_m88_exact_shared32_denominator"]
    candidate = preflight["candidate_scope_reconciliation"]
    baseline_ops = baseline["active_source_vector_ops_all_blocks"]
    baseline_service = baseline_ops * baseline["cycles_per_weight_vector_op"]
    require(baseline_service == baseline["service_cycles"] == 1114383288,
            "M102 baseline service ledger mismatch")

    width_cycles = {"8": 3, "9": 4, "10": 4, "11": 5}
    pwp_service = sum(candidate["pwp_uses_by_width"][width] * cycles
                      for width, cycles in width_cycles.items())
    correction_service = candidate["correction_weight_vector_ops"] * 3
    combined_service = pwp_service + correction_service
    require(pwp_service == candidate["pwp_service_cycles"] == 226222255,
            "M102 PWP service ledger mismatch")
    require(correction_service == candidate["correction_service_cycles"] ==
            564445470, "M102 correction/fallback service ledger mismatch")
    require(combined_service == candidate["combined_candidate_service_cycles"] ==
            790667725, "M102 combined service ledger mismatch")

    cycle_ratio = baseline_service / float(combined_service)
    require(math.isclose(cycle_ratio, candidate["service_only_cycle_ratio"],
                         rel_tol=0.0, abs_tol=1e-15),
            "M102 service-only ratio mismatch")
    require(m88["aggregate"]["bounded_bit_sparse_cycles"] == 1114402488 and
            m88["aggregate"]["bounded_candidate_cycles"] == 790706475,
            "M102 M88 bounded reconciliation mismatch")

    require(sha256(Path(__file__).resolve()) == source_start_sha,
            "M102 analyzer source changed during run")
    identities["analyzer_start_end_sha256"] = source_start_sha
    payload = {
        "schema": "m102_matched_vector_service_islands_cycle_ledger_v1",
        "status": "PASS_M102_MATCHED_DIRECTED_VCS_AND_CYCLE_LEDGER_PRE_DC",
        "identity": identities,
        "vcs_functional_evidence": {
            "baseline": {
                "vectors": 90,
                "accepted_beats": 274,
                "fixed8_ii3_checks": 23,
                "protocol_attacks": 6,
                "reset_recoveries": 7,
                "signed_lane_range": [-128, 127],
            },
            "candidate": {
                "vectors": 8,
                "accepted_beats": 28,
                "pwp_vectors": 4,
                "correction_vectors": 2,
                "fallback_vectors": 2,
                "shared_slot_ii_checks": 7,
                "protocol_attacks": 6,
            },
        },
        "service_cycle_ledger": {
            "baseline_active_source_vector_ops": baseline_ops,
            "baseline_service_cycles": baseline_service,
            "candidate_correction_or_fallback_vector_ops":
                candidate["correction_weight_vector_ops"],
            "candidate_correction_or_fallback_service_cycles": correction_service,
            "candidate_pwp_uses_by_width": candidate["pwp_uses_by_width"],
            "candidate_pwp_service_cycles": pwp_service,
            "candidate_combined_service_cycles": combined_service,
            "same_clock_cycle_ratio": cycle_ratio,
        },
        "physical_throughput": {
            "formula": "(1114383288/f_bit_sparse_weight_service)/(790667725/f_combined_candidate_service)",
            "baseline_frequency_hz": None,
            "candidate_frequency_hz": None,
            "frequency_normalized_speedup": None,
            "status": "PENDING_MATCHED_DC_STA",
            "forbidden_formula": "1.409375695*(f_M99/f_M85)",
        },
        "scope": {
            "matched_aggregate_service_slots": 1,
            "service_slot_bits": 256,
            "precompacted_active_source_input": True,
            "matcher_enumerator_dma_memory_accumulator_port_cuts": True,
            "actual_record_rtl_replay": False,
            "logic_only_vcs_no_macros": True,
        },
        "admission": {
            "directed_vcs_functional": True,
            "cycle_only_same_clock_service_island_ratio": True,
            "physical_frequency_normalized_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "full_network_or_system_speedup": False,
            "accuracy": False,
            "paper_ppa_ready": False,
            "date_or_best_paper_headline": False,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M102 matched service cycle ratio={:.12f} physical_speedup=pending".format(
        cycle_ratio))


if __name__ == "__main__":
    main()
