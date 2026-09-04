#!/usr/bin/env python3
"""Fail-closed validator for the independent M52-r1 hammer review."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
REVIEW_DIR = HW_ROOT / (
    "results/m52_high_fanout_context16_dse_r1_independent_hammer_20260823")
REVIEW = REVIEW_DIR / "m52_r1_independent_hammer_review.json"
EXPECTED = {
    "review": "2a40398bdf5acf4ff9d853e8eba954fe5b62f2e8209c3f8d36548994826797b4",
    "contract": "9aab440911d8a1dbe5b0465ca4af31427131b7a27bebb4f9cfc9451689e5a173",
    "analyzer": "9202a7021bfaa993a3028fcf39a679dbafcfb9fee836ddbd2435f3fdc044fdbc",
    "result": "d60567fecd891e9da0fc1b5bb0d88f4bb7e8e93faa92092037fc46d63dcde50b",
    "producer_validator": "9fa9475d5248e4f77643641cc2e72950e050be5d0e5167e54a00c4d915f63731",
    "checker": "f2a8a8a00febf0532d874f9fef2c5e24a36e71cecc5ead7abcb86dd54edaa58e",
    "reconstruction": "dcbe0130e0a64270ffd0d873c3609897d612f657856ff043e32db35d6abcfd1c",
    "attack_runner": "cdfe84193ad2d257e82b9a6c52153c96d19d624677587cd4a7f9434395dcbaaf",
    "attack_receipt": "8ebe8acf9e3829d1d3836f1c3cc30d449415d49228184a36acafe4ea9e1935e7",
    "durable_rerun": "d60567fecd891e9da0fc1b5bb0d88f4bb7e8e93faa92092037fc46d63dcde50b",
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


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def resolve_hw(row):
    return (HW_ROOT / row["path"]).resolve()


def check_call(command, label):
    result = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    require(result.returncode == 0,
            "{} failed rc={} stdout={} stderr={}".format(
                label, result.returncode, result.stdout[-2000:],
                result.stderr[-2000:]))
    return result


def validate(rerun):
    require(REVIEW.is_file() and sha256_path(REVIEW) == EXPECTED["review"],
            "review SHA mismatch")
    review = strict_json(REVIEW)
    require(review["schema"] == "m52_r1_independent_hammer_review_v1" and
            review["status"] ==
            "PASS_INDEPENDENT_HAMMER_LEDGER_GO_EXPERIMENT_ONLY_WITH_P1" and
            review["verdict"] ==
            "GO_K4_CTX16_RTL_EXPERIMENT_ONLY_NO_GO_HARDWARE_PERFORMANCE_OR_SYSTEM" and
            review["score_0_to_100"] == 84 and
            review["severity_counts"] == {"P0": 0, "P1": 1, "P2": 6},
            "review verdict/score/severity mismatch")
    require([row["id"] for row in review["findings"]] == [
        "M52-P1-01", "M52-P2-01", "M52-P2-02", "M52-P2-03",
        "M52-P2-04", "M52-P2-05", "M52-P2-06"],
        "review finding identity mismatch")
    p1 = review["findings"][0]
    require(p1["severity"] == "P1" and
            "min(16, len(ready))" in p1["finding"] and
            "not an enqueue/dequeue occupancy ledger" in p1["finding"] and
            "does not invalidate the frozen abstract transaction replay" in
            p1["impact"] and
            "blocks treating all promotion gates as hardware-feasibility gates"
            in p1["impact"],
            "metadata ready-window P1 scope mismatch")

    expected_producer = {
        "contract": EXPECTED["contract"],
        "analyzer": EXPECTED["analyzer"],
        "result": EXPECTED["result"],
        "producer_validator": EXPECTED["producer_validator"],
    }
    for name, row in review["producer_anchors"].items():
        path = resolve_hw(row)
        require(path.is_file() and row["sha256"] == expected_producer[name] and
                sha256_path(path) == row["sha256"],
                "producer anchor mismatch: {}".format(name))

    expected_independent = {
        "reconstruction_checker": EXPECTED["checker"],
        "reconstruction": EXPECTED["reconstruction"],
        "attack_runner": EXPECTED["attack_runner"],
        "attack_receipt": EXPECTED["attack_receipt"],
        "durable_all4_rerun": EXPECTED["durable_rerun"],
    }
    for name, row in review["independent_evidence"].items():
        path = resolve_hw(row)
        require(path.is_file() and row["sha256"] == expected_independent[name] and
                sha256_path(path) == row["sha256"],
                "independent evidence mismatch: {}".format(name))

    reconstruction = strict_json(resolve_hw(
        review["independent_evidence"]["reconstruction"]))
    require(reconstruction["status"] ==
            "PASS_INDEPENDENT_RECONSTRUCTION_LEDGER_ONLY" and
            reconstruction["independent_mismatch_count"] == 0 and
            reconstruction["identity"]["rebuilt_result_sha256"] ==
            EXPECTED["result"] and
            reconstruction["scope"] == {
                "capacity_and_complexity_arithmetic_independent": True,
                "durable_producer_rerun_byte_exact": True,
                "record_and_percentile_reconstruction_independent": True,
                "rtl_or_synopsys_run_performed": False,
                "system_speedup_admitted": False,
            }, "reconstruction identity/scope mismatch")
    configs = dict((row["name"], row)
                   for row in reconstruction["configurations"])
    require(set(configs) == {"K2_CTX16", "K4_CTX8", "K4_CTX16", "K8_CTX16"},
            "reconstructed configuration set mismatch")
    k4 = configs["K4_CTX16"]
    k8 = configs["K8_CTX16"]
    require(k4["aggregate_source_only_cycles"] == 70821488 and
            k4["aggregate_integrated_cycles"] == 81921184 and
            k4["integrated_p95_nearest_rank"] == 8376280 and
            k4["capacity_bytes"] == 176688 and
            k4["headroom_bytes"] == 17040 and
            k4["headroom_threshold_surplus_bytes"] == 656 and
            k4["rmw_paths"] == 384 and k4["signed_bank_terms"] == 3072 and
            k4["atomic_push_vectors"] == 4 and
            k4["atomic_payload_bits_excluding_tags"] == 7296 and
            k8["integrated_p95_nearest_rank"] == 8029048 and
            k8["rmw_paths"] == 768 and k8["atomic_push_vectors"] == 8,
            "reconstructed K4/K8 metric mismatch")
    guard = reconstruction["guard_extension_audit"]
    require(guard["guard_replacement_occurrences"] == 1 and
            guard["only_source_length_delta_bytes"] == 1 and
            guard["new_maximum_context_capacity"] == 16 and
            guard["reported_metadata_occupancy_is_ready_window_clamp"] is True and
            guard["response_metadata_enqueue_dequeue_event_ledger_present"] is False,
            "guard/metadata semantic audit mismatch")
    gates = reconstruction["gate_reconstruction"]
    require(gates["selected"] == "K4_CTX16" and gates["k8_killed"] is True and
            gates["k4_ctx16_headroom_threshold_surplus_bytes"] == 656 and
            gates["k4_rmw_and_atomic_width_ratio_vs_k2"] == 2 and
            gates["k8_rmw_and_atomic_width_ratio_vs_k4"] == 2 and
            abs(gates["k8_incremental_p95_improvement_fraction"] -
                0.04145420162649768) < 1e-15 and
            abs(gates["k8_source_cycle_change_fraction"] -
                0.008705041611099727) < 1e-15,
            "gate reconstruction mismatch")
    conditional = reconstruction["conditional_model_reconstruction"]
    require(conditional == {
        "address_timed_pair_replayed": False,
        "conditional_compute_ratio": 3.081290603812283,
        "conditional_denominator_cycles": 201496166,
        "conditional_three_x_crossing": True,
        "pair_p95_upper_bound_cycles": 10035160,
        "system_or_end_to_end_speedup_admitted": False,
    }, "conditional reconstruction mismatch")

    attack = strict_json(resolve_hw(
        review["independent_evidence"]["attack_receipt"]))
    expected_attacks = [
        "aggregate_cycle_increment", "p95_nearest_rank_increment",
        "capacity_headroom_increment", "rmw_path_width_decrement",
        "selected_configuration_changed_to_k8",
        "k8_source_regression_kill_removed",
        "conditional_ratio_promoted_to_system", "unmeasured_rtl_admitted",
        "guard_edit_not_exact_8_to_16", "mutated_canonical_result_sha",
        "occupied_output_no_overwrite",
    ]
    require(attack["status"] == "PASS_BASELINE_AND_FAIL_CLOSED_MUTATIONS" and
            attack["attack_count"] == 11 and
            [row["name"] for row in attack["attacks"]] == expected_attacks and
            all(row["rejected"] is True for row in attack["attacks"]) and
            attack["producer_files_modified"] is False,
            "attack receipt mismatch")
    require(review["review_scope"] == {
        "durable_four_configuration_rerun": "PASS_BYTE_EXACT",
        "independent_capacity_and_complexity_reconstruction": "PASS",
        "independent_gate_reconstruction": "PASS",
        "independent_percentile_and_record_reconstruction":
            "PASS_160_RECORDS_ZERO_MISMATCH",
        "mutation_attacks": "PASS_11_OF_11_REJECTED",
        "producer_files_modified": False,
        "rtl_or_synopsys_run_performed": False,
    }, "review scope mismatch")
    require(len(review["admitted_claims"]) == 5 and
            len(review["not_admitted"]) == 4,
            "claim population mismatch")
    forbidden = " ".join(review["not_admitted"])
    for token in ("K4", "RTL", "Synopsys", "system", "3.081290603812283x",
                  "DATE", "best-paper"):
        require(token in forbidden, "missing forbidden claim token: {}".format(token))

    if rerun:
        producer_validator = resolve_hw(
            review["producer_anchors"]["producer_validator"])
        producer = check_call(
            [sys.executable, str(producer_validator), "--rerun"],
            "producer deterministic all4 rerun")
        require("deterministic all10 rerun" in producer.stdout,
                "producer rerun marker mismatch")
        with tempfile.TemporaryDirectory(prefix="m52_review_rerun_") as directory:
            temp = Path(directory)
            checker = resolve_hw(
                review["independent_evidence"]["reconstruction_checker"])
            rebuilt = resolve_hw(
                review["independent_evidence"]["durable_all4_rerun"])
            reconstruction_output = temp / "reconstruction.json"
            command = [
                sys.executable, str(checker),
                "--contract", str(resolve_hw(review["producer_anchors"]["contract"])),
                "--analyzer", str(resolve_hw(review["producer_anchors"]["analyzer"])),
                "--result", str(resolve_hw(review["producer_anchors"]["result"])),
                "--producer-validator", str(producer_validator),
                "--rebuilt-result", str(rebuilt),
                "--output", str(reconstruction_output),
            ]
            check_call(command, "independent reconstruction rerun")
            require(sha256_path(reconstruction_output) == EXPECTED["reconstruction"],
                    "independent reconstruction rerun SHA mismatch")
            attack_runner = resolve_hw(
                review["independent_evidence"]["attack_runner"])
            attack_output = temp / "attacks.json"
            check_call([sys.executable, str(attack_runner),
                        "--output", str(attack_output)],
                       "independent mutation rerun")
            require(sha256_path(attack_output) == EXPECTED["attack_receipt"],
                    "independent attack rerun SHA mismatch")

    return {
        "schema": "m52_r1_independent_hammer_validator_result_v1",
        "status": "PASS_M52_R1_INDEPENDENT_REVIEW_VALIDATED",
        "review_sha256": sha256_path(REVIEW),
        "score_0_to_100": 84,
        "severity_counts": {"P0": 0, "P1": 1, "P2": 6},
        "verdict": review["verdict"],
        "rerun": bool(rerun),
        "independent_mismatch_count": 0,
        "mutation_attacks_rejected": 11,
        "producer_files_modified": False,
        "rtl_or_synopsys_run_performed": False,
        "system_speedup_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.rerun)
    if args.output is not None:
        require(not args.output.exists(), "refusing validator receipt overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print("PASS M52-r1 independent review score=84 P0=0 P1=1 P2=6 "+
          "GO K4-C16 RTL experiment only; NO-GO hardware/system claims")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M52 independent review: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
