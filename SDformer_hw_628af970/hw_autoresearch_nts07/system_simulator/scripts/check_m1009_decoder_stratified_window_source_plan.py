#!/usr/bin/env python3
"""Static M1009 plan validator and finite-population estimator self-test.

This source never generates or schedules decoder transactions.  It validates
the fail-closed source plan and supplies the exact design-based estimator that
a later independently released bounded-window runner must use.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CONTRACT = HW / "contracts/m1009_decoder_stratified_window_source_plan_contract_r1_20260829.json"
SCHEMA = "m1009_decoder_stratified_window_source_plan_contract_v1"
T_CRITICAL = 2.365  # conservative two-sided 95% value at the minimum df=7


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(values):
        output = {}
        for key, value in values:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs)


def verify_flat_seal(directory):
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "seal absent")
    require(outer.read_text(encoding="utf-8") ==
            sha256(manifest) + "  SHA256SUMS\n", "outer seal drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in listed, "duplicate seal member")
        item = directory / name
        require(item.is_file() and not item.is_symlink() and
                sha256(item) == digest, "sealed member drift: " + name)
        listed[name] = digest
    actual = {item.name for item in directory.iterdir()
              if item.is_file() and item.name not in
              ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "seal exact-set drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer)}


def mean(values: Sequence[float]) -> float:
    require(values, "empty sample")
    return sum(float(value) for value in values) / len(values)


def sample_variance(values: Sequence[float]) -> float:
    require(len(values) >= 2, "variance requires two samples")
    center = mean(values)
    return sum((float(value) - center) ** 2 for value in values) / (len(values) - 1)


def sample_covariance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    require(len(lhs) == len(rhs) and len(lhs) >= 2, "paired covariance shape drift")
    lmean, rmean = mean(lhs), mean(rhs)
    return sum((float(x) - lmean) * (float(y) - rmean)
               for x, y in zip(lhs, rhs)) / (len(lhs) - 1)


def estimate_paired_totals(strata: Sequence[Mapping[str, object]],
                           fixed_candidate: float = 0.0,
                           fixed_baseline: float = 0.0) -> Dict[str, object]:
    """Finite-population paired total/ratio estimate for additive blocks."""
    candidate_total, baseline_total = float(fixed_candidate), float(fixed_baseline)
    candidate_var = baseline_var = covariance = 0.0
    rows = []
    for row in strata:
        population = int(row["population_blocks"])
        candidate = [float(value) for value in row["candidate_cycles"]]
        baseline = [float(value) for value in row["baseline_cycles"]]
        sample = len(candidate)
        require(population >= sample >= 2 and len(baseline) == sample,
                "invalid paired stratum sample")
        require(all(value > 0 for value in candidate + baseline),
                "cycle samples must be positive")
        fraction = sample / population
        cmean, bmean = mean(candidate), mean(baseline)
        cvar, bvar = sample_variance(candidate), sample_variance(baseline)
        cov = sample_covariance(candidate, baseline)
        vc = population * population * (1.0 - fraction) * cvar / sample
        vb = population * population * (1.0 - fraction) * bvar / sample
        vcb = population * population * (1.0 - fraction) * cov / sample
        candidate_total += population * cmean
        baseline_total += population * bmean
        candidate_var += vc
        baseline_var += vb
        covariance += vcb
        rows.append({"stratum": row["stratum"], "population_blocks": population,
                     "sample_blocks": sample, "candidate_mean_cycles": cmean,
                     "baseline_mean_cycles": bmean,
                     "finite_population_fraction": fraction})
    require(candidate_total > 0 and baseline_total > 0, "nonpositive total")
    speedup = baseline_total / candidate_total
    log_variance = max(0.0, baseline_var / (baseline_total ** 2) +
                       candidate_var / (candidate_total ** 2) -
                       2.0 * covariance / (baseline_total * candidate_total))
    log_half = T_CRITICAL * math.sqrt(log_variance)
    return {
        "candidate_total_cycles_estimate": candidate_total,
        "candidate_ci95": [max(0.0, candidate_total - T_CRITICAL * math.sqrt(candidate_var)),
                           candidate_total + T_CRITICAL * math.sqrt(candidate_var)],
        "baseline_total_cycles_estimate": baseline_total,
        "baseline_ci95": [max(0.0, baseline_total - T_CRITICAL * math.sqrt(baseline_var)),
                          baseline_total + T_CRITICAL * math.sqrt(baseline_var)],
        "paired_speedup_estimate": speedup,
        "paired_speedup_ci95": [math.exp(math.log(speedup) - log_half),
                                math.exp(math.log(speedup) + log_half)],
        "t_critical": T_CRITICAL,
        "strata": rows,
        "metric": "block-reset executable schedule cycles; not continuous-M785 cycles",
    }


def validate_contract(path=CONTRACT):
    value = strict_json(path)
    require(value.get("schema") == SCHEMA and
            value.get("status") == "SOURCE_PLAN_ONLY__NO_WINDOW_EXECUTION" and
            value.get("launch_now") is False, "M1009 contract identity drift")
    require(value["layers"] == {"included_exact": ["D0", "D2", "D3"],
                                "D1": "STRICT_COMMON_CHARGE_NO_WINDOW"},
            "layer boundary drift")
    require(value["schedule"]["metric"] ==
            "BLOCK_RESET_EXECUTABLE_LAYER_CYCLES" and
            value["schedule"]["continuous_m785_cycle_estimate_allowed"] is False,
            "schedule metric drift")
    require([row["name"] for row in value["strata"]] ==
            ["SOURCE_INIT_CENSUS", "COMPUTE_REGULAR", "DEPENDENCY_STRESS",
             "COMMIT_TAIL"], "strata drift")
    require(value["sampling"]["pilot_per_noncensus_stratum"] == 8 and
            value["sampling"]["window_expanded_request_cap"] == 10000 and
            value["sampling"]["selection_after_index_before_cycles"] is True,
            "sampling plan drift")
    required = set(value["exact_miter_required_fields"])
    require(required == {"total_cycles", "expanded_request_count",
                         "compressed_transaction_count", "scheduled_requests",
                         "compressed_schedule", "transaction_address_sha256",
                         "commit_sequence_sha256", "population_ids", "configs",
                         "cycle_classes", "same_cycle_response_slot_reuse",
                         "terminal_readiness", "terminal_readiness_sha256",
                         "port_calendars"}, "exact field set drift")
    require(value["decision_gates"]["commit_windows_require_positive_commit"] is True and
            value["decision_gates"]["ci95_relative_halfwidth_target"] == 0.05 and
            value["decision_gates"]["ci95_relative_halfwidth_hard_stop"] == 0.10,
            "decision gate drift")
    require(all(value["claim_boundary"][name] is False for name in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "full_row_run_authorized",
                 "transaction_ratio_is_speedup")), "claim boundary expanded")
    for name, item in value["source_identity"].items():
        source = HW / item["path"]
        require(source.is_file() and not source.is_symlink() and
                sha256(source) == item["sha256"], "source drift: " + name)
    evidence = value["m1008_evidence"]
    directory = HW / evidence["directory"]
    sealed = verify_flat_seal(directory)
    require(sha256(directory / "review.json") == evidence["review_sha256"] and
            sealed["manifest_sha256"] == evidence["manifest_sha256"] and
            sealed["outer_seal_file_sha256"] == evidence["outer_seal_file_sha256"],
            "M1008 identity drift")
    return {"status": "PASS_M1009_SOURCE_PLAN_STATIC_VALIDATION__NO_EXECUTION",
            "contract_sha256": sha256(path), "m1008_seal": sealed,
            "window_execution": False, "full_row_execution": False,
            "eda_gpu_remote_used": False}


def self_test():
    census = estimate_paired_totals([
        {"stratum": "COMPUTE_REGULAR", "population_blocks": 4,
         "candidate_cycles": [10, 20, 30, 40],
         "baseline_cycles": [20, 30, 40, 50]},
        {"stratum": "DEPENDENCY_STRESS", "population_blocks": 2,
         "candidate_cycles": [5, 7], "baseline_cycles": [6, 9]},
    ], fixed_candidate=100, fixed_baseline=100)
    require(census["candidate_total_cycles_estimate"] == 212 and
            census["baseline_total_cycles_estimate"] == 255 and
            census["candidate_ci95"] == [212, 212] and
            census["baseline_ci95"] == [255, 255], "census estimator drift")
    partial = estimate_paired_totals([
        {"stratum": "COMMIT_TAIL", "population_blocks": 8,
         "candidate_cycles": [8, 10, 9, 13],
         "baseline_cycles": [12, 15, 14, 18]},
    ], fixed_candidate=1, fixed_baseline=1)
    require(partial["candidate_ci95"][0] <
            partial["candidate_total_cycles_estimate"] <
            partial["candidate_ci95"][1], "partial CI drift")
    return {"status": "PASS_M1009_FINITE_POPULATION_ESTIMATOR_SELFTEST",
            "census": census, "partial": partial,
            "window_execution": False, "full_row_execution": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    require(args.validate_contract != args.self_test, "select exactly one static mode")
    value = validate_contract(args.contract) if args.validate_contract else self_test()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
