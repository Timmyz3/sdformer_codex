#!/usr/bin/env python3
"""M1034 additive r3 recursive CI-publication redaction repair.

M1023 r2 remains frozen.  This source changes only the estimator publication
boundary: all three CI states share one explicit envelope, and the >10% state
contains no numeric point estimate at any nesting depth.  The CLI is still
source/synthetic-only and cannot open real decoder payloads.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Mapping


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
BASE_PATH = HERE / "analyze_m1023_decoder_stratified_block_reset_windows_source_r2.py"
BASE_SHA256 = "8e9ce843499cbcfdfe1856e5f829218e0329cd299ce25d1ba93e3b45cd74d2b2"
CONTRACT = HW / "contracts/m1034_decoder_stratified_block_reset_windows_source_r3_contract_r1_20260829.json"
SCHEMA = "m1034_decoder_stratified_block_reset_windows_source_r3_v1"
PUBLIC_SCHEMA = "m1034_ci_publication_envelope_v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path, expected, name):
    require(Path(path).is_file() and not Path(path).is_symlink(), name + " absent")
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location("m1034_" + name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_pinned(BASE_PATH, BASE_SHA256, "frozen_m1023_r2")
M946, M896, M890, M785 = BASE.M946, BASE.M896, BASE.M890, BASE.M785
CompressedTransaction = BASE.CompressedTransaction
WindowSpec = BASE.WindowSpec
ALLOWED_LAYERS = BASE.ALLOWED_LAYERS
STRATA = BASE.STRATA
PILOT_PER_STRATUM = BASE.PILOT_PER_STRATUM
MAX_PER_STRATUM = BASE.MAX_PER_STRATUM
WINDOW_EXPANDED_REQUEST_CAP = BASE.WINDOW_EXPANDED_REQUEST_CAP

# Re-export the M1023-closed selector and reset semantics without alteration.
validate_metadata_row = BASE.validate_metadata_row
classify_stratum = BASE.classify_stratum
deterministic_select = BASE.deterministic_select
frozen_route = BASE.frozen_route
block_reset_transactions = BASE.block_reset_transactions
exact_replay = BASE.exact_replay
paired_replay = BASE.paired_replay


TOP_KEYS = {
    "schema", "status", "state", "bounds", "uncertainty", "coverage",
    "identity", "admission", "point_estimates",
}
BOUND_KEYS = {
    "candidate_total_cycles_ci95", "baseline_total_cycles_ci95",
    "paired_speedup_ci95",
}
UNCERTAINTY_KEYS = {
    "candidate_cycles_relative_halfwidth",
    "baseline_cycles_relative_halfwidth",
    "paired_speedup_relative_halfwidth", "maximum_relative_halfwidth",
    "t_critical",
}
COVERAGE_ROW_KEYS = {
    "stratum", "population_blocks", "sample_blocks",
    "finite_population_fraction",
}
ADMISSION_KEYS = {
    "point_estimate_admitted", "paper_citable", "adaptive_action",
}
POINT_KEYS = {
    "candidate_total_cycles", "baseline_total_cycles", "paired_speedup",
}


def _relative_halfwidth(point, interval):
    point = float(point)
    require(point > 0 and len(interval) == 2, "invalid CI point/interval")
    low, high = float(interval[0]), float(interval[1])
    require(0 <= low <= point <= high, "CI does not contain point")
    return max(point - low, high - point) / point


def _coverage_projection(raw):
    rows = []
    for row in raw["strata"]:
        require(COVERAGE_ROW_KEYS <= set(row), "raw coverage row incomplete")
        rows.append({key: row[key] for key in sorted(COVERAGE_ROW_KEYS)})
    return {"strata": rows}


def _walk_numeric_paths(value, path=()):
    output = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            output.extend(_walk_numeric_paths(item, path + (str(key),)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.extend(_walk_numeric_paths(item, path + (str(index),)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output.append((".".join(path), float(value)))
    return output


def validate_publication_envelope(value):
    """Recursive schema validator; rejects hidden/unknown numeric leaves."""
    require(isinstance(value, Mapping) and set(value) == TOP_KEYS,
            "publication top-level schema drift")
    require(value["schema"] == PUBLIC_SCHEMA, "publication schema drift")
    require(set(value["bounds"]) == BOUND_KEYS, "bounds schema drift")
    require(set(value["uncertainty"]) == UNCERTAINTY_KEYS,
            "uncertainty schema drift")
    require(set(value["coverage"]) == {"strata"}, "coverage schema drift")
    require(all(set(row) == COVERAGE_ROW_KEYS
                for row in value["coverage"]["strata"]),
            "coverage row schema drift")
    require(set(value["identity"]) == {"metric"}, "identity schema drift")
    require(set(value["admission"]) == ADMISSION_KEYS,
            "admission schema drift")
    points = value["point_estimates"]
    require(points is None or
            (isinstance(points, Mapping) and set(points) == POINT_KEYS),
            "point-estimate schema drift")
    state = value["state"]
    if state == "HARD_STOP_ABOVE_10_PERCENT":
        require(points is None, "hard-stop point estimate must be null")
        require(value["admission"]["point_estimate_admitted"] is False,
                "hard-stop admission drift")
    elif state == "DIAGNOSTIC_5_TO_10_PERCENT":
        require(points is not None and
                value["admission"]["point_estimate_admitted"] is False,
                "diagnostic point/admission drift")
    elif state == "CANDIDATE_AT_MOST_5_PERCENT":
        require(points is not None and
                value["admission"]["point_estimate_admitted"] is True,
                "candidate point/admission drift")
    else:
        raise RuntimeError("unknown publication state")
    require(value["admission"]["paper_citable"] is False,
            "source publication became paper-citable")

    allowed_numeric_prefixes = (
        "bounds.candidate_total_cycles_ci95.",
        "bounds.baseline_total_cycles_ci95.",
        "bounds.paired_speedup_ci95.",
        "uncertainty.",
        "coverage.strata.",
    )
    if state != "HARD_STOP_ABOVE_10_PERCENT":
        allowed_numeric_prefixes += ("point_estimates.",)
    for path, _ in _walk_numeric_paths(value):
        require(path.startswith(allowed_numeric_prefixes),
                "numeric value outside recursive publication allowlist: " + path)
    return True


def publication_projection(raw):
    """Construct a new envelope; never redact a mutable raw object in place."""
    widths = {
        "candidate_cycles_relative_halfwidth": _relative_halfwidth(
            raw["candidate_total_cycles_estimate"], raw["candidate_ci95"]),
        "baseline_cycles_relative_halfwidth": _relative_halfwidth(
            raw["baseline_total_cycles_estimate"], raw["baseline_ci95"]),
        "paired_speedup_relative_halfwidth": _relative_halfwidth(
            raw["paired_speedup_estimate"], raw["paired_speedup_ci95"]),
    }
    worst = max(widths.values())
    uncertainty = dict(widths)
    uncertainty["maximum_relative_halfwidth"] = worst
    uncertainty["t_critical"] = float(raw["t_critical"])
    if worst > 0.10:
        state = "HARD_STOP_ABOVE_10_PERCENT"
        status = "NO_POINT_ESTIMATE_RECURSIVELY_REDACTED_CI95_ABOVE_10_PERCENT"
        points = None
        action = "REPORT_BOUNDS_WIDTH_COVERAGE_COUNT_IDENTITY_ONLY"
        admitted = False
    elif worst > 0.05:
        state = "DIAGNOSTIC_5_TO_10_PERCENT"
        status = "DIAGNOSTIC_POINT_NOT_ADMITTED_CI95_5_TO_10_PERCENT"
        points = {
            "candidate_total_cycles": raw["candidate_total_cycles_estimate"],
            "baseline_total_cycles": raw["baseline_total_cycles_estimate"],
            "paired_speedup": raw["paired_speedup_estimate"],
        }
        action = "ADAPT_SAMPLE_BY_VARIANCE_BELOW_CAP"
        admitted = False
    else:
        state = "CANDIDATE_AT_MOST_5_PERCENT"
        status = "POINT_CANDIDATE_FOR_LATER_INDEPENDENT_RELEASE"
        points = {
            "candidate_total_cycles": raw["candidate_total_cycles_estimate"],
            "baseline_total_cycles": raw["baseline_total_cycles_estimate"],
            "paired_speedup": raw["paired_speedup_estimate"],
        }
        action = "NONE"
        admitted = True
    output = {
        "schema": PUBLIC_SCHEMA,
        "status": status,
        "state": state,
        "bounds": {
            "candidate_total_cycles_ci95": list(raw["candidate_ci95"]),
            "baseline_total_cycles_ci95": list(raw["baseline_ci95"]),
            "paired_speedup_ci95": list(raw["paired_speedup_ci95"]),
        },
        "uncertainty": uncertainty,
        "coverage": _coverage_projection(raw),
        "identity": {"metric": raw["metric"]},
        "admission": {
            "point_estimate_admitted": admitted,
            "paper_citable": False,
            "adaptive_action": action,
        },
        "point_estimates": points,
    }
    validate_publication_envelope(output)
    return output


def estimate_paired_totals(strata, fixed_candidate=0.0, fixed_baseline=0.0):
    # Call the frozen M1009 raw design-based estimator directly.  M1023's
    # top-level-only redaction object is intentionally bypassed.
    raw = BASE.BASE.estimate_paired_totals(
        strata, fixed_candidate=fixed_candidate,
        fixed_baseline=fixed_baseline)
    return publication_projection(raw)


def validate_source(contract_path=CONTRACT):
    contract = M785.strict_json(contract_path)
    require(contract["schema"] == SCHEMA and
            contract["status"] == "R3_SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False, "M1034 contract drift")
    require(sha256(BASE_PATH) == BASE_SHA256, "M1023 r2 identity drift")
    require(contract["repair"] == {
        "single_recursive_publication_projection": True,
        "hard_stop_numeric_point_leaves": "ZERO_AT_ANY_DEPTH",
        "hard_stop_point_estimates": None}, "r3 repair contract drift")
    require(all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "real_window_execution_authorized",
                 "eda_gpu_remote_used")), "claim boundary expanded")
    return {
        "status": "PASS_M1034_R3_SOURCE_VALIDATION__NO_REAL_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "launch_now": False,
        "real_payload_opened": False,
        "real_window_execution": False,
        "eda_gpu_remote_used": False,
    }


def self_test():
    high = estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
        "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
        "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1],
    }])
    require(high["state"] == "HARD_STOP_ABOVE_10_PERCENT" and
            high["point_estimates"] is None, "hard-stop projection drift")
    numeric = _walk_numeric_paths(high)
    require(not any("mean" in path or "estimate" in path or
                    ("speedup" in path and "ci95" not in path and
                     "halfwidth" not in path)
                    for path, _ in numeric), "nested hard-stop point leak")
    require(all("candidate_mean_cycles" not in row and
                "baseline_mean_cycles" not in row
                for row in high["coverage"]["strata"]),
            "M1024 stratum mean leak remains")
    injected = dict(high)
    injected["coverage"] = {"strata": [dict(
        high["coverage"]["strata"][0], candidate_mean_cycles=50.5)]}
    rejected = False
    try:
        validate_publication_envelope(injected)
    except RuntimeError:
        rejected = True
    require(rejected, "recursive injected point leak accepted")
    precise = estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 8,
        "candidate_cycles": [10] * 8,
        "baseline_cycles": [20] * 8,
    }])
    require(precise["state"] == "CANDIDATE_AT_MOST_5_PERCENT" and
            precise["point_estimates"]["paired_speedup"] == 2.0,
            "precise state drift")
    return {
        "status": "PASS_M1034_R3_RECURSIVE_REDACTION_SYNTHETIC_SELFTEST",
        "hard_stop_state": high["state"],
        "hard_stop_numeric_leaf_paths": [path for path, _ in numeric],
        "hard_stop_point_estimates": None,
        "m1024_nested_mean_leaks": [],
        "injected_nested_leak_rejected": True,
        "precise_state": precise["state"],
        "launch_now": False,
        "real_payload_opened": False,
        "real_window_execution": False,
        "eda_gpu_remote_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-source", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    require(args.validate_source != args.self_test,
            "select exactly one source-only mode")
    result = validate_source() if args.validate_source else self_test()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
