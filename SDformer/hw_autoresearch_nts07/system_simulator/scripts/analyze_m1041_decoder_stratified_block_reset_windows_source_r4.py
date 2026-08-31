#!/usr/bin/env python3
"""M1041 additive r4 strong-typed publication-envelope repair.

M1034 r3 remains frozen.  This source wraps its publication projection with
an exact JSON value-shape contract.  It is source/synthetic-only and exposes
no real decoder payload, execution-runner, EDA, GPU, or remote surface.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Mapping


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
BASE_PATH = HERE / "analyze_m1034_decoder_stratified_block_reset_windows_source_r3.py"
BASE_SHA256 = "155ebe3e19cb42e42afe3f26358f0598e8d33bad9558f450237cffc53eb4691a"
CONTRACT = HW / "contracts/m1041_decoder_stratified_block_reset_windows_source_r4_contract_r1_20260829.json"
SCHEMA = "m1041_decoder_stratified_block_reset_windows_source_r4_v1"
PUBLIC_SCHEMA = "m1041_ci_publication_envelope_v2"


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
    spec = importlib.util.spec_from_file_location("m1041_" + name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_pinned(BASE_PATH, BASE_SHA256, "frozen_m1034_r3")
M946, M896, M890, M785 = BASE.M946, BASE.M896, BASE.M890, BASE.M785
CompressedTransaction = BASE.CompressedTransaction
WindowSpec = BASE.WindowSpec
ALLOWED_LAYERS = BASE.ALLOWED_LAYERS
STRATA = BASE.STRATA
PILOT_PER_STRATUM = BASE.PILOT_PER_STRATUM
MAX_PER_STRATUM = BASE.MAX_PER_STRATUM
WINDOW_EXPANDED_REQUEST_CAP = BASE.WINDOW_EXPANDED_REQUEST_CAP

# Re-export the frozen M1034/M1023 selection, routing, reset, and replay path.
validate_metadata_row = BASE.validate_metadata_row
classify_stratum = BASE.classify_stratum
deterministic_select = BASE.deterministic_select
frozen_route = BASE.frozen_route
block_reset_transactions = BASE.block_reset_transactions
exact_replay = BASE.exact_replay
paired_replay = BASE.paired_replay


TOP_KEYS = BASE.TOP_KEYS
BOUND_KEYS = BASE.BOUND_KEYS
UNCERTAINTY_KEYS = BASE.UNCERTAINTY_KEYS
COVERAGE_ROW_KEYS = BASE.COVERAGE_ROW_KEYS
ADMISSION_KEYS = BASE.ADMISSION_KEYS
POINT_KEYS = BASE.POINT_KEYS

SEMANTIC_POINT_TOKENS = {
    "cycle", "cycles", "mean", "sum", "estimate", "speedup", "fps",
    "throughput", "latency", "time", "runtime",
}
ALLOWED_SEMANTIC_PATHS = {
    "bounds.candidate_total_cycles_ci95",
    "bounds.baseline_total_cycles_ci95",
    "bounds.paired_speedup_ci95",
    "uncertainty.candidate_cycles_relative_halfwidth",
    "uncertainty.baseline_cycles_relative_halfwidth",
    "uncertainty.paired_speedup_relative_halfwidth",
    "admission.point_estimate_admitted",
    "point_estimates",
    "point_estimates.candidate_total_cycles",
    "point_estimates.baseline_total_cycles",
    "point_estimates.paired_speedup",
}


def _finite_scalar(value):
    return (isinstance(value, (int, float)) and
            not isinstance(value, bool) and math.isfinite(value))


def _walk_public_json(value, path=()):
    """Return every public JSON node and reject non-JSON/non-finite values."""
    output = [(".".join(path), value)]
    if isinstance(value, Mapping):
        require(type(value) is dict, "public mapping must be an exact dict")
        for key, item in value.items():
            require(type(key) is str and key, "public key must be nonempty string")
            output.extend(_walk_public_json(item, path + (key,)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.extend(_walk_public_json(item, path + (str(index),)))
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        pass
    elif _finite_scalar(value):
        pass
    else:
        raise RuntimeError("non-JSON or non-finite public value at " + ".".join(path))
    return output


def _semantic_tokens(key):
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    tokens = set()
    for raw in re.findall(r"[A-Za-z]+", expanded):
        token = raw.lower()
        tokens.add(token)
        if token.endswith("ies") and len(token) > 3:
            tokens.add(token[:-3] + "y")
        if token.endswith("es") and len(token) > 2:
            tokens.add(token[:-2])
        if token.endswith("s") and len(token) > 1:
            tokens.add(token[:-1])
    return tokens


def _reject_semantic_point_keys(value, path=()):
    """Reject semantic point keys at every mapping depth unless explicit."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = path + (key,)
            dotted = ".".join(child)
            if _semantic_tokens(key) & SEMANTIC_POINT_TOKENS:
                require(dotted in ALLOWED_SEMANTIC_PATHS,
                        "semantic point key forbidden at depth: " + dotted)
            _reject_semantic_point_keys(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_semantic_point_keys(item, path + (str(index),))


def _validate_bounds(bounds):
    require(type(bounds) is dict and set(bounds) == BOUND_KEYS,
            "bounds schema drift")
    for key, interval in bounds.items():
        require(type(interval) is list and len(interval) == 2,
                "bound must be flat finite length-2 scalar interval: " + key)
        require(all(_finite_scalar(item) for item in interval),
                "bound must be flat finite length-2 scalar interval: " + key)
        require(interval[0] <= interval[1], "bound order drift: " + key)


def _validate_uncertainty(uncertainty):
    require(type(uncertainty) is dict and set(uncertainty) == UNCERTAINTY_KEYS,
            "uncertainty schema drift")
    require(all(_finite_scalar(value) for value in uncertainty.values()),
            "uncertainty leaf must be finite scalar")
    for key in UNCERTAINTY_KEYS - {"t_critical"}:
        require(uncertainty[key] >= 0, "relative uncertainty range drift: " + key)
    require(uncertainty["t_critical"] > 0, "t-critical range drift")


def _validate_coverage(coverage):
    require(type(coverage) is dict and set(coverage) == {"strata"} and
            type(coverage["strata"]) is list and coverage["strata"],
            "coverage schema drift")
    seen = set()
    for row in coverage["strata"]:
        require(type(row) is dict and set(row) == COVERAGE_ROW_KEYS,
                "coverage row schema drift")
        require(type(row["stratum"]) is str and row["stratum"] in STRATA and
                row["stratum"] not in seen, "coverage stratum identity drift")
        seen.add(row["stratum"])
        population = row["population_blocks"]
        sample = row["sample_blocks"]
        fraction = row["finite_population_fraction"]
        require(type(population) is int and population > 0,
                "coverage population must be positive exact int")
        require(type(sample) is int and 0 < sample <= population,
                "coverage sample must be positive exact int within population")
        require(_finite_scalar(fraction) and 0 < fraction <= 1,
                "coverage fraction scalar/range drift")
        require(math.isclose(fraction, sample / population,
                             rel_tol=1e-12, abs_tol=1e-12),
                "coverage fraction identity drift")


def _validate_identity(identity):
    require(type(identity) is dict and set(identity) == {"metric"} and
            type(identity["metric"]) is str and identity["metric"],
            "identity schema/type drift")


def _validate_admission(admission):
    require(type(admission) is dict and set(admission) == ADMISSION_KEYS,
            "admission schema drift")
    require(type(admission["point_estimate_admitted"]) is bool and
            admission["paper_citable"] is False and
            type(admission["adaptive_action"]) is str and
            admission["adaptive_action"], "admission type/boundary drift")


def _validate_points(points):
    require(type(points) is dict and set(points) == POINT_KEYS,
            "point-estimate schema drift")
    require(all(_finite_scalar(value) and value > 0 for value in points.values()),
            "point estimate must be positive finite scalar")


def validate_publication_envelope(value):
    """Strong recursive schema and value-shape validator for public JSON."""
    require(type(value) is dict and set(value) == TOP_KEYS,
            "publication top-level schema drift")
    _walk_public_json(value)
    _reject_semantic_point_keys(value)
    require(value["schema"] == PUBLIC_SCHEMA, "publication schema drift")
    require(type(value["status"]) is str and value["status"],
            "publication status type drift")
    require(type(value["state"]) is str, "publication state type drift")
    _validate_bounds(value["bounds"])
    _validate_uncertainty(value["uncertainty"])
    _validate_coverage(value["coverage"])
    _validate_identity(value["identity"])
    _validate_admission(value["admission"])

    state = value["state"]
    points = value["point_estimates"]
    admitted = value["admission"]["point_estimate_admitted"]
    status = value["status"]
    action = value["admission"]["adaptive_action"]
    if state == "HARD_STOP_ABOVE_10_PERCENT":
        require(points is None and admitted is False,
                "hard-stop point/admission drift")
        require(status ==
                "NO_POINT_ESTIMATE_RECURSIVELY_REDACTED_CI95_ABOVE_10_PERCENT" and
                action == "REPORT_BOUNDS_WIDTH_COVERAGE_COUNT_IDENTITY_ONLY",
                "hard-stop status/action drift")
    elif state == "DIAGNOSTIC_5_TO_10_PERCENT":
        _validate_points(points)
        require(admitted is False, "diagnostic admission drift")
        require(status == "DIAGNOSTIC_POINT_NOT_ADMITTED_CI95_5_TO_10_PERCENT" and
                action == "ADAPT_SAMPLE_BY_VARIANCE_BELOW_CAP",
                "diagnostic status/action drift")
    elif state == "CANDIDATE_AT_MOST_5_PERCENT":
        _validate_points(points)
        require(admitted is True, "candidate admission drift")
        require(status == "POINT_CANDIDATE_FOR_LATER_INDEPENDENT_RELEASE" and
                action == "NONE", "candidate status/action drift")
    else:
        raise RuntimeError("unknown publication state")
    require(math.isclose(
        value["uncertainty"]["maximum_relative_halfwidth"],
        max(value["uncertainty"][key] for key in UNCERTAINTY_KEYS
            if key not in {"maximum_relative_halfwidth", "t_critical"}),
        rel_tol=1e-12, abs_tol=1e-12), "maximum uncertainty identity drift")
    return True


def publication_projection(raw):
    """Reuse r3 construction, rename schema, then apply the r4 strong type."""
    output = BASE.publication_projection(raw)
    output["schema"] = PUBLIC_SCHEMA
    validate_publication_envelope(output)
    return output


def estimate_paired_totals(strata, fixed_candidate=0.0, fixed_baseline=0.0):
    raw = BASE.BASE.BASE.estimate_paired_totals(
        strata, fixed_candidate=fixed_candidate,
        fixed_baseline=fixed_baseline)
    return publication_projection(raw)


def _high_result():
    return estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
        "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
        "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1],
    }])


def _m1035_attacks():
    attacks = []
    for key in ("cycle", "mean", "sum", "estimate", "speedup", "FPS",
                "throughput", "latency", "time"):
        attack = copy.deepcopy(_high_result())
        attack["bounds"]["candidate_total_cycles_ci95"] = {
            key: 50.5, "reported_bounds": [1.0, 100.0]}
        attacks.append(("bounds_nested_" + key, attack))
    attack = copy.deepcopy(_high_result())
    attack["uncertainty"]["t_critical"] = {
        "latency_cycles": 99.0, "t_critical": 2.365}
    attacks.append(("uncertainty_nested_latency", attack))
    attack = copy.deepcopy(_high_result())
    attack["coverage"]["strata"][0]["sample_blocks"] = {
        "cycle_sum": 404.0, "count": 8}
    attacks.append(("coverage_nested_cycle_sum", attack))
    return attacks


def validate_source(contract_path=CONTRACT):
    contract = M785.strict_json(contract_path)
    require(contract["schema"] == SCHEMA and
            contract["status"] == "R4_SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False, "M1041 contract drift")
    require(sha256(BASE_PATH) == BASE_SHA256, "M1034 r3 identity drift")
    require(contract["repair"] == {
        "bound_value_shape": "FLAT_FINITE_LENGTH_2_ORDERED_SCALARS",
        "uncertainty_value_shape": "EXACT_KEYS_FINITE_SCALARS",
        "coverage_value_shape": "EXACT_SCALAR_TYPES_AND_RANGES",
        "semantic_point_keys": "REJECT_AT_ANY_DEPTH",
        "m1035_regression_attacks": 11,
    }, "r4 repair contract drift")
    require(all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "real_window_execution_authorized",
                 "eda_gpu_remote_used")), "claim boundary expanded")
    return {
        "status": "PASS_M1041_R4_SOURCE_VALIDATION__NO_REAL_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "launch_now": False,
        "real_payload_opened": False,
        "real_window_execution": False,
        "eda_gpu_remote_used": False,
    }


def self_test():
    high = _high_result()
    require(high["state"] == "HARD_STOP_ABOVE_10_PERCENT" and
            high["point_estimates"] is None and
            validate_publication_envelope(high), "canonical hard stop drift")
    rejected = []
    for name, attack in _m1035_attacks():
        try:
            validate_publication_envelope(attack)
        except RuntimeError:
            rejected.append(name)
    require(len(rejected) == 11, "M1035 recursive attack survived")
    require(BASE.deterministic_select is deterministic_select and
            BASE.block_reset_transactions is block_reset_transactions and
            BASE.paired_replay is paired_replay,
            "M1034 selector/reset function identity drift")
    return {
        "status": "PASS_M1041_R4_STRONG_TYPED_ENVELOPE_SYNTHETIC_SELFTEST",
        "m1035_attacks_rejected": rejected,
        "m1035_attack_count": len(rejected),
        "hard_stop_point_estimates": None,
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
