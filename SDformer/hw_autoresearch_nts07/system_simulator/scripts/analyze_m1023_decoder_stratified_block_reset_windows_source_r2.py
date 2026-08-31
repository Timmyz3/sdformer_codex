#!/usr/bin/env python3
"""M1023 additive r2 repair of the M1014 source-only window protocol.

The frozen M1014 implementation remains byte-identical.  This wrapper closes
the three M1017 P0 findings: strict recursive metadata admission, exact paired
reset semantics/charge equality, and a publication-safe CI admission state
machine.  The CLI remains synthetic/source-only and cannot execute real
decoder windows.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
BASE_PATH = HERE / "analyze_m1014_decoder_stratified_block_reset_windows_source.py"
BASE_SHA256 = "c1fb987bd6d9921286fd9c53f3c9374d9c4779d9b3617946ab9b3d7ab11e2c64"
CONTRACT = HW / "contracts/m1023_decoder_stratified_block_reset_windows_source_r2_contract_r1_20260829.json"
M1017 = HW / "reviews/m1017_m1014_decoder_stratified_block_reset_windows_source_hammer_r1_20260829"
SCHEMA = "m1023_decoder_stratified_block_reset_windows_source_r2_v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def load_pinned(path, expected, name):
    require(Path(path).is_file() and not Path(path).is_symlink(), name + " absent")
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location("m1023_" + name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_pinned(BASE_PATH, BASE_SHA256, "frozen_m1014")
M946, M896, M890, M785 = BASE.M946, BASE.M896, BASE.M890, BASE.M785
CompressedTransaction = BASE.CompressedTransaction
WindowSpec = BASE.WindowSpec
ALLOWED_LAYERS = BASE.ALLOWED_LAYERS
STRATA = BASE.STRATA
PILOT_PER_STRATUM = BASE.PILOT_PER_STRATUM
MAX_PER_STRATUM = BASE.MAX_PER_STRATUM
WINDOW_EXPANDED_REQUEST_CAP = BASE.WINDOW_EXPANDED_REQUEST_CAP
SELECTION_SEED = BASE.SELECTION_SEED


# Exact, scalar, pre-cycle schema.  Unknown fields fail even when their names
# do not look timing-related.  This prevents a future alias from silently
# becoming a selection feature.
METADATA_SCHEMA = {
    "block_id": str,
    "source_init": bool,
    "commit_count": int,
    "psum_external_move_count": int,
    "weight_refill_external_count": int,
    "max_dependency_fan_in": int,
    "compute_count": int,
    "layer": str,
    "sample_id": int,
    "timestep": int,
    "destination": (str, int),
    "output_block": (str, int),
    "subblock": (str, int),
    "population_id": str,
    "config": str,
    "source_service_group_count": int,
    "dense_commit_address_count": int,
    "compressed_transaction_count": int,
    "expanded_request_count": int,
}
REQUIRED_METADATA = {"block_id"}
NONNEGATIVE_INTEGER_FIELDS = {
    "commit_count", "psum_external_move_count",
    "weight_refill_external_count", "max_dependency_fan_in",
    "compute_count", "sample_id", "timestep",
    "source_service_group_count", "dense_commit_address_count",
    "compressed_transaction_count", "expanded_request_count",
}
TIMING_TOKENS = (
    "cycle", "latency", "timing", "runtime", "elapsed", "wallclock",
    "speedup", "throughput", "framespersecond", "fps", "nanosecond",
    "microsecond", "millisecond",
)


def _normalized_key(key):
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def _semantic_field_scan(value, path=()):
    """Recursively reject timing/speed fields at any spelling or depth."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            # `timestep` is frozen identity metadata, not measured time.
            if str(key) != "timestep" and any(
                    token in normalized for token in TIMING_TOKENS):
                raise RuntimeError(
                    "cycle/latency/time/speedup semantic field forbidden: " +
                    ".".join(path + (str(key),)))
            _semantic_field_scan(item, path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _semantic_field_scan(item, path + (str(index),))


def validate_metadata_row(row):
    require(isinstance(row, Mapping), "metadata row must be a mapping")
    _semantic_field_scan(row)
    keys = set(row)
    require(REQUIRED_METADATA <= keys, "required metadata field absent")
    unknown = keys - set(METADATA_SCHEMA)
    require(not unknown, "unknown pre-cycle metadata field: " +
            ",".join(sorted(str(key) for key in unknown)))
    for key, value in row.items():
        expected = METADATA_SCHEMA[key]
        require(not isinstance(value, (dict, list, tuple)),
                "nested metadata value forbidden: " + key)
        require(isinstance(value, expected) and
                not (isinstance(value, bool) and expected is not bool),
                "metadata type drift: " + key)
        if key in NONNEGATIVE_INTEGER_FIELDS:
            require(int(value) >= 0, "negative metadata value: " + key)
    if "layer" in row:
        BASE.reject_d1(str(row["layer"]))
    if "sample_id" in row:
        require(int(row["sample_id"]) == 0, "sample identity drift")
    if "timestep" in row:
        require(int(row["timestep"]) == 0, "timestep identity drift")
    return dict(row)


def classify_stratum(metadata):
    row = validate_metadata_row(metadata)
    return BASE.classify_stratum(row)


def deterministic_select(index, stratum, requested):
    require(stratum in STRATA, "unknown selection stratum")
    requested = int(requested)
    limit = 1 if stratum == "SOURCE_INIT_CENSUS" else MAX_PER_STRATUM
    require(1 <= requested <= limit, "selection count exceeds frozen bound")
    candidates = []
    identities = set()
    for raw in index:
        row = validate_metadata_row(raw)
        identity = str(row["block_id"])
        require(identity not in identities, "duplicate block identity")
        identities.add(identity)
        if BASE.classify_stratum(row) == stratum:
            candidates.append((canonical_sha([SELECTION_SEED, identity]),
                               identity, raw))
    require(candidates, "empty stratum")
    return [row for _, _, row in sorted(candidates)[:min(requested,
                                                          len(candidates))]]


def frozen_route(layer):
    return BASE.frozen_route(layer)


def block_reset_transactions(body, spec, side):
    return BASE.block_reset_transactions(body, spec, side)


def exact_replay(transactions, spec):
    return BASE.exact_replay(transactions, spec)


def _reset_service_semantics(transactions, body_count):
    """Canonical request/service/cycle charge for boundary/fill/drain."""
    require(len(transactions) >= 3 and body_count == len(transactions) - 3,
            "reset/body transaction shape drift")
    reset_rows = (("boundary", transactions[0]),
                  ("fill", transactions[1]),
                  ("drain", transactions[-1]))
    body_last = M890.terminal_token(transactions[-2])
    boundary_token = M890.terminal_token(transactions[0])
    resource = BASE._resource_for(transactions)
    scheduler = M896.RUNGTLSScheduler(resource)
    output = []
    for role, tx in reset_rows:
        resource_name, port, operation = scheduler._resource(tx.kind)
        port_name = scheduler._port_name(port, operation)
        denominator = (resource.external_bytes_per_cycle
                       if resource_name == "external" else port.row_bytes)
        beats = max(1, math.ceil(int(tx.width_bytes) / int(denominator)))
        service = max(int(port.initiation_interval), beats)
        latency = (int(port.read_latency) if operation == "read"
                   else int(port.write_latency))
        distance = latency + beats - 1
        dependency_roles = []
        for dependency in tx.dependency_tokens:
            if role == "fill" and dependency == boundary_token:
                dependency_roles.append("BOUNDARY_READY")
            elif role == "drain" and dependency == body_last:
                dependency_roles.append("BODY_LAST_TERMINAL")
            else:
                dependency_roles.append("UNEXPECTED:" + dependency)
        offsets = (tuple(tx.address_offsets) if tx.address_offsets else
                   tuple(bank * int(tx.width_bytes)
                         for bank in tx.bank_pattern))
        addresses = [int(tx.base_address) + int(offset) for offset in offsets]
        output.append({
            "role": role,
            "kind": tx.kind,
            "operation": operation,
            "resource": resource_name,
            "port": port_name,
            "base_address": int(tx.base_address),
            "address_stride_bytes": int(tx.address_stride_bytes),
            "addresses": addresses,
            "bank_pattern": [int(value) for value in tx.bank_pattern],
            "address_offsets": [int(value) for value in tx.address_offsets],
            "width_bytes": int(tx.width_bytes),
            "count": int(tx.count),
            "earliest_issue_cycle": int(tx.earliest_issue_cycle),
            "dependency_roles": dependency_roles,
            "row_or_external_bytes_per_cycle": int(denominator),
            "beats_per_request": beats,
            "initiation_interval": int(port.initiation_interval),
            "latency_cycles": latency,
            "service_cycle_charge": service,
            "return_distance_cycles": distance,
            "outstanding_per_bank": int(port.outstanding_per_bank),
        })
    require(output[0]["dependency_roles"] == [] and
            output[1]["dependency_roles"] == ["BOUNDARY_READY"] and
            output[2]["dependency_roles"] == ["BODY_LAST_TERMINAL"],
            "reset dependency-role graph drift")
    return output


def paired_replay(candidate_body, baseline_body, spec):
    """Paired replay with exact reset request and service-charge equality."""
    candidate, cmeta = block_reset_transactions(candidate_body, spec,
                                                 "candidate")
    baseline, bmeta = block_reset_transactions(baseline_body, spec,
                                               "baseline")
    creset = _reset_service_semantics(candidate, len(candidate_body))
    breset = _reset_service_semantics(baseline, len(baseline_body))
    require(creset == breset,
            "candidate/baseline reset semantic or cycle charge asymmetry")
    require(cmeta["reset_expanded_request_count"] ==
            bmeta["reset_expanded_request_count"] == 3,
            "paired reset request count drift")
    cresult = exact_replay(candidate, spec)
    bresult = exact_replay(baseline, spec)
    return {
        "window_identity_sha256": spec.identity_sha256,
        "stratum": spec.stratum,
        "candidate_cycles": cresult["total_cycles"],
        "baseline_cycles": bresult["total_cycles"],
        "candidate": cresult,
        "baseline": bresult,
        "candidate_reset": cmeta,
        "baseline_reset": bmeta,
        "paired_reset_service_cycle_sequence": creset,
        "paired_reset_semantics_sha256": canonical_sha(creset),
        "paired_reset_exact_equal": True,
        "transaction_ratio_is_speedup": False,
    }


def _relative_halfwidth(point, interval):
    point = float(point)
    require(point > 0 and len(interval) == 2, "invalid CI point/interval")
    low, high = float(interval[0]), float(interval[1])
    require(0 <= low <= point <= high, "CI does not contain point")
    return max(point - low, high - point) / point


def apply_ci_publication_gate(raw):
    """Only API that can expose an estimated point outside raw internals."""
    widths = {
        "candidate_cycles": _relative_halfwidth(
            raw["candidate_total_cycles_estimate"], raw["candidate_ci95"]),
        "baseline_cycles": _relative_halfwidth(
            raw["baseline_total_cycles_estimate"], raw["baseline_ci95"]),
        "paired_speedup": _relative_halfwidth(
            raw["paired_speedup_estimate"], raw["paired_speedup_ci95"]),
    }
    worst = max(widths.values())
    output = dict(raw)
    output["ci95_relative_halfwidth"] = widths
    output["ci95_relative_halfwidth_max"] = worst
    output["point_estimate_admitted"] = False
    output["paper_citable"] = False
    if worst > 0.10:
        output["status"] = (
            "NO_POINT_ESTIMATE_CI95_RELATIVE_HALFWIDTH_ABOVE_10_PERCENT")
        output["candidate_total_cycles_estimate"] = None
        output["baseline_total_cycles_estimate"] = None
        output["paired_speedup_estimate"] = None
        output["adaptive_action"] = "HARD_STOP_REPORT_BOUNDS_AND_COVERAGE_ONLY"
    elif worst > 0.05:
        output["status"] = (
            "DIAGNOSTIC_POINT_ONLY_CI95_RELATIVE_HALFWIDTH_5_TO_10_PERCENT")
        output["adaptive_action"] = "ADAPT_SAMPLE_BY_VARIANCE_BELOW_CAP"
    else:
        output["status"] = "PRECISE_POINT_ELIGIBLE_FOR_LATER_RELEASE"
        output["point_estimate_admitted"] = True
        output["adaptive_action"] = "NONE"
    return output


def estimate_paired_totals(strata, fixed_candidate=0.0, fixed_baseline=0.0):
    raw = BASE.estimate_paired_totals(
        strata, fixed_candidate=fixed_candidate,
        fixed_baseline=fixed_baseline)
    return apply_ci_publication_gate(raw)


def validate_source(contract_path=CONTRACT):
    contract = M785.strict_json(contract_path)
    require(contract["schema"] == SCHEMA and
            contract["status"] == "R2_SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False, "M1023 contract drift")
    require(sha256(BASE_PATH) == BASE_SHA256, "M1014 r1 identity drift")
    require(contract["repairs"] == {
        "recursive_strict_metadata_allowlist": True,
        "paired_reset_full_semantic_and_cycle_charge_equality": True,
        "ci_publication_state_machine": True}, "repair contract drift")
    require(contract["sampling"]["pilot"] == 8 and
            contract["sampling"]["maximum"] == 32 and
            contract["sampling"]["window_cap"] == 10000,
            "sampling bound drift")
    require(all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "real_window_execution_authorized",
                 "eda_gpu_remote_used")), "claim boundary expanded")
    return {
        "status": "PASS_M1023_R2_SOURCE_VALIDATION__NO_REAL_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "launch_now": False,
        "real_payload_opened": False,
        "real_window_execution": False,
        "eda_gpu_remote_used": False,
    }


def self_test():
    # P0-1: aliases, case changes and nested paths all fail before selection.
    rejected_fields = []
    attacks = (
        {"block_id": "a", "compute_count": 1, "total_cycles": 1},
        {"block_id": "a", "compute_count": 1, "TOTAL_CyClEs": 1},
        {"block_id": "a", "compute_count": 1, "Latency_NS": 1},
        {"block_id": "a", "compute_count": 1,
         "diagnostics": {"Elapsed_Time": 1}},
        {"block_id": "a", "compute_count": 1,
         "nested": [{"SpeedUp": 2.0}]},
    )
    for row in attacks:
        try:
            deterministic_select([row], "COMPUTE_REGULAR", 1)
        except RuntimeError as error:
            require("semantic field forbidden" in str(error),
                    "wrong selector failure mode")
            rejected_fields.append(True)
    require(len(rejected_fields) == len(attacks), "selector attack accepted")

    # P0-2: normal pair passes; a service mutation with count unchanged fails.
    body = M890.synthetic_transactions(448)
    spec = WindowSpec("m1023-synthetic-commit", "D0", "COMMIT_TAIL", 1)
    normal = paired_replay(body, body, spec)
    original = block_reset_transactions
    mutation_rejected = False

    def asymmetric(body_arg, spec_arg, side_arg):
        rows, metadata = original(body_arg, spec_arg, side_arg)
        if side_arg == "baseline":
            from dataclasses import replace
            rows[0] = replace(rows[0], kind="external_read")
        return rows, metadata

    globals()["block_reset_transactions"] = asymmetric
    try:
        paired_replay(body, body, spec)
    except RuntimeError as error:
        mutation_rejected = "reset semantic or cycle charge asymmetry" in str(error)
    finally:
        globals()["block_reset_transactions"] = original
    require(mutation_rejected, "asymmetric reset attack accepted")

    # P0-3: >10% hides all points, 5-10 diagnostic, <=5 admitted.
    hard = estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
        "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
        "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1],
    }])
    require(hard["ci95_relative_halfwidth_max"] > 0.10 and
            hard["candidate_total_cycles_estimate"] is None and
            hard["baseline_total_cycles_estimate"] is None and
            hard["paired_speedup_estimate"] is None and
            hard["status"].startswith("NO_POINT_ESTIMATE"),
            "CI hard-stop attack accepted")
    precise = estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 8,
        "candidate_cycles": [10] * 8,
        "baseline_cycles": [20] * 8,
    }])
    require(precise["status"] ==
            "PRECISE_POINT_ELIGIBLE_FOR_LATER_RELEASE" and
            precise["point_estimate_admitted"] is True,
            "precise CI state drift")
    # Directly exercise the middle state with a valid estimator-shaped value;
    # this isolates publication policy from sample construction.
    diagnostic = apply_ci_publication_gate({
        "candidate_total_cycles_estimate": 100.0,
        "candidate_ci95": [94.0, 106.0],
        "baseline_total_cycles_estimate": 120.0,
        "baseline_ci95": [114.0, 126.0],
        "paired_speedup_estimate": 1.2,
        "paired_speedup_ci95": [1.128, 1.272],
    })
    require(diagnostic["status"].startswith("DIAGNOSTIC_POINT_ONLY") and
            diagnostic["point_estimate_admitted"] is False and
            diagnostic["paired_speedup_estimate"] == 1.2,
            "diagnostic CI state drift")
    return {
        "status": "PASS_M1023_R2_SMALL_SYNTHETIC_P0_REPAIR_SELFTEST",
        "p0_1_recursive_selector_attacks_rejected": len(rejected_fields),
        "p0_2_reset_asymmetry_rejected": mutation_rejected,
        "p0_2_normal_pair_cycles": [normal["candidate_cycles"],
                                    normal["baseline_cycles"]],
        "p0_2_reset_semantics_sha256":
            normal["paired_reset_semantics_sha256"],
        "p0_3_hard_stop_status": hard["status"],
        "p0_3_diagnostic_status": diagnostic["status"],
        "p0_3_precise_status": precise["status"],
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
