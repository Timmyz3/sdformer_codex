#!/usr/bin/env python3
"""M1281 source-only decoder cycle/traffic surrogate calibration framework.

The only executable CLI mode is a synthetic 120-call fixture self-test.  This
source never opens the live M1111DR2 work prefix or a canonical decoder result.
A future separately reviewed adapter may import ``calibrate_payload`` only
after it has independently verified a sealed 120/120 M1111DR2 result.
"""
import copy
from decimal import Decimal, getcontext
import json
import sys
from typing import Any, Dict, Iterable, List, Tuple

getcontext().prec = 50

LAYERS = ("D0", "D1", "D2", "D3")
CALLS = 120
CALLS_PER_LAYER = 30
SLOPE_CYCLES_PER_GROUP = 4
DESCRIPTOR_BYTES_PER_GROUP = 16
WEIGHT_BYTES_PER_ACTIVE_SOURCE_TERM = 16
PSUM_READ_BYTES_PER_GROUP = 288
PSUM_WRITE_BYTES_PER_GROUP = 288
COMMIT_BYTES_PER_CALL = 288
MAX_RELATIVE_ERROR_GATE = Decimal("0.001")  # 0.1%


class CalibrationError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise CalibrationError(message)


def exact_keys(value: Any, expected: Iterable[str], label: str) -> None:
    require(isinstance(value, dict) and set(value) == set(expected),
            label + " key set drift")


def strict_json_text(text: str) -> Dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject(token):
        raise CalibrationError("nonfinite JSON: " + token)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root is not object")
    return value


def positive_int(value: Any, label: str) -> int:
    require(type(value) is int and value > 0, label + " must be positive int")
    return value


def nonnegative_int(value: Any, label: str) -> int:
    require(type(value) is int and value >= 0, label + " must be nonnegative int")
    return value


def sha_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            label + " must be lowercase SHA256")
    return value


def expected_traffic(group_count: int, active_source_terms: int) -> Dict[str, int]:
    positive_int(group_count, "group_count")
    nonnegative_int(active_source_terms, "active_source_terms")
    result = {
        "descriptor_bytes": group_count * DESCRIPTOR_BYTES_PER_GROUP,
        "weight_bytes": active_source_terms * WEIGHT_BYTES_PER_ACTIVE_SOURCE_TERM,
        "psum_read_bytes": group_count * PSUM_READ_BYTES_PER_GROUP,
        "compute_count": group_count,
        "psum_write_bytes": group_count * PSUM_WRITE_BYTES_PER_GROUP,
        "commit_bytes": COMMIT_BYTES_PER_CALL,
    }
    result["total_bytes"] = (result["descriptor_bytes"] + result["weight_bytes"] +
                             result["psum_read_bytes"] +
                             result["psum_write_bytes"] + result["commit_bytes"])
    return result


def validate_authority(authority: Dict[str, Any], synthetic_fixture: bool) -> None:
    exact_keys(authority, ("sealed_call_count", "result_sha256",
        "result_outer_seal_file_sha256", "result_hammer_review_sha256",
        "result_outer_seal_pass", "result_hammer_pass", "synthetic_fixture"),
        "authority")
    require(type(authority["sealed_call_count"]) is int and
            authority["sealed_call_count"] == CALLS,
            "authority is not sealed 120/120")
    sha_string(authority["result_sha256"], "result SHA")
    sha_string(authority["result_outer_seal_file_sha256"], "outer seal SHA")
    sha_string(authority["result_hammer_review_sha256"], "hammer SHA")
    require(authority["result_outer_seal_pass"] is True and
            authority["result_hammer_pass"] is True and
            authority["synthetic_fixture"] is synthetic_fixture,
            "authority pass/fixture boundary drift")


def validate_call(row: Dict[str, Any], ordinal: int) -> Dict[str, int]:
    exact_keys(row, ("global_call_ordinal", "layer", "group_count",
        "active_source_terms", "measured_cycles", "descriptor_bytes",
        "weight_bytes", "psum_read_bytes", "compute_count",
        "psum_write_bytes", "commit_bytes"), "call")
    require(type(row["global_call_ordinal"]) is int and
            row["global_call_ordinal"] == ordinal, "call ordinal drift")
    require(row["layer"] == LAYERS[ordinal % len(LAYERS)], "layer order drift")
    group_count = positive_int(row["group_count"], "group_count")
    terms = nonnegative_int(row["active_source_terms"], "active_source_terms")
    positive_int(row["measured_cycles"], "measured_cycles")
    expected = expected_traffic(group_count, terms)
    for key in ("descriptor_bytes", "weight_bytes", "psum_read_bytes",
                "compute_count", "psum_write_bytes", "commit_bytes"):
        require(type(row[key]) is int and row[key] == expected[key],
                key + " conservation drift")
    return expected


def mean(values: List[Decimal]) -> Decimal:
    require(values, "empty mean")
    return sum(values, Decimal(0)) / Decimal(len(values))


def decimal_string(value: Decimal) -> str:
    return format(value, ".18f")


def calibrate_payload(payload: Dict[str, Any], synthetic_fixture: bool = False) -> Dict[str, Any]:
    """Fit one additive constant per layer with fixed slope 4 cycles/group.

    This function trusts no path. A future adapter must verify actual result and
    result-hammer seals before constructing a non-fixture payload.
    """
    exact_keys(payload, ("schema", "authority", "calls", "claim_boundary"),
               "payload")
    require(payload["schema"] ==
            "m1281_decoder_surrogate_calibration_input_v1", "payload schema drift")
    validate_authority(payload["authority"], synthetic_fixture)
    require(payload["claim_boundary"] == {
        "calibration_only": True,
        "system_speedup_admitted": False,
        "paper_ppa_ready": False,
        "truth_performance_publish_before_sealed_120": False,
    }, "input claim boundary drift")
    rows = payload["calls"]
    require(isinstance(rows, list) and len(rows) == CALLS,
            "calibration requires exactly 120 calls")

    residuals = {layer: [] for layer in LAYERS}
    traffic_total = {key: 0 for key in ("descriptor_bytes", "weight_bytes",
        "psum_read_bytes", "compute_count", "psum_write_bytes", "commit_bytes",
        "total_bytes")}
    for ordinal, row in enumerate(rows):
        expected = validate_call(row, ordinal)
        for key in traffic_total:
            traffic_total[key] += expected[key]
        layer = row["layer"]
        residuals[layer].append(Decimal(row["measured_cycles"]) -
                                Decimal(SLOPE_CYCLES_PER_GROUP * row["group_count"]))
    require(all(len(residuals[layer]) == CALLS_PER_LAYER for layer in LAYERS),
            "per-layer call population drift")

    constants = {layer: mean(residuals[layer]) for layer in LAYERS}
    layer_errors = {}
    all_relative = []
    all_absolute = []
    for layer in LAYERS:
        rel = []
        absolute = []
        for row in rows:
            if row["layer"] != layer:
                continue
            predicted = (Decimal(SLOPE_CYCLES_PER_GROUP * row["group_count"]) +
                         constants[layer])
            error = abs(predicted - Decimal(row["measured_cycles"]))
            relative = error / Decimal(row["measured_cycles"])
            absolute.append(error); rel.append(relative)
            all_absolute.append(error); all_relative.append(relative)
        layer_errors[layer] = {
            "calls": len(rel),
            "constant_cycles": decimal_string(constants[layer]),
            "mean_absolute_error_cycles": decimal_string(mean(absolute)),
            "max_absolute_error_cycles": decimal_string(max(absolute)),
            "mean_relative_error": decimal_string(mean(rel)),
            "max_relative_error": decimal_string(max(rel)),
        }

    maximum = max(all_relative)
    gate = maximum <= MAX_RELATIVE_ERROR_GATE
    annex_allowed = gate and not synthetic_fixture
    status = ("PASS_SYNTHETIC_CALIBRATION_ERROR_GATE__ANNEX_FORBIDDEN"
              if gate and synthetic_fixture else
              "PASS_SEALED_CALIBRATION_ERROR_GATE__ANALYTICAL_CYCLE_ANNEX_ELIGIBLE"
              if gate else
              "STOP_CALIBRATION_ERROR_GATE__NO_ANALYTICAL_CYCLE_ANNEX")
    return {
        "schema": "m1281_decoder_cycle_traffic_surrogate_calibration_v1",
        "status": status,
        "calibration_only": True,
        "synthetic_fixture": synthetic_fixture,
        "population": {"calls": CALLS, "calls_per_layer": CALLS_PER_LAYER,
                       "layers": list(LAYERS)},
        "cycle_surrogate": {
            "equation": "cycles = 4 * group_count + layer_constant",
            "slope_cycles_per_group": SLOPE_CYCLES_PER_GROUP,
            "fit": "per-layer arithmetic mean of measured_cycles - 4*group_count",
            "layer_results": layer_errors,
            "global_mean_absolute_error_cycles": decimal_string(mean(all_absolute)),
            "global_max_absolute_error_cycles": decimal_string(max(all_absolute)),
            "global_mean_relative_error": decimal_string(mean(all_relative)),
            "global_max_relative_error": decimal_string(maximum),
            "gate_max_relative_error": decimal_string(MAX_RELATIVE_ERROR_GATE),
            "error_gate_pass": gate,
            "analytical_cycle_annex_allowed": annex_allowed,
        },
        "traffic_surrogate": {
            "equation": {
                "descriptor_bytes": "16 * group_count",
                "weight_bytes": "16 * active_source_terms",
                "psum_read_bytes": "288 * group_count",
                "compute_count": "group_count",
                "psum_write_bytes": "288 * group_count",
                "commit_bytes": "288 per call",
            },
            "exact_fixture_conservation": True,
            "totals": traffic_total,
        },
        "authority_projection": copy.deepcopy(payload["authority"]),
        "claim_boundary": {
            "calibration_only": True,
            "diagnostic_only": True,
            "analytical_cycle_annex": annex_allowed,
            "truth_performance_numbers_published": False,
            "speedup_admitted": False,
            "system_speedup_admitted": False,
            "paper_ppa_ready": False,
            "energy": False,
            "final_checkpoint_rebind_required": True,
            "independent_result_hammer_required": True,
        },
    }


def synthetic_payload(noisy: bool = False) -> Dict[str, Any]:
    constants = {"D0": 17, "D1": 23, "D2": 31, "D3": 41}
    rows = []
    for ordinal in range(CALLS):
        layer = LAYERS[ordinal % 4]
        group_count = 1000 + ordinal * 37 + (ordinal % 7)
        active_terms = group_count * 3 + ordinal % 11
        traffic = expected_traffic(group_count, active_terms)
        measured = SLOPE_CYCLES_PER_GROUP * group_count + constants[layer]
        if noisy and ordinal == 119:
            measured += 1000
        rows.append({
            "global_call_ordinal": ordinal,
            "layer": layer,
            "group_count": group_count,
            "active_source_terms": active_terms,
            "measured_cycles": measured,
            "descriptor_bytes": traffic["descriptor_bytes"],
            "weight_bytes": traffic["weight_bytes"],
            "psum_read_bytes": traffic["psum_read_bytes"],
            "compute_count": traffic["compute_count"],
            "psum_write_bytes": traffic["psum_write_bytes"],
            "commit_bytes": traffic["commit_bytes"],
        })
    return {
        "schema": "m1281_decoder_surrogate_calibration_input_v1",
        "authority": {
            "sealed_call_count": 120,
            "result_sha256": "1" * 64,
            "result_outer_seal_file_sha256": "2" * 64,
            "result_hammer_review_sha256": "3" * 64,
            "result_outer_seal_pass": True,
            "result_hammer_pass": True,
            "synthetic_fixture": True,
        },
        "calls": rows,
        "claim_boundary": {
            "calibration_only": True,
            "system_speedup_admitted": False,
            "paper_ppa_ready": False,
            "truth_performance_publish_before_sealed_120": False,
        },
    }


def expect_reject(name: str, payload: Dict[str, Any], mutation) -> str:
    attacked = copy.deepcopy(payload)
    mutation(attacked)
    try:
        calibrate_payload(attacked, synthetic_fixture=True)
    except CalibrationError:
        return name
    raise CalibrationError("attack accepted: " + name)


def run_self_test() -> Dict[str, Any]:
    fixture = synthetic_payload()
    result = calibrate_payload(fixture, synthetic_fixture=True)
    require(result["cycle_surrogate"]["error_gate_pass"] is True and
            result["cycle_surrogate"]["analytical_cycle_annex_allowed"] is False and
            result["cycle_surrogate"]["global_max_relative_error"] ==
                "0.000000000000000000" and
            result["claim_boundary"]["analytical_cycle_annex"] is False,
            "exact fixture gate/boundary drift")
    noisy = calibrate_payload(synthetic_payload(noisy=True), synthetic_fixture=True)
    require(noisy["cycle_surrogate"]["analytical_cycle_annex_allowed"] is False and
            noisy["status"].startswith("STOP_"), "noisy fixture should miss gate")
    attacks = []
    attacks.append(expect_reject("119_calls", fixture,
        lambda value: value["calls"].pop()))
    attacks.append(expect_reject("ordinal_drift", fixture,
        lambda value: value["calls"][1].__setitem__("global_call_ordinal", 0)))
    attacks.append(expect_reject("layer_order_drift", fixture,
        lambda value: value["calls"][0].__setitem__("layer", "D1")))
    attacks.append(expect_reject("descriptor_byte_drift", fixture,
        lambda value: value["calls"][0].__setitem__("descriptor_bytes", 1)))
    attacks.append(expect_reject("weight_byte_drift", fixture,
        lambda value: value["calls"][0].__setitem__("weight_bytes", 1)))
    attacks.append(expect_reject("psum_read_drift", fixture,
        lambda value: value["calls"][0].__setitem__("psum_read_bytes", 1)))
    attacks.append(expect_reject("compute_count_drift", fixture,
        lambda value: value["calls"][0].__setitem__("compute_count", 1)))
    attacks.append(expect_reject("psum_write_drift", fixture,
        lambda value: value["calls"][0].__setitem__("psum_write_bytes", 1)))
    attacks.append(expect_reject("commit_byte_drift", fixture,
        lambda value: value["calls"][0].__setitem__("commit_bytes", 1)))
    attacks.append(expect_reject("unsealed_authority", fixture,
        lambda value: value["authority"].__setitem__("result_outer_seal_pass", False)))
    attacks.append(expect_reject("unhammered_authority", fixture,
        lambda value: value["authority"].__setitem__("result_hammer_pass", False)))
    attacks.append(expect_reject("claim_promotion", fixture,
        lambda value: value["claim_boundary"].__setitem__("system_speedup_admitted", True)))
    attacks.append(expect_reject("negative_group", fixture,
        lambda value: value["calls"][0].__setitem__("group_count", -1)))
    try:
        strict_json_text('{"x":1,"x":2}')
        raise CalibrationError("duplicate JSON accepted")
    except CalibrationError:
        attacks.append("duplicate_json")
    try:
        strict_json_text('{"x":NaN}')
        raise CalibrationError("nonfinite JSON accepted")
    except CalibrationError:
        attacks.append("nonfinite_json")
    require(len(attacks) == 15 and len(set(attacks)) == 15,
            "self-test attack coverage drift")
    return {
        "schema": "m1281_decoder_surrogate_calibration_selftest_v1",
        "status": "PASS_M1281_SYNTHETIC_FIXTURE_AND_FAIL_CLOSED_SELFTEST",
        "synthetic_fixture_only": True,
        "exact_fixture": result,
        "noisy_fixture_gate_rejected": True,
        "attack_cases_rejected": len(attacks),
        "attacks": attacks,
        "live_work_prefix_opened": False,
        "calibration_only": True,
        "system_speedup_admitted": False,
        "paper_ppa_ready": False,
    }


def main() -> int:
    require(sys.argv == [sys.argv[0], "--self-test"],
            "M1281 CLI permits only --self-test")
    print(json.dumps(run_self_test(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalibrationError as error:
        print("M1281_FAIL_CLOSED: " + str(error))
        raise SystemExit(2)
