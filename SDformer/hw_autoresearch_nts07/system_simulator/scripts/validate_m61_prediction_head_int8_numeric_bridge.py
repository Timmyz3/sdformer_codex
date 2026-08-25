#!/usr/bin/env python3
"""Fail-closed validator for the sealed M61 INT8 numeric bridge."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


PARENTS = ("zero", "left", "up", "previous_timestep")


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


def minimum_signed_bits(lower, upper):
    for bits in range(2, 65):
        if lower >= -(1 << (bits - 1)) and upper <= (1 << (bits - 1)) - 1:
            return bits
    raise ValueError("signed width overflow")


def finite_number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)), "non-finite/non-number {}".format(name))
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--m60-result", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--gpu-blocker-observation", required=True, type=Path)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-gpu-blocker-sha256", required=True)
    arguments = parser.parse_args()

    contract = strict_json(arguments.contract)
    identities = contract["identity"]
    require(contract["schema"] ==
            "m61_prediction_head_int8_numeric_bridge_contract_v1" and
            contract["status"] == "FROZEN_EXACT_HEAD_INT8_NUMERIC_ONLY",
            "contract schema/status mismatch")
    require(sha256_path(arguments.analyzer) == identities["analyzer_sha256"],
            "analyzer identity mismatch")
    require(sha256_path(arguments.manifest) == identities["manifest_sha256"],
            "manifest identity mismatch")
    require(sha256_path(arguments.m60_result) == identities["m60_result_sha256"],
            "M60 result identity mismatch")
    manifest = strict_json(arguments.manifest)
    module = manifest["module_identities"][identities["module_name"]]
    require(module["weight"]["content_sha256"] ==
            identities["float_weight_sha256"] and
            module["bias"]["content_sha256"] ==
            identities["float_bias_sha256"],
            "float head identity mismatch")

    filenames = contract["artifact_filenames"]
    result_path = arguments.result_dir / filenames["result_json"]
    require(result_path.is_file() and
            sha256_path(result_path) == arguments.expected_result_sha256,
            "sealed result SHA mismatch")
    result = strict_json(result_path)
    require(result["schema"] ==
            "m61_prediction_head_int8_numeric_bridge_result_v1" and
            result["status"] ==
            "PASS_INT8_NUMERIC_BRIDGE_VALID825_QUANTIZED_OPEN",
            "result schema/status mismatch")
    require(result["contract_sha256"] == sha256_path(arguments.contract) and
            result["upstream_identity"] == identities and
            result["claim_boundary"] == contract["claim_boundary"],
            "result contract/identity/claim mismatch")

    expected_sizes = {"weight_int8": 192, "bias_int32": 8,
                      "scale_float32": 8}
    artifact_paths = {}
    for key, expected_size in expected_sizes.items():
        path = arguments.result_dir / filenames[key]
        artifact_paths[key] = path
        row = result["artifacts"][key]
        require(path.is_file() and path.stat().st_size == expected_size and
                row == {"bytes": expected_size, "filename": filenames[key],
                        "sha256": sha256_path(path)},
                "artifact identity mismatch: {}".format(key))
    weights = struct.unpack("192b", artifact_paths["weight_int8"].read_bytes())
    bias = struct.unpack("<2i", artifact_paths["bias_int32"].read_bytes())
    scales = struct.unpack("<2f", artifact_paths["scale_float32"].read_bytes())
    quant = result["quantization"]
    require(list(bias) == quant["bias_int32"] and
            list(scales) == quant["per_output_scale_float32"] and
            quant["scheme"] ==
            "PER_OUTPUT_SYMMETRIC_SIGNED_INT8_NARROW_RANGE" and
            quant["rounding"] == "RNE_TIES_TO_EVEN",
            "quantization artifact/metadata mismatch")
    rows = (weights[:96], weights[96:])
    require([min(row) for row in rows] == quant["weight_int8_min_per_output"] and
            [max(row) for row in rows] == quant["weight_int8_max_per_output"] and
            all(value != -128 for value in weights) and
            all(scale > 0 and math.isfinite(scale) for scale in scales),
            "INT8 range/scale mismatch")

    proof = result["accumulator_proof"]
    required_bits = []
    for output, row in enumerate(rows):
        lower = sum(value for value in row if value < 0) + bias[output]
        upper = sum(value for value in row if value > 0) + bias[output]
        bits = minimum_signed_bits(lower, upper)
        expected = proof["per_output"][output]
        require(expected["bias_int32"] == bias[output] and
                expected["lower_bound_all_binary_inputs_inclusive"] == lower and
                expected["upper_bound_all_binary_inputs_inclusive"] == upper and
                expected["required_signed_bits"] == bits and
                expected["observed_min_ten_samples"] >= lower and
                expected["observed_max_ten_samples"] <= upper,
                "accumulator proof mismatch")
        required_bits.append(bits)
    require(proof["declared_signed_bits"] == max(required_bits) == 13 and
            "hybrid binary vector" in proof["sequential_update_proof"],
            "aggregate/sequential accumulator proof mismatch")

    m60 = strict_json(arguments.m60_result)
    tile = contract["tile_policy"]
    require(m60["selected_capacity_feasible_tile"] == tile,
            "M60 tile mismatch")
    selected = [row for row in m60["configurations"]
                if row["tile_h"] == tile["tile_h"] and
                row["tile_w"] == tile["tile_w"]]
    require(len(selected) == 1, "M60 selected row population mismatch")
    selected = selected[0]
    reconstruction = result["integer_parent_delta_reconstruction"]
    require(reconstruction == {
        "choice_counts": selected["choice_counts"],
        "mismatches": 0,
        "outputs": 15360000,
        "max_abs_difference": 0,
        "negative_residual_events": selected["negative_residual_events"],
        "positive_residual_events": selected["positive_residual_events"],
        "signed_values": [-1, 0, 1],
        "source_events": selected["source_bits"],
        "tile_h": tile["tile_h"],
        "tile_w": tile["tile_w"],
    }, "integer reconstruction/M60 ledger mismatch")
    require(sum(reconstruction["choice_counts"].values()) == 7680000 and
            reconstruction["positive_residual_events"] +
            reconstruction["negative_residual_events"] ==
            reconstruction["source_events"], "signed population mismatch")

    per_record = result["per_record"]
    require(len(per_record) == 10 and
            [row["sample_id"] for row in per_record] == list(range(10)),
            "per-record population/order mismatch")
    sum_choice = dict((name, 0) for name in PARENTS)
    sum_positive = sum_negative = sum_source = sum_outputs = 0
    for row in per_record:
        require(sum(row["choice_counts"].values()) == 768000 and
                len(row["signed_residual_sha256"]) == 64 and
                len(row["choice_sha256"]) == 64 and
                row["integer_reconstruction"]["mismatches"] == 0 and
                row["integer_reconstruction"]["max_abs_difference"] == 0,
                "per-record reconstruction/identity mismatch")
        for name in PARENTS:
            sum_choice[name] += int(row["choice_counts"][name])
        sum_positive += int(row["positive_residual_events"])
        sum_negative += int(row["negative_residual_events"])
        sum_source += int(row["source_events"])
        sum_outputs += int(row["integer_reconstruction"]["outputs"])
    require(sum_choice == reconstruction["choice_counts"] and
            sum_positive == reconstruction["positive_residual_events"] and
            sum_negative == reconstruction["negative_residual_events"] and
            sum_source == reconstruction["source_events"] and
            sum_outputs == reconstruction["outputs"],
            "per-record aggregate mismatch")

    numeric = result["numeric_perturbation_ten_real_samples"]
    overall = numeric["all_outputs"]
    channels = numeric["per_output"]
    require(overall["count"] == 15360000 and len(channels) == 2 and
            all(row["count"] == 7680000 for row in channels),
            "numeric population mismatch")
    for name in ("mae", "max_abs", "rmse", "reference_rms",
                 "normalized_rmse_over_reference_rms",
                 "sign_agreement_nonzero_reference"):
        finite_number(overall[name], "overall." + name)
        for index, row in enumerate(channels):
            finite_number(row[name], "channel{}.{}".format(index, name))
    require(abs(overall["mae"] -
                sum(row["mae"] for row in channels) / 2.0) < 1e-15 and
            abs(overall["rmse"] - math.sqrt(
                sum(row["rmse"] ** 2 for row in channels) / 2.0)) < 1e-15 and
            overall["normalized_rmse_over_reference_rms"] < 0.01 and
            overall["sign_agreement_nonzero_reference"] > 0.99 and
            finite_number(numeric[
                "two_channel_argmax_agreement_diagnostic_not_accuracy"],
                "argmax diagnostic") > 0.99,
            "numeric consistency/qualification mismatch")
    epe = numeric["vector_endpoint_perturbation_not_ground_truth_aee"]
    require(epe["count"] == 7680000 and
            0 <= finite_number(epe["mean"], "epe.mean") <=
            finite_number(epe["max"], "epe.max"),
            "endpoint perturbation mismatch")

    capacity = result["capacity_requalification"]
    original_component = selected["capacity"]["components_bytes"][
        "conditional_19b_output_accumulator_tile_pair"]
    values = 2 * tile["tile_h"] * tile["tile_w"] * 2
    qualified_bytes = (values * 13 + 7) // 8
    corrected = (selected["capacity"]["combined_capacity_bytes"] -
                 original_component + qualified_bytes)
    require(capacity["m60_conditional_accumulator_bits"] == 19 and
            capacity["qualified_accumulator_bits"] == 13 and
            capacity["qualified_accumulator_tile_pair_bytes"] == qualified_bytes and
            capacity["corrected_combined_capacity_bytes"] == corrected and
            capacity["corrected_headroom_bytes"] ==
            selected["capacity"]["maximum_combined_capacity_bytes"] - corrected and
            capacity["performance_ratio_unchanged"] is True,
            "capacity requalification mismatch")

    valid = result["valid825_accuracy_closure"]
    require(valid["original_profile_samples"] == 825 and
            valid["original_profile_sha256"] ==
            identities["valid825_profile_sha256"] and
            valid["quantized_head_valid825_status"] ==
            "OPEN_NOT_RUN_GPU_OCCUPIED_BY_EXISTING_TRAINING" and
            valid["ten_sample_numeric_perturbation_is_not_a_valid825_substitute"]
            is True, "valid825 claim boundary mismatch")
    blocker_path = arguments.gpu_blocker_observation
    require(blocker_path.is_file() and sha256_path(blocker_path) ==
            arguments.expected_gpu_blocker_sha256,
            "GPU blocker observation SHA mismatch")
    blocker = strict_json(blocker_path)
    require(blocker["schema"] ==
            "m61_quantized_valid825_gpu_blocker_observation_v1" and
            blocker["status"] == "OBSERVED_GPU_CONFLICT_DO_NOT_PREEMPT" and
            blocker["m61_result_sha256"] == arguments.expected_result_sha256 and
            blocker["gpu"]["exit_code"] == 0 and
            blocker["compute_apps"]["exit_code"] == 0 and
            len(blocker["training_process_lines"]) > 0 and
            len(blocker["training_configs"]) == 1 and
            blocker["training_configs"][0]["sha256"] ==
            "fac4b833014478324f224aded5784e99853d4bf0291423eb1ab83c319ce2865c" and
            "47175" in blocker["gpu"]["stdout"] and
            "47166" in blocker["compute_apps"]["stdout"],
            "GPU blocker evidence mismatch")

    print(json.dumps({
        "accumulator_bits": proof["declared_signed_bits"],
        "blocker_observation_sha256": sha256_path(blocker_path),
        "integer_mismatches": reconstruction["mismatches"],
        "numeric_mae": overall["mae"],
        "numeric_rmse": overall["rmse"],
        "result_sha256": sha256_path(result_path),
        "status": "PASS_M61_SEALED_INT8_NUMERIC_BRIDGE_VALID825_OPEN_WITH_OBSERVED_BLOCKER",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
