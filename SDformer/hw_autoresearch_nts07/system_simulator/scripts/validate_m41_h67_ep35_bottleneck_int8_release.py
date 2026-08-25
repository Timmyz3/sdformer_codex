#!/usr/bin/env python3
"""Pure-stdlib, Python-3.6-compatible validator for the frozen M41 release."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_PIN = HW_ROOT / "contracts/m41_h67_ep35_bottleneck_int8_release_pin_r1_20260823.json"
OPERATORS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
DELTAS = (121, 144, 97, 588)
BLOCK_SIZES = (16, 24, 48, 96)


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
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    def reject(token):
        raise ValueError("non-standard JSON number: {}".format(token))

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def exact_int(value, name, minimum=None, maximum=None):
    require(type(value) is int, "{} must be an exact integer".format(name))
    if minimum is not None:
        require(value >= minimum, "{} below minimum".format(name))
    if maximum is not None:
        require(value <= maximum, "{} above maximum".format(name))
    return value


def finite_float(value, name):
    require(type(value) in (int, float) and type(value) is not bool,
            "{} must be numeric".format(name))
    value = float(value)
    require(math.isfinite(value), "{} must be finite".format(name))
    return value


def exact_keys(value, expected, name):
    require(type(value) is dict, "{} must be an object".format(name))
    require(set(value) == set(expected), "{} key drift".format(name))


def signed_bits_for_symmetric_magnitude(magnitude):
    magnitude = exact_int(magnitude, "signed magnitude", 0)
    bits = 1
    while magnitude > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def resolve_under_root(relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute(),
            "release path must be a nonempty relative path")
    require(".." not in Path(relative).parts, "release path escape rejected")
    path = (ROOT / relative).resolve()
    require(str(path).startswith(str(ROOT.resolve()) + "/"),
            "release realpath escape rejected")
    return path


def validate_pinned_files(pin):
    exact_keys(pin, {"schema", "status", "files", "claim_boundary"}, "release pin")
    require(pin["schema"] == "m41_h67_ep35_bottleneck_int8_release_pin_v1",
            "release pin schema drift")
    require(pin["status"] == "FROZEN_VALIDATE_BEFORE_USE",
            "release pin status drift")
    files = pin["files"]
    require(type(files) is list and len(files) == 23,
            "release pin file population drift")
    observed = set()
    resolved = {}
    for index, receipt in enumerate(files):
        exact_keys(receipt, {"role", "path", "bytes", "sha256"},
                   "release pin file {}".format(index))
        path = resolve_under_root(receipt["path"])
        require(receipt["path"] not in observed, "duplicate pinned path")
        observed.add(receipt["path"])
        require(path.is_file(), "pinned file missing: {}".format(path))
        require(path.stat().st_size == exact_int(
            receipt["bytes"], "pinned bytes", 1), "pinned byte count drift")
        require(type(receipt["sha256"]) is str and len(receipt["sha256"]) == 64,
                "pinned SHA syntax drift")
        require(sha256(path) == receipt["sha256"],
                "pinned SHA mismatch: {}".format(receipt["path"]))
        require(receipt["role"] not in resolved,
                "duplicate pinned role: {}".format(receipt["role"]))
        resolved[receipt["role"]] = path
    required_roles = {
        "contract", "exporter", "validator", "tests", "result",
        "per_output_channel_csv", "per_source_term_multicast_csv",
    }
    required_roles.update("o{}_weight".format(i) for i in range(4))
    required_roles.update("o{}_scale_f32".format(i) for i in range(4))
    required_roles.update("o{}_scale_uq31".format(i) for i in range(4))
    required_roles.update("o{}_acc_init".format(i) for i in range(4))
    require(set(resolved) == required_roles, "release pin role population drift")
    return resolved


def validate_upstream_receipt(receipt, name):
    exact_keys(receipt, {"path", "sha256"}, name)
    path = resolve_under_root(receipt["path"])
    require(path.is_file() and sha256(path) == receipt["sha256"],
            "{} SHA drift".format(name))


def validate_contract(contract):
    require(contract["schema"] == "m41_h67_ep35_bottleneck_int8_bridge_contract_v1",
            "contract schema drift")
    require(contract["status"] == "FROZEN_FAIL_CLOSED_INPUT_AND_QUANTIZATION_POLICY",
            "contract status drift")
    identity = contract["identity"]
    for key in ("checkpoint", "m40_source_manifest", "m40_schedule_result",
                "m40_independent_review", "m35_canonical_descriptor_contract",
                "m35_r4_independent_review"):
        validate_upstream_receipt(identity[key], "contract.{}".format(key))
    require(tuple(identity["operators"]) == OPERATORS, "operator identity drift")
    require(identity["bn_policy"] == "no_running", "BN policy drift")
    quant = contract["quantization"]
    require(quant["weight_domain"] == "PER_OUTPUT_CHANNEL_SYMMETRIC_SIGNED_INT8",
            "quantization domain drift")
    require(quant["quantized_range"] == [-127, 127] and
            exact_int(quant["reserved_code"], "reserved code") == -128 and
            exact_int(quant["zero_point"], "zero point") == 0,
            "quantized code contract drift")
    require(quant["runtime_scale_application_admitted"] is False,
            "runtime scale claim opened")
    geometry = contract["geometry"]
    require(geometry["weight_shape_o_i_ky_kx"] == [768, 768, 3, 3] and
            geometry["conv_bias_present"] is False and
            exact_int(geometry["records"], "records") == 40,
            "geometry contract drift")
    require(contract["m40_layout_bridge"]["physical_layout_selected"] is False and
            contract["m40_layout_bridge"]["weight_load_evict_schedule_admitted"] is False,
            "M40 physical schedule claim opened")
    require(tuple(contract["value_mask_multicast_audit"][
        "output_channel_block_sizes"]) == BLOCK_SIZES,
        "multicast block-size drift")
    require(contract["late_scale_elision_audit"][
        "m35_amplitude_delta_by_operator"] == dict(zip(OPERATORS, DELTAS)),
        "late-scale delta mapping drift")


def signed_values(payload):
    return [value if value < 128 else value - 256 for value in bytearray(payload)]


def validate_weight_payload(payload, layer):
    require(len(payload) == 768 * 768 * 3 * 3, "weight payload extent mismatch")
    require(b"\x80" not in payload, "reserved -128 weight code")
    values = signed_values(payload)
    sums = [0] * 768
    zero_count = 0
    endpoint_count = 0
    q_min = 127
    q_max = -127
    for offset, value in enumerate(values):
        output_channel = offset % 768
        sums[output_channel] += abs(value)
        zero_count += int(value == 0)
        endpoint_count += int(abs(value) == 127)
        q_min = min(q_min, value)
        q_max = max(q_max, value)
    bound = layer["accumulator_bound"]
    require(min(sums) == bound["per_channel_sum_abs_q_minimum"] and
            max(sums) == bound["per_channel_sum_abs_q_maximum"] and
            abs(sum(sums) / 768.0 - bound["per_channel_sum_abs_q_mean"]) < 1e-12,
            "checkpoint accumulator-bound mismatch")
    require(max(signed_bits_for_symmetric_magnitude(value) for value in sums) ==
            exact_int(bound["checkpoint_tight_required_signed_bits"],
                      "tight accumulator bits"),
            "tight accumulator-width mismatch")
    require(q_min == layer["q_min"] and q_max == layer["q_max"] and
            zero_count == layer["q_zero_count"] and
            endpoint_count == layer["q_endpoint_count"] and
            layer["reserved_negative_128_count"] == 0,
            "weight payload population mismatch")
    return values, sums


def validate_scale_payloads(scale_bytes, scale_fixed_bytes, layer):
    require(len(scale_bytes) == len(scale_fixed_bytes) == 768 * 4,
            "scale payload extent mismatch")
    scales = [row[0] for row in struct.iter_unpack("<f", scale_bytes)]
    fixed = [row[0] for row in struct.iter_unpack("<I", scale_fixed_bytes)]
    require(all(math.isfinite(value) and value > 0.0 for value in scales),
            "scale payload has NaN/Infinity/nonpositive value")
    expected = [int(round(float(value) * (1 << 31))) for value in scales]
    require(fixed == expected, "UQ0.31 scale rounding mismatch")
    require(min(scales) == layer["scale"]["minimum_f32"] and
            max(scales) == layer["scale"]["maximum_f32"],
            "scale-range summary mismatch")
    require(layer["scale"]["runtime_application_admitted"] is False,
            "layer runtime scale claim opened")
    return scales, fixed


def parse_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_csv_int(row, key, minimum=None, maximum=None):
    try:
        value = int(row[key], 10)
    except (KeyError, TypeError, ValueError):
        raise ValueError("CSV integer parse failed: {}".format(key))
    if minimum is not None:
        require(value >= minimum, "CSV integer below minimum: {}".format(key))
    if maximum is not None:
        require(value <= maximum, "CSV integer above maximum: {}".format(key))
    return value


def parse_csv_float(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError("CSV float parse failed: {}".format(key))
    require(math.isfinite(value), "CSV NaN/Infinity rejected: {}".format(key))
    return value


def validate_channel_csv(rows, all_sums, layers):
    require(len(rows) == 3072, "per-channel CSV population mismatch")
    seen = set()
    for row in rows:
        operator_index = parse_csv_int(row, "operator_index", 0, 3)
        channel = parse_csv_int(row, "output_channel", 0, 767)
        key = (operator_index, channel)
        require(key not in seen, "duplicate per-channel CSV key")
        seen.add(key)
        require(row["operator"] == OPERATORS[operator_index], "channel operator drift")
        require(parse_csv_int(row, "weight_count") == 768 * 3 * 3,
                "channel weight count drift")
        require(parse_csv_int(row, "preclip_violation_count") == 0,
                "channel preclip violation")
        require(parse_csv_int(row, "sum_abs_q") == all_sums[operator_index][channel],
                "channel sum(abs(q)) mismatch")
        require(parse_csv_int(row, "tight_required_signed_bits") ==
                signed_bits_for_symmetric_magnitude(all_sums[operator_index][channel]),
                "channel accumulator bits mismatch")
        for name in ("weight_max_abs", "scale_f32", "mae", "rmse",
                     "max_abs_error", "normalized_l2_error"):
            require(parse_csv_float(row, name) >= 0.0, "negative channel metric")
    require(len(seen) == 3072, "per-channel key coverage mismatch")


def validate_reuse_csv(rows, all_values, layers):
    require(len(rows) == 4 * 768 * 9, "source-term CSV population mismatch")
    seen = set()
    aggregate = []
    for _ in range(4):
        aggregate.append({size: {
            "blocks": 0, "nonzero_destination_updates": 0,
            "value_mask_commands": 0, "exact_uncompressed_value_mask_bytes": 0,
            "dense_int8_bytes": 0,
        } for size in BLOCK_SIZES})
    unique_populations = [[] for _ in range(4)]
    top_populations = [[] for _ in range(4)]
    for row in rows:
        operator_index = parse_csv_int(row, "operator_index", 0, 3)
        ci = parse_csv_int(row, "input_channel", 0, 767)
        ky = parse_csv_int(row, "kernel_y", 0, 2)
        kx = parse_csv_int(row, "kernel_x", 0, 2)
        key = (operator_index, ci, ky, kx)
        require(key not in seen, "duplicate source-term CSV key")
        seen.add(key)
        require(row["operator"] == OPERATORS[operator_index], "reuse operator drift")
        base = ((ci * 3 + ky) * 3 + kx) * 768
        vector = all_values[operator_index][base:base + 768]
        counts = {}
        for value in vector:
            counts[value] = counts.get(value, 0) + 1
        unique_count = len(counts)
        top_value, top_count = sorted(
            counts.items(), key=lambda item: (-item[1], item[0]))[0]
        require(parse_csv_int(row, "unique_value_count_768") == unique_count and
                parse_csv_int(row, "top_value_s8", -127, 127) == top_value and
                parse_csv_int(row, "top_value_count") == top_count,
                "source-term reuse statistic mismatch")
        require(abs(parse_csv_float(row, "repeat_factor_768_div_unique") -
                    768.0 / unique_count) < 1e-12 and
                abs(parse_csv_float(row, "top_value_fraction") -
                    top_count / 768.0) < 1e-12,
                "source-term reuse ratio mismatch")
        unique_populations[operator_index].append(unique_count)
        top_populations[operator_index].append(top_count)
        for size in BLOCK_SIZES:
            commands = 0
            updates = 0
            for start in range(0, 768, size):
                block = vector[start:start + size]
                distinct_nonzero = set(value for value in block if value != 0)
                commands += len(distinct_nonzero)
                updates += sum(value != 0 for value in block)
            encoded = commands * (1 + size // 8)
            require(parse_csv_int(row, "b{}_nonzero_updates".format(size)) == updates and
                    parse_csv_int(row, "b{}_value_mask_commands".format(size)) == commands and
                    parse_csv_int(row, "b{}_value_mask_bytes".format(size)) == encoded and
                    parse_csv_int(row, "b{}_dense_int8_bytes".format(size)) == 768,
                    "blocked value-mask statistic mismatch")
            values = aggregate[operator_index][size]
            values["blocks"] += 768 // size
            values["nonzero_destination_updates"] += updates
            values["value_mask_commands"] += commands
            values["exact_uncompressed_value_mask_bytes"] += encoded
            values["dense_int8_bytes"] += 768
    require(len(seen) == 4 * 768 * 9, "source-term key coverage mismatch")
    for operator_index, layer in enumerate(layers):
        reuse = layer["value_mask_multicast"]
        require(reuse["unique_value_count"]["minimum"] ==
                min(unique_populations[operator_index]) and
                reuse["unique_value_count"]["maximum"] ==
                max(unique_populations[operator_index]) and
                reuse["top_value_count"]["minimum"] ==
                min(top_populations[operator_index]) and
                reuse["top_value_count"]["maximum"] ==
                max(top_populations[operator_index]),
                "layer reuse distribution boundary mismatch")
        for size in BLOCK_SIZES:
            expected = aggregate[operator_index][size]
            observed = reuse["blocked_value_mask"][str(size)]
            for key in expected:
                require(observed[key] == expected[key],
                        "layer blocked multicast aggregate mismatch")
            require(abs(observed["destination_updates_per_command"] -
                        expected["nonzero_destination_updates"] /
                        float(expected["value_mask_commands"])) < 1e-15,
                    "layer multicast command ratio mismatch")
            require(abs(observed["dense_int8_bytes_div_value_mask_bytes"] -
                        expected["dense_int8_bytes"] /
                        float(expected["exact_uncompressed_value_mask_bytes"])) < 1e-15,
                    "layer multicast byte ratio mismatch")


def validate_result(result, resolved, contract):
    require(result["schema"] == "m41_h67_ep35_bottleneck_int8_bridge_result_v1",
            "result schema drift")
    require(result["status"].startswith("PASS_CHECKPOINT_BOUND_FOUR_LAYER_INT8"),
            "result status drift")
    population = result["population"]
    require(exact_int(population["weights_audited"], "weights audited") == 21233664 and
            exact_int(population["raw_conv_output_values_compared"],
                      "raw conv outputs") == 92160000 and
            exact_int(population["preclip_violation_count"], "preclip") == 0,
            "result population drift")
    layers = result["layers"]
    require(type(layers) is list and len(layers) == 4, "result layer population drift")
    all_values = []
    all_sums = []
    for operator_index, layer in enumerate(layers):
        require(layer["operator"] == OPERATORS[operator_index] and
                exact_int(layer["operator_index"], "operator index") == operator_index,
                "result operator order drift")
        weight_bytes = resolved["o{}_weight".format(operator_index)].read_bytes()
        values, sums = validate_weight_payload(weight_bytes, layer)
        all_values.append(values)
        all_sums.append(sums)
        validate_scale_payloads(
            resolved["o{}_scale_f32".format(operator_index)].read_bytes(),
            resolved["o{}_scale_uq31".format(operator_index)].read_bytes(), layer)
        acc = resolved["o{}_acc_init".format(operator_index)].read_bytes()
        require(len(acc) == 3072 and not any(bytearray(acc)),
                "accumulator init must be all-zero signed32")
        error = layer["weight_error"]
        require(0.0 <= finite_float(error["normalized_l2_error"], "weight NRMSE") < 0.02 and
                0.0 <= finite_float(error["max_error_div_stored_scale"],
                                    "weight max-error/scale") <= 0.5,
                "weight error bound drift")
    validate_channel_csv(parse_csv(resolved["per_output_channel_csv"]), all_sums, layers)
    validate_reuse_csv(parse_csv(resolved["per_source_term_multicast_csv"]),
                       all_values, layers)

    local = result["local_raw_convolution_audit"]
    require(len(local["rows"]) == 40 and local["full_network_accuracy"] is None and
            local["valid825_accuracy"] is None,
            "local numeric scope drift")
    for name in OPERATORS:
        row = local["aggregate_by_layer"][name]
        require(row["gate_pass"] is True and
                finite_float(row["normalized_l2_error"], "local NRMSE") <= 0.03 and
                finite_float(row["cosine_similarity"], "local cosine") >= 0.999 and
                exact_int(row["values"], "local values") == 23040000,
                "local raw-convolution gate drift")

    late = result["late_scale_elision_audit"]
    require(late["m35_rounding_rtl_admitted"] is False and
            late["late_scale_elision_rtl_admitted"] is False and
            late["cycle_reduction_admitted"] is False,
            "late-scale claim boundary opened")
    for operator_index, name in enumerate(OPERATORS):
        row = late["aggregate_by_layer"][name]
        require(exact_int(row["delta"], "late-scale delta") == DELTAS[operator_index] and
                exact_int(row["values"], "late-scale values") == 23040000 and
                row["nonzero_accumulators"] + row["zero_accumulators"] == row["values"] and
                row["rne_exact_bypass"] + row["rne_changed"] == row["values"] and
                row["floor_exact_bypass"] + row["floor_changed"] == row["values"],
                "late-scale count conservation mismatch")
        require(abs(row["rne_exact_bypass_fraction_all"] -
                    row["rne_exact_bypass"] / float(row["values"])) < 1e-15 and
                abs(row["rne_exact_bypass_fraction_nonzero"] -
                    row["rne_exact_bypass_nonzero"] /
                    float(row["nonzero_accumulators"])) < 1e-15,
                "late-scale bypass ratio mismatch")

    bridge = result["m40_schedule_bridge"]
    require(bridge["total_int8_weight_bytes"] == 21233664 and
            bridge["checkpoint_tight_accumulator_signed_bits"] == 19 and
            bridge["dense_envelope_accumulator_signed_bits"] == 21 and
            bridge["full_96_output_tile_fits_current_residency"] is False and
            bridge["physical_layout_selected"] is False and
            bridge["real_address_cycle_schedule_admitted"] is False,
            "M40 bridge boundary drift")
    admission = result["admission"]
    for key in ("dynamic_no_running_batchnorm_admitted",
                "full_network_accuracy_admitted",
                "real_m40_address_cycle_schedule_admitted",
                "multicast_accumulator_cycle_reduction_admitted",
                "system_speedup_admitted", "integrated_rtl_or_synopsys_admitted",
                "ppa_power_energy_admitted", "date_best_paper_readiness_admitted"):
        require(admission[key] is False, "forbidden admission opened: {}".format(key))


def validate(pin_path=DEFAULT_PIN):
    pin_path = Path(pin_path).resolve()
    pin = strict_json(pin_path)
    resolved = validate_pinned_files(pin)
    require(resolved["validator"].resolve() == Path(__file__).resolve(),
            "validator path identity drift")
    contract = strict_json(resolved["contract"])
    validate_contract(contract)
    result = strict_json(resolved["result"])
    validate_result(result, resolved, contract)
    require(result["identity"]["contract_sha256"] == sha256(resolved["contract"]) and
            result["identity"]["exporter_sha256"] == sha256(resolved["exporter"]),
            "result producer identity drift")
    return {
        "status": "PASS_M41_RELEASE_PIN_RECURSIVE_PAYLOAD_AND_CLAIM_BOUNDARY_VALIDATION",
        "pin_path": str(pin_path),
        "pin_sha256": sha256(pin_path),
        "files_validated": len(pin["files"]),
        "weights_validated": result["population"]["weights_audited"],
        "raw_conv_output_values_anchored": result["population"][
            "raw_conv_output_values_compared"],
        "source_term_multicast_rows_recomputed": 4 * 768 * 9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    args = parser.parse_args()
    print(json.dumps(validate(args.pin), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
