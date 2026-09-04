#!/usr/bin/env python3
"""Pure-stdlib Python-3.6 validator for the canonical M41-r2 release."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m41_r2_integer_oracle_contract_r1_20260823.json")
RESULT_PATH = ROOT / (
    "hw_autoresearch_nts07/results/m41_r2_integer_oracle_r1_20260823/"
    "m41_r2_integer_oracle.json")
BITMAP_PATH = ROOT / (
    "hw_autoresearch_nts07/results/m41_r2_integer_oracle_r1_20260823/"
    "m41_r2_exception_bitmap_1b_lsb0.zlib")
CANONICAL_CONTRACT_SHA256 = (
    "c4bc078b9136aa716b32b6b2802f28a32d17359d61139547dd9e544f66f26be9")
CANONICAL_RESULT_SHA256 = (
    "89fc07790234f01c9532f01d806a766ca37a386390503074151a6eff37f1704a")
CANONICAL_BITMAP_SHA256 = (
    "8da84b9be10f22cecf80250f708aea1df173a2e52e181ea6880ccda3b901396a")
OPERATORS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
WINDOWS = (8, 16, 32, 64, 256)
SERVICE_RATES = (1, 2, 4, 8)
POPCOUNT = tuple(bin(value).count("1") for value in range(256))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def strict_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: {}".format(key))
            output[key] = value
        return output

    def reject(token):
        raise ValueError("non-finite JSON token: {}".format(token))

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


def resolve_under_root(relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute(),
            "path must be nonempty and relative")
    require(".." not in Path(relative).parts, "dot-dot path escape")
    path = (ROOT / relative).resolve()
    require(str(path).startswith(str(ROOT.resolve()) + "/"),
            "realpath escape")
    return path


def validate_receipt(receipt, role):
    require(type(receipt) is dict and set(receipt) == {
        "path", "bytes", "sha256"}, "{} receipt key drift".format(role))
    path = resolve_under_root(receipt["path"])
    require(path.is_file(), "{} missing".format(role))
    require(path.stat().st_size == exact_int(
        receipt["bytes"], "{} bytes".format(role), 1),
        "{} byte mismatch".format(role))
    require(type(receipt["sha256"]) is str and len(receipt["sha256"]) == 64,
            "{} SHA syntax".format(role))
    require(sha256(path) == receipt["sha256"], "{} SHA mismatch".format(role))
    return path


def validate_contract(contract):
    require(contract["schema"] == "m41_r2_integer_oracle_contract_v1",
            "contract schema")
    require(contract["status"] ==
            "FROZEN_EXHAUSTIVE_INTEGER_ORACLE_FAIL_CLOSED", "contract status")
    require(tuple(contract["geometry"]["operators"]) == OPERATORS,
            "contract operators")
    require(contract["geometry"]["records"] == 40 and
            contract["geometry"]["total_accumulators"] == 92160000,
            "contract population")
    oracle = contract["integer_oracle"]
    require(oracle["authoritative_operator"] ==
            "torch._int_mm_signed_int8_to_signed_int32", "oracle identity")
    require(oracle["predeclared_exception_counts"] is None,
            "exception count was predeclared")
    require(oracle["required_population"] == 92160000 and
            oracle["required_integer_oracle_mismatches"] == 0 and
            oracle["required_dual_rne_mismatches"] == 0,
            "oracle gate drift")
    producer = validate_receipt(contract["identity"]["producer"], "producer")
    require(producer.name == "run_m41_r2_integer_oracle.py", "producer identity")
    inputs = {}
    for role, receipt in contract["identity"]["frozen_inputs"].items():
        inputs[role] = validate_receipt(receipt, role)
    required = {
        "checkpoint", "m40_source_manifest", "m41_r1_contract",
        "m41_r1_release_pin", "m41_r1_result", "m41_r1_independent_review",
    }
    for index in range(4):
        required.update({
            "o{}_weight_s8".format(index), "o{}_scale_f32".format(index),
            "o{}_scale_uq31".format(index), "o{}_acc_init".format(index),
        })
    require(set(inputs) == required, "frozen input role population")
    for index in range(4):
        require(inputs["o{}_weight_s8".format(index)].stat().st_size == 5308416,
                "weight extent")
        require(inputs["o{}_scale_f32".format(index)].stat().st_size == 3072 and
                inputs["o{}_scale_uq31".format(index)].stat().st_size == 3072 and
                inputs["o{}_acc_init".format(index)].stat().st_size == 3072,
                "scale/acc-init extent")
    return inputs


def validate_m40_sources(manifest_path):
    manifest = strict_json(manifest_path)
    require(manifest["schema"] == "m40_bottleneck_packed_source_trace_v1",
            "M40 manifest schema")
    records = sorted(manifest["records"],
                     key=lambda row: (row["operator_index"], row["sample_id"]))
    require(len(records) == 40, "M40 record count")
    require([(row["operator_index"], row["sample_id"]) for row in records] ==
            [(operator, sample) for operator in range(4) for sample in range(10)],
            "M40 record key population")
    for row in records:
        require(row["operator"] == OPERATORS[row["operator_index"]],
                "M40 operator mapping")
        require(row["shape"] == [10, 1, 768, 15, 20] and
                row["output_shape"] == [10, 1, 768, 15, 20],
                "M40 record geometry")
        path = manifest_path.parent / row["value_payload_file"]
        require(path.is_file() and
                path.stat().st_size == row["value_payload_compressed_bytes"] and
                sha256(path) == row["value_payload_sha256"],
                "M40 compressed source receipt")
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(path.read_bytes()) + decompressor.flush()
        require(decompressor.eof and not decompressor.unused_data and
                not decompressor.unconsumed_tail, "M40 source zlib framing")
        require(len(raw) == row["input_content_bytes"] and
                sha256_bytes(raw) == row["input_content_sha256"],
                "M40 raw source receipt")
    return records


def longest_true_run_bits(payload):
    longest = 0
    current = 0
    for byte in bytearray(payload):
        for bit in range(8):
            if byte & (1 << bit):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
    return longest


def longest_nonzero_run(values):
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def maximum_window_sum(values, width):
    if len(values) <= width:
        return sum(values)
    current = sum(values[:width])
    maximum = current
    for index in range(width, len(values)):
        current += values[index] - values[index - width]
        maximum = max(maximum, current)
    return maximum


def fifo_trace(values, rate):
    backlog = 0
    peak_before = 0
    peak_after = 0
    for count in values:
        before = backlog + count
        peak_before = max(peak_before, before)
        backlog = max(0, before - rate)
        peak_after = max(peak_after, backlog)
    return {
        "scalar_exceptions_serviced_per_vector_cycle": rate,
        "minimum_fifo_entries_arrivals_before_service": peak_before,
        "peak_backlog_after_service": peak_after,
        "end_of_stream_backlog": backlog,
        "drain_cycles_after_stream": (
            (backlog + rate - 1) // rate if backlog else 0),
    }


def bitmap_stream(payload):
    require(len(payload) % 12 == 0, "96-lane bitmap alignment")
    raw = bytearray(payload)
    arrivals = []
    for offset in range(0, len(raw), 12):
        arrivals.append(sum(POPCOUNT[value] for value in raw[offset:offset + 12]))
    histogram = {}
    for value in arrivals:
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1
    exceptions = sum(arrivals)
    active = sum(1 for value in arrivals if value)
    vectors = len(arrivals)
    return {
        "vectors": vectors,
        "exceptions": exceptions,
        "exceptions_per_vector": float(exceptions) / vectors if vectors else 0.0,
        "active_vectors": active,
        "active_vector_fraction": float(active) / vectors if vectors else 0.0,
        "peak_exceptions_in_one_vector": max(arrivals) if arrivals else 0,
        "arrival_histogram": histogram,
        "longest_consecutive_active_vector_run": longest_nonzero_run(arrivals),
        "longest_consecutive_scalar_exception_run": longest_true_run_bits(payload),
        "maximum_exception_arrivals_by_window_vectors": {
            str(width): maximum_window_sum(arrivals, width) for width in WINDOWS
        },
        "fixed_scalar_service_fifo_requirements": {
            str(rate): fifo_trace(arrivals, rate) for rate in SERVICE_RATES
        },
    }


def compare_stream(observed, expected, name):
    require(set(observed) == set(expected), "{} stream key drift".format(name))
    for key, value in expected.items():
        candidate = observed[key]
        if type(value) is float:
            require(abs(finite_float(candidate, name + "." + key) - value) <= 1e-15,
                    "{} float mismatch".format(name + "." + key))
        else:
            require(candidate == value, "{} mismatch".format(name + "." + key))


def validate_result_semantics(result, contract, manifest_records, raw_bitmap):
    require(result["schema"] == "m41_r2_integer_oracle_result_v1",
            "result schema")
    require(result["status"] ==
            "PASS_EXHAUSTIVE_SIGNED_INT8_BINARY_SUPPORT_TO_SIGNED_INT32_ORACLE_"
            "AND_DUAL_RNE_EXCEPTION_TRACE_SIDE_CAR_REMAINS_UNIMPLEMENTED",
            "result status")
    identity = result["identity"]
    require(identity["contract_sha256"] == CANONICAL_CONTRACT_SHA256,
            "result contract identity")
    require(identity["producer_sha256"] ==
            contract["identity"]["producer"]["sha256"], "result producer identity")
    inputs = contract["identity"]["frozen_inputs"]
    require(identity["checkpoint_sha256"] == inputs["checkpoint"]["sha256"] and
            identity["m40_source_manifest_sha256"] ==
            inputs["m40_source_manifest"]["sha256"] and
            identity["m41_r1_result_sha256"] == inputs["m41_r1_result"]["sha256"] and
            identity["m41_r1_independent_review_sha256"] ==
            inputs["m41_r1_independent_review"]["sha256"],
            "result upstream identity")
    tool = result["toolchain"]
    require(tool["authoritative_operator"] == "aten::_int_mm" and
            tool["authoritative_input_dtypes"] == ["torch.int8", "torch.int8"] and
            tool["authoritative_output_dtype"] == "torch.int32",
            "toolchain integer operator")
    for key in ("python_version", "torch_version", "torch_git_version",
                "numpy_version", "cuda_runtime_compiled_version",
                "nvidia_driver_version", "gpu_name"):
        require(type(tool[key]) is str and tool[key], "toolchain {}".format(key))
    require(exact_int(tool["cudnn_runtime_version"], "cuDNN version", 1) > 0 and
            tool["gpu_compute_capability"] == [8, 6] and
            tool["cudnn_benchmark"] is False and
            tool["cudnn_deterministic"] is True and
            tool["cuda_matmul_allow_tf32"] is True and
            tool["cudnn_allow_tf32"] is True and
            tool["deterministic_algorithms"] is True and
            tool["cublas_workspace_config"] == ":4096:8",
            "toolchain arithmetic/determinism policy")
    population = result["population"]
    require(population == {
        "operators": 4, "records": 40, "timesteps_per_record": 10,
        "output_positions_per_timestep": 300, "output_channels": 768,
        "values_per_record": 2304000,
        "accumulators_exhaustively_checked": 92160000,
        "integer_float_accumulator_mismatches": 0,
        "dual_rne_correction_mismatches": 0,
    }, "result population")
    proof = result["integer_exactness_proof"]
    require(proof["all_accumulator_integer_float_digest_matches"] is True and
            proof["all_dual_rne_correction_matches"] is True,
            "exactness proof flags")
    require(set(proof["weight_proofs"]) == set(OPERATORS), "weight proof layers")
    for index, name in enumerate(OPERATORS):
        row = proof["weight_proofs"][name]
        require(row["weight_payload_sha256"] ==
                inputs["o{}_weight_s8".format(index)]["sha256"] and
                row["reserved_negative_128_count"] == 0 and
                row["per_output_channel_sum_abs_maximum"] < (1 << 24) and
                row["int32_no_overflow_proved"] is True and
                row["fp32_all_partial_integer_sums_exact_proved"] is True,
                "weight exactness proof")

    records = result["records"]
    require(type(records) is list and len(records) == 40, "result record count")
    observed_counts = [0, 0, 0, 0]
    observed_values = [0, 0, 0, 0]
    for index, row in enumerate(records):
        operator_index = index // 10
        sample_id = index % 10
        source = manifest_records[index]
        require(row["record_index"] == index and
                row["operator_index"] == operator_index and
                row["sample_id"] == sample_id and
                row["operator"] == OPERATORS[operator_index],
                "result record identity/order")
        require(row["source_content_sha256"] == source["input_content_sha256"] and
                row["source_value_payload_sha256"] == source["value_payload_sha256"],
                "result record source identity")
        require(row["values"] == 2304000 and
                row["integer_float_accumulator_mismatches"] == 0 and
                row["dual_rne_correction_mismatches"] == 0 and
                row["integer_oracle_s32le_sha256"] ==
                row["float_crosscheck_rounded_s32le_sha256"],
                "record exhaustive oracle evidence")
        require(type(row["integer_oracle_s32le_sha256"]) is str and
                len(row["integer_oracle_s32le_sha256"]) == 64,
                "record accumulator digest")
        offset = index * 288000
        portion = raw_bitmap[offset:offset + 288000]
        require(row["exception_bitmap_raw_offset_bytes"] == offset and
                row["exception_bitmap_raw_bytes"] == 288000 and
                row["exception_bitmap_record_sha256"] == sha256_bytes(portion),
                "record bitmap receipt")
        expected_stream = bitmap_stream(portion)
        compare_stream(row["exception_stream"], expected_stream,
                       "record {}".format(index))
        require(row["exception_count"] == expected_stream["exceptions"],
                "record exception count")
        histogram = row["correction_histogram"]
        require(set(histogram).issubset({"-1", "0", "1"}),
                "record correction alphabet")
        require(sum(exact_int(value, "correction count", 0)
                    for value in histogram.values()) == 2304000,
                "record correction population")
        require(sum(value for key, value in histogram.items() if key != "0") ==
                row["exception_count"], "record correction/bitmap conservation")
        require(row["maximum_absolute_correction_lsb"] ==
                (1 if row["exception_count"] else 0), "record maximum correction")
        observed_counts[operator_index] += row["exception_count"]
        observed_values[operator_index] += row["values"]

    require(result["observed_exception_counts_by_operator"] == observed_counts,
            "observed exception layer counts")
    diagnostic = result["comparison_to_r1_diagnostic"]
    require(diagnostic["r2_integer_oracle_counts"] == observed_counts and
            diagnostic["counts_were_not_predeclared_or_used_as_an_acceptance_gate"]
            is True and
            diagnostic["counts_equal"] ==
            (diagnostic["r1_reported_counts"] == observed_counts),
            "r1 diagnostic semantics")
    aggregate = result["aggregate_by_layer"]
    require(set(aggregate) == set(OPERATORS), "aggregate layer population")
    for operator_index, name in enumerate(OPERATORS):
        row = aggregate[name]
        start = operator_index * 10 * 288000
        end = start + 10 * 288000
        stream = bitmap_stream(raw_bitmap[start:end])
        compare_stream(row["exception_stream"], stream, "layer {}".format(name))
        require(row["values"] == observed_values[operator_index] == 23040000 and
                row["exception_count"] == observed_counts[operator_index] ==
                stream["exceptions"] and
                row["exact_bypass_count"] == row["values"] - row["exception_count"] and
                abs(row["exact_bypass_fraction"] -
                    float(row["exact_bypass_count"]) / row["values"]) <= 1e-15,
                "layer aggregate conservation")

    bitmap = result["exception_bitmap"]
    require(bitmap["canonical_order"] ==
            "OPERATOR_THEN_SAMPLE_THEN_T_THEN_H_THEN_W_THEN_O" and
            bitmap["packing"] == "one_bit_per_canonical_accumulator_lsb0" and
            bitmap["raw_bits"] == 92160000 and
            bitmap["raw_bytes"] == 11520000 and
            bitmap["raw_sha256"] == sha256_bytes(raw_bitmap),
            "global bitmap semantics")
    global_stream = bitmap_stream(raw_bitmap)
    sidecar = result["exception_sidecar_trace_thresholds"]
    compare_stream(sidecar["global_trace"], global_stream, "global")
    require(sidecar["zero_backpressure_parallel_correction_lane_lower_bound"] ==
            global_stream["peak_exceptions_in_one_vector"], "sidecar lane threshold")
    require(sidecar["fixed_service_fifo_depths_are_exact_for_frozen_stream_only"]
            is True and sidecar["rtl_admitted"] is False and
            sidecar["cycle_reduction_admitted"] is False and
            sidecar["ppa_or_power_admitted"] is False and
            sidecar["system_speedup_admitted"] is False,
            "sidecar claim boundary")
    admission = result["admission"]
    require(admission["backend_independent_integer_accumulator_population_admitted"]
            is True and admission["exhaustive_rne_exception_population_admitted"]
            is True and admission["m41_r1_p1_backend_identity_repaired"] is True,
            "r2 admission flags")
    for key in ("exception_sidecar_rtl_admitted",
                "integrated_cycle_reduction_admitted",
                "synopsys_ppa_power_energy_admitted",
                "dynamic_batchnorm_or_full_network_accuracy_admitted",
                "system_speedup_admitted", "date_or_best_paper_readiness_admitted"):
        require(admission[key] is False, "claim opened: {}".format(key))
    return {
        "status": "PASS_M41_R2_CANONICAL_INTEGER_ORACLE_AND_EXCEPTION_TRACE",
        "files_validated": 26,
        "m40_value_payloads_validated": 40,
        "accumulators_anchored": 92160000,
        "integer_float_mismatches": 0,
        "dual_rne_mismatches": 0,
        "observed_exception_counts_by_operator": observed_counts,
        "exception_bitmap_popcount": global_stream["exceptions"],
        "peak_exceptions_per_96_lane_vector":
            global_stream["peak_exceptions_in_one_vector"],
    }


def canonical_validate(contract_path=CONTRACT_PATH, result_path=RESULT_PATH,
                       bitmap_path=BITMAP_PATH):
    require(sha256(contract_path) == CANONICAL_CONTRACT_SHA256,
            "canonical contract SHA mismatch")
    require(sha256(result_path) == CANONICAL_RESULT_SHA256,
            "canonical result SHA mismatch")
    require(sha256(bitmap_path) == CANONICAL_BITMAP_SHA256,
            "canonical bitmap SHA mismatch")
    contract = strict_json(contract_path)
    inputs = validate_contract(contract)
    manifest_records = validate_m40_sources(inputs["m40_source_manifest"])
    result = strict_json(result_path)
    compressed = bitmap_path.read_bytes()
    bitmap = result["exception_bitmap"]
    require(len(compressed) == bitmap["compressed_bytes"] and
            sha256_bytes(compressed) == bitmap["compressed_sha256"],
            "compressed bitmap receipt")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(compressed) + decompressor.flush()
    require(decompressor.eof and not decompressor.unused_data and
            not decompressor.unconsumed_tail, "bitmap zlib framing")
    require(len(raw) == 11520000, "bitmap raw extent")
    return validate_result_semantics(result, contract, manifest_records, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--bitmap", type=Path, default=BITMAP_PATH)
    args = parser.parse_args()
    summary = canonical_validate(args.contract.resolve(), args.result.resolve(),
                                 args.bitmap.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
