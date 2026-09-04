#!/usr/bin/env python3
"""M41-r2 exhaustive integer-accumulator and late-scale oracle.

The authoritative convolution path is CUDA ``aten::_int_mm`` with signed int8
operands and signed int32 outputs.  A second, exhaustive float32 Conv2d path is
used only as a cross-check.  It is mathematically exact for this workload:
support is binary, weights are signed int8, and every channel's sum(abs(q)) is
below 2**24, so all products and every possible partial integer sum are exactly
representable in float32 (and signed int8 is exactly representable in TF32).

The source is Python-3.6 syntax compatible.  Generation needs the frozen
PyTorch/CUDA environment; the companion validator is pure stdlib Python 3.6.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as torch_functional


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m41_r2_integer_oracle_contract_r1_20260823.json")
DEFAULT_OUTPUT = ROOT / (
    "hw_autoresearch_nts07/results/"
    "m41_r2_integer_oracle_r1_20260823")
OPERATORS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
DENOMINATOR = 1 << 24
VECTOR_LANES = 96
WINDOWS = (8, 16, 32, 64, 256)
SERVICE_RATES = (1, 2, 4, 8)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
        raise RuntimeError("non-finite JSON token: {}".format(token))

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def resolve_input(receipt, role):
    require(type(receipt) is dict and set(receipt) == {
        "path", "bytes", "sha256"}, "{} receipt shape".format(role))
    relative = receipt["path"]
    require(type(relative) is str and relative and not Path(relative).is_absolute(),
            "{} path must be relative".format(role))
    require(".." not in Path(relative).parts, "{} path escape".format(role))
    path = (ROOT / relative).resolve()
    require(str(path).startswith(str(ROOT.resolve()) + "/"),
            "{} realpath escape".format(role))
    require(path.is_file(), "{} missing".format(role))
    require(path.stat().st_size == receipt["bytes"],
            "{} byte mismatch".format(role))
    require(sha256(path) == receipt["sha256"],
            "{} SHA mismatch".format(role))
    return path


def rne_signed(numerator, denominator):
    """Signed round-to-nearest, ties-to-even using floor quotient."""
    require(numerator.dtype == np.int64, "RNE numerator must be int64")
    quotient = np.floor_divide(numerator, denominator)
    remainder = numerator - quotient * denominator
    half = denominator // 2
    increment = ((remainder > half) |
                 ((remainder == half) & ((quotient & np.int64(1)) != 0)))
    return quotient + increment.astype(np.int64)


def rne_signed_magnitude(numerator, denominator):
    """Independent signed RNE implementation using magnitude and sign."""
    require(numerator.dtype == np.int64, "magnitude RNE input must be int64")
    sign = np.where(numerator < 0, np.int64(-1), np.int64(1))
    magnitude = np.abs(numerator)
    quotient = magnitude // denominator
    remainder = magnitude % denominator
    twice = remainder * np.int64(2)
    increment = ((twice > denominator) |
                 ((twice == denominator) & ((quotient & np.int64(1)) != 0)))
    return sign * (quotient + increment.astype(np.int64))


def longest_true_run(values):
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def maximum_window_sum(arrivals, width):
    if arrivals.size == 0:
        return 0
    if arrivals.size <= width:
        return int(np.sum(arrivals, dtype=np.int64))
    prefix = np.empty(arrivals.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(arrivals, dtype=np.int64, out=prefix[1:])
    return int(np.max(prefix[width:] - prefix[:-width]))


def fifo_trace(arrivals, service_rate):
    """FIFO occupancy with arrivals before a fixed scalar service each cycle."""
    backlog = 0
    peak_before_service = 0
    peak_after_service = 0
    for count in arrivals:
        before = backlog + int(count)
        peak_before_service = max(peak_before_service, before)
        backlog = max(0, before - service_rate)
        peak_after_service = max(peak_after_service, backlog)
    drain_cycles = int(math.ceil(float(backlog) / service_rate)) if backlog else 0
    return {
        "scalar_exceptions_serviced_per_vector_cycle": service_rate,
        "minimum_fifo_entries_arrivals_before_service": peak_before_service,
        "peak_backlog_after_service": peak_after_service,
        "end_of_stream_backlog": backlog,
        "drain_cycles_after_stream": drain_cycles,
    }


def stream_statistics(arrivals, scalar_flags):
    histogram = {}
    unique, counts = np.unique(arrivals, return_counts=True)
    for value, count in zip(unique, counts):
        histogram[str(int(value))] = int(count)
    active = arrivals != 0
    vectors = int(arrivals.size)
    exceptions = int(np.sum(arrivals, dtype=np.int64))
    return {
        "vectors": vectors,
        "exceptions": exceptions,
        "exceptions_per_vector": float(exceptions) / vectors if vectors else 0.0,
        "active_vectors": int(np.count_nonzero(active)),
        "active_vector_fraction": (
            float(np.count_nonzero(active)) / vectors if vectors else 0.0),
        "peak_exceptions_in_one_vector": int(np.max(arrivals)) if vectors else 0,
        "arrival_histogram": histogram,
        "longest_consecutive_active_vector_run": longest_true_run(active),
        "longest_consecutive_scalar_exception_run": longest_true_run(scalar_flags),
        "maximum_exception_arrivals_by_window_vectors": {
            str(width): maximum_window_sum(arrivals, width) for width in WINDOWS
        },
        "fixed_scalar_service_fifo_requirements": {
            str(rate): fifo_trace(arrivals, rate) for rate in SERVICE_RATES
        },
    }


def driver_version():
    try:
        value = subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ], universal_newlines=True).strip().splitlines()[0]
        return value.strip()
    except Exception as error:  # identity remains explicit if nvidia-smi is absent
        return "UNAVAILABLE:{}".format(type(error).__name__)


def validate_contract(contract, contract_path):
    require(contract["schema"] == "m41_r2_integer_oracle_contract_v1",
            "contract schema drift")
    require(contract["status"] ==
            "FROZEN_EXHAUSTIVE_INTEGER_ORACLE_FAIL_CLOSED",
            "contract status drift")
    require(tuple(contract["geometry"]["operators"]) == OPERATORS,
            "operator identity drift")
    require(contract["geometry"]["records"] == 40 and
            contract["geometry"]["values_per_record"] == 2304000 and
            contract["geometry"]["total_accumulators"] == 92160000,
            "population contract drift")
    require(contract["geometry"]["source_shape"] == [10, 1, 768, 15, 20] and
            contract["geometry"]["output_shape"] == [10, 1, 768, 15, 20] and
            contract["geometry"]["weight_matrix_shape"] == [6912, 768],
            "geometry drift")
    require(contract["integer_oracle"]["authoritative_operator"] ==
            "torch._int_mm_signed_int8_to_signed_int32",
            "integer oracle drift")
    require(contract["integer_oracle"]["predeclared_exception_counts"] is None,
            "exception conclusion was predeclared")
    producer = resolve_input(contract["identity"]["producer"], "producer")
    require(producer == Path(__file__).resolve(), "producer path mismatch")
    resolved = {}
    for role, receipt in contract["identity"]["frozen_inputs"].items():
        resolved[role] = resolve_input(receipt, role)
    require(set(resolved) == {
        "checkpoint", "m40_source_manifest", "m41_r1_contract",
        "m41_r1_release_pin", "m41_r1_result", "m41_r1_independent_review",
        "o0_weight_s8", "o1_weight_s8", "o2_weight_s8", "o3_weight_s8",
        "o0_scale_f32", "o1_scale_f32", "o2_scale_f32", "o3_scale_f32",
        "o0_scale_uq31", "o1_scale_uq31", "o2_scale_uq31", "o3_scale_uq31",
        "o0_acc_init", "o1_acc_init", "o2_acc_init", "o3_acc_init",
    }, "frozen input role population drift")
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite output directory")
    contract = strict_json(contract_path)
    inputs = validate_contract(contract, contract_path)

    manifest = strict_json(inputs["m40_source_manifest"])
    require(manifest["schema"] == "m40_bottleneck_packed_source_trace_v1" and
            len(manifest["records"]) == 40,
            "M40 manifest identity/population drift")
    records = sorted(manifest["records"],
                     key=lambda row: (row["operator_index"], row["sample_id"]))
    require([(row["operator_index"], row["sample_id"]) for row in records] ==
            [(operator, sample) for operator in range(4) for sample in range(10)],
            "M40 record key population drift")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    require(torch.cuda.is_available(), "CUDA is required for exhaustive r2 oracle")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")

    output_dir.mkdir(parents=True)
    bitmap_path = output_dir / "m41_r2_exception_bitmap_1b_lsb0.zlib"
    bitmap_raw = bytearray()
    record_rows = []
    layer_accumulators = {}
    layer_exception_flags = {}
    layer_arrivals = {}
    layer_correction_histograms = {}
    qweight_matrices = {}
    qweight_conv = {}
    weight_proofs = {}

    for operator_index, name in enumerate(OPERATORS):
        role = "o{}_weight_s8".format(operator_index)
        payload = inputs[role].read_bytes()
        require(len(payload) == 6912 * 768 and b"\x80" not in payload,
                "{} int8 payload extent/reserved-code failure".format(role))
        matrix = np.frombuffer(payload, dtype=np.int8).reshape(6912, 768).copy()
        sum_abs = np.sum(np.abs(matrix.astype(np.int16)), axis=0, dtype=np.int64)
        require(int(np.max(sum_abs)) < (1 << 24),
                "float cross-check exactness bound failed")
        require(int(np.max(sum_abs)) < (1 << 31),
                "int32 oracle overflow bound failed")
        qweight_matrices[name] = torch.from_numpy(matrix).to(device).contiguous()
        conv = matrix.reshape(768, 3, 3, 768).transpose(3, 0, 1, 2).copy()
        qweight_conv[name] = torch.from_numpy(conv.astype(np.float32)).to(device)
        weight_proofs[name] = {
            "weight_payload_sha256": sha256(inputs[role]),
            "signed_int8_minimum": int(np.min(matrix)),
            "signed_int8_maximum": int(np.max(matrix)),
            "reserved_negative_128_count": int(np.count_nonzero(matrix == -128)),
            "per_output_channel_sum_abs_minimum": int(np.min(sum_abs)),
            "per_output_channel_sum_abs_maximum": int(np.max(sum_abs)),
            "int32_no_overflow_proved": True,
            "fp32_all_partial_integer_sums_exact_proved": True,
        }
        layer_accumulators[name] = 0
        layer_exception_flags[name] = []
        layer_arrivals[name] = []
        layer_correction_histograms[name] = {}

    with torch.no_grad():
        for record_index, record in enumerate(records):
            operator_index = int(record["operator_index"])
            name = OPERATORS[operator_index]
            require(record["operator"] == name and
                    record["shape"] == [10, 1, 768, 15, 20] and
                    record["output_shape"] == [10, 1, 768, 15, 20],
                    "M40 record geometry/operator drift")
            value_path = inputs["m40_source_manifest"].parent / record[
                "value_payload_file"]
            require(value_path.is_file() and
                    value_path.stat().st_size == record["value_payload_compressed_bytes"] and
                    sha256(value_path) == record["value_payload_sha256"],
                    "M40 compressed value payload drift")
            raw = zlib.decompress(value_path.read_bytes())
            require(len(raw) == record["input_content_bytes"] and
                    sha256_bytes(raw) == record["input_content_sha256"],
                    "M40 uncompressed value payload drift")
            source = np.frombuffer(raw, dtype="<f4").reshape(
                10, 1, 768, 15, 20)
            require(bool(np.isfinite(source).all()), "non-finite M40 source")
            support_np = (source.reshape(10, 768, 15, 20) != 0).astype(
                np.float32)
            support = torch.from_numpy(support_np).to(device)
            patches = torch_functional.unfold(
                support, kernel_size=(3, 3), dilation=1, padding=1,
                stride=1).transpose(1, 2).reshape(-1, 6912)
            patches_i8 = patches.to(dtype=torch.int8).contiguous()
            require(int(torch.min(patches_i8).item()) == 0 and
                    int(torch.max(patches_i8).item()) == 1,
                    "unfolded support is not binary int8")

            exact_gpu = torch._int_mm(patches_i8, qweight_matrices[name])
            require(exact_gpu.dtype == torch.int32 and
                    tuple(exact_gpu.shape) == (3000, 768),
                    "integer oracle output dtype/shape drift")
            float_gpu = torch_functional.conv2d(
                support, qweight_conv[name], bias=None, stride=1, padding=1,
                dilation=1, groups=1).permute(0, 2, 3, 1).contiguous().reshape(
                    3000, 768)
            rounded_float_gpu = torch.round(float_gpu).to(dtype=torch.int32)
            mismatch = exact_gpu != rounded_float_gpu
            mismatch_count = int(torch.count_nonzero(mismatch).item())
            require(mismatch_count == 0,
                    "exhaustive integer/float oracle accumulator mismatch")
            fractional = float_gpu != torch.round(float_gpu)
            fractional_count = int(torch.count_nonzero(fractional).item())
            maximum_fractional_tail = float(torch.max(
                torch.abs(float_gpu - torch.round(float_gpu))).item())

            accumulators = exact_gpu.cpu().numpy().astype("<i4", copy=False)
            float_accumulators = rounded_float_gpu.cpu().numpy().astype(
                "<i4", copy=False)
            require(accumulators.tobytes(order="C") ==
                    float_accumulators.tobytes(order="C"),
                    "CPU canonical accumulator byte mismatch")
            accumulator_i64 = accumulators.astype(np.int64, copy=False)
            delta = int(contract["late_scale"]["delta_by_operator"][name])
            full = rne_signed(
                accumulator_i64 * np.int64(DENOMINATOR - delta), DENOMINATOR)
            full_independent = accumulator_i64 + rne_signed_magnitude(
                -accumulator_i64 * np.int64(delta), DENOMINATOR)
            correction_mismatch = int(np.count_nonzero(full != full_independent))
            require(correction_mismatch == 0,
                    "exhaustive dual-RNE correction oracle mismatch")
            correction = full - accumulator_i64
            changed = correction != 0
            require(int(np.max(np.abs(correction))) <= 1,
                    "observed correction exceeds one LSB")

            scalar_flags = changed.reshape(-1)
            vectors = changed.reshape(-1, VECTOR_LANES)
            arrivals = np.count_nonzero(vectors, axis=1).astype(np.int16)
            packed = np.packbits(scalar_flags.astype(np.uint8), bitorder="little")
            require(packed.size == 2304000 // 8,
                    "exception bitmap record extent drift")
            bitmap_offset = len(bitmap_raw)
            bitmap_payload = packed.tobytes(order="C")
            bitmap_raw.extend(bitmap_payload)

            histogram = {}
            unique, counts = np.unique(correction, return_counts=True)
            for value, count in zip(unique, counts):
                histogram[str(int(value))] = int(count)
                target = layer_correction_histograms[name]
                key = str(int(value))
                target[key] = target.get(key, 0) + int(count)
            statistics = stream_statistics(arrivals, scalar_flags)
            row = {
                "record_index": record_index,
                "operator_index": operator_index,
                "operator": name,
                "sample_id": int(record["sample_id"]),
                "sample_key": record["sample_key"],
                "source_content_sha256": record["input_content_sha256"],
                "source_value_payload_sha256": record["value_payload_sha256"],
                "values": int(accumulators.size),
                "accumulator_minimum": int(np.min(accumulator_i64)),
                "accumulator_maximum": int(np.max(accumulator_i64)),
                "integer_oracle_s32le_sha256": sha256_bytes(
                    accumulators.tobytes(order="C")),
                "float_crosscheck_rounded_s32le_sha256": sha256_bytes(
                    float_accumulators.tobytes(order="C")),
                "integer_float_accumulator_mismatches": mismatch_count,
                "float_fractional_output_count": fractional_count,
                "float_maximum_fractional_tail": maximum_fractional_tail,
                "delta": delta,
                "dual_rne_correction_mismatches": correction_mismatch,
                "correction_histogram": histogram,
                "exception_count": int(np.count_nonzero(changed)),
                "maximum_absolute_correction_lsb": int(np.max(
                    np.abs(correction))),
                "exception_bitmap_raw_offset_bytes": bitmap_offset,
                "exception_bitmap_raw_bytes": len(bitmap_payload),
                "exception_bitmap_record_sha256": sha256_bytes(bitmap_payload),
                "exception_stream": statistics,
            }
            record_rows.append(row)
            layer_accumulators[name] += int(accumulators.size)
            layer_exception_flags[name].append(scalar_flags.copy())
            layer_arrivals[name].append(arrivals.copy())
            print("[M41-r2] {}/40 o{} s{:02d} exceptions={}".format(
                record_index + 1, operator_index, record["sample_id"],
                row["exception_count"]), flush=True)
            del support, patches, patches_i8, exact_gpu, float_gpu
            del rounded_float_gpu, mismatch, fractional

    torch.cuda.synchronize(device)
    require(len(record_rows) == 40 and
            sum(row["values"] for row in record_rows) == 92160000,
            "exhaustive accumulator population mismatch")
    require(len(bitmap_raw) == 92160000 // 8,
            "global exception bitmap extent mismatch")
    compressed_bitmap = zlib.compress(bytes(bitmap_raw), 9)
    with bitmap_path.open("wb") as handle:
        handle.write(compressed_bitmap)

    aggregate_by_layer = {}
    all_scalar_flags = []
    all_arrivals = []
    for name in OPERATORS:
        flags = np.concatenate(layer_exception_flags[name])
        arrivals = np.concatenate(layer_arrivals[name])
        statistics = stream_statistics(arrivals, flags)
        aggregate_by_layer[name] = {
            "delta": int(contract["late_scale"]["delta_by_operator"][name]),
            "values": layer_accumulators[name],
            "correction_histogram": layer_correction_histograms[name],
            "exception_count": int(np.count_nonzero(flags)),
            "exact_bypass_count": int(flags.size - np.count_nonzero(flags)),
            "exact_bypass_fraction": float(
                flags.size - np.count_nonzero(flags)) / flags.size,
            "exception_stream": statistics,
        }
        all_scalar_flags.append(flags)
        all_arrivals.append(arrivals)
    global_flags = np.concatenate(all_scalar_flags)
    global_arrivals = np.concatenate(all_arrivals)
    global_stream = stream_statistics(global_arrivals, global_flags)
    observed_counts = [aggregate_by_layer[name]["exception_count"]
                       for name in OPERATORS]
    r1_result = strict_json(inputs["m41_r1_result"])
    r1_counts = [int(r1_result["late_scale_elision_audit"]["aggregate_by_layer"][
        name]["rne_changed"]) for name in OPERATORS]

    properties = torch.cuda.get_device_properties(device)
    identity = {
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256(contract_path),
        "producer_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "producer_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_sha256": sha256(inputs["checkpoint"]),
        "m40_source_manifest_sha256": sha256(inputs["m40_source_manifest"]),
        "m41_r1_result_sha256": sha256(inputs["m41_r1_result"]),
        "m41_r1_independent_review_sha256": sha256(
            inputs["m41_r1_independent_review"]),
    }
    toolchain = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "numpy_version": np.__version__,
        "cuda_runtime_compiled_version": torch.version.cuda,
        "cudnn_runtime_version": int(torch.backends.cudnn.version()),
        "nvidia_driver_version": driver_version(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "gpu_total_memory_bytes": int(properties.total_memory),
        "authoritative_operator": "aten::_int_mm",
        "authoritative_input_dtypes": ["torch.int8", "torch.int8"],
        "authoritative_output_dtype": "torch.int32",
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "random_seed": 0,
    }
    result = {
        "schema": "m41_r2_integer_oracle_result_v1",
        "status": (
            "PASS_EXHAUSTIVE_SIGNED_INT8_BINARY_SUPPORT_TO_SIGNED_INT32_ORACLE_"
            "AND_DUAL_RNE_EXCEPTION_TRACE_SIDE_CAR_REMAINS_UNIMPLEMENTED"),
        "identity": identity,
        "toolchain": toolchain,
        "population": {
            "operators": 4,
            "records": 40,
            "timesteps_per_record": 10,
            "output_positions_per_timestep": 300,
            "output_channels": 768,
            "values_per_record": 2304000,
            "accumulators_exhaustively_checked": 92160000,
            "integer_float_accumulator_mismatches": sum(
                row["integer_float_accumulator_mismatches"] for row in record_rows),
            "dual_rne_correction_mismatches": sum(
                row["dual_rne_correction_mismatches"] for row in record_rows),
        },
        "integer_exactness_proof": {
            "authoritative_expression": (
                "signed_int8(binary_support_matrix[3000,6912]) @ "
                "signed_int8(weight_matrix[6912,768]) -> signed_int32"),
            "convolution_lowering_order": "T_THEN_H_THEN_W_BY_I_THEN_KY_THEN_KX_TO_O",
            "int32_overflow_rule": "max_per_channel_sum_abs_q < 2^31",
            "float_crosscheck_exactness_rule": (
                "0/1 and signed-int8 products are exact; every partial absolute "
                "sum is bounded by channel sum(abs(q)) < 2^24"),
            "weight_proofs": weight_proofs,
            "all_accumulator_integer_float_digest_matches": True,
            "all_dual_rne_correction_matches": True,
        },
        "records": record_rows,
        "aggregate_by_layer": aggregate_by_layer,
        "observed_exception_counts_by_operator": observed_counts,
        "comparison_to_r1_diagnostic": {
            "r1_reported_counts": r1_counts,
            "r2_integer_oracle_counts": observed_counts,
            "counts_equal": observed_counts == r1_counts,
            "counts_were_not_predeclared_or_used_as_an_acceptance_gate": True,
        },
        "exception_bitmap": {
            "path": str(bitmap_path.relative_to(ROOT)),
            "codec": "zlib_level9",
            "packing": "one_bit_per_canonical_accumulator_lsb0",
            "canonical_order": "OPERATOR_THEN_SAMPLE_THEN_T_THEN_H_THEN_W_THEN_O",
            "raw_bits": 92160000,
            "raw_bytes": len(bitmap_raw),
            "raw_sha256": sha256_bytes(bytes(bitmap_raw)),
            "compressed_bytes": len(compressed_bitmap),
            "compressed_sha256": sha256(bitmap_path),
        },
        "exception_sidecar_trace_thresholds": {
            "scope": (
                "trace-derived arrival/FIFO requirements only; not an RTL, cycle, "
                "area, power, energy, or system-speedup result"),
            "input_vector_contract": (
                "one canonical 96-output accumulator vector arrives per abstract "
                "vector cycle in OPERATOR/SAMPLE/T/H/W/O_BLOCK order"),
            "correction_semantics": (
                "for each set lane, replace bypass accumulator by accumulator plus "
                "the exact signed one-LSB correction"),
            "required_association_state": (
                "vector order plus lane id plus correction sign; tag width and "
                "integration protocol remain open"),
            "global_trace": global_stream,
            "zero_backpressure_parallel_correction_lane_lower_bound":
                global_stream["peak_exceptions_in_one_vector"],
            "fixed_service_fifo_depths_are_exact_for_frozen_stream_only": True,
            "rtl_admitted": False,
            "cycle_reduction_admitted": False,
            "ppa_or_power_admitted": False,
            "system_speedup_admitted": False,
        },
        "admission": {
            "backend_independent_integer_accumulator_population_admitted": True,
            "exhaustive_rne_exception_population_admitted": True,
            "m41_r1_p1_backend_identity_repaired": True,
            "exception_sidecar_trace_requirements_admitted": True,
            "exception_sidecar_rtl_admitted": False,
            "integrated_cycle_reduction_admitted": False,
            "synopsys_ppa_power_energy_admitted": False,
            "dynamic_batchnorm_or_full_network_accuracy_admitted": False,
            "system_speedup_admitted": False,
            "date_or_best_paper_readiness_admitted": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    result_path = output_dir / "m41_r2_integer_oracle.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("[M41-r2] wrote {}".format(result_path), flush=True)
    print("[M41-r2] observed counts {}".format(observed_counts), flush=True)
    print("[M41-r2] exception bitmap SHA {}".format(sha256(bitmap_path)), flush=True)


if __name__ == "__main__":
    main()
