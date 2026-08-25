#!/usr/bin/env python3
"""Freeze the M256 PAFT-ep4 running-BN INT8 bottleneck-weight bridge.

The exporter is intentionally a one-shot evidence producer.  It refuses an
existing output directory, pins every upstream byte identity before loading the
checkpoint, and retains only raw-convolution numeric claims.  It does not
implement or claim dynamic BatchNorm, an address/cycle schedule, system
speedup, or any EDA result.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as torch_functional


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = HW_ROOT / (
    "contracts/m256_paft_ep4_running_bn_bottleneck_int8_contract_r1_20260825.json")


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


def exact_int(value, name):
    require(type(value) is int, "{} must be an exact integer".format(name))
    return value


def finite_float(value, name):
    value = float(value)
    require(math.isfinite(value), "{} is not finite".format(name))
    return value


def resolve_receipt(receipt):
    require(set(receipt) == {"path", "sha256"}, "input receipt schema drift")
    path = ROOT / receipt["path"]
    require(path.is_file(), "missing input: {}".format(path))
    require(sha256(path) == receipt["sha256"],
            "input SHA drift: {}".format(path))
    return path


def percentile_nearest_rank(values, percentile):
    require(values, "empty percentile population")
    ordered = sorted(values)
    rank = max(1, int(math.ceil(float(percentile) * len(ordered))))
    return ordered[rank - 1]


def summarize(values):
    require(values, "empty summary population")
    clean = [finite_float(value, "summary value") for value in values]
    return {
        "count": len(clean),
        "minimum": min(clean),
        "mean": math.fsum(clean) / len(clean),
        "p50_nearest_rank": percentile_nearest_rank(clean, 0.50),
        "p95_nearest_rank": percentile_nearest_rank(clean, 0.95),
        "maximum": max(clean),
    }


def signed_bits_for_symmetric_magnitude(magnitude):
    magnitude = exact_int(magnitude, "signed magnitude")
    require(magnitude >= 0, "negative signed magnitude")
    bits = 1
    while magnitude > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def canonical_weight_quantize(weight):
    require(weight.dtype == np.float32, "weight must be exact float32")
    require(weight.ndim == 4, "weight rank drift")
    require(bool(np.isfinite(weight).all()), "weight has NaN/Infinity")
    max_abs = np.max(np.abs(weight), axis=(1, 2, 3)).astype(np.float32)
    zero_channel = max_abs == np.float32(0.0)
    scale = (max_abs / np.float32(127.0)).astype(np.float32)
    scale[zero_channel] = np.float32(1.0)
    ratio = weight.astype(np.float64) / scale[:, None, None, None].astype(np.float64)
    rounded = np.rint(ratio)
    preclip = int(np.count_nonzero((rounded < -127.0) | (rounded > 127.0)))
    qweight = np.clip(rounded, -127.0, 127.0).astype(np.int8)
    require(not bool(np.any(qweight == np.int8(-128))), "reserved -128 emitted")
    dequant = (qweight.astype(np.float64) *
               scale[:, None, None, None].astype(np.float64))
    error = dequant - weight.astype(np.float64)
    return {
        "scale": scale,
        "qweight": qweight,
        "dequant_float64": dequant,
        "error_float64": error,
        "zero_channel": zero_channel,
        "preclip_violation_count": preclip,
    }


def write_new_bytes(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite: {}".format(path))
    path.write_bytes(payload)


def payload_receipt(path, role, dtype, shape, layout):
    path = Path(path)
    return {
        "file": path.name,
        "role": role,
        "dtype": dtype,
        "shape": list(shape),
        "layout": layout,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def source_term_reuse(qweight, operator, operator_index, writer):
    # [O,I,KY,KX] -> one row per [I,KY,KX], contiguous O vector.
    vectors = qweight.transpose(1, 2, 3, 0).reshape(-1, qweight.shape[0])
    block_sizes = (16, 24, 48, 96)
    unique_counts = []
    repeat_factors = []
    top_counts = []
    top_fractions = []
    aggregates = {
        size: {
            "blocks": 0,
            "nonzero_destination_updates": 0,
            "value_mask_commands": 0,
            "exact_uncompressed_value_mask_bytes": 0,
            "dense_int8_bytes": 0,
        }
        for size in block_sizes
    }
    for term_index, vector in enumerate(vectors):
        values, counts = np.unique(vector, return_counts=True)
        unique_count = int(values.size)
        top_count = int(np.max(counts))
        unique_counts.append(unique_count)
        repeat_factors.append(float(vector.size) / unique_count)
        top_counts.append(top_count)
        top_fractions.append(float(top_count) / vector.size)
        row = {
            "operator": operator,
            "operator_index": operator_index,
            "input_channel": term_index // 9,
            "kernel_y": (term_index % 9) // 3,
            "kernel_x": term_index % 3,
            "unique_value_count_768": unique_count,
            "repeat_factor_768_div_unique": "{:.17g}".format(
                float(vector.size) / unique_count),
            "top_value_s8": int(values[int(np.argmax(counts))]),
            "top_value_count": top_count,
            "top_value_fraction": "{:.17g}".format(
                float(top_count) / vector.size),
        }
        for block_size in block_sizes:
            blocks = vector.reshape(vector.size // block_size, block_size)
            commands = 0
            updates = int(np.count_nonzero(blocks))
            for block in blocks:
                distinct = np.unique(block)
                commands += int(np.count_nonzero(distinct))
            encoded_bytes = commands * (1 + int(math.ceil(block_size / 8.0)))
            dense_bytes = int(blocks.size)
            aggregates[block_size]["blocks"] += int(blocks.shape[0])
            aggregates[block_size]["nonzero_destination_updates"] += updates
            aggregates[block_size]["value_mask_commands"] += commands
            aggregates[block_size][
                "exact_uncompressed_value_mask_bytes"] += encoded_bytes
            aggregates[block_size]["dense_int8_bytes"] += dense_bytes
            row["b{}_nonzero_updates".format(block_size)] = updates
            row["b{}_value_mask_commands".format(block_size)] = commands
            row["b{}_value_mask_bytes".format(block_size)] = encoded_bytes
            row["b{}_dense_int8_bytes".format(block_size)] = dense_bytes
        writer.writerow(row)
    block_result = {}
    for block_size, values in aggregates.items():
        commands = values["value_mask_commands"]
        encoded = values["exact_uncompressed_value_mask_bytes"]
        values["destination_updates_per_command"] = (
            float(values["nonzero_destination_updates"]) / commands)
        values["dense_int8_bytes_div_value_mask_bytes"] = (
            float(values["dense_int8_bytes"]) / encoded)
        values["encoded_bytes_per_command"] = 1 + int(math.ceil(block_size / 8.0))
        block_result[str(block_size)] = values
    return {
        "source_terms": int(vectors.shape[0]),
        "output_channels_per_term": int(vectors.shape[1]),
        "unique_value_count": summarize(unique_counts),
        "repeat_factor_768_div_unique": summarize(repeat_factors),
        "top_value_count": summarize(top_counts),
        "top_value_fraction": summarize(top_fractions),
        "blocked_value_mask": block_result,
    }


def round_divide_rne_signed(values, denominator):
    """Round signed int64 values/positive denominator to nearest, ties even."""
    require(values.dtype == np.int64, "RNE input must be signed int64")
    quotient = np.floor_divide(values, denominator)
    remainder = values - quotient * denominator
    half = denominator // 2
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    return quotient + increment.astype(np.int64)


def local_conv_metrics(source, float_weight, quant_weight, qweight, delta, device):
    x = torch.from_numpy(source.reshape(10, 768, 15, 20)).to(device=device)
    wf = torch.from_numpy(float_weight).to(device=device)
    wq = torch.from_numpy(quant_weight).to(device=device)
    wi = torch.from_numpy(qweight.astype(np.float32)).to(device=device)
    support = (x != 0).to(dtype=torch.float32)
    with torch.no_grad():
        reference = torch_functional.conv2d(x, wf, bias=None, stride=1, padding=1)
        candidate = torch_functional.conv2d(x, wq, bias=None, stride=1, padding=1)
        integer_acc_float = torch_functional.conv2d(
            support, wi, bias=None, stride=1, padding=1)
        require(bool(torch.equal(integer_acc_float, torch.round(integer_acc_float))),
                "integer accumulator Conv2d lost exact integrality")
        difference = candidate - reference
        ref64 = reference.double()
        cand64 = candidate.double()
        diff64 = difference.double()
        count = int(difference.numel())
        sum_abs_error = float(torch.sum(torch.abs(diff64)).item())
        sum_squared_error = float(torch.sum(diff64 * diff64).item())
        maximum_abs_error = float(torch.max(torch.abs(diff64)).item())
        reference_sum_squared = float(torch.sum(ref64 * ref64).item())
        candidate_sum_squared = float(torch.sum(cand64 * cand64).item())
        dot_product = float(torch.sum(ref64 * cand64).item())
        integer_acc = integer_acc_float.to(dtype=torch.int64).cpu().numpy().reshape(-1)
    product = integer_acc * np.int64((1 << 24) - delta)
    full_rne = round_divide_rne_signed(product, 1 << 24)
    full_floor = np.floor_divide(product, 1 << 24)
    nonzero = integer_acc != 0
    rne_equal = full_rne == integer_acc
    floor_equal = full_floor == integer_acc
    late_scale = {
        "delta": delta,
        "values": int(integer_acc.size),
        "nonzero_accumulators": int(np.count_nonzero(nonzero)),
        "zero_accumulators": int(np.count_nonzero(~nonzero)),
        "rne_exact_bypass": int(np.count_nonzero(rne_equal)),
        "rne_exact_bypass_nonzero": int(np.count_nonzero(rne_equal & nonzero)),
        "rne_changed": int(np.count_nonzero(~rne_equal)),
        "rne_maximum_absolute_integer_correction": int(np.max(
            np.abs(full_rne - integer_acc))),
        "floor_exact_bypass": int(np.count_nonzero(floor_equal)),
        "floor_exact_bypass_nonzero": int(np.count_nonzero(floor_equal & nonzero)),
        "floor_changed": int(np.count_nonzero(~floor_equal)),
        "floor_maximum_absolute_integer_correction": int(np.max(
            np.abs(full_floor - integer_acc))),
        "observed_accumulator_minimum": int(np.min(integer_acc)),
        "observed_accumulator_maximum": int(np.max(integer_acc)),
    }
    del x, wf, wq, wi, support, reference, candidate, difference
    del ref64, cand64, diff64, integer_acc_float, integer_acc
    return {
        "values": count,
        "sum_abs_error": sum_abs_error,
        "sum_squared_error": sum_squared_error,
        "maximum_abs_error": maximum_abs_error,
        "reference_sum_squared": reference_sum_squared,
        "candidate_sum_squared": candidate_sum_squared,
        "dot_product": dot_product,
    }, late_scale


def finalize_numeric_metrics(raw):
    count = raw["values"]
    ref_sq = raw["reference_sum_squared"]
    cand_sq = raw["candidate_sum_squared"]
    require(count > 0 and ref_sq > 0.0 and cand_sq > 0.0,
            "degenerate local numeric metric population")
    return {
        "values": count,
        "mae": raw["sum_abs_error"] / count,
        "rmse": math.sqrt(raw["sum_squared_error"] / count),
        "max_abs_error": raw["maximum_abs_error"],
        "normalized_l2_error": math.sqrt(raw["sum_squared_error"] / ref_sq),
        "cosine_similarity": raw["dot_product"] / math.sqrt(ref_sq * cand_sq),
    }


def merge_numeric_metrics(destination, source):
    for key in ("values", "sum_abs_error", "sum_squared_error",
                "reference_sum_squared", "candidate_sum_squared", "dot_product"):
        destination[key] += source[key]
    destination["maximum_abs_error"] = max(
        destination["maximum_abs_error"], source["maximum_abs_error"])


def load_checkpoint_model(checkpoint_path, config_path):
    profile_path = (ROOT / "neuron_experiments/H9_bipolar_self_attention/"
                    "entrypoints/profile_nts11_hardware_p0.py")
    require(profile_path.is_file(), "missing frozen H9 profile entrypoint")
    spec = importlib.util.spec_from_file_location("m256_h9_profile", profile_path)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen H9 profile entrypoint")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    config, device = profile.load_config(config_path)
    require(device.type == "cuda" and torch.cuda.is_available(),
            "M256 checkpoint overlay requires CUDA")
    model = profile.build_model(config, checkpoint_path, device)
    load_audit = profile.validate_h9_load_audit(model, config)
    require(load_audit.get("missing_count") == 0 and
            load_audit.get("unexpected_count") == 0,
            "M256 checkpoint overlay mismatch")
    require(not hasattr(model, "_m71_pattern_paft_state"),
            "training-only PAFT hook leaked into M256 export")
    return model, load_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    require(sys.byteorder == "little", "M256 payloads require a little-endian host")
    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract["schema"] == "m256_paft_ep4_running_bn_bottleneck_int8_contract_v1",
            "contract schema drift")
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite output directory")
    output_dir.mkdir(parents=True)

    checkpoint_path = resolve_receipt(contract["identity"]["checkpoint"])
    config_path = resolve_receipt(contract["identity"]["config"])
    source_manifest_path = resolve_receipt(
        contract["identity"]["m248_source_manifest"])
    resolve_receipt(contract["identity"]["m251r2_range_correction"])
    source_manifest = strict_json(source_manifest_path)
    require(source_manifest["identity"]["checkpoint_sha256"] ==
            contract["identity"]["checkpoint"]["sha256"],
            "M248/checkpoint identity mismatch")
    require(source_manifest["identity"]["capture_bn_policy"] ==
            contract["identity"]["bn_policy"], "BN policy drift")
    operators = contract["identity"]["operators"]
    require(type(operators) is list and len(operators) == 4 and
            len(set(operators)) == 4, "operator population drift")
    require(source_manifest["cohort"]["operators"] == operators,
            "M248 operator order drift")
    require(exact_int(source_manifest["cohort"]["records"], "M248 records") == 40,
            "M248 record population drift")

    model, checkpoint_load_audit = load_checkpoint_model(
        checkpoint_path, config_path)
    require("{}.{}".format(model.__class__.__module__, model.__class__.__name__) ==
            contract["identity"]["checkpoint_model_type"],
            "checkpoint model type drift")
    modules = dict(model.named_modules())
    records_by_operator = {
        name: sorted([row for row in source_manifest["records"]
                      if row["operator"] == name], key=lambda row: row["sample_id"])
        for name in operators
    }
    require(all(len(rows) == 10 for rows in records_by_operator.values()),
            "M248 per-operator record population drift")

    channel_fields = [
        "operator", "operator_index", "output_channel", "weight_count",
        "weight_max_abs", "scale_f32", "scale_f32_bits_hex", "scale_uq0p31_raw",
        "q_min", "q_max", "q_zero_count", "q_endpoint_count",
        "preclip_violation_count", "mae", "rmse", "max_abs_error",
        "normalized_l2_error", "sum_abs_q", "tight_required_signed_bits",
    ]
    reuse_fields = [
        "operator", "operator_index", "input_channel", "kernel_y", "kernel_x",
        "unique_value_count_768", "repeat_factor_768_div_unique",
        "top_value_s8", "top_value_count", "top_value_fraction",
    ]
    for size in (16, 24, 48, 96):
        reuse_fields.extend([
            "b{}_nonzero_updates".format(size),
            "b{}_value_mask_commands".format(size),
            "b{}_value_mask_bytes".format(size),
            "b{}_dense_int8_bytes".format(size),
        ])
    channel_csv_path = output_dir / "per_output_channel_quantization.csv"
    reuse_csv_path = output_dir / "per_source_term_value_mask_multicast.csv"
    require(not channel_csv_path.exists() and not reuse_csv_path.exists(),
            "refusing to overwrite CSV evidence")

    quantized = {}
    layer_rows = []
    total_weights = 0
    total_preclip = 0
    with channel_csv_path.open("w", newline="", encoding="utf-8") as channel_handle, \
            reuse_csv_path.open("w", newline="", encoding="utf-8") as reuse_handle:
        channel_writer = csv.DictWriter(channel_handle, fieldnames=channel_fields)
        reuse_writer = csv.DictWriter(reuse_handle, fieldnames=reuse_fields)
        channel_writer.writeheader()
        reuse_writer.writeheader()
        for operator_index, name in enumerate(operators):
            require(name in modules and isinstance(modules[name], torch.nn.Conv2d),
                    "target module missing or not Conv2d: {}".format(name))
            module = modules[name]
            require(tuple(module.weight.shape) == (768, 768, 3, 3),
                    "weight geometry drift: {}".format(name))
            require(module.bias is None, "bias unexpectedly present: {}".format(name))
            weight = module.weight.detach().cpu().contiguous().numpy()
            require(weight.dtype == np.float32, "weight dtype drift")
            raw_weight_bytes = weight.astype("<f4", copy=False).tobytes(order="C")
            weight_sha = hashlib.sha256(raw_weight_bytes).hexdigest()
            require(weight_sha == contract["identity"]["expected_weight_sha256"][name],
                    "checkpoint weight SHA drift: {}".format(name))
            require(all(row["module_geometry"]["weight_content_sha256"] == weight_sha
                        for row in records_by_operator[name]),
                    "M248 recursive weight SHA drift: {}".format(name))

            audit = canonical_weight_quantize(weight)
            scale = audit["scale"]
            qweight = audit["qweight"]
            error = audit["error_float64"]
            require(audit["preclip_violation_count"] == 0,
                    "unexpected weight saturation before clip")
            scale_uq31 = np.clip(
                np.rint(scale.astype(np.float64) * float(1 << 31)),
                0, (1 << 32) - 1).astype("<u4")
            scale_fixed = scale_uq31.astype(np.float64) / float(1 << 31)
            scale_relative_error = np.abs(
                scale_fixed - scale.astype(np.float64)) / scale.astype(np.float64)

            # M248/M251 source-major layout: O is the contiguous lane dimension.
            hardware_q = qweight.transpose(1, 2, 3, 0).copy(order="C")
            weight_path = output_dir / "o{}_weight_i_ky_kx_o_s8.bin".format(operator_index)
            scale_path = output_dir / "o{}_scale_f32le.bin".format(operator_index)
            scale_fixed_path = output_dir / "o{}_scale_uq0p31_u32le.bin".format(
                operator_index)
            acc_path = output_dir / "o{}_acc_init_s32le.bin".format(operator_index)
            write_new_bytes(weight_path, hardware_q.tobytes(order="C"))
            write_new_bytes(scale_path, scale.astype("<f4", copy=False).tobytes(order="C"))
            write_new_bytes(scale_fixed_path, scale_uq31.tobytes(order="C"))
            write_new_bytes(acc_path, np.zeros(768, dtype="<i4").tobytes(order="C"))
            require(np.array_equal(
                np.frombuffer(weight_path.read_bytes(), dtype=np.int8).reshape(
                    768, 3, 3, 768).transpose(3, 0, 1, 2), qweight),
                "hardware-layout weight roundtrip mismatch")

            sum_abs_q = np.sum(np.abs(qweight.astype(np.int16)),
                               axis=(1, 2, 3), dtype=np.int64)
            tight_bits = [signed_bits_for_symmetric_magnitude(int(value))
                          for value in sum_abs_q]
            weight_norm = float(np.linalg.norm(weight.astype(np.float64).reshape(-1)))
            error_norm = float(np.linalg.norm(error.reshape(-1)))
            for channel in range(768):
                channel_weight = weight[channel].astype(np.float64)
                channel_error = error[channel]
                channel_q = qweight[channel]
                channel_norm = float(np.linalg.norm(channel_weight.reshape(-1)))
                channel_writer.writerow({
                    "operator": name,
                    "operator_index": operator_index,
                    "output_channel": channel,
                    "weight_count": int(channel_weight.size),
                    "weight_max_abs": "{:.17g}".format(float(np.max(np.abs(channel_weight)))),
                    "scale_f32": "{:.17g}".format(float(scale[channel])),
                    "scale_f32_bits_hex": "{:08x}".format(
                        struct.unpack("<I", scale[channel].tobytes())[0]),
                    "scale_uq0p31_raw": int(scale_uq31[channel]),
                    "q_min": int(np.min(channel_q)),
                    "q_max": int(np.max(channel_q)),
                    "q_zero_count": int(np.count_nonzero(channel_q == 0)),
                    "q_endpoint_count": int(np.count_nonzero(np.abs(
                        channel_q.astype(np.int16)) == 127)),
                    "preclip_violation_count": 0,
                    "mae": "{:.17g}".format(float(np.mean(np.abs(channel_error)))),
                    "rmse": "{:.17g}".format(float(np.sqrt(np.mean(channel_error ** 2)))),
                    "max_abs_error": "{:.17g}".format(float(np.max(np.abs(channel_error)))),
                    "normalized_l2_error": "{:.17g}".format(
                        float(np.linalg.norm(channel_error.reshape(-1))) / channel_norm),
                    "sum_abs_q": int(sum_abs_q[channel]),
                    "tight_required_signed_bits": tight_bits[channel],
                })
            reuse = source_term_reuse(
                qweight, name, operator_index, reuse_writer)
            layer_row = {
                "operator": name,
                "operator_index": operator_index,
                "source_float_weight_sha256": weight_sha,
                "source_float_weight_bytes": len(raw_weight_bytes),
                "weight_shape_o_i_ky_kx": list(weight.shape),
                "conv_bias_present": False,
                "weights_audited": int(weight.size),
                "zero_output_channels": int(np.count_nonzero(audit["zero_channel"])),
                "preclip_violation_count": audit["preclip_violation_count"],
                "q_min": int(np.min(qweight)),
                "q_max": int(np.max(qweight)),
                "reserved_negative_128_count": int(np.count_nonzero(qweight == -128)),
                "q_zero_count": int(np.count_nonzero(qweight == 0)),
                "q_endpoint_count": int(np.count_nonzero(
                    np.abs(qweight.astype(np.int16)) == 127)),
                "weight_error": {
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "max_abs_error": float(np.max(np.abs(error))),
                    "normalized_l2_error": error_norm / weight_norm,
                    "max_error_div_stored_scale": float(np.max(
                        np.abs(error) / scale[:, None, None, None].astype(np.float64))),
                },
                "scale": {
                    "minimum_f32": float(np.min(scale)),
                    "maximum_f32": float(np.max(scale)),
                    "uq0p31_max_relative_error": float(np.max(scale_relative_error)),
                    "runtime_application_admitted": False,
                },
                "accumulator_bound": {
                    "per_channel_sum_abs_q_minimum": int(np.min(sum_abs_q)),
                    "per_channel_sum_abs_q_mean": float(np.mean(sum_abs_q)),
                    "per_channel_sum_abs_q_maximum": int(np.max(sum_abs_q)),
                    "checkpoint_tight_required_signed_bits": max(tight_bits),
                    "dense_int8_envelope_magnitude": 768 * 9 * 127,
                    "dense_int8_envelope_required_signed_bits": 21,
                },
                "payloads": [
                    payload_receipt(weight_path, "weight", "signed_int8",
                                    (768, 3, 3, 768), "I_KY_KX_O_C_ORDER"),
                    payload_receipt(scale_path, "weight_scale", "float32_le",
                                    (768,), "OUTPUT_CHANNEL"),
                    payload_receipt(scale_fixed_path, "weight_scale_fixed_candidate",
                                    "uint32_le_uq0p31", (768,), "OUTPUT_CHANNEL"),
                    payload_receipt(acc_path, "accumulator_init", "signed_int32_le",
                                    (768,), "OUTPUT_CHANNEL_ALL_ZERO"),
                ],
                "value_mask_multicast": reuse,
            }
            quantized[name] = {
                "weight": weight,
                "dequant_weight_f32": audit["dequant_float64"].astype(np.float32),
                "qweight": qweight,
            }
            layer_rows.append(layer_row)
            total_weights += int(weight.size)
            total_preclip += audit["preclip_violation_count"]
            print("[M256 quant] {}/4 {}".format(operator_index + 1, name), flush=True)

    require(total_weights == 4 * 768 * 768 * 3 * 3,
            "all-weight audit population mismatch")
    require(total_preclip == 0, "quantization clipping present")

    require(torch.cuda.is_available(),
            "M256 frozen local raw-convolution audit requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    local_rows = []
    layer_numeric = {
        name: {
            "values": 0, "sum_abs_error": 0.0, "sum_squared_error": 0.0,
            "maximum_abs_error": 0.0, "reference_sum_squared": 0.0,
            "candidate_sum_squared": 0.0, "dot_product": 0.0,
        }
        for name in operators
    }
    layer_late_scale = {
        name: {
            "delta": contract["late_scale_elision_audit"][
                "m35_amplitude_delta_by_operator"][name],
            "values": 0,
            "nonzero_accumulators": 0,
            "zero_accumulators": 0,
            "rne_exact_bypass": 0,
            "rne_exact_bypass_nonzero": 0,
            "rne_changed": 0,
            "rne_maximum_absolute_integer_correction": 0,
            "floor_exact_bypass": 0,
            "floor_exact_bypass_nonzero": 0,
            "floor_changed": 0,
            "floor_maximum_absolute_integer_correction": 0,
            "observed_accumulator_minimum": None,
            "observed_accumulator_maximum": None,
        }
        for name in operators
    }
    for operator_index, name in enumerate(operators):
        for record in records_by_operator[name]:
            value_path = source_manifest_path.parent / record["value_payload_file"]
            require(value_path.is_file() and sha256(value_path) ==
                    record["value_payload_sha256"],
                    "M248 value-payload SHA drift")
            compressed = value_path.read_bytes()
            raw = zlib.decompress(compressed)
            require(len(raw) == record["input_content_bytes"] and
                    hashlib.sha256(raw).hexdigest() == record["input_content_sha256"],
                    "M248 value payload decompression mismatch")
            source = np.frombuffer(raw, dtype="<f4").reshape(
                tuple(record["shape"])).copy()
            require(bool(np.isfinite(source).all()), "M248 source has NaN/Infinity")
            delta = layer_late_scale[name]["delta"]
            metrics_raw, late_scale = local_conv_metrics(
                source, quantized[name]["weight"],
                quantized[name]["dequant_weight_f32"],
                quantized[name]["qweight"], delta, device)
            merge_numeric_metrics(layer_numeric[name], metrics_raw)
            late_aggregate = layer_late_scale[name]
            for key in ("values", "nonzero_accumulators", "zero_accumulators",
                        "rne_exact_bypass", "rne_exact_bypass_nonzero", "rne_changed",
                        "floor_exact_bypass", "floor_exact_bypass_nonzero",
                        "floor_changed"):
                late_aggregate[key] += late_scale[key]
            for key in ("rne_maximum_absolute_integer_correction",
                        "floor_maximum_absolute_integer_correction"):
                late_aggregate[key] = max(late_aggregate[key], late_scale[key])
            low = late_aggregate["observed_accumulator_minimum"]
            high = late_aggregate["observed_accumulator_maximum"]
            late_aggregate["observed_accumulator_minimum"] = (
                late_scale["observed_accumulator_minimum"] if low is None else
                min(low, late_scale["observed_accumulator_minimum"]))
            late_aggregate["observed_accumulator_maximum"] = (
                late_scale["observed_accumulator_maximum"] if high is None else
                max(high, late_scale["observed_accumulator_maximum"]))
            metrics = finalize_numeric_metrics(metrics_raw)
            metrics.update({
                "operator": name,
                "operator_index": operator_index,
                "sample_id": exact_int(record["sample_id"], "sample id"),
                "sample_key": record["sample_key"],
                "source_float32_sha256": record["input_content_sha256"],
            })
            local_rows.append(metrics)
            print("[M256 local conv] o{} s{:02d}".format(
                operator_index, record["sample_id"]), flush=True)
    torch.cuda.synchronize(device)
    local_layer = {}
    gate = contract["local_numeric_audit"]["predeclared_layer_aggregate_gate"]
    for name in operators:
        metrics = finalize_numeric_metrics(layer_numeric[name])
        metrics["normalized_l2_gate_maximum"] = gate[
            "normalized_l2_error_maximum"]
        metrics["cosine_gate_minimum"] = gate["cosine_similarity_minimum"]
        metrics["gate_pass"] = bool(
            metrics["normalized_l2_error"] <= gate["normalized_l2_error_maximum"]
            and metrics["cosine_similarity"] >= gate["cosine_similarity_minimum"])
        require(metrics["gate_pass"], "local raw-convolution numeric gate failed")
        local_layer[name] = metrics
        late = layer_late_scale[name]
        late["rne_exact_bypass_fraction_all"] = (
            float(late["rne_exact_bypass"]) / late["values"])
        late["rne_exact_bypass_fraction_nonzero"] = (
            float(late["rne_exact_bypass_nonzero"]) /
            late["nonzero_accumulators"])
        late["floor_exact_bypass_fraction_all"] = (
            float(late["floor_exact_bypass"]) / late["values"])
        late["floor_exact_bypass_fraction_nonzero"] = (
            float(late["floor_exact_bypass_nonzero"]) /
            late["nonzero_accumulators"])

    del model
    payload_receipts = []
    for layer in layer_rows:
        payload_receipts.extend(layer["payloads"])
    csv_receipts = [
        payload_receipt(channel_csv_path, "per_output_channel_audit", "csv", (3072,),
                        "OPERATOR_THEN_OUTPUT_CHANNEL"),
        payload_receipt(reuse_csv_path, "per_source_term_value_mask_audit", "csv",
                        (4 * 768 * 9,), "OPERATOR_THEN_INPUT_CHANNEL_KY_KX"),
    ]
    result = {
        "schema": "m256_paft_ep4_running_bn_bottleneck_int8_result_v1",
        "status": (
            "PASS_PAFT_EP4_RUNNING_BN_FOUR_LAYER_INT8_WEIGHT_BRIDGE_AND_S10_RAW_"
            "CONV_LOCAL_NUMERIC_GATE_INTEGRATED_RTL_AND_FULL_NETWORK_REMAIN_BLOCKED"),
        "identity": {
            "contract_path": str(contract_path.relative_to(ROOT)),
            "contract_sha256": sha256(contract_path),
            "exporter_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "exporter_sha256": sha256(Path(__file__).resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "config_sha256": sha256(config_path),
            "checkpoint_load_audit": checkpoint_load_audit,
            "m248_source_manifest_sha256": sha256(source_manifest_path),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "deterministic_algorithms": True,
        },
        "population": {
            "operators": 4,
            "output_channels": 3072,
            "weights_audited": total_weights,
            "m248_source_records": len(local_rows),
            "raw_conv_output_values_compared": sum(row["values"] for row in local_rows),
            "preclip_violation_count": total_preclip,
        },
        "quantization_contract": contract["quantization"],
        "layers": layer_rows,
        "local_raw_convolution_audit": {
            "scope": contract["local_numeric_audit"]["scope"],
            "rows": local_rows,
            "aggregate_by_layer": local_layer,
            "full_network_accuracy": None,
            "valid825_accuracy": None,
        },
        "m251_schedule_bridge": {
            "payload_layout": contract["quantization"]["weight_payload_layout"],
            "candidate_linear_address": contract["m251_layout_bridge"][
                "candidate_linear_address"],
            "total_int8_weight_bytes": sum(
                item["bytes"] for item in payload_receipts if item["role"] == "weight"),
            "total_scale_float32_bytes": sum(
                item["bytes"] for item in payload_receipts
                if item["role"] == "weight_scale"),
            "total_scale_uq0p31_bytes": sum(
                item["bytes"] for item in payload_receipts
                if item["role"] == "weight_scale_fixed_candidate"),
            "total_accumulator_init_bytes": sum(
                item["bytes"] for item in payload_receipts
                if item["role"] == "accumulator_init"),
            "checkpoint_tight_accumulator_signed_bits": max(
                layer["accumulator_bound"]["checkpoint_tight_required_signed_bits"]
                for layer in layer_rows),
            "dense_envelope_accumulator_signed_bits": 21,
            "current_weight_residency_bytes": contract["m251_layout_bridge"][
                "current_weight_residency_bytes"],
            "full_96_output_tile_bytes": contract["m251_layout_bridge"][
                "bytes_per_full_output_lane_tile"],
            "full_96_output_tile_fits_current_residency": False,
            "physical_layout_selected": False,
            "real_address_cycle_schedule_admitted": False,
        },
        "value_mask_multicast_scope": contract["value_mask_multicast_audit"],
        "late_scale_elision_audit": {
            "contract": contract["late_scale_elision_audit"],
            "aggregate_by_layer": layer_late_scale,
            "m35_rounding_rtl_admitted": False,
            "late_scale_elision_rtl_admitted": False,
            "cycle_reduction_admitted": False,
        },
        "payloads": payload_receipts + csv_receipts,
        "admission": {
            "checkpoint_bound_int8_weight_payload_admitted": True,
            "all_weight_error_and_saturation_audit_admitted": True,
            "bias_free_zero_accumulator_init_admitted": True,
            "checkpoint_tight_accumulator_bound_admitted": True,
            "s10_raw_convolution_local_numeric_audit_admitted": True,
            "weight_value_mask_fetch_potential_statistics_admitted": True,
            "rounding_aware_late_scale_elision_statistics_admitted": True,
            "running_batchnorm_hardware_admitted": False,
            "full_network_accuracy_admitted": False,
            "real_m251_address_cycle_schedule_admitted": False,
            "multicast_accumulator_cycle_reduction_admitted": False,
            "system_speedup_admitted": False,
            "integrated_rtl_or_synopsys_admitted": False,
            "ppa_power_energy_admitted": False,
            "date_best_paper_readiness_admitted": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    result_path = output_dir / "m256_paft_ep4_running_bn_bottleneck_int8_result_r1.json"
    require(not result_path.exists(), "refusing to overwrite result")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
