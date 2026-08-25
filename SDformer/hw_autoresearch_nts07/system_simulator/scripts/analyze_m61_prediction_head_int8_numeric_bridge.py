#!/usr/bin/env python3
"""Checkpoint-bound INT8/numeric bridge for the M60 prediction head.

The analysis is intentionally kernel-local.  It qualifies exact integer
arithmetic and output perturbation on the ten frozen M51 head inputs, but does
not turn either number into a system speedup or a valid825 accuracy result.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch


PARENTS = ("zero", "left", "up", "previous_timestep")


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def product(values):
    value = 1
    for item in values:
        value *= int(item)
    return value


def unpack_little(path, elements):
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(int(packed.size) == (int(elements) + 7) // 8,
            "packed-byte mismatch: {}".format(path))
    # Avoid depending on the numpy bitorder keyword on older hosts.
    lut = ((np.arange(256, dtype=np.uint16)[:, None] >>
            np.arange(8, dtype=np.uint16)[None, :]) & 1).astype(np.uint8)
    bits = lut[packed].reshape(-1)[:int(elements)]
    if int(elements) % 8:
        used_mask = (1 << (int(elements) % 8)) - 1
        require((int(packed[-1]) & ~used_mask) == 0,
                "nonzero tail padding: {}".format(path))
    return bits


def as_tbhwc(bits, record):
    shape = [int(value) for value in record["input_shape"]]
    require(record["operator"] == "Conv2d" and len(shape) == 5 and
            product(shape) == int(bits.size), "head shape mismatch")
    array = bits.reshape(shape).transpose(0, 1, 3, 4, 2)
    require(list(array.shape) == [10, 1, 240, 320, 96],
            "unexpected prediction-head geometry")
    return array


def apply_candidate(best, choice, signed, candidate_signed, candidate_id,
                    target_slice, validity):
    target_best = best[target_slice]
    target_choice = choice[target_slice]
    target_signed = signed[target_slice]
    cost = np.abs(candidate_signed).sum(axis=-1, dtype=np.int32)
    take = np.logical_and(cost < target_best, validity)
    target_best[take] = cost[take]
    target_choice[take] = candidate_id
    target_signed[:] = np.where(take[..., None], candidate_signed,
                                target_signed)


def bounded_signed_residual(array, tile_h, tile_w):
    """Independent reimplementation of the frozen M60 tie/edge policy."""
    current = array.astype(np.int8, copy=False)
    signed = current.copy()
    best = np.abs(signed).sum(axis=-1, dtype=np.int32)
    choice = np.zeros(best.shape, dtype=np.uint8)

    left = (current[:, :, :, 1:, :] -
            current[:, :, :, :-1, :]).astype(np.int8, copy=False)
    left_w = np.arange(1, array.shape[3], dtype=np.int32)
    left_valid = (left_w % int(tile_w) != 0)[None, None, None, :]
    apply_candidate(best, choice, signed, left, 1,
                    (slice(None), slice(None), slice(None), slice(1, None)),
                    left_valid)
    del left

    up = (current[:, :, 1:, :, :] -
          current[:, :, :-1, :, :]).astype(np.int8, copy=False)
    up_h = np.arange(1, array.shape[2], dtype=np.int32)
    up_valid = (up_h % int(tile_h) != 0)[None, None, :, None]
    apply_candidate(best, choice, signed, up, 2,
                    (slice(None), slice(None), slice(1, None), slice(None)),
                    up_valid)
    del up

    previous = (current[1:] - current[:-1]).astype(np.int8, copy=False)
    apply_candidate(best, choice, signed, previous, 3,
                    (slice(1, None), slice(None), slice(None), slice(None)),
                    True)
    del previous
    require(np.all(np.logical_or(signed == -1,
                                 np.logical_or(signed == 0, signed == 1))),
            "signed residual outside -1/0/+1")
    require(np.array_equal(np.abs(signed).sum(axis=-1, dtype=np.int32), best),
            "selected signed residual does not match selected cost")
    return signed, choice


def counts(choice):
    values = np.bincount(choice.reshape(-1), minlength=4)
    return dict((PARENTS[index], int(values[index])) for index in range(4))


def minimum_signed_bits(lower, upper):
    for bits in range(2, 65):
        if int(lower) >= -(1 << (bits - 1)) and int(upper) <= ((1 << (bits - 1)) - 1):
            return bits
    raise ValueError("accumulator bound exceeds signed64")


def empty_error_stats():
    return {
        "count": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "max_abs": 0.0,
        "reference_sum_sq": 0.0,
        "sign_compared": 0,
        "sign_equal": 0,
    }


def add_error(stats, reference, candidate):
    error = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(error)
    stats["count"] += int(error.size)
    stats["sum_abs"] += float(absolute.sum(dtype=np.float64))
    stats["sum_sq"] += float(np.square(error).sum(dtype=np.float64))
    stats["max_abs"] = max(stats["max_abs"], float(absolute.max(initial=0.0)))
    stats["reference_sum_sq"] += float(
        np.square(reference.astype(np.float64)).sum(dtype=np.float64))
    nonzero = reference != 0.0
    stats["sign_compared"] += int(nonzero.sum(dtype=np.int64))
    stats["sign_equal"] += int(np.equal(np.signbit(reference[nonzero]),
                                        np.signbit(candidate[nonzero])).sum(
                                            dtype=np.int64))


def finish_error(stats):
    count = int(stats["count"])
    require(count > 0, "empty error population")
    rmse = (float(stats["sum_sq"]) / float(count)) ** 0.5
    reference_rms = (float(stats["reference_sum_sq"]) / float(count)) ** 0.5
    return {
        "count": count,
        "mae": float(stats["sum_abs"]) / float(count),
        "max_abs": float(stats["max_abs"]),
        "rmse": rmse,
        "normalized_rmse_over_reference_rms": (
            rmse / reference_rms if reference_rms else None),
        "reference_rms": reference_rms,
        "sign_agreement_nonzero_reference": (
            float(stats["sign_equal"]) / float(stats["sign_compared"])
            if stats["sign_compared"] else None),
        "sign_compared": int(stats["sign_compared"]),
    }


def bind_model(repo_root, config_path, checkpoint_path, module_name):
    exp = repo_root / "neuron_experiments/H9_bipolar_self_attention"
    for path in (exp / "entrypoints", exp / "overlay",
                 repo_root / "third_party/SDformerFlow", repo_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import profile_nts11_hardware_p0 as profile

    config, _ = profile.load_config(config_path)
    model = profile.build_model(config, checkpoint_path, torch.device("cpu"))
    audit = profile.validate_h9_load_audit(model, config)
    require(audit is not None and int(audit["missing_count"]) == 0 and
            int(audit["unexpected_count"]) == 0,
            "checkpoint load audit is incomplete")
    named = dict(model.named_modules())
    require(module_name in named and isinstance(named[module_name], torch.nn.Conv2d),
            "prediction-head module missing/type mismatch")
    module = named[module_name]
    weight = module.weight.detach().cpu().contiguous().numpy().astype(
        np.float32, copy=False).reshape(2, 96)
    bias = module.bias.detach().cpu().contiguous().numpy().astype(
        np.float32, copy=False)
    return weight, bias, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--m60-result", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--valid825-profile", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-vectors", type=int, default=32768)
    arguments = parser.parse_args()

    contract = strict_json(arguments.contract)
    manifest = strict_json(arguments.manifest)
    m60 = strict_json(arguments.m60_result)
    require(contract["schema"] ==
            "m61_prediction_head_int8_numeric_bridge_contract_v1",
            "contract schema mismatch")
    identities = contract["identity"]
    require(sha256_path(Path(__file__).resolve()) ==
            identities["analyzer_sha256"], "analyzer SHA mismatch")
    require(sha256_path(arguments.manifest) == identities["manifest_sha256"] and
            sha256_path(arguments.m60_result) == identities["m60_result_sha256"],
            "M51/M60 identity mismatch")
    require(arguments.checkpoint.stat().st_size ==
            identities["checkpoint_size_bytes"] and
            sha256_path(arguments.checkpoint) == identities["checkpoint_sha256"],
            "checkpoint identity mismatch")
    require(sha256_path(arguments.config) == identities["deploy_config_sha256"],
            "deployment config identity mismatch")
    require(sha256_path(arguments.valid825_profile) ==
            identities["valid825_profile_sha256"],
            "valid825 profile identity mismatch")
    require(not arguments.output_dir.exists(), "refusing existing output directory")
    require(arguments.chunk_vectors > 0, "invalid chunk-vectors")

    module_name = identities["module_name"]
    module_index = int(identities["module_index"])
    module_identity = manifest["module_identities"][module_name]
    require(module_identity["weight"]["content_sha256"] ==
            identities["float_weight_sha256"] and
            module_identity["bias"]["content_sha256"] ==
            identities["float_bias_sha256"],
            "manifest head parameter identity mismatch")
    require(m60["selected_capacity_feasible_tile"] ==
            contract["tile_policy"], "M60 selected tile mismatch")
    selected_key = "H{}_W{}".format(contract["tile_policy"]["tile_h"],
                                     contract["tile_policy"]["tile_w"])
    selected_m60 = None
    for row in m60["configurations"]:
        if "H{}_W{}".format(row["tile_h"], row["tile_w"]) == selected_key:
            selected_m60 = row
    require(selected_m60 is not None, "selected M60 row missing")

    repo_root = arguments.repo_root.resolve()
    weight, bias, load_audit = bind_model(
        repo_root, arguments.config.resolve(), arguments.checkpoint.resolve(),
        module_name)
    require(hashlib.sha256(weight.reshape(2, 96, 1, 1).tobytes(
        order="C")).hexdigest() == identities["float_weight_sha256"] and
        hashlib.sha256(bias.tobytes(order="C")).hexdigest() ==
        identities["float_bias_sha256"],
        "loaded checkpoint parameter content mismatch")

    # Per-output symmetric INT8; np.rint is round-to-nearest, ties-to-even.
    scales = (np.max(np.abs(weight), axis=1).astype(np.float32) /
              np.float32(127.0)).astype("<f4")
    require(np.all(scales > 0), "zero quantization scale")
    quant_weight = np.rint(weight / scales[:, None]).clip(-127, 127).astype(np.int8)
    quant_bias = np.rint(bias / scales).astype("<i4")
    require(np.all(quant_weight >= -127) and np.all(quant_weight <= 127),
            "INT8 range failure")

    lower_bounds = []
    upper_bounds = []
    required_bits = []
    for output in range(2):
        row = quant_weight[output].astype(np.int32)
        lower = int(row[row < 0].sum(dtype=np.int64)) + int(quant_bias[output])
        upper = int(row[row > 0].sum(dtype=np.int64)) + int(quant_bias[output])
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        required_bits.append(minimum_signed_bits(lower, upper))
    accumulator_bits = max(required_bits)
    require(accumulator_bits <= int(contract["quantization"][
        "maximum_qualified_accumulator_bits"]),
            "accumulator exceeds contract")

    target_records = [row for row in manifest["records"]
                      if int(row["module_index"]) == module_index]
    target_records.sort(key=lambda row: int(row["sample_id"]))
    require(len(target_records) == 10 and
            [int(row["sample_id"]) for row in target_records] == list(range(10)),
            "head record population mismatch")
    m60_records = dict((int(row["sample_id"]), row["configs"][selected_key])
                       for row in m60["per_record"])

    total_error = empty_error_stats()
    channel_error = [empty_error_stats(), empty_error_stats()]
    integer_outputs = 0
    integer_mismatches = 0
    integer_max_abs_difference = 0
    argmax_equal = 0
    vector_count = 0
    epe_sum = 0.0
    epe_max = 0.0
    observed_min = [None, None]
    observed_max = [None, None]
    aggregate_choice = dict((name, 0) for name in PARENTS)
    positive_events = 0
    negative_events = 0
    source_events = 0
    per_record = []
    q32 = quant_weight.astype(np.int32)
    tile_h = int(contract["tile_policy"]["tile_h"])
    tile_w = int(contract["tile_policy"]["tile_w"])

    for record in target_records:
        payload = arguments.payload_root.resolve() / record["relative_path"]
        require(payload.is_file() and payload.stat().st_size ==
                int(record["packed_bytes"]) and
                sha256_path(payload) == record["file_sha256"],
                "payload identity mismatch: {}".format(payload))
        bits = unpack_little(payload, record["input_elements"])
        require(int(bits.sum(dtype=np.int64)) == int(record["active_elements"]),
                "payload popcount mismatch")
        array = as_tbhwc(bits, record)
        signed, choice = bounded_signed_residual(array, tile_h, tile_w)
        current = array.reshape(-1, 96)
        delta = signed.reshape(-1, 96)
        choice_row = counts(choice)
        positive = int((signed > 0).sum(dtype=np.int64))
        negative = int((signed < 0).sum(dtype=np.int64))
        source = positive + negative
        upstream = m60_records[int(record["sample_id"])]
        require(choice_row == upstream["choice_counts"] and
                positive == int(upstream["positive_residual_events"]) and
                negative == int(upstream["negative_residual_events"]) and
                source == int(upstream["source_bits"]),
                "M60 signed ledger mismatch")
        for name in PARENTS:
            aggregate_choice[name] += int(choice_row[name])
        positive_events += positive
        negative_events += negative
        source_events += source

        sample_error = empty_error_stats()
        sample_mismatches = 0
        sample_max_difference = 0
        sample_argmax = 0
        sample_vectors = int(current.shape[0])
        sample_epe_sum = 0.0
        sample_epe_max = 0.0
        sample_observed_min = [None, None]
        sample_observed_max = [None, None]
        for start in range(0, sample_vectors, arguments.chunk_vectors):
            stop = min(sample_vectors, start + arguments.chunk_vectors)
            x8 = current[start:stop]
            d8 = delta[start:stop]
            parent16 = x8.astype(np.int16) - d8.astype(np.int16)
            require(np.all(np.logical_or(parent16 == 0, parent16 == 1)),
                    "selected parent is not binary")
            x32 = x8.astype(np.int32)
            d32 = d8.astype(np.int32)
            parent32 = parent16.astype(np.int32)
            dense_int = np.matmul(x32, q32.T) + quant_bias[None, :]
            reconstructed_int = (np.matmul(parent32, q32.T) +
                                 np.matmul(d32, q32.T) +
                                 quant_bias[None, :])
            difference = reconstructed_int.astype(np.int64) - dense_int.astype(np.int64)
            mismatches = int(np.count_nonzero(difference))
            max_difference = int(np.abs(difference).max(initial=0))
            sample_mismatches += mismatches
            sample_max_difference = max(sample_max_difference, max_difference)

            reference = (np.matmul(x8.astype(np.float32), weight.T) +
                         bias[None, :]).astype(np.float32)
            candidate = (dense_int.astype(np.float32) *
                         scales[None, :]).astype(np.float32)
            add_error(sample_error, reference, candidate)
            add_error(total_error, reference, candidate)
            for output in range(2):
                add_error(channel_error[output], reference[:, output],
                          candidate[:, output])
                current_min = int(dense_int[:, output].min(initial=0))
                current_max = int(dense_int[:, output].max(initial=0))
                if sample_observed_min[output] is None:
                    sample_observed_min[output] = current_min
                    sample_observed_max[output] = current_max
                else:
                    sample_observed_min[output] = min(
                        sample_observed_min[output], current_min)
                    sample_observed_max[output] = max(
                        sample_observed_max[output], current_max)
                if observed_min[output] is None:
                    observed_min[output] = current_min
                    observed_max[output] = current_max
                else:
                    observed_min[output] = min(observed_min[output], current_min)
                    observed_max[output] = max(observed_max[output], current_max)
            sample_argmax += int(np.equal(np.argmax(reference, axis=1),
                                          np.argmax(candidate, axis=1)).sum(
                                              dtype=np.int64))
            endpoint_error = np.sqrt(np.square(
                candidate.astype(np.float64) - reference.astype(np.float64)
            ).sum(axis=1, dtype=np.float64))
            sample_epe_sum += float(endpoint_error.sum(dtype=np.float64))
            sample_epe_max = max(sample_epe_max,
                                 float(endpoint_error.max(initial=0.0)))

        integer_outputs += sample_vectors * 2
        integer_mismatches += sample_mismatches
        integer_max_abs_difference = max(integer_max_abs_difference,
                                         sample_max_difference)
        argmax_equal += sample_argmax
        vector_count += sample_vectors
        epe_sum += sample_epe_sum
        epe_max = max(epe_max, sample_epe_max)
        per_record.append({
            "choice_counts": choice_row,
            "file_sha256": record["file_sha256"],
            "float_error": finish_error(sample_error),
            "integer_reconstruction": {
                "mismatches": sample_mismatches,
                "outputs": sample_vectors * 2,
                "max_abs_difference": sample_max_difference,
            },
            "negative_residual_events": negative,
            "observed_integer_accumulator_min": sample_observed_min,
            "observed_integer_accumulator_max": sample_observed_max,
            "positive_residual_events": positive,
            "relative_path": record["relative_path"],
            "sample_id": int(record["sample_id"]),
            "signed_residual_sha256": hashlib.sha256(
                signed.tobytes(order="C")).hexdigest(),
            "choice_sha256": hashlib.sha256(
                choice.tobytes(order="C")).hexdigest(),
            "source_events": source,
            "two_channel_argmax_agreement_diagnostic": (
                float(sample_argmax) / float(sample_vectors)),
            "vector_endpoint_perturbation": {
                "count": sample_vectors,
                "mean": sample_epe_sum / float(sample_vectors),
                "max": sample_epe_max,
            },
        })
        del bits, array, signed, choice

    require(integer_outputs == 15360000 and integer_mismatches == 0 and
            integer_max_abs_difference == 0,
            "integer parent-delta reconstruction is not exact")
    require(aggregate_choice == selected_m60["choice_counts"] and
            positive_events == int(selected_m60["positive_residual_events"]) and
            negative_events == int(selected_m60["negative_residual_events"]) and
            source_events == int(selected_m60["source_bits"]),
            "aggregate M60 signed ledger mismatch")
    for output in range(2):
        require(observed_min[output] >= lower_bounds[output] and
                observed_max[output] <= upper_bounds[output],
                "observed accumulator outside theoretical bound")

    profile = strict_json(arguments.valid825_profile)
    require(int(profile["samples"]) == 825 and
            profile["artifact_identity"]["checkpoint_sha256"] ==
            identities["checkpoint_sha256"],
            "valid825 source profile population/identity mismatch")

    partial = arguments.output_dir.with_name(
        arguments.output_dir.name + ".partial.{}".format(os.getpid()))
    require(not partial.exists(), "partial output already exists")
    partial.mkdir(parents=True)
    artifact_names = contract["artifact_filenames"]
    quant_weight.astype(np.int8).tofile(str(partial / artifact_names["weight_int8"]))
    quant_bias.astype("<i4").tofile(str(partial / artifact_names["bias_int32"]))
    scales.astype("<f4").tofile(str(partial / artifact_names["scale_float32"]))
    artifacts = {}
    for key in ("weight_int8", "bias_int32", "scale_float32"):
        filename = artifact_names[key]
        path = partial / filename
        artifacts[key] = {
            "bytes": path.stat().st_size,
            "filename": filename,
            "sha256": sha256_path(path),
        }

    weight_error = quant_weight.astype(np.float32) * scales[:, None] - weight
    bias_error = quant_bias.astype(np.float32) * scales - bias
    capacity_bits = int(selected_m60["capacity"]["components_bytes"][
        "conditional_19b_output_accumulator_tile_pair"]) * 8
    # Remove M60's 19-bit accumulator tile pair and replace it with the now
    # qualified exact width.  All ceil-byte boundaries are exact here.
    accumulator_values = 2 * tile_h * tile_w * 2
    require(capacity_bits == accumulator_values * 19,
            "M60 accumulator capacity identity mismatch")
    qualified_accumulator_bytes = (accumulator_values * accumulator_bits + 7) // 8
    corrected_combined = (int(selected_m60["capacity"]["combined_capacity_bytes"]) -
                          capacity_bits // 8 + qualified_accumulator_bytes)

    result = {
        "accumulator_proof": {
            "declared_signed_bits": accumulator_bits,
            "per_output": [{
                "bias_int32": int(quant_bias[index]),
                "lower_bound_all_binary_inputs_inclusive": lower_bounds[index],
                "observed_max_ten_samples": observed_max[index],
                "observed_min_ten_samples": observed_min[index],
                "required_signed_bits": required_bits[index],
                "upper_bound_all_binary_inputs_inclusive": upper_bounds[index],
            } for index in range(2)],
            "sequential_update_proof": (
                "With residual source channels issued in ascending channel order, "
                "every partial parent-plus-delta accumulator is the dot product of "
                "the INT8 weight row with a hybrid binary vector, plus quantized "
                "bias; therefore the exhaustive all-binary lower/upper sums bound "
                "every intermediate as well as every final value."),
        },
        "artifacts": artifacts,
        "capacity_requalification": {
            "m60_conditional_accumulator_bits": 19,
            "qualified_accumulator_bits": accumulator_bits,
            "qualified_accumulator_tile_pair_bytes": qualified_accumulator_bytes,
            "corrected_combined_capacity_bytes": corrected_combined,
            "corrected_headroom_bytes": int(selected_m60["capacity"][
                "maximum_combined_capacity_bytes"]) - corrected_combined,
            "performance_ratio_unchanged": True,
        },
        "checkpoint_load_audit": load_audit,
        "claim_boundary": contract["claim_boundary"],
        "contract_sha256": sha256_path(arguments.contract),
        "float_parameter_identity": {
            "bias_float32": [float(value) for value in bias],
            "bias_sha256": identities["float_bias_sha256"],
            "weight_sha256": identities["float_weight_sha256"],
        },
        "integer_parent_delta_reconstruction": {
            "choice_counts": aggregate_choice,
            "mismatches": integer_mismatches,
            "outputs": integer_outputs,
            "max_abs_difference": integer_max_abs_difference,
            "negative_residual_events": negative_events,
            "positive_residual_events": positive_events,
            "signed_values": [-1, 0, 1],
            "source_events": source_events,
            "tile_h": tile_h,
            "tile_w": tile_w,
        },
        "numeric_perturbation_ten_real_samples": {
            "all_outputs": finish_error(total_error),
            "per_output": [finish_error(row) for row in channel_error],
            "two_channel_argmax_agreement_diagnostic_not_accuracy": (
                float(argmax_equal) / float(vector_count)),
            "vector_endpoint_perturbation_not_ground_truth_aee": {
                "count": vector_count,
                "mean": epe_sum / float(vector_count),
                "max": epe_max,
            },
        },
        "per_record": per_record,
        "quantization": {
            "bias_int32": [int(value) for value in quant_bias],
            "bias_quantization_error": [float(value) for value in bias_error],
            "per_output_scale_float32": [float(value) for value in scales],
            "rounding": "RNE_TIES_TO_EVEN",
            "scheme": "PER_OUTPUT_SYMMETRIC_SIGNED_INT8_NARROW_RANGE",
            "weight_dequant_error": {
                "mae": float(np.abs(weight_error).mean(dtype=np.float64)),
                "max_abs": float(np.abs(weight_error).max()),
                "rmse": float(np.sqrt(np.square(
                    weight_error.astype(np.float64)).mean(dtype=np.float64))),
            },
            "weight_int8_min_per_output": [
                int(value) for value in quant_weight.min(axis=1)],
            "weight_int8_max_per_output": [
                int(value) for value in quant_weight.max(axis=1)],
        },
        "schema": "m61_prediction_head_int8_numeric_bridge_result_v1",
        "status": "PASS_INT8_NUMERIC_BRIDGE_VALID825_QUANTIZED_OPEN",
        "upstream_identity": identities,
        "valid825_accuracy_closure": {
            "original_fp32_checkpoint_metrics_read_only": profile["metrics"],
            "original_profile_samples": int(profile["samples"]),
            "original_profile_sha256": identities["valid825_profile_sha256"],
            "quantized_head_valid825_status": "OPEN_NOT_RUN_GPU_OCCUPIED_BY_EXISTING_TRAINING",
            "ten_sample_numeric_perturbation_is_not_a_valid825_substitute": True,
        },
    }
    result_path = partial / artifact_names["result_json"]
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    os.rename(str(partial), str(arguments.output_dir))
    print(json.dumps({
        "accumulator_bits": accumulator_bits,
        "integer_mismatches": integer_mismatches,
        "output": str(arguments.output_dir / artifact_names["result_json"]),
        "result_sha256": sha256_path(arguments.output_dir /
                                      artifact_names["result_json"]),
        "status": result["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
