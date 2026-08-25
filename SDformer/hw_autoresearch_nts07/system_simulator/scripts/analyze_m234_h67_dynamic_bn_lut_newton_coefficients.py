#!/usr/bin/env python3
"""DSE an integer LUT+Newton coefficient path on M233 H67 BN ranges."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
INCOMING = HW / "system_handoff/incoming/m233_h67_ffn_dynamic_bn_ranges_s10_r1_20260825"
PATHS = {
    "m233_manifest": INCOMING / "manifest.sha256",
    "m233_summary": INCOMING / "m233_h67_ffn_dynamic_bn_range_summary_r1.json",
    "m233_npz": INCOMING / "m233_h67_ffn_dynamic_bn_ranges_s10.npz",
    "m233_records": INCOMING / "per_sample_module_records.csv",
    "m233_samples": INCOMING / "samples.csv",
    "m233_capture": HW / "system_handoff/scripts/capture_m233_h67_ffn_dynamic_bn_ranges.py",
    "m233_contract": HW / "contracts/m233_h67_ffn_dynamic_bn_range_capture_contract_r1_20260825.json",
    "m232_correction": HW / "results/m232_r1_storage_and_first_latency_correction_overlay_r1_20260825/manifest.sha256",
    "m232_review": HW / "results/m232_independent_hammer_review_r1_20260825/SHA256SUMS",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "m233_manifest": "4f4914aa5c167751ba7d4db4f00b5eeda4dced660ce668144f3613b2c4cd42ad",
    "m233_summary": "e4ff7b29617484e2d1a71d5faa4964d176bb46486e129a984f28fa0095a43f9f",
    "m233_npz": "d08aa6f49ec246a89008138c421a1bd6bf9274c7995dee48f245777237e9f7a3",
    "m233_records": "3db8c0ec1b4f3734464fba0436749db26082ebbb32c1a06331a640c2ab640a6d",
    "m233_samples": "c41ca08151e970c3a220cc0aede28fad86183765f2c136a189c54b7b35f9b687",
    "m233_capture": "10c1c7de0d78b7dc1f0491e020119143e75a6b3d036a4219208a28e41193afc8",
    "m233_contract": "fb47bfade5b86f75880c30ea12115ad35d7387e8d09a601a7209ac45720fdb4b",
    "m232_correction": "b4159d3beeb4ec91acc95e1b34bede0da3ff2eb7fd936c82015af2c4ab185cc3",
    "m232_review": "60fd58ad6bf32151f2199b90face9a2add8436be20d5c2834c902b06ecb3e7df",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

VAR_FRAC = 16
VAR_BITS = 22
MEAN_FRAC = 14
MEAN_BITS = 18
PARAM_FRAC = 14
PARAM_BITS = 16
MANTISSA_FRAC = 16
LUT_FRAC = 18
LUT_BITS = 19
NEWTON_FRAC = 18
INV_FRAC = 16
INV_BITS = 20
COEFF_FRAC = 16
COEFF_BITS = 20
EPSILON = 1.0e-5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(
            handle, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + value)))


def quantize(values, bits, frac, signed):
    lower = -(1 << (bits - 1)) if signed else 0
    upper = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    raw = np.rint(np.asarray(values, dtype=np.float64) * (1 << frac))
    rails = np.logical_or(raw < lower, raw > upper)
    return np.clip(raw, lower, upper).astype(np.int64), int(np.count_nonzero(rails))


def rshift_rne_unsigned(values, shift):
    values = np.asarray(values, dtype=np.int64)
    if shift == 0:
        return values.copy()
    require(shift > 0, "negative RNE shift")
    quotient = values >> shift
    remainder = values & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = np.logical_or(
        remainder > half,
        np.logical_and(remainder == half, (quotient & 1) != 0))
    return quotient + increment.astype(np.int64)


def rshift_rne_signed(values, shift):
    values = np.asarray(values, dtype=np.int64)
    signs = np.where(values < 0, -1, 1).astype(np.int64)
    magnitude = np.abs(values)
    return signs * rshift_rne_unsigned(magnitude, shift)


def variable_power2(values, shifts):
    """Return values * 2**shifts with RNE for negative shifts."""
    values = np.asarray(values, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64)
    result = np.empty_like(values)
    for shift in np.unique(shifts):
        mask = shifts == shift
        if shift >= 0:
            result[mask] = values[mask] << int(shift)
        else:
            result[mask] = rshift_rne_unsigned(
                values[mask], int(-shift))
    return result


def make_lut(per_segment):
    require(per_segment > 0 and per_segment & (per_segment - 1) == 0,
            "per-segment LUT depth must be power of two")
    first = 1.0 + (np.arange(per_segment) + 0.5) / per_segment
    second = 2.0 + (np.arange(per_segment) + 0.5) * 2.0 / per_segment
    values, rails = quantize(
        1.0 / np.sqrt(np.concatenate((first, second))),
        LUT_BITS, LUT_FRAC, False)
    require(rails == 0, "LUT rail")
    return values


def integer_model(variance, mean, gamma, beta, per_segment, newton_steps):
    x_q, x_rails = quantize(
        variance + EPSILON, VAR_BITS, VAR_FRAC, False)
    mean_q, mean_rails = quantize(mean, MEAN_BITS, MEAN_FRAC, True)
    gamma_q, gamma_rails = quantize(gamma, PARAM_BITS, PARAM_FRAC, True)
    beta_q, beta_rails = quantize(beta, PARAM_BITS, PARAM_FRAC, True)
    require(np.all(x_q > 0), "quantized variance+epsilon reached zero")

    msb = np.floor(np.log2(x_q.astype(np.float64))).astype(np.int64)
    exponent = msb - VAR_FRAC
    even_exponent = exponent - (exponent & 1)
    mantissa_q = variable_power2(x_q, -even_exponent)
    require(np.all(mantissa_q >= (1 << MANTISSA_FRAC)) and
            np.all(mantissa_q < (4 << MANTISSA_FRAC)),
            "normalized mantissa out of [1,4)")

    log_depth = int(math.log2(per_segment))
    first_shift = MANTISSA_FRAC - log_depth
    second_shift = MANTISSA_FRAC + 1 - log_depth
    first_half = mantissa_q < (2 << MANTISSA_FRAC)
    lut_index = np.empty_like(mantissa_q)
    lut_index[first_half] = (
        mantissa_q[first_half] - (1 << MANTISSA_FRAC)) >> first_shift
    lut_index[~first_half] = per_segment + ((
        mantissa_q[~first_half] - (2 << MANTISSA_FRAC)) >> second_shift)
    require(np.all(lut_index >= 0) and
            np.all(lut_index < 2 * per_segment), "LUT index out of range")
    lut = make_lut(per_segment)
    y_q = lut[lut_index]

    for _ in range(newton_steps):
        y2_q = rshift_rne_unsigned(y_q * y_q, NEWTON_FRAC)
        my2_q = rshift_rne_unsigned(
            mantissa_q * y2_q, MANTISSA_FRAC)
        half_my2_q = rshift_rne_unsigned(my2_q, 1)
        term_q = (3 << (NEWTON_FRAC - 1)) - half_my2_q
        require(np.all(term_q > 0), "Newton term non-positive")
        y_q = rshift_rne_unsigned(y_q * term_q, NEWTON_FRAC)

    # y is Q18.  Convert to Q16 and apply 2**(-even_exponent/2).
    inv_shift = -2 - (even_exponent // 2)
    inv_q = variable_power2(y_q, inv_shift)
    inv_upper = (1 << INV_BITS) - 1
    inv_rails = int(np.count_nonzero(inv_q > inv_upper))
    inv_q = np.clip(inv_q, 0, inv_upper)

    alpha_q = rshift_rne_signed(gamma_q * inv_q, PARAM_FRAC)
    coeff_lower = -(1 << (COEFF_BITS - 1))
    coeff_upper = (1 << (COEFF_BITS - 1)) - 1
    alpha_rails = int(np.count_nonzero(np.logical_or(
        alpha_q < coeff_lower, alpha_q > coeff_upper)))
    alpha_q = np.clip(alpha_q, coeff_lower, coeff_upper)
    alpha_mean_q = rshift_rne_signed(alpha_q * mean_q, MEAN_FRAC)
    offset_q = (beta_q << (COEFF_FRAC - PARAM_FRAC)) - alpha_mean_q
    offset_rails = int(np.count_nonzero(np.logical_or(
        offset_q < coeff_lower, offset_q > coeff_upper)))
    offset_q = np.clip(offset_q, coeff_lower, coeff_upper)
    return {
        "variance_q": x_q,
        "mean_q": mean_q,
        "gamma_q": gamma_q,
        "beta_q": beta_q,
        "even_exponent": even_exponent,
        "mantissa_q": mantissa_q,
        "lut_index": lut_index,
        "invstd_q": inv_q,
        "alpha_q": alpha_q,
        "offset_q": offset_q,
        "rail_counts": {
            "variance": x_rails, "mean": mean_rails,
            "gamma": gamma_rails, "beta": beta_rails,
            "invstd": inv_rails, "alpha": alpha_rails,
            "offset": offset_rails,
        },
        "lut": lut,
    }


def metric(values):
    values = np.asarray(values, dtype=np.float64)
    require(np.all(np.isfinite(values)), "non-finite metric")
    return {
        "mean": float(np.mean(values)),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
        "maximum": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(not output.exists(), "refusing to overwrite M234 output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M234 input identity drift")
    summary = strict_json(PATHS["m233_summary"])
    require(summary["status"] ==
            "PASS_CHECKPOINT_BOUND_S10_DYNAMIC_BN_RANGE_CAPTURE" and
            summary["capture"]["records"] == 240,
            "M233 capture admission drift")

    with np.load(PATHS["m233_npz"], allow_pickle=False) as archive:
        module_keys = sorted({key.rsplit("__", 1)[0] for key in archive.files
                              if key.endswith("__variance")})
        require(len(module_keys) == 24 and len(archive.files) == 264,
                "M233 NPZ topology drift")

        def dynamic(metric_name):
            return np.concatenate([
                archive[key + "__" + metric_name].reshape(-1)
                for key in module_keys]).astype(np.float64)

        variance = dynamic("variance")
        mean = dynamic("mean")
        ref_invstd = dynamic("invstd")
        ref_alpha = dynamic("alpha")
        ref_offset = dynamic("offset")
        input_min = dynamic("input_min")
        input_max = dynamic("input_max")
        gamma = np.concatenate([
            np.tile(archive[key + "__gamma"], (10, 1)).reshape(-1)
            for key in module_keys]).astype(np.float64)
        beta = np.concatenate([
            np.tile(archive[key + "__beta"], (10, 1)).reshape(-1)
            for key in module_keys]).astype(np.float64)

    require(variance.size == 220_800 and
            all(array.size == variance.size for array in (
                mean, ref_invstd, ref_alpha, ref_offset,
                input_min, input_max, gamma, beta)),
            "M234 flattened population drift")
    require(float(np.max(np.abs(
        ref_invstd - 1.0 / np.sqrt(variance + EPSILON)))) < 1.0e-6,
        "M233 invstd identity drift")
    require(float(np.max(np.abs(ref_alpha - gamma * ref_invstd))) < 1.0e-6,
            "M233 alpha identity drift")
    require(float(np.max(np.abs(
        ref_offset - (beta - ref_alpha * mean)))) < 1.0e-6,
        "M233 offset identity drift")

    candidates = []
    selected = None
    for per_segment in (8, 16, 32, 64):
        for newton_steps in (0, 1, 2):
            model = integer_model(
                variance, mean, gamma, beta, per_segment, newton_steps)
            invstd = model["invstd_q"] / float(1 << INV_FRAC)
            alpha = model["alpha_q"] / float(1 << COEFF_FRAC)
            offset = model["offset_q"] / float(1 << COEFF_FRAC)
            inv_error = np.abs(invstd - ref_invstd)
            alpha_error = np.abs(alpha - ref_alpha)
            offset_error = np.abs(offset - ref_offset)
            delta_alpha = alpha - ref_alpha
            delta_offset = offset - ref_offset
            # The difference is affine in x, so its maximum over each captured
            # channel interval occurs at one of the two observed endpoints.
            output_error_bound = np.maximum(
                np.abs(input_min * delta_alpha + delta_offset),
                np.abs(input_max * delta_alpha + delta_offset))
            row = {
                "lut_entries": 2 * per_segment,
                "per_segment": per_segment,
                "newton_steps": newton_steps,
                "lut_payload_bits": 2 * per_segment * LUT_BITS,
                "invstd_abs_error": metric(inv_error),
                "alpha_abs_error": metric(alpha_error),
                "offset_abs_error": metric(offset_error),
                "captured_interval_affine_output_abs_error_bound": metric(
                    output_error_bound),
                "rail_counts": model["rail_counts"],
            }
            candidates.append(row)
            if per_segment == 32 and newton_steps == 1:
                selected = (row, model, output_error_bound)
    require(selected is not None, "selected M234 candidate missing")
    selected_row, selected_model, selected_error = selected
    require(all(value == 0 for value in selected_row["rail_counts"].values()),
            "selected M234 candidate hits numeric rail")
    require(selected_row[
        "captured_interval_affine_output_abs_error_bound"]["maximum"]
        < 0.0018, "selected M234 output error boundary drift")

    output.mkdir(parents=True)
    lut_path = output / "m234_rsqrt_segmented64_uq1p18.mem"
    lut_path.write_text("".join(
        "{:05x}\n".format(int(value)) for value in selected_model["lut"]),
        encoding="ascii")

    priority = set()
    for values in (variance, mean, ref_invstd, ref_alpha, ref_offset,
                   input_min, input_max, selected_error):
        priority.add(int(np.argmin(values)))
        priority.add(int(np.argmax(values)))
    priority.update(int(value) for value in np.linspace(
        0, variance.size - 1, 768, dtype=np.int64))
    rng = np.random.RandomState(234)
    priority.update(int(value) for value in rng.choice(
        variance.size, size=512, replace=False))
    selected_indices = sorted(priority)[:1024]
    require(len(selected_indices) == 1024,
            "M234 vector selection population drift")
    vector_path = output / "m234_selected_coefficient_vectors.csv"
    fields = [
        "vector_id", "flat_source_index", "variance_plus_epsilon_q22_uq6p16",
        "mean_q18_sq3p14", "gamma_q16_sq1p14", "beta_q16_sq1p14",
        "even_exponent", "mantissa_q18_uq2p16", "lut_index",
        "expected_invstd_q20_uq4p16", "expected_alpha_q20_sq3p16",
        "expected_offset_q20_sq3p16", "reference_invstd_float32",
        "reference_alpha_float32", "reference_offset_float32",
        "captured_interval_output_error_bound",
    ]
    with vector_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for vector_id, index in enumerate(selected_indices):
            writer.writerow({
                "vector_id": vector_id,
                "flat_source_index": index,
                "variance_plus_epsilon_q22_uq6p16": int(
                    selected_model["variance_q"][index]),
                "mean_q18_sq3p14": int(selected_model["mean_q"][index]),
                "gamma_q16_sq1p14": int(selected_model["gamma_q"][index]),
                "beta_q16_sq1p14": int(selected_model["beta_q"][index]),
                "even_exponent": int(
                    selected_model["even_exponent"][index]),
                "mantissa_q18_uq2p16": int(
                    selected_model["mantissa_q"][index]),
                "lut_index": int(selected_model["lut_index"][index]),
                "expected_invstd_q20_uq4p16": int(
                    selected_model["invstd_q"][index]),
                "expected_alpha_q20_sq3p16": int(
                    selected_model["alpha_q"][index]),
                "expected_offset_q20_sq3p16": int(
                    selected_model["offset_q"][index]),
                "reference_invstd_float32": float(ref_invstd[index]),
                "reference_alpha_float32": float(ref_alpha[index]),
                "reference_offset_float32": float(ref_offset[index]),
                "captured_interval_output_error_bound": float(
                    selected_error[index]),
            })

    result = {
        "schema": "m234_h67_dynamic_bn_lut_newton_coefficient_dse_v1",
        "status": "PASS_CHECKPOINT_BOUND_INTEGER_COEFFICIENT_DSE",
        "scope": "M233 H67 ep35 ten-sample FFN BN1+BN2 coefficient population; mean/variance are inputs to the candidate engine",
        "population": {
            "modules": 24,
            "samples": 10,
            "coefficient_pairs": int(variance.size),
            "unique_channels_per_frame": 22080,
        },
        "selected_format": {
            "variance_plus_epsilon": "UQ6.16/22b",
            "mean": "signed Q3.14/18b",
            "gamma_beta": "signed Q1.14/16b",
            "normalized_mantissa": "UQ2.16/18b",
            "segmented_lut": "32 bins over [1,2) plus 32 bins over [2,4), midpoint UQ1.18/19b",
            "newton": "one y*(1.5-0.5*m*y*y) step with RNE after each multiply at Q18",
            "invstd": "UQ4.16/20b",
            "alpha_offset": "signed Q3.16/20b",
            "rounding": "round-to-nearest ties-to-even",
            "saturation": "explicit at every declared external format",
        },
        "selected_candidate": selected_row,
        "candidate_sweep": candidates,
        "hardware_mapping": {
            "lut_entries": 64,
            "lut_payload_bits": 64 * LUT_BITS,
            "shared_multiplier_schedule": [
                "newton_y_squared", "newton_m_times_y_squared",
                "newton_y_times_term", "gamma_times_invstd",
                "alpha_times_mean"],
            "target_output_interval_cycles": 16,
            "target_first_result_latency_cycles": 16,
            "m232_rate_matched_if_rtl_meets_target": True,
            "moment_sum_sumsq_to_mean_variance_divider_included": False,
        },
        "numeric_qualification": {
            "captured_interval_bound": "For each captured sample/channel the coefficient-only affine error is maximized at its observed input minimum or maximum; this is not a full element replay or threshold-event proof.",
            "float32_bn_affine_baseline_max_error": summary["capture"][
                "maximum_float32_affine_reconstruction_error"],
            "atlif_threshold_event_equivalence": False,
            "bn2_residual_end_to_end_equivalence": False,
        },
        "identity": {
            "inputs_sha256": observed,
            "analyzer_start_sha256": script_start,
            "lut_sha256": sha256(lut_path),
            "vectors_sha256": sha256(vector_path),
            "docs359_sha256_unchanged": EXPECTED["docs359"],
        },
        "admission": {
            "checkpoint_bound_integer_dse": True,
            "selected_format_no_rails_s10": True,
            "selected_coefficient_error_screen": True,
            "rtl": False,
            "vcs": False,
            "synopsys_dc": False,
            "moment_finalizer": False,
            "atlif_event_equivalence": False,
            "valid825_accuracy": False,
            "cycle_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "paper_safe_statement": "Across 220,800 checkpoint-bound H67 FFN BN coefficient pairs, a 64-entry segmented reciprocal-square-root LUT followed by one fixed-point Newton step has zero format rails and a maximum 0.0018-bounded coefficient-only affine deviation over each captured input interval. RTL latency, moment finalization and event equivalence remain unproved.",
    }
    result_path = output / "m234_h67_dynamic_bn_lut_newton_coefficient_dse_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    readme_path = output / "README.md"
    readme_path.write_text(
        "# M234 H67 dynamic-BN LUT+Newton coefficient DSE\n\n"
        "M233 provides 220,800 checkpoint-bound FFN BN coefficient pairs. The "
        "selected integer path uses a hardware-addressable segmented 64-entry "
        "UQ1.18 rsqrt LUT, one RNE Newton step, and 20-bit Q16 invstd/alpha/offset "
        "outputs. It has zero rails on the captured population.\n\n"
        "The maximum coefficient-only affine deviation over every captured "
        "per-channel input interval is below 0.0018. This uses an exact endpoint "
        "bound for the approximate-minus-reference affine function, but it does "
        "not prove ATLIF threshold/event equivalence or BN2 residual accuracy.\n\n"
        "The candidate schedules five multiplies onto one scalar multiplier and "
        "targets first-result latency and output interval of 16 cycles, matching "
        "the corrected M232 service target. Those timing values require RTL/VCS/DC. "
        "The sum/sumsq-to-mean/variance divider remains outside this module.\n",
        encoding="utf-8")
    require(sha256(script_path) == script_start, "M234 analyzer changed during run")
    manifest_path = output / "manifest.sha256"
    entries = [readme_path, result_path, lut_path, vector_path]
    manifest_path.write_text("".join(
        "{}  {}\n".format(sha256(path), path.name) for path in entries),
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "pairs": int(variance.size),
        "selected_lut_entries": 64,
        "selected_newton_steps": 1,
        "maximum_output_bound": selected_row[
            "captured_interval_affine_output_abs_error_bound"]["maximum"],
        "rail_counts": selected_row["rail_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
