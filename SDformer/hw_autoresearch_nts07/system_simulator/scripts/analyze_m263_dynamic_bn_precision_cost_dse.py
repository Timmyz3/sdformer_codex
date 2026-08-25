#!/usr/bin/env python3
"""Sweep dynamic-BN coefficient precision after the M260 downstream failure."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
INCOMING = HW / "system_handoff/incoming/m233_h67_ffn_dynamic_bn_ranges_s10_r1_20260825"
PATHS = {
    "m233_npz": INCOMING / "m233_h67_ffn_dynamic_bn_ranges_s10.npz",
    "m233_manifest": INCOMING / "manifest.sha256",
    "m234_model": HW / "system_simulator/scripts/analyze_m234_h67_dynamic_bn_lut_newton_coefficients.py",
    "m235_dc": HW / "dc_handoff/runs/m235r2_synthesis_safe_logic_only_dc_3p000ns_r1_20260825/evidence_manifest.sha256",
    "m260_negative_seal": HW / "results/m260_m235_ffn_bn_paired_s10_r1_20260825/LOCAL_SHA256SUMS",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "m233_npz": "d08aa6f49ec246a89008138c421a1bd6bf9274c7995dee48f245777237e9f7a3",
    "m233_manifest": "4f4914aa5c167751ba7d4db4f00b5eeda4dced660ce668144f3613b2c4cd42ad",
    "m234_model": "8ec3b3ca594962c5f7a5a050df030a4a1dddccc768d791975148a2d895985430",
    "m235_dc": "8452150d71f6c261be7887b5d0237219b254d79eb444a83546a6084de52cb94b",
    "m260_negative_seal": "9852d3216445e0c9f4ef902706081806d3867cd5951f7842382b9f9cb201b1fe",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

VAR_FRAC = 16
VAR_BITS = 22
MEAN_FRAC = 14
MEAN_BITS = 18
PARAM_FRAC = 14
PARAM_BITS = 16
MANTISSA_FRAC = 16
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
    sign = np.where(values < 0, -1, 1).astype(np.int64)
    return sign * rshift_rne_unsigned(np.abs(values), shift)


def variable_power2(values, shifts):
    values = np.asarray(values, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64)
    result = np.empty_like(values)
    for shift in np.unique(shifts):
        mask = shifts == shift
        if shift >= 0:
            result[mask] = values[mask] << int(shift)
        else:
            result[mask] = rshift_rne_unsigned(values[mask], int(-shift))
    return result


def make_lut(per_segment, work_frac):
    require(per_segment > 0 and per_segment & (per_segment - 1) == 0,
            "LUT segment depth must be a power of two")
    first = 1.0 + (np.arange(per_segment) + 0.5) / per_segment
    second = 2.0 + (np.arange(per_segment) + 0.5) * 2.0 / per_segment
    return np.rint(
        (1.0 / np.sqrt(np.concatenate((first, second)))) *
        (1 << work_frac)).astype(np.int64)


def integer_model(variance, mean, gamma, beta, per_segment,
                  newton_steps, precision):
    variance_frac = precision["variance_frac"]
    mean_frac = precision["mean_frac"]
    param_frac = precision["param_frac"]
    coefficient_frac = precision["coefficient_frac"]
    work_frac = precision["work_frac"]
    variance_bits = variance_frac + 6
    mean_bits = mean_frac + 4
    param_bits = param_frac + 2
    coefficient_bits = coefficient_frac + 4
    invstd_bits = coefficient_frac + 4
    x_q, x_rails = quantize(
        variance + EPSILON, variance_bits, variance_frac, False)
    mean_q, mean_rails = quantize(mean, mean_bits, mean_frac, True)
    gamma_q, gamma_rails = quantize(gamma, param_bits, param_frac, True)
    beta_q, beta_rails = quantize(beta, param_bits, param_frac, True)
    require(np.all(x_q > 0), "quantized variance reached zero")

    msb = np.floor(np.log2(x_q.astype(np.float64))).astype(np.int64)
    exponent = msb - variance_frac
    even_exponent = exponent - (exponent & 1)
    mantissa_q = variable_power2(
        x_q, MANTISSA_FRAC - variance_frac - even_exponent)
    require(np.all(mantissa_q >= (1 << MANTISSA_FRAC)) and
            np.all(mantissa_q < (4 << MANTISSA_FRAC)),
            "normalized mantissa out of range")

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
    lut = make_lut(per_segment, work_frac)
    y_q = lut[lut_index]

    for _ in range(newton_steps):
        y2_q = rshift_rne_unsigned(y_q * y_q, work_frac)
        my2_q = rshift_rne_unsigned(mantissa_q * y2_q, MANTISSA_FRAC)
        half_my2_q = rshift_rne_unsigned(my2_q, 1)
        term_q = (3 << (work_frac - 1)) - half_my2_q
        require(np.all(term_q > 0), "Newton term non-positive")
        y_q = rshift_rne_unsigned(y_q * term_q, work_frac)

    inv_shift = coefficient_frac - work_frac - (even_exponent // 2)
    invstd_q = variable_power2(y_q, inv_shift)
    invstd_upper = (1 << invstd_bits) - 1
    invstd_rails = int(np.count_nonzero(invstd_q > invstd_upper))
    invstd_q = np.clip(invstd_q, 0, invstd_upper)
    alpha_q = rshift_rne_signed(gamma_q * invstd_q, param_frac)
    coefficient_lower = -(1 << (coefficient_bits - 1))
    coefficient_upper = (1 << (coefficient_bits - 1)) - 1
    alpha_rails = int(np.count_nonzero(np.logical_or(
        alpha_q < coefficient_lower, alpha_q > coefficient_upper)))
    alpha_q = np.clip(alpha_q, coefficient_lower, coefficient_upper)
    alpha_mean_q = rshift_rne_signed(alpha_q * mean_q, mean_frac)
    offset_q = (beta_q << (coefficient_frac - param_frac)) - alpha_mean_q
    offset_rails = int(np.count_nonzero(np.logical_or(
        offset_q < coefficient_lower, offset_q > coefficient_upper)))
    offset_q = np.clip(offset_q, coefficient_lower, coefficient_upper)
    return {
        "invstd_q": invstd_q,
        "alpha_q": alpha_q,
        "offset_q": offset_q,
        "rail_counts": {
            "variance": x_rails,
            "mean": mean_rails,
            "gamma": gamma_rails,
            "beta": beta_rails,
            "invstd": invstd_rails,
            "alpha": alpha_rails,
            "offset": offset_rails,
        },
        "lut": lut,
        "coefficient_bits": coefficient_bits,
        "invstd_bits": invstd_bits,
        "variance_bits": variance_bits,
        "mean_bits": mean_bits,
        "param_bits": param_bits,
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
    require(not output.exists(), "refusing to overwrite M263 output")
    observed = {name: sha256(path) for name, path in PATHS.items()}
    require(observed == EXPECTED, "M263 frozen identity drift")

    with np.load(PATHS["m233_npz"], allow_pickle=False) as archive:
        module_keys = sorted({key.rsplit("__", 1)[0] for key in archive.files
                              if key.endswith("__variance")})
        require(len(module_keys) == 24, "M233 module population drift")

        def dynamic(metric_name):
            return np.concatenate([
                archive[key + "__" + metric_name].reshape(-1)
                for key in module_keys]).astype(np.float64)

        variance = dynamic("variance")
        mean = dynamic("mean")
        reference_invstd = dynamic("invstd")
        reference_alpha = dynamic("alpha")
        reference_offset = dynamic("offset")
        input_min = dynamic("input_min")
        input_max = dynamic("input_max")
        gamma = np.concatenate([
            np.tile(archive[key + "__gamma"], (10, 1)).reshape(-1)
            for key in module_keys]).astype(np.float64)
        beta = np.concatenate([
            np.tile(archive[key + "__beta"], (10, 1)).reshape(-1)
            for key in module_keys]).astype(np.float64)
    require(variance.size == 220800 and all(
        array.size == variance.size for array in (
            mean, reference_invstd, reference_alpha, reference_offset,
            input_min, input_max, gamma, beta)),
        "M263 vector population drift")

    precision_tiers = (
        {
            "name": "legacy_q16",
            "variance_frac": 16,
            "mean_frac": 14,
            "param_frac": 14,
            "coefficient_frac": 16,
            "work_frac": 18,
        },
        {
            "name": "coefficient_only_q18",
            "variance_frac": 16,
            "mean_frac": 14,
            "param_frac": 14,
            "coefficient_frac": 18,
            "work_frac": 18,
        },
        {
            "name": "affine_q18",
            "variance_frac": 16,
            "mean_frac": 16,
            "param_frac": 16,
            "coefficient_frac": 18,
            "work_frac": 18,
        },
        {
            "name": "balanced_q18",
            "variance_frac": 18,
            "mean_frac": 16,
            "param_frac": 16,
            "coefficient_frac": 18,
            "work_frac": 18,
        },
        {
            "name": "balanced_q20",
            "variance_frac": 20,
            "mean_frac": 18,
            "param_frac": 18,
            "coefficient_frac": 20,
            "work_frac": 20,
        },
    )
    candidates = []
    for precision in precision_tiers:
        for per_segment in (16, 32):
            for newton_steps in (1, 2):
                model = integer_model(
                    variance, mean, gamma, beta,
                    per_segment, newton_steps, precision)
                coefficient_frac = precision["coefficient_frac"]
                scale = float(1 << coefficient_frac)
                invstd = model["invstd_q"] / scale
                alpha = model["alpha_q"] / scale
                offset = model["offset_q"] / scale
                invstd_error = np.abs(invstd - reference_invstd)
                alpha_error = np.abs(alpha - reference_alpha)
                offset_error = np.abs(offset - reference_offset)
                delta_alpha = alpha - reference_alpha
                delta_offset = offset - reference_offset
                output_error = np.maximum(
                    np.abs(input_min * delta_alpha + delta_offset),
                    np.abs(input_max * delta_alpha + delta_offset))
                candidates.append({
                    "precision_tier": precision["name"],
                    "variance_frac": precision["variance_frac"],
                    "variance_bits": model["variance_bits"],
                    "mean_frac": precision["mean_frac"],
                    "mean_bits": model["mean_bits"],
                    "param_frac": precision["param_frac"],
                    "param_bits": model["param_bits"],
                    "coefficient_frac": coefficient_frac,
                    "coefficient_bits": model["coefficient_bits"],
                    "invstd_bits": model["invstd_bits"],
                    "work_frac": precision["work_frac"],
                    "per_segment": per_segment,
                    "lut_entries": 2 * per_segment,
                    "lut_payload_bits": int(
                        2 * per_segment * (precision["work_frac"] + 1)),
                    "newton_steps": newton_steps,
                    "scalar_multiplier_operations_per_pair":
                        2 + 3 * newton_steps,
                    "invstd_absolute_error": metric(invstd_error),
                    "alpha_absolute_error": metric(alpha_error),
                    "offset_absolute_error": metric(offset_error),
                    "captured_interval_output_absolute_error_bound": metric(
                        output_error),
                    "rail_counts": model["rail_counts"],
                })
    require(all(all(value == 0 for value in row["rail_counts"].values())
                for row in candidates), "candidate numeric rail")

    qualified = [row for row in candidates
                 if row["captured_interval_output_absolute_error_bound"]
                 ["maximum"] < 2.5e-4]
    require(qualified, "no candidate meets M263 local error gate")
    selected = min(qualified, key=lambda row: (
        row["scalar_multiplier_operations_per_pair"],
        max(row["coefficient_bits"], row["variance_bits"],
            row["mean_bits"], row["param_bits"]),
        row["coefficient_bits"], row["lut_payload_bits"]))

    result = {
        "schema": "m263_dynamic_bn_precision_cost_dse_v1",
        "status": "PASS_LOCAL_PRECISION_COST_DSE_REQUIRES_NETWORK_GATE",
        "identity": observed,
        "population": {
            "modules": 24,
            "samples": 10,
            "coefficient_pairs": int(variance.size),
        },
        "candidate_sweep": candidates,
        "local_error_gate": {
            "maximum_captured_interval_output_absolute_error": 2.5e-4,
            "reason": "heuristic screening threshold after M260; not a downstream accuracy proof"
        },
        "selected_for_first_network_test": selected,
        "selection_reason": (
            "Fewest scalar multiplier operations, then narrowest coefficient "
            "width and smallest LUT among candidates passing the local bound."
        ),
        "m260_lesson": (
            "Q16 one-Newton passed a 0.0018 local bound but worsened paired "
            "first-ten AEE by 1.730869%; every candidate still requires a "
            "trace-executable downstream event and flow gate."
        ),
        "admission": {
            "local_numeric_dse": True,
            "network_accuracy": False,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    output.mkdir(parents=True)
    result_path = output / "m263_dynamic_bn_precision_cost_dse_r1.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    readme = output / "README.md"
    readme.write_text(
        "# M263 dynamic-BN precision/cost DSE\n\n"
        "This is a frozen 220,800-pair local numeric screen after the M260 "
        "downstream failure.  It is not an accuracy or hardware admission.\n\n"
        "Selected first network candidate: Q{}.{} coefficients, {} LUT "
        "entries, {} Newton step(s), {} scalar multiply operations/pair; "
        "captured-interval maximum bound {:.9g}.\n\n"
        "All candidates must pass paired downstream event/AEE evaluation; no "
        "speedup, PPA, valid825, or headline claim is admitted.\n".format(
            selected["coefficient_bits"],
            selected["coefficient_frac"],
            selected["lut_entries"],
            selected["newton_steps"],
            selected["scalar_multiplier_operations_per_pair"],
            selected["captured_interval_output_absolute_error_bound"][
                "maximum"],
        ),
        encoding="utf-8")
    manifest = output / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(path), path.name)
        for path in (readme, result_path)), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "selected": selected,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
