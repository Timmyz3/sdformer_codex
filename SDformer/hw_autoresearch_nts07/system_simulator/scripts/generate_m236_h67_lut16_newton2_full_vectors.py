#!/usr/bin/env python3
"""Generate all M233 H67 pairs for the M236 16-entry/two-Newton RTL."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M233 = HW / "system_handoff/incoming/m233_h67_ffn_dynamic_bn_ranges_s10_r1_20260825"
PATHS = {
    "m233_npz": M233 / "m233_h67_ffn_dynamic_bn_ranges_s10.npz",
    "m233_manifest": M233 / "manifest.sha256",
    "m233_summary": M233 / "m233_h67_ffn_dynamic_bn_range_summary_r1.json",
    "m234_review_seal": HW / "results/m234_independent_hammer_review_r1_20260825/SHA256SUMS",
    "m235r2_vcs_seal": HW / "results/m235r2_synthesis_safe_directed_vcs_r1_exact_20260825/SHA256SUMS",
    "m235_interface_source": HW / "rtl_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine.sv",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "m233_npz": "d08aa6f49ec246a89008138c421a1bd6bf9274c7995dee48f245777237e9f7a3",
    "m233_manifest": "4f4914aa5c167751ba7d4db4f00b5eeda4dced660ce668144f3613b2c4cd42ad",
    "m233_summary": "e4ff7b29617484e2d1a71d5faa4964d176bb46486e129a984f28fa0095a43f9f",
    "m234_review_seal": "18a57e92dde575c680646ae020fb0d4ae5f8a0d6a4bac7cadf07eaf13dd32404",
    "m235r2_vcs_seal": "b813ac5f8fcb5b3273f580db9a70b230df72d18a9c646964dd0b8bee7927fff5",
    "m235_interface_source": "ec0bf05540433ecfc436eac63b41a4cecf4cc53b46533f2fd4f44c7eb70bd611",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
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


def quantize(values, width, frac, signed):
    raw = np.rint(np.asarray(values, dtype=np.float64) * (1 << frac)).astype(np.int64)
    lower = -(1 << (width - 1)) if signed else 0
    upper = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
    rails = int(np.count_nonzero((raw < lower) | (raw > upper)))
    return np.clip(raw, lower, upper), rails


def rshift_rne(values, shift):
    values = np.asarray(values, dtype=np.int64)
    require(shift > 0, "RNE shift must be positive")
    negative = values < 0
    magnitude = np.abs(values)
    quotient = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    rounded = quotient + increment.astype(np.int64)
    return np.where(negative, -rounded, rounded)


def variable_scale_unsigned(values, shifts):
    values = np.asarray(values, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64)
    result = np.empty_like(values)
    for shift in np.unique(shifts):
        mask = shifts == shift
        if shift >= 0:
            result[mask] = values[mask] << int(shift)
        else:
            result[mask] = rshift_rne(values[mask], int(-shift))
    return result


def lut16():
    points = np.concatenate((
        1.0 + (np.arange(8) + 0.5) / 8.0,
        2.0 + (np.arange(8) + 0.5) * 2.0 / 8.0,
    ))
    payload, rails = quantize(1.0 / np.sqrt(points), 19, 18, False)
    require(rails == 0, "LUT rail")
    return payload


def model(variance, mean, gamma, beta):
    variance_q, variance_rails = quantize(variance + EPSILON, 22, 16, False)
    mean_q, mean_rails = quantize(mean, 18, 14, True)
    gamma_q, gamma_rails = quantize(gamma, 16, 14, True)
    beta_q, beta_rails = quantize(beta, 16, 14, True)
    require(np.all(variance_q > 0), "variance quantizes to zero")
    msb = np.floor(np.log2(variance_q.astype(np.float64))).astype(np.int64)
    exponent = msb - 16
    even_exponent = exponent - (exponent & 1)
    mantissa_q = variable_scale_unsigned(variance_q, -even_exponent)
    require(np.all((mantissa_q >= 65536) & (mantissa_q < 262144)),
            "normalized mantissa range")
    first_half = mantissa_q < 131072
    lut_index = np.empty_like(mantissa_q)
    lut_index[first_half] = (mantissa_q[first_half] - 65536) >> 13
    lut_index[~first_half] = 8 + ((mantissa_q[~first_half] - 131072) >> 14)
    require(np.all((lut_index >= 0) & (lut_index < 16)), "LUT index range")
    y_q = lut16()[lut_index]
    for _ in range(2):
        y2_q = rshift_rne(y_q * y_q, 18)
        my2_q = rshift_rne(mantissa_q * y2_q, 16)
        half_my2_q = rshift_rne(my2_q, 1)
        term_q = (3 << 17) - half_my2_q
        require(np.all(term_q > 0), "Newton term")
        y_q = rshift_rne(y_q * term_q, 18)
    invstd_q = variable_scale_unsigned(y_q, -2 - even_exponent // 2)
    invstd_rails = int(np.count_nonzero(invstd_q > 1048575))
    invstd_q = np.clip(invstd_q, 0, 1048575)
    alpha_q = rshift_rne(gamma_q * invstd_q, 14)
    alpha_rails = int(np.count_nonzero((alpha_q < -524288) | (alpha_q > 524287)))
    alpha_q = np.clip(alpha_q, -524288, 524287)
    alpha_mean_q = rshift_rne(alpha_q * mean_q, 14)
    offset_q = (beta_q << 2) - alpha_mean_q
    offset_rails = int(np.count_nonzero((offset_q < -524288) | (offset_q > 524287)))
    offset_q = np.clip(offset_q, -524288, 524287)
    return {
        "variance_q": variance_q,
        "mean_q": mean_q,
        "gamma_q": gamma_q,
        "beta_q": beta_q,
        "even_exponent": even_exponent,
        "mantissa_q": mantissa_q,
        "lut_index": lut_index,
        "invstd_q": invstd_q,
        "alpha_q": alpha_q,
        "offset_q": offset_q,
        "lut": lut16(),
        "rails": {
            "variance": variance_rails,
            "mean": mean_rails,
            "gamma": gamma_rails,
            "beta": beta_rails,
            "invstd": invstd_rails,
            "alpha": alpha_rails,
            "offset": offset_rails,
        },
    }


def metric(values):
    values = np.asarray(values, dtype=np.float64)
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
    require(not output.exists(), "refusing to overwrite output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    observed = {name: sha256(path) for name, path in PATHS.items()}
    require(observed == EXPECTED, "strict input SHA drift")

    with np.load(PATHS["m233_npz"], allow_pickle=False) as archive:
        module_keys = sorted({
            key.rsplit("__", 1)[0] for key in archive.files
            if key.endswith("__variance")
        })
        require(len(module_keys) == 24 and len(archive.files) == 264,
                "M233 topology drift")

        def dynamic(field):
            return np.concatenate([
                archive[key + "__" + field].reshape(-1) for key in module_keys
            ]).astype(np.float64)

        variance = dynamic("variance")
        mean = dynamic("mean")
        ref_invstd = dynamic("invstd")
        ref_alpha = dynamic("alpha")
        ref_offset = dynamic("offset")
        input_min = dynamic("input_min")
        input_max = dynamic("input_max")
        gamma = np.concatenate([
            np.tile(archive[key + "__gamma"], (10, 1)).reshape(-1)
            for key in module_keys
        ]).astype(np.float64)
        beta = np.concatenate([
            np.tile(archive[key + "__beta"], (10, 1)).reshape(-1)
            for key in module_keys
        ]).astype(np.float64)
    require(variance.size == 220800, "population count drift")
    candidate = model(variance, mean, gamma, beta)
    require(all(value == 0 for value in candidate["rails"].values()), "format rail")
    invstd = candidate["invstd_q"] / 65536.0
    alpha = candidate["alpha_q"] / 65536.0
    offset = candidate["offset_q"] / 65536.0
    delta_alpha = alpha - ref_alpha
    delta_offset = offset - ref_offset
    endpoint_error = np.maximum(
        np.abs(input_min * delta_alpha + delta_offset),
        np.abs(input_max * delta_alpha + delta_offset),
    )
    require(float(np.max(endpoint_error)) < 0.0016, "M236 numeric bound drift")

    output.mkdir(parents=True)
    vector_path = output / "m236_h67_lut16_newton2_full220800_vectors.csv"
    fields = [
        "vector_id", "flat_source_index",
        "variance_plus_epsilon_q22_uq6p16", "mean_q18_sq3p14",
        "gamma_q16_sq1p14", "beta_q16_sq1p14", "even_exponent",
        "mantissa_q18_uq2p16", "lut_index",
        "expected_invstd_q20_uq4p16", "expected_alpha_q20_sq3p16",
        "expected_offset_q20_sq3p16",
    ]
    with vector_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        arrays = [
            candidate["variance_q"], candidate["mean_q"], candidate["gamma_q"],
            candidate["beta_q"], candidate["even_exponent"],
            candidate["mantissa_q"], candidate["lut_index"],
            candidate["invstd_q"], candidate["alpha_q"], candidate["offset_q"],
        ]
        for index, values in enumerate(zip(*arrays)):
            writer.writerow((index, index) + tuple(int(value) for value in values))

    missing_six = [175162, 175604, 176110, 182167, 190728, 219956]
    require(all(index < variance.size for index in missing_six), "tail index range")
    summary = {
        "schema": "m236_h67_lut16_newton2_full_vector_set_v1",
        "status": "PASS_FULL220800_CHECKPOINT_BOUND_INTEGER_VECTORS",
        "population": {
            "modules": 24,
            "samples": 10,
            "coefficient_pairs": 220800,
            "vector_rows": 220800,
            "source_index_min": 0,
            "source_index_max": 220799,
            "previously_missing_six_extrema_included": missing_six,
        },
        "candidate": {
            "lut_entries": 16,
            "lut_payload_bits": 304,
            "newton_steps": 2,
            "single_shared_multiplier_operations_per_pair": 8,
            "rail_counts": candidate["rails"],
            "lut_address_histogram": {
                str(index): int(np.count_nonzero(candidate["lut_index"] == index))
                for index in range(16)
            },
            "invstd_abs_error": metric(np.abs(invstd - ref_invstd)),
            "alpha_abs_error": metric(np.abs(alpha - ref_alpha)),
            "offset_abs_error": metric(np.abs(offset - ref_offset)),
            "captured_interval_affine_output_abs_error_bound": metric(endpoint_error),
        },
        "identity": {
            "inputs_sha256": observed,
            "generator_start_sha256": script_start,
            "vectors_sha256": sha256(vector_path),
            "docs359_sha256_unchanged": EXPECTED["docs359"],
        },
        "admission": {
            "checkpoint_bound_full_integer_vectors": True,
            "rtl": False,
            "vcs": False,
            "synopsys_dc": False,
            "moment_finalizer": False,
            "event_equivalence": False,
            "valid825": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    summary_path = output / "m236_h67_lut16_newton2_full_vector_summary_r1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    readme_path = output / "README.md"
    readme_path.write_text(
        "# M236 full checkpoint-bound vector set\n\n"
        "This set contains every one of the 220,800 M233 H67 FFN BN coefficient "
        "pairs for the independent 16-entry LUT plus two-Newton candidate. It "
        "therefore includes the six high-index extrema omitted by the M234 "
        "1,024-vector selector. It is an integer coefficient-engine reference, "
        "not a moment-finalizer, event-equivalence, cycle-speedup or system result.\n",
        encoding="utf-8",
    )
    require(sha256(script_path) == script_start, "generator changed during run")
    manifest_path = output / "manifest.sha256"
    manifest_path.write_text("".join(
        f"{sha256(path)}  {path.name}\n"
        for path in (readme_path, summary_path, vector_path)
    ), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "vectors": 220800,
        "maximum_endpoint_bound": summary["candidate"]
            ["captured_interval_affine_output_abs_error_bound"]["maximum"],
        "vectors_sha256": summary["identity"]["vectors_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
