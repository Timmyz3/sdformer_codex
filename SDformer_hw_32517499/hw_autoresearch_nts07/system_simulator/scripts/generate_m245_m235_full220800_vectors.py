#!/usr/bin/env python3
"""Generate every M233 coefficient pair for the unchanged M235 64+1 RTL."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_m234_h67_dynamic_bn_lut_newton_coefficients import integer_model


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M233 = HW / "system_handoff/incoming/m233_h67_ffn_dynamic_bn_ranges_s10_r1_20260825"
PATHS = {
    "m233_npz": M233 / "m233_h67_ffn_dynamic_bn_ranges_s10.npz",
    "m233_manifest": M233 / "manifest.sha256",
    "m233_summary": M233 / "m233_h67_ffn_dynamic_bn_range_summary_r1.json",
    "m234_integer_model": HW / "system_simulator/scripts/analyze_m234_h67_dynamic_bn_lut_newton_coefficients.py",
    "m240_review_seal": HW / "results/m240_bn_pareto_independent_hammer_r1_20260825/SHA256SUMS",
    "m235_rtl": HW / "rtl_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine.sv",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "m233_npz": "d08aa6f49ec246a89008138c421a1bd6bf9274c7995dee48f245777237e9f7a3",
    "m233_manifest": "4f4914aa5c167751ba7d4db4f00b5eeda4dced660ce668144f3613b2c4cd42ad",
    "m233_summary": "e4ff7b29617484e2d1a71d5faa4964d176bb46486e129a984f28fa0095a43f9f",
    "m234_integer_model": "8ec3b3ca594962c5f7a5a050df030a4a1dddccc768d791975148a2d895985430",
    "m240_review_seal": "f9baa9402116b487c6a81be80d6f5d85db2250a0d684712cf1048e7d161d5f09",
    "m235_rtl": "ec0bf05540433ecfc436eac63b41a4cecf4cc53b46533f2fd4f44c7eb70bd611",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
                archive[key + "__" + field].reshape(-1)
                for key in module_keys
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
    candidate = integer_model(
        variance, mean, gamma, beta, per_segment=32, newton_steps=1)
    require(all(value == 0 for value in candidate["rail_counts"].values()),
            "format rail")
    invstd = candidate["invstd_q"] / 65536.0
    alpha = candidate["alpha_q"] / 65536.0
    offset = candidate["offset_q"] / 65536.0
    delta_alpha = alpha - ref_alpha
    delta_offset = offset - ref_offset
    endpoint_error = np.maximum(
        np.abs(input_min * delta_alpha + delta_offset),
        np.abs(input_max * delta_alpha + delta_offset))
    require(float(np.max(endpoint_error)) < 0.0018,
            "M235 numeric bound drift")

    output.mkdir(parents=True)
    vector_path = output / "m245_m235_full220800_vectors.csv"
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
            candidate["variance_q"], candidate["mean_q"],
            candidate["gamma_q"], candidate["beta_q"],
            candidate["even_exponent"], candidate["mantissa_q"],
            candidate["lut_index"], candidate["invstd_q"],
            candidate["alpha_q"], candidate["offset_q"],
        ]
        for index, values in enumerate(zip(*arrays)):
            writer.writerow((index, index) + tuple(int(value) for value in values))

    missing_six = [175162, 175604, 176110, 182167, 190728, 219956]
    require(all(index < variance.size for index in missing_six),
            "tail index range")
    summary = {
        "schema": "m245_m235_full220800_vector_set_v1",
        "status": "PASS_M235_FULL220800_CHECKPOINT_BOUND_INTEGER_VECTORS",
        "population": {
            "modules": 24,
            "samples": 10,
            "coefficient_pairs": 220800,
            "source_index_min": 0,
            "source_index_max": 220799,
            "previously_missing_six_extrema_included": missing_six,
        },
        "candidate": {
            "lut_entries": 64,
            "lut_payload_bits": 1216,
            "newton_steps": 1,
            "single_shared_multiplier_operations_per_pair": 5,
            "rail_counts": candidate["rail_counts"],
            "invstd_abs_error": metric(np.abs(invstd - ref_invstd)),
            "alpha_abs_error": metric(np.abs(alpha - ref_alpha)),
            "offset_abs_error": metric(np.abs(offset - ref_offset)),
            "captured_interval_affine_output_abs_error_bound":
                metric(endpoint_error),
        },
        "identity": {
            "inputs_sha256": observed,
            "generator_start_sha256": script_start,
            "vectors_sha256": sha256(vector_path),
            "docs359_sha256_unchanged": EXPECTED["docs359"],
        },
        "admission": {
            "checkpoint_bound_full_integer_vectors": True,
            "unchanged_m235_rtl": True,
            "vcs": False,
            "moment_finalizer": False,
            "event_equivalence": False,
            "valid825": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    summary_path = output / "m245_m235_full220800_vector_summary_r1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    readme_path = output / "README.md"
    readme_path.write_text(
        "# M245 unchanged-M235 full checkpoint vector set\n\n"
        "All 220,800 M233 coefficient pairs are enumerated for the M235 64-entry "
        "LUT plus one-Newton integer recurrence. This is a vector-generation "
        "milestone; VCS and downstream event/accuracy admission remain false.\n",
        encoding="utf-8")
    require(sha256(script_path) == script_start,
            "generator changed during run")
    manifest_path = output / "manifest.sha256"
    manifest_path.write_text("".join(
        "{}  {}\n".format(sha256(path), path.name)
        for path in (readme_path, summary_path, vector_path)),
        encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "vectors": 220800,
        "maximum_endpoint_bound": summary["candidate"]
            ["captured_interval_affine_output_abs_error_bound"]["maximum"],
        "vectors_sha256": summary["identity"]["vectors_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
