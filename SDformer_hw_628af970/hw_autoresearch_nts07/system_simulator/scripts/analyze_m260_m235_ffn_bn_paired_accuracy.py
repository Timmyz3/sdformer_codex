#!/usr/bin/env python3
"""Seal the paired first-ten M260 downstream accuracy experiment."""

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m260_m235_ffn_bn_paired_s10_r1_20260825"
MILESTONE = RESULT.name
PATHS = {
    "contract": HW / "contracts/m260_m235_ffn_bn_downstream_accuracy_contract_r1_20260825.json",
    "evaluator": HW / "system_handoff/scripts/eval_m260_m235_approx_ffn_bn_DSEC.py",
    "wrapper": HW / "system_handoff/run_m260_m235_ffn_bn_paired_eval_when_gpu_idle_20260825.sh",
    "upstream": ROOT / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
    "m234": HW / "system_simulator/scripts/analyze_m234_h67_dynamic_bn_lut_newton_coefficients.py",
    "m245_seal": HW / "results/m245_m235_full220800_vcs_r1_exact_20260825/SHA256SUMS",
    "m246_seal": HW / "results/m246_m245_full220800_independent_hammer_r1_20260825/SHA256SUMS",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "contract": "feb472dd7b29677dec285790b15826c72d468bf3485a9be5704bacb837fdafd2",
    "evaluator": "5716068204442df606f5bec785ed14e5d3a2628e6152e75c2227efb86c23b37b",
    "wrapper": "64954aaf5373d6e9fc9193060ac0b9da0ddafebc433e58165cdf011fe7b4cfec",
    "upstream": "ba40b42c7395fd703c59a183a19b6a4fd38fa08ed75201008f03fd71b82aaef1",
    "m234": "8ec3b3ca594962c5f7a5a050df030a4a1dddccc768d791975148a2d895985430",
    "m245_seal": "a10da0a8ffe7b30665cb8fb3270603448166f8ac3f6e51d4831765a210b35272",
    "m246_seal": "8a0f07a74d49229019dde0ae7c69ea2fdc1040d4723d82d5ccaefe49790795eb",
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


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + value)),
        )


def validate_remote_manifest(path):
    checked = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        expected, remote_path = line.split("  ", 1)
        marker = "/" + MILESTONE + "/"
        require(marker in remote_path, "remote manifest path escaped result")
        relative = remote_path.split(marker, 1)[1]
        local_path = RESULT / relative
        observed = sha256(local_path)
        require(observed == expected, "remote payload SHA drift: " + relative)
        checked.append({
            "path": relative,
            "sha256": observed,
            "bytes": local_path.stat().st_size,
        })
    require(len(checked) == 11, "remote manifest population drift")
    return checked


def load_frames(mode):
    with (RESULT / mode / "per_frame.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 10, mode + " sample count drift")
    return rows


def metric_pair(reference, candidate, field, lower_is_better):
    lhs = [float(row[field]) for row in reference]
    rhs = [float(row[field]) for row in candidate]
    delta = [b - a for a, b in zip(lhs, rhs)]
    lhs_mean = sum(lhs) / len(lhs)
    rhs_mean = sum(rhs) / len(rhs)
    return {
        "reference_mean": lhs_mean,
        "m235_approx_mean": rhs_mean,
        "mean_delta": sum(delta) / len(delta),
        "relative_delta_percent": (rhs_mean / lhs_mean - 1.0) * 100.0,
        "maximum_absolute_paired_delta": max(abs(value) for value in delta),
        "candidate_wins": sum(
            (b < a) if lower_is_better else (b > a)
            for a, b in zip(lhs, rhs)),
        "candidate_losses": sum(
            (b > a) if lower_is_better else (b < a)
            for a, b in zip(lhs, rhs)),
        "candidate_ties": sum(a == b for a, b in zip(lhs, rhs)),
    }


def main():
    observed = {name: sha256(path) for name, path in PATHS.items()}
    require(observed == EXPECTED, "M260 frozen source identity drift")
    remote_manifest = RESULT / "SHA256SUMS"
    remote_manifest_sha = sha256(remote_manifest)
    require(remote_manifest_sha ==
            "b6fc711cc890f11fc5129bd54e872591aa20e310d3080af88d69c208e6baf7ad",
            "M260 remote manifest seal drift")
    payloads = validate_remote_manifest(remote_manifest)

    reference = load_frames("reference")
    candidate = load_frames("m235_approx")
    reference_ids = [(row["sequence"], row["file"]) for row in reference]
    candidate_ids = [(row["sequence"], row["file"]) for row in candidate]
    require(reference_ids == candidate_ids, "paired sample identity drift")
    require(len(set(reference_ids)) == 10, "duplicate paired sample identity")

    reference_receipt = strict_json(
        RESULT / "reference/m260_runtime_receipt.json")
    candidate_receipt = strict_json(
        RESULT / "m235_approx/m260_runtime_receipt.json")
    require(reference_receipt["status"] ==
            "PASS_M260_REFERENCE_EVALUATION_RUNTIME" and
            reference_receipt["samples"] == 10 and
            reference_receipt["hook_calls"] == 0,
            "reference runtime receipt drift")
    require(candidate_receipt["status"] ==
            "PASS_M260_M235_APPROX_EVALUATION_RUNTIME" and
            candidate_receipt["samples"] == 10 and
            candidate_receipt["hook_calls"] == 240 and
            candidate_receipt["coefficient_pairs"] == 220800 and
            candidate_receipt["output_values"] == 4377600000 and
            not candidate_receipt["moment_finalizer_modeled"] and
            all(value == 0 for value in
                candidate_receipt["rail_counts"].values()),
            "candidate runtime receipt drift")

    expected_runtime_identity = {
        "upstream_evaluator": EXPECTED["upstream"],
        "config": "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
        "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
        "m234_integer_model": EXPECTED["m234"],
        "m245_vcs_seal": EXPECTED["m245_seal"],
        "m246_review_seal": EXPECTED["m246_seal"],
        "docs359": EXPECTED["docs359"],
    }
    require(candidate_receipt["identity"] == expected_runtime_identity,
            "candidate runtime identity drift")

    metrics = {
        "AEE": metric_pair(reference, candidate, "AEE", True),
        "AAE": metric_pair(reference, candidate, "AAE", True),
        "AAE_Benchmark": metric_pair(
            reference, candidate, "AAE_Benchmark", True),
        "DSEC_Fl": metric_pair(reference, candidate, "DSEC_Fl", True),
        "spikes": metric_pair(reference, candidate, "spikes", True),
    }
    require(metrics["AEE"]["relative_delta_percent"] > 1.0 and
            metrics["AEE"]["candidate_losses"] == 7 and
            metrics["DSEC_Fl"]["relative_delta_percent"] > 5.0,
            "M260 negative downstream accuracy observation drift")

    ref_profile = strict_json(RESULT / "reference/spike_profile.json")
    can_profile = strict_json(RESULT / "m235_approx/spike_profile.json")
    require(ref_profile["samples"] == can_profile["samples"] == 10,
            "spike profile sample population drift")
    require(ref_profile["total_spikes"] == int(sum(
        float(row["spikes"]) for row in reference)) and
            can_profile["total_spikes"] == int(sum(
                float(row["spikes"]) for row in candidate)),
            "per-frame/profile spike total drift")

    layer_deltas = []
    for name in sorted(set(ref_profile["layer_firing_rates"]) &
                       set(can_profile["layer_firing_rates"])):
        before = ref_profile["layer_firing_rates"][name]
        after = can_profile["layer_firing_rates"][name]
        delta = int(after["spikes"]) - int(before["spikes"])
        if delta:
            layer_deltas.append({
                "module": name,
                "reference_spikes": int(before["spikes"]),
                "m235_approx_spikes": int(after["spikes"]),
                "delta_spikes": delta,
            })
    layer_deltas.sort(key=lambda row: abs(row["delta_spikes"]), reverse=True)

    output = {
        "schema": "m260_m235_ffn_bn_paired_accuracy_analysis_v1",
        "status": "PASS_PAIRED_S10_NEGATIVE_DOWNSTREAM_ACCURACY_GATE",
        "identity": observed,
        "remote_payload_manifest_sha256": remote_manifest_sha,
        "remote_payloads": payloads,
        "population": {
            "samples": 10,
            "sequences": sorted(set(row["sequence"] for row in reference)),
            "ffn_bn_modules": 24,
            "hook_calls": candidate_receipt["hook_calls"],
            "coefficient_pairs": candidate_receipt["coefficient_pairs"],
            "bn_output_values": candidate_receipt["output_values"],
        },
        "coefficient_path_output_error": {
            "mean_absolute": candidate_receipt["mean_abs_output_delta"],
            "rmse": candidate_receipt["rmse_output_delta"],
            "maximum_absolute": candidate_receipt[
                "maximum_abs_output_delta"],
            "rail_counts": candidate_receipt["rail_counts"],
        },
        "paired_metrics": metrics,
        "top_absolute_layer_spike_deltas": layer_deltas[:20],
        "finding": (
            "The M235 Q16 coefficient path is numerically bounded but is not "
            "downstream-safe: mean AEE worsens by {:.6f}% and DSEC Fl worsens "
            "by {:.6f}% on the paired first ten frames, despite only {:.6f}% "
            "total-spike change. Small BN perturbations cross recurrent SNN "
            "thresholds and amplify into prediction differences."
        ).format(
            metrics["AEE"]["relative_delta_percent"],
            metrics["DSEC_Fl"]["relative_delta_percent"],
            metrics["spikes"]["relative_delta_percent"],
        ),
        "algorithm_feedback": [
            "Do not optimize BN only for coefficient RMSE or rail freedom; train and select against downstream event flips and AEE.",
            "Profile per-module threshold margins, then allocate more coefficient precision only to sensitive BN1/BN2 blocks.",
            "Consider quantization-aware fine-tuning with the exact LUT/Newton recurrence if mixed precision alone cannot recover the paired gate."
        ],
        "next_hardware_experiment": (
            "Run a trace-executable mixed-precision sweep over invstd/alpha/offset "
            "fraction bits and optional second Newton step; retain the cheapest "
            "configuration that restores paired downstream metrics before valid825."
        ),
        "admission": {
            "paired_first10_executed": True,
            "m235_q16_downstream_accuracy_safe": False,
            "valid825_recommended_for_m235_q16": False,
            "moment_finalizer_modeled": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    out_path = RESULT / "m260_m235_ffn_bn_paired_accuracy_analysis_r1.json"
    require(not out_path.exists(), "refusing to overwrite M260 analysis")
    out_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "aee_relative_delta_percent": metrics["AEE"][
            "relative_delta_percent"],
        "dsec_fl_relative_delta_percent": metrics["DSEC_Fl"][
            "relative_delta_percent"],
        "spike_relative_delta_percent": metrics["spikes"][
            "relative_delta_percent"],
        "maximum_bn_output_delta": candidate_receipt[
            "maximum_abs_output_delta"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
