#!/usr/bin/env python3
"""Fail-closed analysis for the six M266 dynamic-BN sensitivity subsets."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m266_q20n2_ffn_bn_stage_ablation_s10_r1_20260825"
REFERENCE = HW / "results/m263_balanced_q20n2_ffn_bn_paired_s10_r1_20260825/reference"
SELECTORS = {"stage0": 4, "stage1": 4, "stage2": 12, "stage3": 4,
             "bn1": 12, "bn2": 12}
PATHS = {
    "contract": HW / "contracts/m266_q20n2_ffn_bn_stage_ablation_contract_r1_20260825.json",
    "evaluator": HW / "system_handoff/scripts/eval_m266_q20n2_ffn_bn_stage_ablation_DSEC.py",
    "wrapper": HW / "system_handoff/run_m266_q20n2_ffn_bn_stage_ablation_when_gpu_idle_20260825.sh",
    "m263_network_seal": HW / "results/m263_balanced_q20n2_ffn_bn_paired_s10_r1_20260825/LOCAL_SHA256SUMS",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "contract": "a4337e92fca104d638bafd90386343925d6a162e033ba1bb9a05df720d322444",
    "evaluator": "7d0bfdd1adc2c0db44f19130a49aada212c733797cfb12e6d6a09a046b0ced6f",
    "wrapper": "c585d1e3b1fd5b063adce814259f24ce3aaa38bd012d922dbcf1cab477dd449f",
    "m263_network_seal": "c82c50f886c5af45cbc8c185abd33e23f3a5e30733dcac17da8e9a7c4ef78f16",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
REMOTE_MANIFEST_SHA = "dc9cd77d6a4db81f391f3c3787e0a6819b4c2b6489bc6d7be918c7f3f8cf7cf7"


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
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             RuntimeError("non-finite JSON: " + value)))


def read_frames(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 10, "sample count drift: " + str(path))
    return rows


def paired_metric(reference, candidate, field):
    before = [float(row[field]) for row in reference]
    after = [float(row[field]) for row in candidate]
    require(all(value == value and abs(value) != float("inf")
                for value in before + after), "non-finite metric: " + field)
    before_mean = sum(before) / len(before)
    after_mean = sum(after) / len(after)
    return {
        "reference_mean": before_mean,
        "candidate_mean": after_mean,
        "relative_delta_percent": (after_mean / before_mean - 1.0) * 100.0,
        "candidate_wins": sum(a < b for a, b in zip(after, before)),
        "candidate_losses": sum(a > b for a, b in zip(after, before)),
        "candidate_ties": sum(a == b for a, b in zip(after, before)),
        "maximum_absolute_paired_delta": max(abs(a - b)
                                               for a, b in zip(after, before)),
    }


def validate_remote_manifest():
    manifest = RESULT / "SHA256SUMS"
    require(sha256(manifest) == REMOTE_MANIFEST_SHA,
            "remote manifest seal drift")
    payloads = []
    marker = "/" + RESULT.name + "/"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, remote_path = line.split("  ", 1)
        require(marker in remote_path, "remote manifest escaped result")
        relative = remote_path.split(marker, 1)[1]
        local_path = RESULT / relative
        require(local_path.is_file(), "missing remote payload: " + relative)
        observed = sha256(local_path)
        require(observed == expected, "remote payload SHA drift: " + relative)
        payloads.append({"path": relative, "sha256": observed,
                         "bytes": local_path.stat().st_size})
    require(len(payloads) == 27, "remote payload population drift")
    return payloads


def main():
    identity = {name: sha256(path) for name, path in PATHS.items()}
    require(identity == EXPECTED, "frozen input identity drift")
    payloads = validate_remote_manifest()
    reference = read_frames(REFERENCE / "per_frame.csv")
    reference_ids = [(row["sequence"], row["file"]) for row in reference]
    require(len(set(reference_ids)) == 10, "reference identity duplicate")

    subsets = {}
    for selector, expected_count in SELECTORS.items():
        directory = RESULT / selector
        candidate = read_frames(directory / "per_frame.csv")
        candidate_ids = [(row["sequence"], row["file"]) for row in candidate]
        require(candidate_ids == reference_ids, selector + " paired identity drift")
        receipt = strict_json(directory / "m266_runtime_receipt.json")
        require(receipt["status"] ==
                "PASS_M266_BALANCED_Q20N2_EVALUATION_RUNTIME",
                selector + " status drift")
        require(receipt["samples"] == 10 and
                receipt["target_selector"] == selector and
                receipt["target_count"] == expected_count and
                receipt["hook_calls"] == expected_count * 10,
                selector + " target/hook population drift")
        require(len(receipt["target_modules"]) == expected_count and
                len(set(receipt["target_modules"])) == expected_count,
                selector + " target module list drift")
        require(receipt["m266_python_candidate_exact"] and
                not receipt["m235_q16_recurrence"] and
                not receipt["moment_finalizer_modeled"] and
                all(value == 0 for value in receipt["rail_counts"].values()),
                selector + " numeric contract drift")
        metrics = {field: paired_metric(reference, candidate, field)
                   for field in ("AEE", "DSEC_Fl", "spikes")}
        subsets[selector] = {
            "target_count": expected_count,
            "target_modules": receipt["target_modules"],
            "hook_calls": receipt["hook_calls"],
            "coefficient_pairs": receipt["coefficient_pairs"],
            "bn_output_values": receipt["output_values"],
            "maximum_local_bn_output_delta": receipt["maximum_abs_output_delta"],
            "mean_local_bn_output_delta": receipt["mean_abs_output_delta"],
            "rail_counts": receipt["rail_counts"],
            "paired_metrics": metrics,
            "gates": {
                "local_delta_le_0p00025":
                    receipt["maximum_abs_output_delta"] <= 0.00025,
                "aee_regression_le_0p25_percent":
                    metrics["AEE"]["relative_delta_percent"] <= 0.25,
                "dsec_fl_regression_le_1_percent":
                    metrics["DSEC_Fl"]["relative_delta_percent"] <= 1.0,
            },
        }

    require(all(item["gates"]["local_delta_le_0p00025"]
                for item in subsets.values()), "local error gate observation drift")
    require(not any(item["gates"]["aee_regression_le_0p25_percent"]
                    for item in subsets.values()), "AEE subset observation drift")
    require(subsets["stage2"]["paired_metrics"]["AEE"]
            ["relative_delta_percent"] <
            subsets["stage1"]["paired_metrics"]["AEE"]
            ["relative_delta_percent"], "non-additivity observation drift")

    ranking = sorted(SELECTORS, key=lambda name: (
        subsets[name]["paired_metrics"]["AEE"]["relative_delta_percent"],
        subsets[name]["paired_metrics"]["DSEC_Fl"]["relative_delta_percent"]))
    output = {
        "schema": "m266_q20n2_ffn_bn_stage_ablation_analysis_v1",
        "status": "PASS_SIX_SUBSETS_NO_ACCURACY_SAFE_APPROXIMATE_SET",
        "identity": identity,
        "remote_payload_manifest_sha256": REMOTE_MANIFEST_SHA,
        "remote_payloads": payloads,
        "reference": {
            "samples": 10,
            "sequence": reference[0]["sequence"],
            "source": str(REFERENCE.relative_to(ROOT)),
        },
        "subsets": subsets,
        "aee_ranking_best_to_worst": ranking,
        "findings": [
            "Every isolated subset passes the local BN output-error gate, but all six fail the paired AEE regression gate.",
            "Stage/position effects are non-additive: stage2 approximates three times as many modules as stage1 yet has a smaller AEE regression.",
            "BN1-only and BN2-only both regress AEE more than the all-24-module M263 run, so subset deltas cannot be summed or used as fallback attribution.",
            "Uniform Q20N2 coefficient approximation is therefore not admitted for dynamic BN; preserve an exact/high-precision coefficient path or obtain quantization-aware algorithm recovery.",
        ],
        "hardware_direction": {
            "admitted": "Share an exact/high-precision dynamic-BN moment/reciprocal engine across layers and optimize its utilization before reducing arithmetic precision.",
            "algorithm_feedback": "Train or calibrate with the exact deployed coefficient recurrence and threshold-aware loss; widening local error alone is not an accuracy guarantee.",
            "rejected": "Selecting a low-sensitivity stage or BN position from this first10 ablation.",
        },
        "boundaries": {
            "overlapping_subsets_non_additive": True,
            "single_sequence_first10": True,
            "moment_finalizer_modeled": False,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "valid825": False,
            "speedup": False,
            "system_speedup": False,
            "energy": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    output_path = RESULT / "m266_q20n2_ffn_bn_stage_ablation_analysis_r1.json"
    require(not output_path.exists(), "refusing to overwrite M266 analysis")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "aee_ranking": ranking,
        "aee_relative_delta_percent": {
            name: subsets[name]["paired_metrics"]["AEE"]
            ["relative_delta_percent"] for name in SELECTORS},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
