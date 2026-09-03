#!/usr/bin/env python3
"""Read-only independent result hammer for the canonical M2045 run."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / (
    "results/m2045_ep34_valid825_sdformerflow_env_successor_r1_20260902"
)
SOURCE = HW / (
    "system_handoff/scripts/"
    "run_m2045_ep34_valid825_sdformerflow_env_successor.py"
)
CONTRACT = HW / (
    "contracts/"
    "m2045_ep34_valid825_sdformerflow_env_successor_contract_r1_20260902.json"
)
ENGINE = HW / (
    "system_handoff/scripts/"
    "run_m2044_ep34_valid825_attention_eight_operator_qdq.py"
)
ENGINE_CONTRACT = HW / (
    "contracts/m2044_ep34_valid825_attention_eight_operator_qdq_contract_r1_20260902.json"
)
BUNDLE = HW / (
    "system_handoff/generated/"
    "m2044_ep34_attention_hw_order_qdq8_bundle_r1_20260902"
)
INPUTS = HW / "system_handoff/incoming/m2041_ep34_quant_binding_inputs"
BASELINE = INPUTS / "spike_profile.json"
SOURCE_CONFIG = INPUTS / "dsec_c12_alpha0125_ep29_resume5_20260830.yml"
EVALUATOR = ROOT / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
PREFLIGHT = HW / (
    "reviews/m2045_ep34_valid825_sdformerflow_env_preflight_r1_20260902/"
    "preflight.json"
)
PREFLIGHT_REVIEW = HW / (
    "reviews/"
    "m2045_ep34_valid825_sdformerflow_env_preflight_independent_hammer_"
    "r1_20260902"
)
TENSOR_REVIEW = HW / "reviews/m2044_ep34_derived_bundle_tensor_audit_r1_20260902"

SHA = {
    "result_manifest": "c25a4857b5cd40616aa94324b396ed9a96d457a1453307a29eb99918fadf59fa",
    "result_outer": "a926d722381df3c1f2961adf81fb2bf5cbcf4963082227c554ea16b2711bea93",
    "result_json": "bf73e27cba9c69461d5cfc0ff97fb30b4ceadb08ac726fb52443278a6629a831",
    "candidate_profile": "3b9d5fe7adf2156ebf4f2d0df286a629e9f19b9df3a47b0d47a70c0e87d37e33",
    "eval_log": "a4bb5d7a24c6a9ce68ad01d267c5ebe1ad0c542f29ab5b797450392a17818a95",
    "source": "890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1",
    "contract": "4c3222055a7fa7b8b246ab43caf7b37a7eeb8554021f3556d9998942d302bdb0",
    "engine": "edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20",
    "engine_contract": "03f13063493d563cf0b26363498d18bde60c8bee5e785a4dfca95845555757d2",
    "bundle_manifest": "ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c",
    "bundle_outer": "32cf8a7f4a7c015bcf0086fd7676bc0b5360710981be7c425e14ae62475d06a2",
    "bundle_json": "01e7aadb454e82ce8fb04d25c4dc40f05bedd59cfd03d7e3835cdb2b967c3aee",
    "bundle_checkpoint": "daec6c188e7045ca3867c16cfcee5b25d2680eb4a7f1933541dfea17f0ac8371",
    "bundle_config": "977d8f654e7aa5d528ca77a3a374d5d6554cc51b7773c1e579c08a79bcc6646d",
    "baseline": "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c",
    "evaluator": "84daee48291d8ab2ee644f43458b909e96190c0dce7f5ff4d4179b61be30faac",
    "preflight": "41da22f4b5745e5919f6177267bfefeb4d168815f43196c53755e89cad74079a",
    "preflight_review_manifest": "9de7d091af3bbfa6be9bb0177abaf83a3e886eec4cb1f2118f897187075e80a8",
    "preflight_review_outer": "d57263553debd0ae660ced98ce93c8fdc28763495b3906285c6fc72bef3d4a92",
    "tensor_review_manifest": "e2714d4a841e86fba30265d97e537c8f98a19af521a5ece8d8a47b9c33ae3ce9",
    "tensor_review_outer": "d7a102fd964d7a1109fa309dc4e45a296d99646b2f23f23241bccb7e25548bea",
    "tensor_audit": "0e8905bde3d54b53518b0795ea42656a16c7da305788d200e06e16261b415fe6",
}

METRICS = ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
TARGETS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
    "sttmultires_unet.decoders.0.deconv.0",
    "sttmultires_unet.decoders.1.deconv.0",
    "sttmultires_unet.decoders.2.deconv.0",
    "sttmultires_unet.decoders.3.deconv.0",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    assert stat.S_ISREG(mode) and not path.is_symlink(), path
    assert sha256(path) == expected, path


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            assert key not in result, "duplicate key: " + key
            result[key] = value
        return result

    result = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError("nonfinite token: " + token)))
    assert type(result) is dict
    return result


def verify_manifest(directory: Path, expected_manifest: str,
                    expected_outer: str, expected_members: set[str]) -> None:
    regular_exact(directory / "SHA256SUMS", expected_manifest)
    regular_exact(directory / "SHA256SUMS.seal.sha256", expected_outer)
    assert (directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").split() == [expected_manifest, "SHA256SUMS"]
    seen: set[str] = set()
    for line in (directory / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        assert len(fields) == 2 and len(fields[0]) == 64
        digest, name = fields
        assert name not in seen and "/" not in name and name not in {".", ".."}
        regular_exact(directory / name, digest)
        seen.add(name)
    assert seen == expected_members
    actual = {path.name for path in directory.iterdir()}
    assert actual == expected_members | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    for path in directory.iterdir():
        mode = path.lstat().st_mode
        assert stat.S_ISREG(mode) and not path.is_symlink()


def close(a: float, b: float, tolerance: float = 5e-12) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def recompute_aggregates(profile: dict[str, Any]) -> dict[str, dict[str, float]]:
    audit = profile["metric_aggregation_audit"]
    rows = audit["per_sequence"]
    frames = math.fsum(float(row["frame_count"]) for row in rows.values())
    pixels = math.fsum(float(row["valid_pixels"]) for row in rows.values())
    return {
        "frame_equal_mean": {
            metric: math.fsum(
                float(row["frame_equal_mean"][metric]) *
                float(row["frame_count"]) for row in rows.values()) / frames
            for metric in METRICS
        },
        "pixel_global_mean": {
            metric: math.fsum(
                float(row["pixel_global_mean"][metric]) *
                float(row["valid_pixels"]) for row in rows.values()) / pixels
            for metric in METRICS
        },
        "sequence_balanced_mean": {
            metric: math.fsum(
                float(row["pixel_global_mean"][metric])
                for row in rows.values()) / len(rows)
            for metric in METRICS
        },
    }


def main() -> int:
    tests: list[dict[str, Any]] = []

    def check(name: str, value: bool) -> None:
        tests.append({"name": name, "pass": bool(value)})
        assert value, name

    verify_manifest(
        RESULT, SHA["result_manifest"], SHA["result_outer"],
        {"RUN_COMPLETE.txt", "eval.log", "result.json", "spike_profile.json"})
    check("canonical_result_exhaustive_double_seal", True)
    regular_exact(RESULT / "result.json", SHA["result_json"])
    regular_exact(RESULT / "spike_profile.json", SHA["candidate_profile"])
    regular_exact(RESULT / "eval.log", SHA["eval_log"])

    for path, key in ((SOURCE, "source"), (CONTRACT, "contract"),
                      (ENGINE, "engine"), (ENGINE_CONTRACT, "engine_contract"),
                      (BASELINE, "baseline"), (EVALUATOR, "evaluator"),
                      (PREFLIGHT, "preflight")):
        regular_exact(path, SHA[key])
    check("successor_engine_evaluator_identity", True)
    verify_manifest(
        BUNDLE, SHA["bundle_manifest"], SHA["bundle_outer"],
        {"RUN_COMPLETE.txt", "bundle.json",
         "checkpoint_epoch34_m2044_qdq8.pth",
         "m2044_ep34_attention_hw_order_qdq8_valid825.yml"})
    regular_exact(BUNDLE / "bundle.json", SHA["bundle_json"])
    regular_exact(BUNDLE / "checkpoint_epoch34_m2044_qdq8.pth",
                  SHA["bundle_checkpoint"])
    regular_exact(BUNDLE / "m2044_ep34_attention_hw_order_qdq8_valid825.yml",
                  SHA["bundle_config"])
    check("reviewed_bundle_exhaustive_double_seal", True)
    regular_exact(PREFLIGHT_REVIEW / "SHA256SUMS",
                  SHA["preflight_review_manifest"])
    regular_exact(PREFLIGHT_REVIEW / "SHA256SUMS.seal.sha256",
                  SHA["preflight_review_outer"])
    regular_exact(TENSOR_REVIEW / "SHA256SUMS", SHA["tensor_review_manifest"])
    regular_exact(TENSOR_REVIEW / "SHA256SUMS.seal.sha256",
                  SHA["tensor_review_outer"])
    regular_exact(TENSOR_REVIEW / "remote_cpu_audit.json", SHA["tensor_audit"])
    tensor = strict_json(TENSOR_REVIEW / "remote_cpu_audit.json")
    counts = tensor["counts"]
    check("bundle_tensor_audit", tensor["status"] ==
          "PASS_M2044_EP34_DERIVED_BUNDLE_TENSOR_AUDIT" and
          counts["mismatches"] == 0 and
          counts["non_target_torch_equal"] == 913 and
          counts["target_qdq_torch_equal"] == 8 and
          counts["tensor_keys_checked"] == 921)

    result = strict_json(RESULT / "result.json")
    candidate = strict_json(RESULT / "spike_profile.json")
    baseline = strict_json(BASELINE)
    bundle = strict_json(BUNDLE / "bundle.json")
    contract = strict_json(CONTRACT)
    check("result_schema_status", result["schema"] ==
          "m2044_ep34_valid825_attention_eight_operator_qdq_result_r1_v1"
          and result["status"] == "PASS_M2044_VALID825_ACCURACY_GATE"
          and result["automatic_retry"] is False)
    check("wrapper_contract_engine_binding", contract["claim_boundary"][
          "environment_only_successor"] is True and
          contract["frozen_execution_engine"]["sha256"] == SHA["engine"] and
          result["producer_source_sha256"] == SHA["engine"])
    check("bundle_identity_in_result", result["derived_bundle"] == {
        "bundle_json_sha256": SHA["bundle_json"],
        "reviewed_manifest_sha256": SHA["bundle_manifest"],
        "checkpoint_sha256": SHA["bundle_checkpoint"],
        "config_sha256": SHA["bundle_config"],
        "weights_modified": 8})
    check("bundle_exact_eight_targets", len(bundle["modified_weights"]) == 8 and
          {row["checkpoint_key"] for row in bundle["modified_weights"]} ==
          {target + ".weight" for target in TARGETS})

    log = (RESULT / "eval.log").read_text(encoding="utf-8")
    first = log.splitlines()[0]
    prefix = "M2044 exact command argv: "
    argv = json.loads(first[len(prefix):]) if first.startswith(prefix) else []
    check("sdformerflow_interpreter_executed", argv[:3] == [
        "/opt/conda/envs/sdformerflow/bin/python", "-u",
        "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"])
    check("evaluator_success", log.count("M2044 evaluator exit_code=0") == 1 and
          "Traceback (most recent call last)" not in log)

    ba = baseline["metric_aggregation_audit"]
    ca = candidate["metric_aggregation_audit"]
    check("population_825_18_48152523", result["population"]["samples"] == 825 and
          result["population"]["aggregation_frame_count"] == 825 and
          result["population"]["aggregation_sequence_count"] == 18 and
          result["population"]["aggregation_valid_pixels"] == 48152523.0 and
          candidate["samples"] == 825 and ca["frame_count"] == 825 and
          ca["sequence_count"] == 18 and ca["valid_pixels"] == 48152523.0)
    check("paired_protocol_and_filelist", candidate["eval_protocol"] ==
          baseline["eval_protocol"] and candidate["metric_contract"] ==
          baseline["metric_contract"] and
          candidate["validation_file_list"] == baseline["validation_file_list"] and
          candidate["validation_file_list"]["sha256"] ==
          result["validation_file_list_sha256"])
    check("per_sequence_population_equal", set(ba["per_sequence"]) ==
          set(ca["per_sequence"]) and all(
              ba["per_sequence"][name]["frame_count"] ==
              ca["per_sequence"][name]["frame_count"] and
              ba["per_sequence"][name]["valid_pixels"] ==
              ca["per_sequence"][name]["valid_pixels"]
              for name in ba["per_sequence"]))

    recomputed = {
        "baseline": recompute_aggregates(baseline),
        "candidate": recompute_aggregates(candidate),
    }
    check("independent_aggregate_recompute", all(
        close(recomputed[side][mode][metric],
              profile["metric_aggregation_audit"][mode][metric])
        for side, profile in (("baseline", baseline), ("candidate", candidate))
        for mode in ("frame_equal_mean", "pixel_global_mean",
                     "sequence_balanced_mean") for metric in METRICS))

    headline = {
        side: {metric: float(profile["metrics"][metric]) for metric in METRICS}
        for side, profile in (("baseline", baseline), ("candidate", candidate))
    }
    deltas = {metric: headline["candidate"][metric] -
              headline["baseline"][metric] for metric in METRICS}
    check("headline_metrics_recomputed", all(
        close(headline["baseline"][metric], result["baseline_metrics"][metric],
              1e-15) and
        close(headline["candidate"][metric], result["candidate_metrics"][metric],
              1e-15) and
        close(deltas[metric], result["candidate_minus_baseline"][metric], 1e-15)
        for metric in METRICS))
    check("headline_vs_aggregation_consistent", all(
        abs(headline[side][metric] - recomputed[side]["frame_equal_mean"][metric])
        <= 5e-7 for side in ("baseline", "candidate") for metric in METRICS))
    check("aee_gate", deltas["AEE"] == -0.0021467093987899144 and
          deltas["AEE"] <= 0.02 and result["accuracy_gate"] == {
              "metric": "candidate_minus_baseline_AEE",
              "observed": deltas["AEE"], "pass": True, "threshold": 0.02})

    load = candidate["checkpoint_load_audit"]
    check("checkpoint_load_audit", load["missing_count"] == 0 and
          load["unexpected_count"] == 0 and load["overlay_missing_count"] == 0 and
          load["overlay_unexpected_count"] == 0 and load["remap"] == "v1")
    check("backend_audit", candidate["runtime_backend_audit"] == {
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "cudnn_benchmark": False} and
        result["runtime_backend_audit"] == candidate["runtime_backend_audit"])
    forward = candidate["deployment_operator_forward_audit"]
    check("eight_operator_forward_audit", tuple(forward["targets"]) == TARGETS and
          forward["all_targets_reached"] is True and
          all(forward["calls"][target] == 825 and
              forward["output_elements"][target] > 0 for target in TARGETS) and
          result["deployment_operator_forward_audit"] == forward)
    check("module_population", candidate["module_counts"] == {
        "ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12} and
        result["population"]["ATLIF_modules_configured"] == 105 and
        result["population"]["attention_blocks"] == 12)

    deployment = candidate["deployment_contract"]
    check("deployment_scope", deployment["scope"] ==
          "attention_hardware_order_plus_eight_operator_weight_qdq" and
          deployment["retained_weights"] ==
          "four_C1_Conv3x3_plus_four_decoder_ConvTranspose_dyadic_INT8_QDQ" and
          deployment["untouched_network_operators"] == "checkpoint_precision" and
          deployment["full_network_INT8"] is False and
          deployment["SystemVerilog_equivalent_full_network"] is False)
    boundary = result["claim_boundary"]
    check("positive_claim_boundary", boundary[
          "paired_valid825_subset_deployment_result"] is True and
          boundary["attention_hardware_order_full_valid825"] is True and
          boundary["eight_operator_weight_QDQ_full_valid825"] is True and
          boundary["operator_integer_bridge_separate_M2043"] is True)
    check("negative_claim_boundary", all(boundary[key] is False for key in (
        "full_network_INT8", "whole_network_hardware_order_equivalence",
        "SystemVerilog_equivalent_full_network", "hardware_cycles",
        "hardware_speedup", "system_speedup", "energy", "PPA")))

    source_cfg = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    candidate_cfg = yaml.safe_load((BUNDLE /
        "m2044_ep34_attention_hw_order_qdq8_valid825.yml").read_text(
            encoding="utf-8"))
    backend_difference = {
        "baseline_config_allow_tf32": source_cfg["runtime"]["allow_tf32"],
        "baseline_config_cudnn_benchmark":
            source_cfg["runtime"]["cudnn_benchmark"],
        "candidate_allow_tf32": candidate_cfg["runtime"]["allow_tf32"],
        "candidate_cudnn_benchmark":
            candidate_cfg["runtime"]["cudnn_benchmark"],
    }
    check("backend_difference_detected", backend_difference == {
        "baseline_config_allow_tf32": True,
        "baseline_config_cudnn_benchmark": True,
        "candidate_allow_tf32": False,
        "candidate_cudnn_benchmark": False})

    output = {
        "schema": "m2045_canonical_result_independent_hammer_r1_v1",
        "status": "PASS_PAIRED_TASK_ACCURACY_CITABLE_WITH_BOUNDARY",
        "score": {"passed": len(tests), "total": len(tests),
                  "review_score": 95},
        "severity": {"P0": 0, "P1": 2, "P2": 3},
        "headline_metrics": headline,
        "candidate_minus_baseline": deltas,
        "independent_aggregate_recompute": recomputed,
        "population": {"samples": 825, "sequences": 18,
                       "valid_pixels": 48152523},
        "backend_config_difference": backend_difference,
        "citation_admission": {
            "candidate_valid825_task_accuracy": True,
            "frozen_baseline_referenced_delta": True,
            "strict_backend_matched_causal_attribution": False,
            "hardware_performance_claim": False,
        },
        "fixed_sha256": SHA,
        "tests": tests,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
