#!/usr/bin/env python3
"""Independent, read-only audit of the canonical M2045 valid825 result."""
from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m2045_ep34_valid825_sdformerflow_env_successor_r1_20260902"
RESULT_MANIFEST = "c25a4857b5cd40616aa94324b396ed9a96d457a1453307a29eb99918fadf59fa"
RESULT_OUTER = "a926d722381df3c1f2961adf81fb2bf5cbcf4963082227c554ea16b2711bea93"
RESULT_JSON = "bf73e27cba9c69461d5cfc0ff97fb30b4ceadb08ac726fb52443278a6629a831"
PROFILE_JSON = "3b9d5fe7adf2156ebf4f2d0df286a629e9f19b9df3a47b0d47a70c0e87d37e33"
EVAL_LOG = "a4bb5d7a24c6a9ce68ad01d267c5ebe1ad0c542f29ab5b797450392a17818a95"
RUN_COMPLETE = "dbcfe480312ed17260e2310879c0aa65c535096d78f046c3110f286f5c26a1dc"

M2045_SOURCE = HW / "system_handoff/scripts/run_m2045_ep34_valid825_sdformerflow_env_successor.py"
M2045_SOURCE_SHA = "890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1"
M2045_CONTRACT = HW / "contracts/m2045_ep34_valid825_sdformerflow_env_successor_contract_r1_20260902.json"
M2045_CONTRACT_SHA = "4c3222055a7fa7b8b246ab43caf7b37a7eeb8554021f3556d9998942d302bdb0"
M2044_SOURCE = HW / "system_handoff/scripts/run_m2044_ep34_valid825_attention_eight_operator_qdq.py"
M2044_SOURCE_SHA = "edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20"
M2044_CONTRACT = HW / "contracts/m2044_ep34_valid825_attention_eight_operator_qdq_contract_r1_20260902.json"
M2044_CONTRACT_SHA = "03f13063493d563cf0b26363498d18bde60c8bee5e785a4dfca95845555757d2"

BUNDLE = HW / "system_handoff/generated/m2044_ep34_attention_hw_order_qdq8_bundle_r1_20260902"
BUNDLE_MANIFEST = "ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c"
BUNDLE_OUTER = "32cf8a7f4a7c015bcf0086fd7676bc0b5360710981be7c425e14ae62475d06a2"
BUNDLE_JSON = "01e7aadb454e82ce8fb04d25c4dc40f05bedd59cfd03d7e3835cdb2b967c3aee"
BUNDLE_CHECKPOINT = "daec6c188e7045ca3867c16cfcee5b25d2680eb4a7f1933541dfea17f0ac8371"
BUNDLE_CONFIG = "977d8f654e7aa5d528ca77a3a374d5d6554cc51b7773c1e579c08a79bcc6646d"

BASELINE = HW / "system_handoff/incoming/m2041_ep34_quant_binding_inputs/spike_profile.json"
BASELINE_SHA = "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c"
SOURCE_CHECKPOINT_SHA = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
SOURCE_CONFIG_SHA = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
VALIDATION_LIST = ROOT / "data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_split_seq.csv"
VALIDATION_LIST_SHA = "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0"

IDENTITY_FILES = (
    (ROOT / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
     "84daee48291d8ab2ee644f43458b909e96190c0dce7f5ff4d4179b61be30faac",
     "evaluator"),
    (ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py",
     "0f77f66dbd331daa77a284199cda33125a1959a005b6f4d592e2e6cda5317187",
     "BSA attention"),
    (ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/h9_load_audit.py",
     "172b3b8086cfe5c43bf9627fe92f947ca63148f9bbe8c50bca729b23c6273e68",
     "H9 load audit"),
    (ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py",
     "5873063b98eb4a267afa6513d03b86621f3fb6a885b310b4c5569ef5448ae657",
     "ATLIF installer"),
    (ROOT / "third_party/SDformerFlow/utils/metric_aggregation.py",
     "a34c31eaae52fafdb3442fbca82aac956e46d0fc040ccabb2f9d905e3dd8d379",
     "metric aggregation"),
    (ROOT / "third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py",
     "01dec420d4b97bd9ea97b5ab8fb54fb801fea79c52f2b37f5bedd40b7ff03e68",
     "dataset loader"),
)

OLD_FAILURE = HW / "results/m2044_ep34_valid825_attention_eight_operator_qdq_r1_20260902_FAILED_DO_NOT_CITE"
OLD_FAILURE_MANIFEST = "6d366ccb3121a9b72e4e38bf12a112f6241e1be4e6fe341269685d7ceba6af58"
OLD_FAILURE_OUTER = "ae7ebf05d56e4f409f09e1107f3c79fcebb7e61ced028593f282e1d7de8110a1"

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
EXPECTED_OUTPUT_ELEMENTS = {
    TARGETS[0]: 1900800000,
    TARGETS[1]: 1900800000,
    TARGETS[2]: 1900800000,
    TARGETS[3]: 1900800000,
    TARGETS[4]: 3801600000,
    TARGETS[5]: 7603200000,
    TARGETS[6]: 15206400000,
    TARGETS[7]: 60825600000,
}


class AuditError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise AuditError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise AuditError("missing " + label)
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AuditError("nonfinite JSON token " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def verify_sealed(directory, manifest_sha, outer_sha, expected_members):
    regular_exact(directory / "SHA256SUMS", manifest_sha,
                  directory.name + "/SHA256SUMS")
    regular_exact(directory / "SHA256SUMS.seal.sha256", outer_sha,
                  directory.name + "/outer seal")
    require((directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
        directory.name + " outer content drift")
    members = {}
    for line in (directory / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "malformed manifest line")
        digest, name = fields
        require(name not in members and Path(name).parts == (name,),
                "unsafe/duplicate manifest member")
        regular_exact(directory / name, digest, directory.name + "/" + name)
        members[name] = digest
    require(set(members) == set(expected_members),
            directory.name + " manifest member population drift")
    actual = set()
    for path in directory.iterdir():
        require(path.is_file() and not path.is_symlink(),
                directory.name + " has non-regular child")
        actual.add(path.name)
    require(actual == set(expected_members) |
            {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            directory.name + " exhaustive topology drift")
    return members


def close(left, right, tolerance):
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def recompute_aggregation(profile, label):
    audit = profile["metric_aggregation_audit"]
    sequences = audit["per_sequence"]
    require(audit["schema"] == "flow_metric_aggregation_audit_v1",
            label + " aggregation schema drift")
    require(audit["frame_count"] == 825 and
            audit["sequence_count"] == 18 and
            audit["valid_pixels"] == 48152523.0 and
            len(sequences) == 18,
            label + " aggregate population drift")
    require(sum(row["frame_count"] for row in sequences.values()) == 825,
            label + " sequence frame conservation")
    require(sum(row["valid_pixels"] for row in sequences.values()) == 48152523.0,
            label + " sequence pixel conservation")
    maxima = {"frame_recompute_error": 0.0,
              "pixel_recompute_error": 0.0,
              "sequence_recompute_error": 0.0,
              "profile_metric_vs_audit_error": 0.0}
    for metric in METRICS:
        frame_value = math.fsum(
            row["frame_count"] * row["frame_equal_mean"][metric]
            for row in sequences.values()) / 825.0
        pixel_value = math.fsum(
            row["valid_pixels"] * row["pixel_global_mean"][metric]
            for row in sequences.values()) / 48152523.0
        sequence_value = math.fsum(
            row["pixel_global_mean"][metric]
            for row in sequences.values()) / 18.0
        frame_error = abs(frame_value - audit["frame_equal_mean"][metric])
        pixel_error = abs(pixel_value - audit["pixel_global_mean"][metric])
        sequence_error = abs(
            sequence_value - audit["sequence_balanced_mean"][metric])
        profile_error = abs(
            float(profile["metrics"][metric]) -
            audit["frame_equal_mean"][metric])
        require(frame_error <= 1e-12 and pixel_error <= 1e-12 and
                sequence_error <= 1e-12,
                label + " metric aggregation recomputation failed: " + metric)
        require(profile_error <= 1e-6,
                label + " profile/audit metric mismatch: " + metric)
        maxima["frame_recompute_error"] = max(
            maxima["frame_recompute_error"], frame_error)
        maxima["pixel_recompute_error"] = max(
            maxima["pixel_recompute_error"], pixel_error)
        maxima["sequence_recompute_error"] = max(
            maxima["sequence_recompute_error"], sequence_error)
        maxima["profile_metric_vs_audit_error"] = max(
            maxima["profile_metric_vs_audit_error"], profile_error)
    for name, row in sequences.items():
        require(row["frame_count"] > 0 and row["valid_pixels"] > 0,
                label + " empty sequence: " + name)
        for mode in ("frame_equal_mean", "pixel_global_mean"):
            require(set(row[mode]) == set(METRICS),
                    label + " per-sequence metric keys drift")
            for metric in METRICS:
                value = float(row[mode][metric])
                require(math.isfinite(value) and value >= 0.0,
                        label + " nonfinite/negative per-sequence metric")
                if metric == "DSEC_Fl":
                    require(value <= 100.0,
                            label + " DSEC_Fl out of percentage range")
    return maxima


def main_audit():
    result_members = verify_sealed(
        RESULT, RESULT_MANIFEST, RESULT_OUTER,
        {"RUN_COMPLETE.txt", "eval.log", "result.json", "spike_profile.json"})
    require(result_members["result.json"] == RESULT_JSON and
            result_members["spike_profile.json"] == PROFILE_JSON and
            result_members["eval.log"] == EVAL_LOG and
            result_members["RUN_COMPLETE.txt"] == RUN_COMPLETE,
            "canonical member pin drift")
    require((RESULT / "RUN_COMPLETE.txt").read_text(
        encoding="utf-8").strip() == "PASS_M2044_VALID825_ACCURACY_GATE",
        "RUN_COMPLETE status drift")

    for path, digest, label in (
        (M2045_SOURCE, M2045_SOURCE_SHA, "M2045 wrapper"),
        (M2045_CONTRACT, M2045_CONTRACT_SHA, "M2045 contract"),
        (M2044_SOURCE, M2044_SOURCE_SHA, "frozen M2044 engine"),
        (M2044_CONTRACT, M2044_CONTRACT_SHA, "M2044 contract"),
        (BASELINE, BASELINE_SHA, "paired baseline profile"),
        (VALIDATION_LIST, VALIDATION_LIST_SHA, "validation file list"),
    ) + IDENTITY_FILES:
        regular_exact(path, digest, label)
    verify_sealed(
        BUNDLE, BUNDLE_MANIFEST, BUNDLE_OUTER,
        {"RUN_COMPLETE.txt", "bundle.json",
         "checkpoint_epoch34_m2044_qdq8.pth",
         "m2044_ep34_attention_hw_order_qdq8_valid825.yml"})
    regular_exact(BUNDLE / "bundle.json", BUNDLE_JSON, "bundle JSON")
    regular_exact(BUNDLE / "checkpoint_epoch34_m2044_qdq8.pth",
                  BUNDLE_CHECKPOINT, "derived checkpoint")
    regular_exact(BUNDLE / "m2044_ep34_attention_hw_order_qdq8_valid825.yml",
                  BUNDLE_CONFIG, "derived config")
    verify_sealed(
        OLD_FAILURE, OLD_FAILURE_MANIFEST, OLD_FAILURE_OUTER,
        {"FAILURE.txt", "eval.log"})

    result = strict_json(RESULT / "result.json")
    candidate = strict_json(RESULT / "spike_profile.json")
    baseline = strict_json(BASELINE)
    contract = strict_json(M2045_CONTRACT)
    require(result["schema"] ==
            "m2044_ep34_valid825_attention_eight_operator_qdq_result_r1_v1" and
            result["status"] == "PASS_M2044_VALID825_ACCURACY_GATE",
            "result schema/status drift")
    require(contract["accuracy_gate"] == {
        "baseline_AEE": 1.1995140134204518,
        "maximum_candidate_minus_baseline_AEE": 0.02,
        "unchanged_from_m2044": True,
    }, "M2045 accuracy contract drift")

    expected_identity = {
        "producer_source_sha256": M2044_SOURCE_SHA,
        "evaluator_sha256": IDENTITY_FILES[0][1],
        "bsa_attention_sha256": IDENTITY_FILES[1][1],
        "h9_load_audit_sha256": IDENTITY_FILES[2][1],
        "atlif_installer_sha256": IDENTITY_FILES[3][1],
        "metric_aggregation_sha256": IDENTITY_FILES[4][1],
        "dataset_loader_sha256": IDENTITY_FILES[5][1],
        "validation_file_list_sha256": VALIDATION_LIST_SHA,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
        "source_config_sha256": SOURCE_CONFIG_SHA,
        "baseline_profile_sha256": BASELINE_SHA,
    }
    for key, value in expected_identity.items():
        require(result.get(key) == value, "result identity drift: " + key)
    require(result["derived_bundle"] == {
        "bundle_json_sha256": BUNDLE_JSON,
        "reviewed_manifest_sha256": BUNDLE_MANIFEST,
        "checkpoint_sha256": BUNDLE_CHECKPOINT,
        "config_sha256": BUNDLE_CONFIG,
        "weights_modified": 8,
    }, "result bundle identity drift")

    require(candidate["samples"] == baseline["samples"] == 825,
            "paired sample count drift")
    require(candidate["eval_protocol"] == baseline["eval_protocol"] and
            candidate["metric_contract"] == baseline["metric_contract"],
            "paired evaluation/metric protocol drift")
    require(candidate["validation_file_list"]["sha256"] ==
            baseline["validation_file_list"]["sha256"] ==
            VALIDATION_LIST_SHA, "paired validation list drift")
    candidate_agg = candidate["metric_aggregation_audit"]
    baseline_agg = baseline["metric_aggregation_audit"]
    require(set(candidate_agg["per_sequence"]) ==
            set(baseline_agg["per_sequence"]), "paired sequence set drift")
    for sequence in baseline_agg["per_sequence"]:
        c_row = candidate_agg["per_sequence"][sequence]
        b_row = baseline_agg["per_sequence"][sequence]
        require(c_row["frame_count"] == b_row["frame_count"] and
                c_row["valid_pixels"] == b_row["valid_pixels"],
                "paired per-sequence population drift: " + sequence)
    baseline_recompute = recompute_aggregation(baseline, "baseline")
    candidate_recompute = recompute_aggregation(candidate, "candidate")

    for metric in METRICS:
        b_value = float(baseline["metrics"][metric])
        c_value = float(candidate["metrics"][metric])
        require(result["baseline_metrics"][metric] == b_value and
                result["candidate_metrics"][metric] == c_value,
                "result/profile metric drift: " + metric)
        require(close(result["candidate_minus_baseline"][metric],
                      c_value - b_value, 1e-15),
                "metric delta recomputation failed: " + metric)
    aee_delta = (float(candidate["metrics"]["AEE"]) -
                 float(baseline["metrics"]["AEE"]))
    require(close(aee_delta, -0.0021467093987899144, 1e-15) and
            result["accuracy_gate"]["observed"] == aee_delta and
            result["accuracy_gate"]["threshold"] == 0.02 and
            result["accuracy_gate"]["pass"] is True and aee_delta <= 0.02,
            "AEE accuracy gate recomputation failed")

    require(candidate["checkpoint_load_audit"]["missing_count"] == 0 and
            candidate["checkpoint_load_audit"]["unexpected_count"] == 0 and
            candidate["checkpoint_load_audit"]["overlay_missing_count"] == 0 and
            candidate["checkpoint_load_audit"]["overlay_unexpected_count"] == 0,
            "checkpoint load audit failed")
    require(candidate["artifact_identity"]["checkpoint_sha256"] ==
            BUNDLE_CHECKPOINT and
            candidate["artifact_identity"]["config_sha256"] == BUNDLE_CONFIG,
            "candidate artifact identity drift")
    require(candidate["module_counts"] == {
        "ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
        "candidate module count drift")
    require(candidate["runtime_backend_audit"] == {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
    }, "candidate backend audit drift")
    forward = candidate["deployment_operator_forward_audit"]
    require(tuple(forward["targets"]) == TARGETS and
            forward["all_targets_reached"] is True and
            forward["calls"] == dict((target, 825) for target in TARGETS) and
            forward["output_elements"] == EXPECTED_OUTPUT_ELEMENTS,
            "eight-target forward audit drift")
    require(result["deployment_operator_forward_audit"] == forward,
            "result/profile forward audit drift")
    require(candidate["deployment_contract"] == {
        "scope": "attention_hardware_order_plus_eight_operator_weight_qdq",
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
        "m2042_result_sha256":
            "455c9fe7036779b890d4b85911cc42dc47bcb62c9fb6f6a6ce9c28a2c833cf29",
        "attention_score": "Q7_RNE_step_2^-7_clip_-2_2",
        "attention_gate": "existing_Q8_LUT_next_pow2_rowsum_Q1.7_RNE",
        "retained_weights":
            "four_C1_Conv3x3_plus_four_decoder_ConvTranspose_dyadic_INT8_QDQ",
        "untouched_network_operators": "checkpoint_precision",
        "full_network_INT8": False,
        "SystemVerilog_equivalent_full_network": False,
    }, "candidate deployment contract drift")

    require(result["population"] == {
        "samples": 825,
        "attention_blocks": 12,
        "ATLIF_modules_configured": 105,
        "operator_weights_quantized": 8,
        "aggregation_frame_count": 825,
        "aggregation_valid_pixels": 48152523.0,
        "aggregation_sequence_count": 18,
    }, "result population summary drift")
    require(result["automatic_retry"] is False, "automatic retry drift")
    expected_boundary = {
        "paired_valid825_subset_deployment_result": True,
        "attention_hardware_order_full_valid825": True,
        "eight_operator_weight_QDQ_full_valid825": True,
        "operator_integer_bridge_separate_M2043": True,
        "full_network_INT8": False,
        "whole_network_hardware_order_equivalence": False,
        "SystemVerilog_equivalent_full_network": False,
        "hardware_cycles": False,
        "hardware_speedup": False,
        "system_speedup": False,
        "energy": False,
        "PPA": False,
        "paper_accuracy_result": False,
        "paper_accuracy_result_requires_independent_result_hammer": True,
    }
    require(result["claim_boundary"] == expected_boundary,
            "result claim boundary drift")

    log_text = (RESULT / "eval.log").read_text(encoding="utf-8")
    require(log_text.startswith(
        'M2044 exact command argv: ["/opt/conda/envs/sdformerflow/bin/python"')
        and "M2044 evaluator exit_code=0" in log_text,
        "M2045 interpreter/exit evidence drift")

    sequence_deltas = {}
    for sequence in sorted(baseline_agg["per_sequence"]):
        sequence_deltas[sequence] = (
            candidate_agg["per_sequence"][sequence]["frame_equal_mean"]["AEE"] -
            baseline_agg["per_sequence"][sequence]["frame_equal_mean"]["AEE"])
    improving = sum(value < 0.0 for value in sequence_deltas.values())
    regressing = sum(value > 0.0 for value in sequence_deltas.values())
    minimum_sequence = min(sequence_deltas, key=sequence_deltas.get)
    maximum_sequence = max(sequence_deltas, key=sequence_deltas.get)
    return {
        "schema": "m2045_ep34_valid825_result_independent_audit_r1_v1",
        "status": "PASS_M2045_EP34_VALID825_RESULT_INDEPENDENT_AUDIT",
        "canonical_result_manifest_sha256": RESULT_MANIFEST,
        "canonical_result_outer_sha256": RESULT_OUTER,
        "m2045_wrapper_sha256": M2045_SOURCE_SHA,
        "m2045_contract_sha256": M2045_CONTRACT_SHA,
        "frozen_m2044_engine_sha256": M2044_SOURCE_SHA,
        "bundle_manifest_sha256": BUNDLE_MANIFEST,
        "paired_population": {
            "samples": 825,
            "frames": 825,
            "sequences": 18,
            "valid_pixels": 48152523,
            "exactly_equal": True,
        },
        "metric_recomputation": {
            "baseline": baseline_recompute,
            "candidate": candidate_recompute,
            "baseline_AEE": float(baseline["metrics"]["AEE"]),
            "candidate_AEE": float(candidate["metrics"]["AEE"]),
            "candidate_minus_baseline_AEE": aee_delta,
            "gate_threshold": 0.02,
            "gate_pass": True,
        },
        "aee_sequence_distribution": {
            "improving_sequences": improving,
            "regressing_sequences": regressing,
            "minimum_delta": sequence_deltas[minimum_sequence],
            "minimum_delta_sequence": minimum_sequence,
            "maximum_delta": sequence_deltas[maximum_sequence],
            "maximum_delta_sequence": maximum_sequence,
        },
        "load_backend_forward_audit": {
            "load_mismatches": 0,
            "TF32": False,
            "cudnn_benchmark": False,
            "targets": 8,
            "calls_per_target": 825,
            "all_targets_reached": True,
        },
        "paper_admission": {
            "paired_attention_hardware_order_plus_eight_operator_QDQ_accuracy":
                True,
            "full_network_INT8": False,
            "whole_network_hardware_order_equivalence": False,
            "hardware_speedup": False,
            "energy": False,
            "PPA": False,
        },
    }


def main():
    try:
        result = main_audit()
    except Exception as error:
        print(json.dumps({
            "schema": "m2045_ep34_valid825_result_independent_audit_r1_v1",
            "status": "FAIL_M2045_EP34_VALID825_RESULT_INDEPENDENT_AUDIT",
            "error_type": type(error).__name__,
            "error": str(error),
        }, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
