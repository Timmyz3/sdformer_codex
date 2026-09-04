#!/usr/bin/env python3
"""Run one frozen ep34 valid825 deployment-candidate evaluation.

The candidate combines the existing Q7/Q1.7 hardware-order attention path with
the eight M2042 Conv/ConvTranspose dyadic-INT8 weights, stored dequantized in a
portable state-dict checkpoint.  It deliberately leaves every other network
operator in its checkpoint precision.  Therefore a successful run admits a
paired valid825 result for this *subset deployment transform*, not a full-network
INT8 or SystemVerilog-equivalent claim.

The program has two explicit phases.  ``--prepare-only`` atomically publishes a
stable derived checkpoint/config bundle without using the GPU.  ``--run`` first
rehashes that bundle and then consumes the only authorized valid825 production
attempt.  Failure evidence is retained and automatic retry is forbidden.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m2044_ep34_valid825_attention_eight_operator_qdq_contract_r1_20260902.json"
CONTRACT_SHA256 = "03f13063493d563cf0b26363498d18bde60c8bee5e785a4dfca95845555757d2"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M2042 = HW / "results/m2042_ep34_s40_eight_operator_int8_export_r1_20260902"
M2042_RESULT_SHA256 = "455c9fe7036779b890d4b85911cc42dc47bcb62c9fb6f6a6ce9c28a2c833cf29"
M2042_SHA256SUMS_SHA256 = "519b8621a0c16f67ed33c8c624adc6bbfbc1c4a27224b2812542da3d92fc3881"
M2042_OUTER_SEAL_SHA256 = "da977b9effab3accaff229877bc4d9f0e930f82de1c0833be5c872e63aee142b"
M2041 = HW / "system_handoff/incoming/m2041_ep34_quant_binding_inputs"
ORIGINAL_CHECKPOINT = M2041 / "checkpoint_epoch34.pth"
ORIGINAL_CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
ORIGINAL_CONFIG = M2041 / "dsec_c12_alpha0125_ep29_resume5_20260830.yml"
ORIGINAL_CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
BASELINE_PROFILE = M2041 / "spike_profile.json"
BASELINE_PROFILE_SHA256 = "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c"
EVALUATOR = ROOT / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
EVALUATOR_SHA256 = "84daee48291d8ab2ee644f43458b909e96190c0dce7f5ff4d4179b61be30faac"
BSA_ATTENTION = (ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay/"
                 "models/STSwinNet_SNN/bsa_attention.py")
BSA_ATTENTION_SHA256 = "0f77f66dbd331daa77a284199cda33125a1959a005b6f4d592e2e6cda5317187"
H9_LOAD_AUDIT = (ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay/"
                 "models/STSwinNet_SNN/h9_load_audit.py")
H9_LOAD_AUDIT_SHA256 = "172b3b8086cfe5c43bf9627fe92f947ca63148f9bbe8c50bca729b23c6273e68"
ATLIF_INSTALLER = (ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay/"
                   "models/STSwinNet_SNN/atlif_ternary_psn/installer.py")
ATLIF_INSTALLER_SHA256 = "5873063b98eb4a267afa6513d03b86621f3fb6a885b310b4c5569ef5448ae657"
METRIC_AGGREGATION = ROOT / "third_party/SDformerFlow/utils/metric_aggregation.py"
METRIC_AGGREGATION_SHA256 = "a34c31eaae52fafdb3442fbca82aac956e46d0fc040ccabb2f9d905e3dd8d379"
DATASET_LOADER = ROOT / "third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py"
DATASET_LOADER_SHA256 = "01dec420d4b97bd9ea97b5ab8fb54fb801fea79c52f2b37f5bedd40b7ff03e68"
VALIDATION_FILE_LIST_SHA256 = "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0"
TARGET_MODULES = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
    "sttmultires_unet.decoders.0.deconv.0",
    "sttmultires_unet.decoders.1.deconv.0",
    "sttmultires_unet.decoders.2.deconv.0",
    "sttmultires_unet.decoders.3.deconv.0",
)
TARGET_KEYS = tuple(name + ".weight" for name in TARGET_MODULES)


class M2044Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M2044Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M2044Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root must be an object: " + str(path))
    return value


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M2044Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def verify_manifest(directory: Path, expected_manifest: str,
                    expected_outer: str) -> list[str]:
    regular_exact(directory / "SHA256SUMS", expected_manifest,
                  directory.name + " SHA256SUMS")
    regular_exact(directory / "SHA256SUMS.seal.sha256", expected_outer,
                  directory.name + " outer seal")
    require((directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").split() == [expected_manifest, "SHA256SUMS"],
        directory.name + " outer seal content drift")
    names: list[str] = []
    for line in (directory / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed SHA256SUMS line")
        expected, name = fields
        require(name not in names and "/" not in name and name not in {".", ".."},
                "duplicate or unsafe sealed member")
        regular_exact(directory / name, expected, directory.name + "/" + name)
        names.append(name)
    return names


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_seal(directory: Path, names: list[str]) -> tuple[str, str]:
    require("SHA256SUMS" not in names and "SHA256SUMS.seal.sha256" not in names,
            "seal list contains seal files")
    manifest = directory / "SHA256SUMS"
    manifest.write_text("\n".join(
        sha256(directory / name) + "  " + name for name in sorted(names)) + "\n",
        encoding="utf-8")
    manifest_sha = sha256(manifest)
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(manifest_sha + "  SHA256SUMS\n", encoding="utf-8")
    return manifest_sha, sha256(outer)


def retain_failure(temporary: Path, failed: Path, status: str,
                   error: BaseException) -> None:
    """Best-effort fail-closed publication for every started phase."""
    require(temporary.is_dir(), "failure temp directory is missing")
    require(not failed.exists(), "failure namespace already exists")
    for seal_name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
        seal_path = temporary / seal_name
        if seal_path.exists():
            seal_path.unlink()
    require(not any(path.is_dir() or path.is_symlink()
                    for path in temporary.iterdir()),
            "failure temp contains an unsealable child")
    (temporary / "FAILURE.txt").write_text(
        status + "\nautomatic_retry=false\nerror_type="
        + type(error).__name__ + "\nerror=" + str(error) + "\n",
        encoding="utf-8")
    write_seal(temporary, [path.name for path in temporary.iterdir()
                           if path.is_file()])
    os.replace(temporary, failed)


def load_contract(expected_source_sha256: str) -> dict[str, Any]:
    regular_exact(CONTRACT, CONTRACT_SHA256, "M2044 contract")
    contract = strict_json(CONTRACT)
    require(contract.get("schema") ==
            "m2044_ep34_valid825_attention_eight_operator_qdq_contract_r1_v1",
            "M2044 contract schema drift")
    require(contract.get("status") ==
            "LOCKED_SOURCE_REVIEW_REQUIRED__ONE_VALID825_ATTEMPT_ONLY",
            "M2044 contract status drift")
    running_sha = sha256(Path(__file__).resolve())
    require(running_sha == expected_source_sha256, "running source SHA drift")
    require(contract["producer"] == {
        "path": "hw_autoresearch_nts07/system_handoff/scripts/run_m2044_ep34_valid825_attention_eight_operator_qdq.py",
        "independent_source_review_must_pin_actual_source_sha256": True,
        "required_python_on_A800": "/opt/conda/bin/python3",
        "automatic_retry": False,
        "valid825_production_attempts_authorized_after_independent_source_review": 1,
    }, "M2044 producer contract drift")
    require(tuple(contract["candidate_contract"][
        "deployment_operator_forward_audit_targets"]) == TARGET_MODULES,
        "M2044 forward-audit target contract drift")
    require(contract.get("phase_admission") == {
        "prepare_only_requires_independent_source_review_with_zero_P0": True,
        "valid825_run_requires_independent_derived_bundle_review": True,
        "valid825_run_must_supply_reviewed_bundle_manifest_sha256": True,
        "automatic_retry_after_any_valid825_attempt": False,
    }, "M2044 phase-admission contract drift")
    require(contract.get("accuracy_gate") == {
        "baseline_AEE": 1.1995140134204518,
        "maximum_candidate_minus_baseline_AEE": 0.02,
        "required_metrics": ["AEE", "AAE", "AAE_Benchmark", "DSEC_Fl"],
        "gate_semantics": (
            "candidate_AEE_minus_baseline_AEE_must_be_less_than_or_equal_to_"
            "threshold; improvements are admitted"
        ),
        "gate_failure_policy": (
            "publish_as_executed_negative_result_and_do_not_retry"
        ),
    }, "M2044 accuracy-gate contract drift")
    return contract


def verify_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    regular_exact(ORIGINAL_CHECKPOINT, ORIGINAL_CHECKPOINT_SHA256,
                  "ep34 source checkpoint")
    regular_exact(ORIGINAL_CONFIG, ORIGINAL_CONFIG_SHA256, "ep34 source config")
    regular_exact(BASELINE_PROFILE, BASELINE_PROFILE_SHA256,
                  "ep34 baseline profile")
    for path, digest, label in (
        (EVALUATOR, EVALUATOR_SHA256, "DSEC evaluator"),
        (BSA_ATTENTION, BSA_ATTENTION_SHA256, "H9 BSA attention"),
        (H9_LOAD_AUDIT, H9_LOAD_AUDIT_SHA256, "H9 load audit"),
        (ATLIF_INSTALLER, ATLIF_INSTALLER_SHA256, "ATLIF installer"),
        (METRIC_AGGREGATION, METRIC_AGGREGATION_SHA256, "metric aggregation"),
        (DATASET_LOADER, DATASET_LOADER_SHA256, "DSEC dataset loader"),
    ):
        regular_exact(path, digest, label)
    members = verify_manifest(M2042, M2042_SHA256SUMS_SHA256,
                              M2042_OUTER_SEAL_SHA256)
    require(len(members) == 26 and set(members) ==
            set(["result.json", "RUN_COMPLETE.txt"] + [
                f"{index:02d}_{family}_{suffix}.{dtype}.npy"
                for index, family in enumerate(
                    ["c1_conv3x3"] * 4 + ["decoder_convtranspose"] * 4)
                for suffix, dtype in (
                    ("canonical_o_i_ky_kx", "int8"),
                    ("hardware_i_ky_kx_o", "int8"),
                    ("scale_exp2", "int16"))
            ]), "M2042 exact member population drift")
    regular_exact(M2042 / "result.json", M2042_RESULT_SHA256, "M2042 result")
    m2042 = strict_json(M2042 / "result.json")
    require(m2042.get("status") ==
            "PASS_M2042_EP34_EIGHT_OPERATOR_INT8_WEIGHT_EXPORT" and
            len(m2042.get("layers", [])) == 8 and
            m2042.get("checkpoint_sha256") == ORIGINAL_CHECKPOINT_SHA256,
            "M2042 result identity/population drift")
    for row in m2042["layers"]:
        regular_exact(M2042 / row["canonical_code_file"],
                      row["canonical_code_sha256"],
                      "M2042 canonical code " + row["checkpoint_key"])
        regular_exact(M2042 / row["scale_exponent_file"],
                      row["scale_exponent_sha256"],
                      "M2042 scale exponent " + row["checkpoint_key"])
    baseline = strict_json(BASELINE_PROFILE)
    require(int(baseline.get("samples", -1)) == 825 and
            baseline["artifact_identity"]["checkpoint_sha256"] ==
            ORIGINAL_CHECKPOINT_SHA256 and
            baseline["artifact_identity"]["config_sha256"] ==
            ORIGINAL_CONFIG_SHA256, "baseline valid825 identity drift")
    require(contract["inputs"] == {
        "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "config_sha256": ORIGINAL_CONFIG_SHA256,
        "baseline_profile_sha256": BASELINE_PROFILE_SHA256,
        "m2042_result_sha256": M2042_RESULT_SHA256,
        "m2042_sha256sums_sha256": M2042_SHA256SUMS_SHA256,
        "docs359_sha256": DOCS359_SHA256,
        "evaluator_sha256": EVALUATOR_SHA256,
        "bsa_attention_sha256": BSA_ATTENTION_SHA256,
        "h9_load_audit_sha256": H9_LOAD_AUDIT_SHA256,
        "atlif_installer_sha256": ATLIF_INSTALLER_SHA256,
        "metric_aggregation_sha256": METRIC_AGGREGATION_SHA256,
        "dataset_loader_sha256": DATASET_LOADER_SHA256,
        "validation_file_list_sha256": VALIDATION_FILE_LIST_SHA256,
    }, "contract input pins drift")
    return {"m2042": m2042, "baseline": baseline}


def output_paths(contract: dict[str, Any]) -> tuple[Path, Path]:
    bundle = (ROOT / contract["outputs"]["derived_bundle"]).resolve()
    result = (ROOT / contract["outputs"]["result_directory"]).resolve()
    for path in (bundle, result):
        require(path.is_relative_to(ROOT.resolve()), "output escapes repository")
    return bundle, result


def make_config() -> dict[str, Any]:
    config = yaml.safe_load(ORIGINAL_CONFIG.read_text(encoding="utf-8")) or {}
    require(type(config) is dict, "source config root drift")
    candidate = copy.deepcopy(config)
    attention = candidate["bsa_attention"]
    require(attention["mode"] == "h60" and float(attention["alpha0"]) == 0.02 and
            float(attention["binary_motion_xor_alpha"]) == 0.125,
            "selected attention identity drift")
    attention.update({
        "hardware_quant_enabled": True,
        "hardware_mu_pow2_shift": 0,
        "hardware_score_step": 1.0 / 128.0,
        "hardware_score_min": -2.0,
        "hardware_score_max": 2.0,
        "hardware_gate_step": 1.0 / 128.0,
        "hardware_gate_min": 0.0,
        "hardware_gate_max": 2.0,
        "hardware_rtl_shiftmax_enabled": True,
    })
    runtime = candidate.setdefault("runtime", {})
    runtime["allow_tf32"] = False
    runtime["cudnn_benchmark"] = False
    runtime["deployment_operator_forward_audit_targets"] = list(TARGET_MODULES)
    vis = candidate["vis"]
    require(vis.get("enabled") is False and vis.get("store") is False and
            vis.get("store_att", False) is False and
            vis.get("monitor_fr", False) is False and
            vis.get("monitor_v", False) is False,
            "source config would create unsealed visualization children")
    runtime["deployment_contract"] = {
        "scope": "attention_hardware_order_plus_eight_operator_weight_qdq",
        "source_checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "m2042_result_sha256": M2042_RESULT_SHA256,
        "attention_score": "Q7_RNE_step_2^-7_clip_-2_2",
        "attention_gate": "existing_Q8_LUT_next_pow2_rowsum_Q1.7_RNE",
        "retained_weights": "four_C1_Conv3x3_plus_four_decoder_ConvTranspose_dyadic_INT8_QDQ",
        "untouched_network_operators": "checkpoint_precision",
        "full_network_INT8": False,
        "SystemVerilog_equivalent_full_network": False,
    }
    candidate["experiment"] = "m2044_ep34_attention_hw_order_eight_operator_qdq_valid825"
    candidate["note"] = (
        "M2044 subset deployment candidate: preserve ep34 alpha0=0.02 and "
        "Motion-XOR alpha=0.125; enable existing Q7/Q1.7 RTL-order Shiftmax; "
        "replace only four C1 and four decoder weights with M2042 dyadic-INT8 "
        "QDQ. This is not full-network INT8."
    )
    return candidate


def prepare_bundle(contract: dict[str, Any], inputs: dict[str, Any],
                   bundle: Path, producer_source_sha256: str) -> None:
    import torch

    require(not bundle.exists(), "derived bundle already exists")
    temporary = bundle.parent / ("." + bundle.name + ".tmp")
    failed = bundle.parent / (bundle.name + "_FAILED_DO_NOT_CITE")
    require(not temporary.exists(), "stale derived-bundle temp exists")
    require(not failed.exists(), "derived-bundle failure already consumed this phase")
    temporary.mkdir(parents=True)
    try:
        payload = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu",
                             weights_only=False)
        require(type(payload) is dict and set(payload) == {"model_state_dict"},
                "checkpoint container drift")
        state = payload["model_state_dict"]
        require(len(state) == 921, "checkpoint state population drift")
        modified: list[dict[str, Any]] = []
        for row in inputs["m2042"]["layers"]:
            key = row["checkpoint_key"]
            require(key in state and type(state[key]) is torch.Tensor,
                    "missing target weight: " + key)
            source = state[key].detach().cpu().contiguous()
            require(hashlib.sha256(source.numpy().tobytes(order="C")).hexdigest() ==
                    row["source_weight_sha256"], "source weight identity drift: " + key)
            code = np.load(M2042 / row["canonical_code_file"], allow_pickle=False)
            exponent = np.load(M2042 / row["scale_exponent_file"], allow_pickle=False)
            require(code.dtype == np.int8 and exponent.dtype == np.int16 and
                    tuple(code.shape) == tuple(row["canonical_shape"]) and
                    tuple(exponent.shape) == (code.shape[0],),
                    "M2042 array dtype/shape drift: " + key)
            canonical = (code.astype(np.float64) *
                         np.exp2(exponent.astype(np.float64))[:, None, None, None])
            native = (canonical if int(row["native_output_axis"]) == 0 else
                      np.moveaxis(canonical, 0, 1))
            require(tuple(native.shape) == tuple(row["native_shape"]),
                    "native QDQ shape drift: " + key)
            replacement = torch.from_numpy(native.copy(order="C")).to(dtype=source.dtype)
            state[key] = replacement
            modified.append({
                "checkpoint_key": key,
                "family": row["family"],
                "native_output_axis": row["native_output_axis"],
                "canonical_code_sha256": row["canonical_code_sha256"],
                "scale_exponent_sha256": row["scale_exponent_sha256"],
                "source_weight_sha256": row["source_weight_sha256"],
                "qdq_weight_sha256": hashlib.sha256(
                    replacement.numpy().tobytes(order="C")).hexdigest(),
            })
        require(len(modified) == 8 and len({row["checkpoint_key"] for row in modified}) == 8,
                "modified target population drift")
        checkpoint = temporary / "checkpoint_epoch34_m2044_qdq8.pth"
        torch.save({"model_state_dict": state}, checkpoint)
        reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(type(reloaded) is dict and set(reloaded) == {"model_state_dict"} and
                len(reloaded["model_state_dict"]) == 921,
                "derived checkpoint readback drift")
        for row in modified:
            tensor = reloaded["model_state_dict"][row["checkpoint_key"]]
            require(hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes(
                order="C")).hexdigest() == row["qdq_weight_sha256"],
                "derived checkpoint target readback drift")

        config = make_config()
        config_path = temporary / "m2044_ep34_attention_hw_order_qdq8_valid825.yml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False,
                                              allow_unicode=True), encoding="utf-8")
        bundle_result = {
            "schema": "m2044_ep34_qdq8_derived_bundle_r1_v1",
            "status": "PASS_M2044_DERIVED_BUNDLE_NO_ACCURACY_CLAIM",
            "producer_source_sha256": producer_source_sha256,
            "source_checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "source_config_sha256": ORIGINAL_CONFIG_SHA256,
            "m2042_result_sha256": M2042_RESULT_SHA256,
            "derived_checkpoint_file": checkpoint.name,
            "derived_checkpoint_sha256": sha256(checkpoint),
            "derived_config_file": config_path.name,
            "derived_config_sha256": sha256(config_path),
            "modified_weights": modified,
            "claim_boundary": {
                "exactly_eight_weights_replaced": True,
                "attention_hardware_order_configured": True,
                "valid825_executed": False,
                "valid825_AEE": False,
                "full_network_INT8": False,
                "SystemVerilog_equivalent_full_network": False,
                "hardware_cycles": False,
                "system_speedup": False,
            },
        }
        write_json(temporary / "bundle.json", bundle_result)
        (temporary / "RUN_COMPLETE.txt").write_text(
            "PASS_M2044_DERIVED_BUNDLE_NO_ACCURACY_CLAIM\n", encoding="utf-8")
        write_seal(temporary, ["bundle.json", "RUN_COMPLETE.txt",
                               checkpoint.name, config_path.name])
        os.replace(temporary, bundle)
    except Exception as error:
        retain_failure(temporary, failed,
                       "FAILED_M2044_DERIVED_BUNDLE_DO_NOT_CITE", error)
        raise


def verify_bundle(bundle: Path, expected_source_sha256: str,
                  inputs: dict[str, Any],
                  expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    import torch

    outer_words = (bundle / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").split()
    require(len(outer_words) == 2 and outer_words[1] == "SHA256SUMS",
            "derived bundle outer seal malformed")
    if expected_manifest_sha256 is not None:
        require(len(expected_manifest_sha256) == 64 and
                sha256(bundle / "SHA256SUMS") == expected_manifest_sha256,
                "derived bundle differs from independently reviewed manifest")
    members = verify_manifest(bundle, outer_words[0],
                              sha256(bundle / "SHA256SUMS.seal.sha256"))
    require(len(members) == 4 and set(members) == {
        "bundle.json", "RUN_COMPLETE.txt", "checkpoint_epoch34_m2044_qdq8.pth",
        "m2044_ep34_attention_hw_order_qdq8_valid825.yml"},
        "derived bundle member population drift")
    result = strict_json(bundle / "bundle.json")
    require(result.get("status") == "PASS_M2044_DERIVED_BUNDLE_NO_ACCURACY_CLAIM" and
            len(result.get("modified_weights", [])) == 8,
            "derived bundle result drift")
    require(result.get("producer_source_sha256") == expected_source_sha256 and
            result.get("source_checkpoint_sha256") == ORIGINAL_CHECKPOINT_SHA256 and
            result.get("source_config_sha256") == ORIGINAL_CONFIG_SHA256 and
            result.get("m2042_result_sha256") == M2042_RESULT_SHA256,
            "derived bundle authority drift")
    checkpoint = bundle / result["derived_checkpoint_file"]
    config_path = bundle / result["derived_config_file"]
    require(sha256(checkpoint) == result["derived_checkpoint_sha256"] and
            sha256(config_path) == result["derived_config_sha256"],
            "derived bundle file/result SHA drift")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    require(config == make_config(), "derived config semantic drift")
    attention = config["bsa_attention"]
    runtime = config["runtime"]
    require(attention["mode"] == "h60" and float(attention["alpha0"]) == 0.02 and
            float(attention["binary_motion_xor_alpha"]) == 0.125 and
            attention["hardware_quant_enabled"] is True and
            attention["hardware_rtl_shiftmax_enabled"] is True and
            float(attention["hardware_score_step"]) == 1.0 / 128.0 and
            [float(attention["hardware_score_min"]),
             float(attention["hardware_score_max"])] == [-2.0, 2.0] and
            float(attention["hardware_gate_step"]) == 1.0 / 128.0 and
            [float(attention["hardware_gate_min"]),
             float(attention["hardware_gate_max"])] == [0.0, 2.0] and
            runtime["allow_tf32"] is False and
            runtime["cudnn_benchmark"] is False and
            tuple(runtime["deployment_operator_forward_audit_targets"]) ==
            TARGET_MODULES, "derived config deployment fields drift")

    expected_rows = {row["checkpoint_key"]: row
                     for row in inputs["m2042"]["layers"]}
    modified_rows = {row["checkpoint_key"]: row
                     for row in result["modified_weights"]}
    require(tuple(expected_rows) == TARGET_KEYS and set(modified_rows) ==
            set(TARGET_KEYS) and len(modified_rows) == 8,
            "derived bundle exact target-key population drift")
    source_payload = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu",
                                weights_only=False)
    derived_payload = torch.load(checkpoint, map_location="cpu",
                                 weights_only=False)
    require(type(source_payload) is dict and type(derived_payload) is dict and
            set(source_payload) == {"model_state_dict"} and
            set(derived_payload) == {"model_state_dict"},
            "source/derived checkpoint container drift")
    source_state = source_payload["model_state_dict"]
    derived_state = derived_payload["model_state_dict"]
    require(tuple(source_state) == tuple(derived_state) and len(source_state) == 921,
            "source/derived state key population drift")
    for key in source_state:
        source_value = source_state[key]
        derived_value = derived_state[key]
        require(type(source_value) is torch.Tensor and
                type(derived_value) is torch.Tensor,
                "non-tensor state value: " + key)
        if key not in expected_rows:
            require(torch.equal(source_value, derived_value),
                    "non-target state drift: " + key)
            continue
        frozen = expected_rows[key]
        recorded = modified_rows[key]
        require(recorded["family"] == frozen["family"] and
                int(recorded["native_output_axis"]) ==
                int(frozen["native_output_axis"]) and
                recorded["canonical_code_sha256"] ==
                frozen["canonical_code_sha256"] and
                recorded["scale_exponent_sha256"] ==
                frozen["scale_exponent_sha256"] and
                recorded["source_weight_sha256"] ==
                frozen["source_weight_sha256"],
                "modified-weight authority row drift: " + key)
        code = np.load(M2042 / frozen["canonical_code_file"], allow_pickle=False)
        exponent = np.load(M2042 / frozen["scale_exponent_file"], allow_pickle=False)
        canonical = (code.astype(np.float64) *
                     np.exp2(exponent.astype(np.float64))[:, None, None, None])
        native = (canonical if int(frozen["native_output_axis"]) == 0 else
                  np.moveaxis(canonical, 0, 1))
        expected = torch.from_numpy(native.copy(order="C")).to(
            dtype=source_value.dtype)
        require(torch.equal(derived_value, expected),
                "target QDQ content drift: " + key)
        actual_digest = hashlib.sha256(derived_value.detach().cpu().contiguous()
                                       .numpy().tobytes(order="C")).hexdigest()
        require(actual_digest == recorded["qdq_weight_sha256"],
                "target QDQ recorded digest drift: " + key)
    return result


def metric(profile: dict[str, Any], name: str) -> float:
    value = float(profile["metrics"][name])
    require(math.isfinite(value), "nonfinite metric: " + name)
    return value


def verify_population_contract(profile: dict[str, Any],
                               baseline: dict[str, Any]) -> None:
    require(profile.get("eval_protocol") == baseline.get("eval_protocol"),
            "candidate evaluation protocol differs from paired baseline")
    require(profile.get("metric_contract") == baseline.get("metric_contract"),
            "candidate metric contract differs from paired baseline")
    candidate_list = profile.get("validation_file_list")
    baseline_list = baseline.get("validation_file_list")
    require(type(candidate_list) is dict and type(baseline_list) is dict and
            candidate_list.get("sha256") == VALIDATION_FILE_LIST_SHA256 and
            baseline_list.get("sha256") == VALIDATION_FILE_LIST_SHA256,
            "candidate/baseline validation population SHA drift")

    candidate_audit = profile.get("metric_aggregation_audit")
    baseline_audit = baseline.get("metric_aggregation_audit")
    require(type(candidate_audit) is dict and type(baseline_audit) is dict,
            "missing metric aggregation audit")
    for key in ("schema", "definitions", "frame_count", "valid_pixels",
                "sequence_count"):
        require(candidate_audit.get(key) == baseline_audit.get(key),
                "candidate aggregation population drift: " + key)
    for key in ("frame_equal_mean", "pixel_global_mean",
                "sequence_balanced_mean"):
        require(type(candidate_audit.get(key)) is dict and
                set(candidate_audit[key]) ==
                {"AEE", "AAE", "AAE_Benchmark", "DSEC_Fl"},
                "candidate aggregation mode drift: " + key)
    candidate_sequences = candidate_audit.get("per_sequence")
    baseline_sequences = baseline_audit.get("per_sequence")
    require(type(candidate_sequences) is dict and
            type(baseline_sequences) is dict and
            set(candidate_sequences) == set(baseline_sequences),
            "candidate per-sequence population drift")
    for sequence in baseline_sequences:
        candidate_row = candidate_sequences[sequence]
        baseline_row = baseline_sequences[sequence]
        require(type(candidate_row) is dict and
                candidate_row.get("frame_count") ==
                baseline_row.get("frame_count") and
                candidate_row.get("valid_pixels") ==
                baseline_row.get("valid_pixels") and
                set(candidate_row.get("frame_equal_mean", {})) ==
                {"AEE", "AAE", "AAE_Benchmark", "DSEC_Fl"} and
                set(candidate_row.get("pixel_global_mean", {})) ==
                {"AEE", "AAE", "AAE_Benchmark", "DSEC_Fl"},
                "candidate sequence population drift: " + sequence)


def verify_forward_audit(profile: dict[str, Any]) -> dict[str, Any]:
    audit = profile.get("deployment_operator_forward_audit")
    require(type(audit) is dict and
            tuple(audit.get("targets", ())) == TARGET_MODULES,
            "candidate forward-audit target population drift")
    calls = audit.get("calls")
    output_elements = audit.get("output_elements")
    require(type(calls) is dict and type(output_elements) is dict and
            set(calls) == set(TARGET_MODULES) and
            set(output_elements) == set(TARGET_MODULES) and
            audit.get("all_targets_reached") is True,
            "candidate forward-audit map drift")
    for target in TARGET_MODULES:
        require(int(calls[target]) == 825 and int(output_elements[target]) > 0,
                "candidate transformed operator was not reached exactly once "
                "per sample: " + target)
    return audit


def run_valid825(contract: dict[str, Any], inputs: dict[str, Any],
                 bundle: Path, output: Path,
                 expected_source_sha256: str,
                 expected_bundle_manifest_sha256: str) -> None:
    bundle_result = verify_bundle(bundle, expected_source_sha256, inputs,
                                  expected_bundle_manifest_sha256)
    require(not output.exists(), "canonical M2044 result already exists")
    temporary = output.parent / ("." + output.name + ".tmp")
    failed = output.parent / (output.name + "_FAILED_DO_NOT_CITE")
    require(not temporary.exists(), "stale M2044 result temp exists")
    require(not failed.exists(),
            "M2044 failed-attempt namespace exists; automatic retry forbidden")
    temporary.mkdir(parents=True)
    try:
        config = bundle / bundle_result["derived_config_file"]
        checkpoint = bundle / bundle_result["derived_checkpoint_file"]
        log = temporary / "eval.log"
        command = [
            sys.executable, "-u", str(EVALUATOR.relative_to(ROOT)),
            "--config", str(config), "--checkpoint", str(checkpoint),
            "--path_results", str(temporary), "--mode", "valid",
            "--bn-policy", "no_running",
        ]
        environment = os.environ.copy()
        environment.update({
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONHASHSEED": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        })
        with log.open("w", encoding="utf-8") as stream:
            stream.write("M2044 exact command argv: " + json.dumps(command) + "\n")
            stream.flush()
            process = subprocess.run(command, cwd=ROOT, env=environment,
                                     stdout=stream,
                                     stderr=subprocess.STDOUT)
            stream.write("\nM2044 evaluator exit_code={}\n".format(
                process.returncode))
        require(process.returncode == 0,
                "valid825 evaluator exit_code=" + str(process.returncode))

        # Detect source, data, or bundle drift during the long production run
        # before admitting any accuracy result.
        post_inputs = verify_inputs(contract)
        require(post_inputs["baseline"] == inputs["baseline"] and
                post_inputs["m2042"] == inputs["m2042"],
                "frozen input semantics drifted during valid825")
        verify_bundle(bundle, expected_source_sha256, inputs,
                      expected_bundle_manifest_sha256)

        profile = strict_json(temporary / "spike_profile.json")
        require(int(profile.get("samples", -1)) == 825,
                "candidate sample count drift")
        identity = profile["artifact_identity"]
        require(identity["checkpoint_sha256"] ==
                bundle_result["derived_checkpoint_sha256"] and
                identity["config_sha256"] ==
                bundle_result["derived_config_sha256"],
                "candidate artifact identity drift")
        audit = profile["checkpoint_load_audit"]
        require(int(audit["missing_count"]) == 0 and
                int(audit["unexpected_count"]) == 0 and
                int(audit["overlay_missing_count"]) == 0 and
                int(audit["overlay_unexpected_count"]) == 0,
                "candidate checkpoint load audit failed")
        require(profile["module_counts"] == {"ATLIFTernaryPSN": 105,
                                              "ShiftmaxAttention": 12},
                "candidate module-count drift")
        baseline = inputs["baseline"]
        verify_population_contract(profile, baseline)
        require(profile.get("runtime_backend_audit") == {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
        }, "candidate deterministic backend audit failed")
        forward_audit = verify_forward_audit(profile)
        require(profile["deployment_contract"] ==
                make_config()["runtime"]["deployment_contract"],
                "candidate deployment contract drift")

        metric_names = ["AEE", "AAE", "AAE_Benchmark", "DSEC_Fl"]
        baseline_metrics = {name: metric(baseline, name)
                            for name in metric_names}
        candidate_metrics = {name: metric(profile, name)
                             for name in metric_names}
        deltas = {name: candidate_metrics[name] - baseline_metrics[name]
                  for name in metric_names}
        gate = deltas["AEE"] <= float(contract["accuracy_gate"][
            "maximum_candidate_minus_baseline_AEE"])
        aggregation = profile["metric_aggregation_audit"]
        result = {
            "schema": (
                "m2044_ep34_valid825_attention_eight_operator_qdq_result_r1_v1"
            ),
            "status": ("PASS_M2044_VALID825_ACCURACY_GATE" if gate else
                       "PASS_EXECUTION__M2044_ACCURACY_GATE_FAIL"),
            "producer_source_sha256": expected_source_sha256,
            "evaluator_sha256": EVALUATOR_SHA256,
            "bsa_attention_sha256": BSA_ATTENTION_SHA256,
            "h9_load_audit_sha256": H9_LOAD_AUDIT_SHA256,
            "atlif_installer_sha256": ATLIF_INSTALLER_SHA256,
            "metric_aggregation_sha256": METRIC_AGGREGATION_SHA256,
            "dataset_loader_sha256": DATASET_LOADER_SHA256,
            "validation_file_list_sha256": VALIDATION_FILE_LIST_SHA256,
            "source_checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "source_config_sha256": ORIGINAL_CONFIG_SHA256,
            "baseline_profile_sha256": BASELINE_PROFILE_SHA256,
            "m2042_result_sha256": M2042_RESULT_SHA256,
            "derived_bundle": {
                "bundle_json_sha256": sha256(bundle / "bundle.json"),
                "reviewed_manifest_sha256": expected_bundle_manifest_sha256,
                "checkpoint_sha256": bundle_result["derived_checkpoint_sha256"],
                "config_sha256": bundle_result["derived_config_sha256"],
                "weights_modified": 8,
            },
            "population": {
                "samples": 825,
                "attention_blocks": 12,
                "ATLIF_modules_configured": 105,
                "operator_weights_quantized": 8,
                "aggregation_frame_count": aggregation["frame_count"],
                "aggregation_valid_pixels": aggregation["valid_pixels"],
                "aggregation_sequence_count": aggregation["sequence_count"],
            },
            "runtime_backend_audit": profile["runtime_backend_audit"],
            "deployment_operator_forward_audit": forward_audit,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "candidate_minus_baseline": deltas,
            "accuracy_gate": {
                "metric": "candidate_minus_baseline_AEE",
                "threshold": contract["accuracy_gate"][
                    "maximum_candidate_minus_baseline_AEE"],
                "observed": deltas["AEE"],
                "pass": gate,
            },
            "claim_boundary": {
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
            },
            "automatic_retry": False,
        }
        write_json(temporary / "result.json", result)
        (temporary / "RUN_COMPLETE.txt").write_text(
            result["status"] + "\n", encoding="utf-8")
        require(not any(path.is_dir() or path.is_symlink()
                        for path in temporary.iterdir()),
                "M2044 result temp contains unsealable child")
        names_to_seal = [path.name for path in temporary.iterdir()
                         if path.is_file()]
        write_seal(temporary, names_to_seal)
        os.replace(temporary, output)
    except Exception as error:
        if temporary.exists():
            retain_failure(temporary, failed,
                           "FAILED_M2044_VALID825_DO_NOT_CITE", error)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-bundle-manifest-sha256")
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--preflight", action="store_true")
    phase.add_argument("--prepare-only", action="store_true")
    phase.add_argument("--run", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.expected_source_sha256)
    inputs = verify_inputs(contract)
    bundle, output = output_paths(contract)
    if args.preflight:
        print(json.dumps({"status": "PASS_M2044_PREFLIGHT",
                          "source_sha256": args.expected_source_sha256,
                          "bundle_exists": bundle.exists(),
                          "result_exists": output.exists()}, sort_keys=True))
        return 0
    if args.prepare_only:
        prepare_bundle(contract, inputs, bundle, args.expected_source_sha256)
        print("PASS_M2044_DERIVED_BUNDLE_NO_ACCURACY_CLAIM")
        return 0
    require(type(args.expected_bundle_manifest_sha256) is str and
            len(args.expected_bundle_manifest_sha256) == 64,
            "--run requires independently reviewed bundle manifest SHA256")
    run_valid825(contract, inputs, bundle, output,
                 args.expected_source_sha256,
                 args.expected_bundle_manifest_sha256)
    print("PASS_M2044_VALID825_EXECUTION")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M2044Error as error:
        print("FAIL_M2044: " + str(error), file=sys.stderr)
        raise SystemExit(2)
