#!/usr/bin/env python3
"""Source-only gate for a future ep34 decoder INT8 numeric bridge.

M1526 deliberately does not quantize or export weights.  It proves that the
available authorities do *not* select one legal decoder quantization rule:
the old M61 rule is bound to another checkpoint and a prediction-head Conv2d,
while the repository-wide hardware specification selects a different scale
granularity and is not bound by the selected ep34 deployment configuration.
Consequently M1525 must not consume an invented INT8 payload.  The emitted
machine-readable handoff states the minimum algorithm-side work that can
close this gate in a separately reviewed successor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
SOURCE = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1526_ep34_decoder_int8_numeric_bridge_gate_source.py"
CONTRACT = HW / "contracts/m1526_ep34_decoder_int8_numeric_bridge_gate_source_contract_r1_20260831.json"
M1514 = HW / "contracts/m1514_ep34_decoder_weight_identity_export_source_contract_r1_20260831.json"
M1514_SHA256 = "178aadda75ff7a22b0f958a78e28a1ab9260e4134fd8bf2bdca6aeaad2bea7a0"
M1458 = HW / "contracts/m1458_m1434_motion_ep34_live93_production_runner_source_contract_r1_20260831.json"
M1458_SHA256 = "ae3fa89fe0517578e2ef475c675f1c26160d82fc6356e51b54f79e42960bc0b6"
M61 = HW / "contracts/m61_prediction_head_int8_numeric_bridge_contract_r1_20260823.json"
M61_SHA256 = "24133f6b0e1a3fee3635487bc5a099a46f1a3bb6dd86b24c15d171d43c29f728"
QUANT_SPEC = ROOT / "configs/hw/quant_spec.yaml"
QUANT_SPEC_SHA256 = "f95c1edb66f9ec5c2c95d1484d9791a9267b2481ab8658f7ff604a7fb55aa305"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1526_ep34_decoder_int8_numeric_bridge_gate_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__FAIL_CLOSED_NO_AUTHORIZED_EP34_DECODER_INT8_RULE"
ADMISSION_STATUS = (
    "FAIL_CLOSED_M1526_NO_AUTHORIZED_EP34_DECODER_INT8_RULE__"
    "M1525_INT8_REPLAY_BLOCKED"
)
EP34_CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
EP34_CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
M61_CHECKPOINT_SHA256 = "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
LAYER_SHAPES = (
    (1536, 384, 3, 3),
    (770, 192, 3, 3),
    (386, 96, 3, 3),
    (194, 96, 3, 3),
)
LAYER_WEIGHT_SHA256 = (
    "cb1a90a4ff33622024b43ee6b15a3409e2567ea1e7b626715f40cf8a4fbfd83b",
    "35a9214e9fbc2e4e271beea74c4f329c12d6c072cda9252eaae350dd404a51cb",
    "75f9921f3cd9786ece78247115dd07bdda425b4f6e068d43936c884c611d3ef7",
    "6a42dabae358d0048aa46c609c9cb633f1e8d0479e4628e4f85c21e00835ea4e",
)
CLAIM_BOUNDARY = {
    "source_only": True,
    "read_only": True,
    "quantized_weight_payload_written": False,
    "quantization_rule_admitted": False,
    "m1525_int8_replay_admitted": False,
    "production": False,
    "gpu": False,
    "remote": False,
    "eda": False,
    "cycles": False,
    "traffic": False,
    "speedup": False,
    "system_speedup": False,
    "energy": False,
    "ppa": False,
    "table_a": False,
    "paper_result": False,
}


class M1526Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1526Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as error:
        raise M1526Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == digest, label + " SHA drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M1526Error("nonfinite JSON token: " + token)),
    )
    require(type(value) is dict, "JSON root is not an exact object")
    return value


def accumulator_metadata_requirements() -> dict[str, Any]:
    return {
        "accumulator_storage_bits_candidate_not_admitted": 24,
        "bias_policy": "ABSENT_IN_ALL_FOUR_M1514_DECODER_LAYERS",
        "input_contract_must_be_bound_per_layer": {
            "integer_code_set": [0, 1],
            "scale_float32_bits_and_sha256_required": True,
            "zero_point_must_be_zero": True,
            "mixed_parent_scale_or_nonbinary_input_is_rejected": True,
        },
        "weight_tensor_layout": "CONVTRANSPOSE2D_CIN_COUT_KY_KX",
        "per_output_axis_if_selected": 1,
        "per_tensor_scale_count_if_selected": 1,
        "per_output_scale_counts_if_selected": [384, 192, 96, 96],
        "weight_code_requirements": {
            "dtype": "SIGNED_INT8",
            "narrow_range": [-127, 127],
            "zero_point": 0,
            "rounding_and_tie_rule_must_be_explicit": True,
            "code_payload_sha256_per_layer_required": True,
            "scale_float32_bitpattern_sha256_per_layer_required": True,
        },
        "reachable_polyphase_taps_for_k3_s2_p1_op1": {
            "output_even_even": [[1, 1]],
            "output_even_odd": [[1, 0], [1, 2]],
            "output_odd_even": [[0, 1], [2, 1]],
            "output_odd_odd": [[0, 0], [0, 2], [2, 0], [2, 2]],
            "maximum_spatial_taps_per_input_channel": 4,
        },
        "proof_required_per_layer_output_channel_and_polyphase": {
            "lower_bound": "sum(all negative reachable INT8 codes)",
            "upper_bound": "sum(all positive reachable INT8 codes)",
            "sequential_prefix_bound": "every issued partial sum must remain in declared Acc24",
            "observed_min_max": "measure all exact M1458 decoder payload calls",
            "overflow_or_saturation_count_must_equal": 0,
            "minimum_signed_bits_and_headroom_bits_required": True,
        },
        "old_accumulator_numbers_reusable": False,
        "reason_old_numbers_forbidden": (
            "M61 covers a 2x96 Conv2d head under another checkpoint; M519/C2 "
            "Acc24 is a storage/protocol choice, not an ep34 decoder range proof."
        ),
    }


def algorithm_handoff() -> dict[str, Any]:
    return {
        "handoff_schema": "m1526_algorithm_ep34_decoder_int8_closure_request_v1",
        "target_checkpoint_sha256": EP34_CHECKPOINT_SHA256,
        "target_config_sha256": EP34_CONFIG_SHA256,
        "required_actions_in_order": [
            {
                "id": "Q1_CONFIG_AUTHORITY",
                "action": (
                    "Return the exact selected ep34 configuration bytes and an "
                    "algorithm-owned manifest that explicitly authorizes one decoder "
                    "weight PTQ rule. Do not infer it from M61 or the generic spec."
                ),
            },
            {
                "id": "Q2_DETERMINISTIC_PTQ",
                "action": (
                    "For exactly four M1514 ConvTranspose weights, generate INT8 codes, "
                    "float32 scale bit patterns, and a dequantized checkpoint using only "
                    "the Q1 rule; preserve every non-decoder tensor byte-for-byte."
                ),
            },
            {
                "id": "Q3_KERNEL_MITER",
                "action": (
                    "On the exact 40-sample M1458 schedule, compare FP32 and quantized-"
                    "dequantized ConvTranspose outputs and downstream ATLIF events per "
                    "layer; report count/MAE/RMSE/max, support XOR count/rate, and all "
                    "nonfinite/overflow/saturation counts."
                ),
            },
            {
                "id": "Q4_OFFICIAL_ACCURACY",
                "action": (
                    "Run paired FP32 and four-layer-decoder-PTQ inference with identical "
                    "official samples, masks, flow scaling, and evaluator; report both "
                    "AEE values and their signed/absolute delta."
                ),
            },
        ],
        "minimum_return_artifacts": {
            "selected_config": {
                "sha256": EP34_CONFIG_SHA256,
                "bytes_required": True,
            },
            "quantization_authority_manifest": {
                "checkpoint_sha256": EP34_CHECKPOINT_SHA256,
                "exact_rule_fields": [
                    "granularity", "weight_axis", "scale_equation", "scale_dtype",
                    "rounding_tie_rule", "clamp_range", "zero_point",
                    "activation_code_and_scale_contract", "accumulator_bits",
                ],
            },
            "decoder_int8_identity_manifest": {
                "layers": 4,
                "source_float_weight_sha256": list(LAYER_WEIGHT_SHA256),
                "required_per_layer": [
                    "int8_payload_sha256", "scale_bitpattern_sha256",
                    "dequantized_weight_sha256", "shape", "axis",
                    "integer_min", "integer_max",
                ],
            },
            "paired_kernel_miter_json": True,
            "paired_official_accuracy_json": True,
            "portable_dequantized_checkpoint_sha256": True,
            "load_audit_missing_unexpected_must_be": [0, 0],
        },
        "numeric_admission_gate": {
            "source_of_candidate_tolerance": (
                "configs/hw/quant_spec.yaml::interfaces.max_epe_delta"
            ),
            "candidate_tolerance_value": 0.02,
            "tolerance_is_not_ep34_authority_until_Q1_binds_it": True,
            "paired_official_absolute_aee_delta_max": 0.02,
            "paired_population_and_evaluator_identity_required": True,
            "kernel_and_atlif_mismatch_metrics_required_not_silently_zeroed": True,
            "all_nonfinite_overflow_saturation_counts_must_equal": 0,
            "passing_S40_kernel_miter_is_not_accuracy_substitute": True,
        },
        "release_conditions": [
            "Q1 through Q4 are independently sealed and exact-bind ep34 checkpoint/config",
            "one and only one quantization granularity is authorized",
            "all Acc24 phase/channel bounds and observed ranges pass",
            "paired official absolute AEE delta is at most 0.02",
            "a fresh independent hardware-side hammer admits the returned artifacts",
        ],
        "if_any_field_missing": "KEEP_M1525_INT8_REPLAY_BLOCKED",
    }


def build_admission() -> dict[str, Any]:
    regular_exact(M1514, M1514_SHA256, "M1514 decoder weight identity contract")
    regular_exact(M1458, M1458_SHA256, "M1458 production source contract")
    regular_exact(M61, M61_SHA256, "M61 old numeric bridge contract")
    regular_exact(QUANT_SPEC, QUANT_SPEC_SHA256, "generic hardware quant spec")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    m1514 = strict_json(M1514)
    m1458 = strict_json(M1458)
    m61 = strict_json(M61)

    require(m1514.get("checkpoint", {}).get("sha256") == EP34_CHECKPOINT_SHA256,
            "M1514 ep34 checkpoint identity drift")
    require(m1458.get("identity", {}).get("checkpoint_sha256") ==
            EP34_CHECKPOINT_SHA256 and
            m1458.get("identity", {}).get("config_sha256") == EP34_CONFIG_SHA256,
            "M1458 selected checkpoint/config identity drift")
    rows = m1514.get("weight_identities")
    require(type(rows) is list and len(rows) == 4, "M1514 layer population drift")
    require([tuple(row.get("shape", [])) for row in rows] == list(LAYER_SHAPES) and
            [row.get("content_sha256") for row in rows] == list(LAYER_WEIGHT_SHA256) and
            all(row.get("bias") is None for row in rows),
            "M1514 weight identity/shape/bias drift")

    old = m61.get("identity", {})
    require(old.get("checkpoint_sha256") == M61_CHECKPOINT_SHA256 and
            old.get("checkpoint_sha256") != EP34_CHECKPOINT_SHA256 and
            old.get("module_name") == "sttmultires_unet.preds.3.conv.0",
            "M61 mismatch evidence drift")
    require(m61.get("quantization", {}).get("scale", "").endswith(
        "per output channel"), "M61 per-output rule evidence drift")
    quant_text = QUANT_SPEC.read_text(encoding="utf-8")
    require("scale_mode: per_tensor" in quant_text and
            "accumulator_bits: 24" in quant_text and
            "max_epe_delta: 0.02" in quant_text,
            "generic quant spec evidence drift")

    return {
        "schema": SCHEMA,
        "status": ADMISSION_STATUS,
        "decision": {
            "ep34_fp32_decoder_weight_identity": "ADMITTED_BY_M1514",
            "ep34_decoder_int8_rule": "NOT_AUTHORIZED",
            "m1525_fp32_or_bit_opportunity_study": "OUTSIDE_M1526_SCOPE",
            "m1525_int8_or_k8_weighted_replay": "BLOCKED",
        },
        "blocking_findings": [
            {
                "id": "B1_SELECTED_CONFIG_BYTES_NOT_LOCAL_AUTHORITY",
                "detail": (
                    "Only SHA256 of the selected ep34 config is locally bound; its bytes "
                    "and an explicit decoder quantization authorization are absent."
                ),
            },
            {
                "id": "B2_M61_IDENTITY_AND_OBJECT_MISMATCH",
                "detail": (
                    "M61 is bound to a different checkpoint and a 2x96 prediction-head "
                    "Conv2d, so neither its scales nor its accumulator proof may migrate."
                ),
            },
            {
                "id": "B3_GRANULARITY_CONFLICT",
                "detail": (
                    "The generic repository spec says per_tensor; M61 says per-output. "
                    "Neither is selected by an exact ep34 deployment authority."
                ),
            },
            {
                "id": "B4_NUMERIC_AND_AEE_CLOSURE_ABSENT",
                "detail": (
                    "No ep34 four-decoder FP32/INT8 kernel miter, downstream event "
                    "comparison, accumulator range proof, or paired official AEE exists."
                ),
            },
        ],
        "candidate_rules_not_authority": {
            "generic_repository_spec": {
                "granularity": "PER_TENSOR",
                "rounding": "NEAREST_TIE_RULE_UNSPECIFIED",
                "accumulator_bits": 24,
                "zero_point": False,
            },
            "m61_prediction_head_rule": {
                "granularity": "PER_OUTPUT",
                "rounding": "RNE_TIES_TO_EVEN",
                "narrow_range": [-127, 127],
                "reusable_for_ep34_decoder": False,
            },
        },
        "accumulator_metadata_required": accumulator_metadata_requirements(),
        "algorithm_handoff": algorithm_handoff(),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def validate_source_policy() -> dict[str, Any]:
    policy = strict_json(CONTRACT)
    require(policy.get("schema") == SCHEMA and
            policy.get("status") == SOURCE_STATUS,
            "M1526 contract schema/status drift")
    require(policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "M1526 source/test identity drift")
    admission = build_admission()
    handoff = admission["algorithm_handoff"]
    handoff_summary = {
        "handoff_schema": handoff["handoff_schema"],
        "target_checkpoint_sha256": handoff["target_checkpoint_sha256"],
        "target_config_sha256": handoff["target_config_sha256"],
        "required_action_ids": [row["id"] for row in
                                handoff["required_actions_in_order"]],
        "paired_official_absolute_aee_delta_max": handoff[
            "numeric_admission_gate"]["paired_official_absolute_aee_delta_max"],
        "if_any_field_missing": handoff["if_any_field_missing"],
    }
    accumulator = admission["accumulator_metadata_required"]
    accumulator_summary = {
        "candidate_bits_not_admitted": accumulator[
            "accumulator_storage_bits_candidate_not_admitted"],
        "weight_layout": accumulator["weight_tensor_layout"],
        "per_output_axis_if_selected": accumulator["per_output_axis_if_selected"],
        "per_output_scale_counts_if_selected": accumulator[
            "per_output_scale_counts_if_selected"],
        "maximum_spatial_taps_per_input_channel": accumulator[
            "reachable_polyphase_taps_for_k3_s2_p1_op1"][
                "maximum_spatial_taps_per_input_channel"],
        "old_accumulator_numbers_reusable": accumulator[
            "old_accumulator_numbers_reusable"],
    }
    require(policy.get("expected_admission_status") == ADMISSION_STATUS and
            policy.get("claim_boundary") == CLAIM_BOUNDARY and
            policy.get("algorithm_handoff_summary") == handoff_summary and
            policy.get("accumulator_metadata_summary") == accumulator_summary,
            "M1526 decision/handoff/accumulator policy drift")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--emit-handoff", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    validate_source_policy()
    if args.source_self_check:
        print("PASS_M1526_SOURCE_SELF_CHECK__FAIL_CLOSED_NO_INT8_EXPORT")
    else:
        print(json.dumps(build_admission(), indent=2, sort_keys=True,
                         allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1526Error as error:
        print("M1526_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
