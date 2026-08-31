#!/usr/bin/env python3
"""M1525 source-only ep34 decoder multi-baseline replay specification.

This additive source closes a design ambiguity in M1105DR2: a single typed-K8
schedule cannot yield a speedup.  It defines four configurations over one
memory/port envelope and validates the future M1521 positive-plane manifest
plus the M1514 ep34 FP32 weight identity.  It deliberately does not schedule a
production transaction, read a production namespace, or authorize a result.

The product-capture configuration is fail-closed until a separately hammered
ep34 INT8 weight-byte manifest, quantization miter and Acc24 proof exist.
"""
import argparse
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA = "m1525_ep34_decoder_multibaseline_replay_successor_source_r1_v1"
STATUS = "SOURCE_ONLY__FOUR_CONFIG_LADDER__PRODUCTION_FALSE"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
OLD_EP35_CHECKPOINT_SHA256 = "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
OLD_D1_WORD = 0x3F7FFFB3
SCALE_WORDS = (0x3F7FFD6B, 0x3F7FFFA0, 0x3F800000, 0x3F800000)
MODULES = tuple("sttmultires_unet.decoders.{}.deconv.0".format(i) for i in range(4))
SHAPES = (
    (10, 1, 1536, 15, 20),
    (10, 1, 770, 30, 40),
    (10, 1, 386, 60, 80),
    (10, 1, 194, 120, 160),
)
WEIGHT_SHAPES = (
    (1536, 384, 3, 3),
    (770, 192, 3, 3),
    (386, 96, 3, 3),
    (194, 96, 3, 3),
)
WEIGHT_SHA256 = (
    "cb1a90a4ff33622024b43ee6b15a3409e2567ea1e7b626715f40cf8a4fbfd83b",
    "35a9214e9fbc2e4e271beea74c4f329c12d6c072cda9252eaae350dd404a51cb",
    "75f9921f3cd9786ece78247115dd07bdda425b4f6e068d43936c884c611d3ef7",
    "6a42dabae358d0048aa46c609c9cb633f1e8d0479e4628e4f85c21e00835ea4e",
)
CONFIGS = (
    "DENSE_TYPED_K8",
    "BIT_EQUAL_SERVICE_K1X8",
    "BIT_TYPED_K8",
    "PRODUCT_CAPTURE_TYPED_K8",
)
COMMON_RESOURCE = {
    "lanes": 96,
    "accumulator_bits": 24,
    "clock_ns": 3.0,
    "external_bytes_per_cycle": 192,
    "onchip_sram_bytes_macro_rounded": 245760,
    "partitions": {"weight_bytes": 13824, "psum_bytes": 221184,
                   "descriptor_control_bytes": 8192,
                   "reserved_unallocated_bytes": 2560},
    "ports": {
        "weight": {"banks": 8, "mode": "1R1W", "row_bytes": 16,
                   "read_latency_cycles": 4, "initiation_interval": 1,
                   "outstanding_per_bank": 8},
        "psum": {"banks": 6, "mode": "1RW", "row_bytes": 48,
                 "read_latency_cycles": 2, "write_latency_cycles": 1,
                 "initiation_interval": 1, "outstanding_per_bank": 8},
        "external": {"banks": 1, "mode": "1RW", "row_bytes": 192,
                     "read_latency_cycles": 32, "write_latency_cycles": 3,
                     "initiation_interval": 1, "outstanding_per_bank": 16},
        "compute": {"contexts": 1, "row_bytes": 288,
                    "latency_cycles": 1, "initiation_interval": 1},
    },
}


class M1525Error(RuntimeError):
    pass


def product(values: Sequence[int]) -> int:
    """Python-3.6-compatible product for shape tuples."""
    total = 1
    for value in values:
        total *= value
    return total


def require(value: bool, message: str) -> None:
    if not value:
        raise M1525Error(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def lowercase_sha(value: Any, label: str) -> str:
    require(type(value) is str and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), label + " SHA drift")
    return value


def validate_positive_plane_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    require(type(value) is dict, "positive-plane manifest is not exact object")
    capture = value.get("capture")
    population = value.get("population")
    rows = value.get("records")
    require(type(capture) is dict and
            capture.get("checkpoint_sha256") == CHECKPOINT_SHA256,
            "ep34 capture/checkpoint identity drift")
    require(type(population) is dict and population.get("samples") == 30 and
            population.get("calls") == 120 and population.get("modules") == 4 and
            population.get("positive_plane_files") == 120 and
            population.get("negative_plane_files") == 0,
            "ep34 positive-plane population drift")
    require(type(rows) is list and len(rows) == 120,
            "ep34 positive-plane rows are not 120")
    paths = set()
    for ordinal, row in enumerate(rows):
        module = ordinal % 4
        sample = 10 + ordinal // 4
        path = "payloads/c{:03d}_s{:02d}_d{}.positive.le.bitpack".format(
            ordinal, sample, module)
        require(type(row) is dict and row.get("global_call_ordinal") == ordinal and
                row.get("global_sample_id") == sample and
                row.get("replay_sample_ordinal") == sample - 10 and
                row.get("module_ordinal") == module and
                row.get("module") == MODULES[module] and
                tuple(row.get("shape", ())) == SHAPES[module] and
                row.get("elements") == product(SHAPES[module]) and
                row.get("plane_bytes") == (product(SHAPES[module]) + 7) // 8 and
                row.get("positive_output") == path,
                "ep34 positive-plane call/order/path drift")
        require(row.get("layer_scale_word_uint32") == SCALE_WORDS[module] and
                row.get("numeric_encoding") == (
                    "bit_times_layer_constant" if module in (0, 1) else "exact_binary") and
                row.get("weight_folding") is False and
                row.get("normalized") is False and row.get("coerced") is False and
                row.get("negative_plane_output") is None and
                row.get("negative_plane_all_zero") is True,
                "ep34 scale/encoding/no-fold semantics drift")
        lowercase_sha(row.get("positive_output_sha256"), "positive output")
        require(path not in paths, "positive-plane path duplicate")
        paths.add(path)
    require(OLD_D1_WORD not in SCALE_WORDS and len(paths) == 120,
            "old D1 threshold leaked into ep34 manifest")
    return {"checkpoint_sha256": CHECKPOINT_SHA256, "calls": 120,
            "manifest_projection_sha256": digest(rows),
            "scale_words": list(SCALE_WORDS)}


def validate_weight_identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    require(type(value) is dict and value.get("status") ==
            "PASS_M1514_SOURCE_ONLY_DECODER_WEIGHT_IDENTITY__NO_EXPORT",
            "M1514 weight audit status drift")
    checkpoint = value.get("checkpoint")
    rows = value.get("weights")
    require(type(checkpoint) is dict and checkpoint.get("sha256") == CHECKPOINT_SHA256 and
            checkpoint.get("root_keys") == ["model_state_dict"],
            "ep34 weight checkpoint identity drift")
    require(type(rows) is list and len(rows) == 4, "decoder weight rows are not four")
    for ordinal, row in enumerate(rows):
        require(type(row) is dict and row.get("module_ordinal") == ordinal and
                row.get("module") == MODULES[ordinal] and
                tuple(row.get("shape", ())) == WEIGHT_SHAPES[ordinal] and
                row.get("dtype") == "torch.float32" and
                row.get("layout") == "C_ORDER_CONTIGUOUS" and
                row.get("byte_order") == "little" and
                row.get("content_sha256") == WEIGHT_SHA256[ordinal] and
                row.get("content_bytes") == product(WEIGHT_SHAPES[ordinal]) * 4 and
                row.get("bias") is None,
                "ep34 decoder FP32 weight identity drift")
    return {"checkpoint_sha256": CHECKPOINT_SHA256, "layers": 4,
            "fp32_content_bytes": sum(product(shape) * 4 for shape in WEIGHT_SHAPES),
            "weight_projection_sha256": digest(rows)}


def configuration_ladder() -> List[Dict[str, Any]]:
    common = {"resource_manifest_sha256": digest(COMMON_RESOURCE),
              "commit_policy": "DENSE_ALL_OUTPUT_SITES_SAME_ADDRESS_ORDER",
              "d0_d1_value_policy": "TYPED_LAYER_CONSTANT_NO_WEIGHT_FOLD",
              "d2_d3_value_policy": "EXACT_BINARY",
              "production": False}
    return [
        {"name": CONFIGS[0], **common, "activation_policy": "STRUCTURAL_DENSE",
         "source_issue": "BANK_UNIQUE_UP_TO_8", "role": "dense_denominator",
         "frontend_area_matched": True, "product_bridge_required": False},
        {"name": CONFIGS[1], **common, "activation_policy": "EXACT_POSITIVE_BITPLANE",
         "source_issue": "EIGHT_INDEPENDENT_K1_EQUAL_SERVICE",
         "role": "bit_sparsity_equal_service_denominator",
         "frontend_area_matched": False, "product_bridge_required": False},
        {"name": CONFIGS[2], **common, "activation_policy": "EXACT_POSITIVE_BITPLANE",
         "source_issue": "SHARED_TYPED_BANK_UNIQUE_K8", "role": "typed_k8_candidate",
         "frontend_area_matched": True, "product_bridge_required": False},
        {"name": CONFIGS[3], **common, "activation_policy": "EXACT_T10_PATTERN_CLASSES",
         "source_issue": "PARENT_SUM_THEN_SHARED_TYPED_K8",
         "role": "product_capture_candidate", "frontend_area_matched": True,
         "product_bridge_required": True},
    ]


def build_replay_plan(plane_manifest: Mapping[str, Any],
                      weight_audit: Mapping[str, Any],
                      quantized_weight_bridge: Optional[Mapping[str, Any]] = None,
                      request_production: bool = False) -> Dict[str, Any]:
    planes = validate_positive_plane_manifest(plane_manifest)
    weights = validate_weight_identity(weight_audit)
    bridge_ready = False
    if quantized_weight_bridge is not None:
        required = {"checkpoint_sha256", "four_int8_payload_sha256",
                    "quantization_policy", "fp32_to_int8_miter",
                    "acc24_bound", "independent_hammer_pass"}
        require(type(quantized_weight_bridge) is dict and
                set(quantized_weight_bridge) == required and
                quantized_weight_bridge["checkpoint_sha256"] == CHECKPOINT_SHA256 and
                type(quantized_weight_bridge["four_int8_payload_sha256"]) is list and
                len(quantized_weight_bridge["four_int8_payload_sha256"]) == 4 and
                all(lowercase_sha(v, "INT8 payload") for v in
                    quantized_weight_bridge["four_int8_payload_sha256"]) and
                quantized_weight_bridge["fp32_to_int8_miter"] is True and
                quantized_weight_bridge["acc24_bound"] is True and
                quantized_weight_bridge["independent_hammer_pass"] is True,
                "ep34 INT8 weight bridge incomplete")
        bridge_ready = True
    require(request_production is False,
            "M1525 is source-only and cannot authorize production")
    ladder = configuration_ladder()
    return {
        "schema": SCHEMA, "status": STATUS,
        "identity": {"planes": planes, "weights": weights},
        "common_resource": COMMON_RESOURCE,
        "configurations": ladder,
        "readiness": {
            "dense_bit_k1x8_k8_source_plan_ready": True,
            "product_capture_ready": bridge_ready,
            "product_blocker": None if bridge_ready else
                "EP34_INT8_WEIGHT_BYTES_PLUS_MITER_PLUS_ACC24_PROOF_MISSING",
            "production": False,
        },
        "mandatory_charges": [
            "input_descriptor", "weight_read", "parent_scratch_read_write",
            "pattern_class_build", "psum_read_write", "dense_output_commit",
            "external_bandwidth", "bank_and_dependency_stalls"],
        "old_m1105dr2_reuse": {
            "allowed": False,
            "old_checkpoint_sha256": OLD_EP35_CHECKPOINT_SHA256,
            "old_d1_word_uint32": OLD_D1_WORD,
            "reasons": ["checkpoint_and_activity_changed", "D0_is_not_binary_one",
                        "D1_layer_constant_changed", "ep34_weights_not_bound",
                        "single_configuration_has_no_speedup_denominator",
                        "old_cycles_and_traffic_are_diagnostic_only"],
        },
        "claim_boundary": {"source_only": True, "production": False,
            "transactions": False, "cycles": False, "traffic": False,
            "speedup": False, "system_speedup": False, "energy": False,
            "rtl": False, "eda": False, "ppa": False, "table_a": False},
    }


def validate_comparator_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(type(rows) is list and [row.get("configuration") for row in rows] ==
            list(CONFIGS), "four-configuration row order/population drift")
    for key in ("resource_manifest_sha256", "commit_address_hash",
                "population_manifest_sha256", "checkpoint_sha256"):
        require(len({row.get(key) for row in rows}) == 1,
                "comparator {} differs across configurations".format(key))
    require(all(row.get("checkpoint_sha256") == CHECKPOINT_SHA256 for row in rows),
            "comparator checkpoint is not ep34")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    require(args.describe, "M1525 only exposes --describe; production is forbidden")
    print(json.dumps({"schema": SCHEMA, "status": STATUS,
        "configurations": configuration_ladder(),
        "common_resource": COMMON_RESOURCE,
        "production": False}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
