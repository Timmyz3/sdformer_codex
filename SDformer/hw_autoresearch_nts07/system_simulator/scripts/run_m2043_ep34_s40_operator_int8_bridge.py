#!/usr/bin/env python3
"""Replay the frozen ep34 S40 Conv/ConvTranspose inputs against M2042 INT8.

The r2 bridge reports full-output FP32-vs-QDQ error for 40 C1 Conv3x3 calls and
120 decoder ConvTranspose calls.  Its integer path is deliberately not cuDNN:
it uses explicit zero insertion (for deconvolution), unfold, and TF32-disabled
FP32 GEMM.  Since operands are exact integers and the independent absolute-sum
bound is below 2**24, every product and every possible reordered partial sum is
exact in binary32.  Deterministic probes are also recomputed with NumPy/Python
integer addition.  It is an operator-local numerical experiment: M1458 did
not retain the downstream ATLIF states, so this result cannot claim valid825
AEE or whole-network equivalence.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import time
from typing import Any, Iterable
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
SOURCE = Path(__file__).resolve()
AUTHORITY = HW / "contracts/m2042_ep34_s40_operator_int8_authority_r1_20260902.json"
AUTHORITY_SHA256 = "7f4d09b1d7d9bd3ffafb0e03b5d74100ec0082992518306cc8e81b6939c44cd0"
EXECUTION_CONTRACT = HW / "contracts/m2043r2_ep34_s40_operator_bridge_execution_contract_r1_20260902.json"
EXECUTION_CONTRACT_SHA256 = "92fd28fcdbd4cf6f2e6d8d76a3fa28f9e46acabc775d8d1ff5927337bee324e3"
EXPORT = HW / "results/m2042_ep34_s40_eight_operator_int8_export_r1_20260902"
EXPORT_SHA256SUMS_SHA256 = "519b8621a0c16f67ed33c8c624adc6bbfbc1c4a27224b2812542da3d92fc3881"
EXPORT_OUTER_SHA256 = "da977b9effab3accaff229877bc4d9f0e930f82de1c0833be5c872e63aee142b"
EXPORT_RESULT_SHA256 = "455c9fe7036779b890d4b85911cc42dc47bcb62c9fb6f6a6ce9c28a2c833cf29"
CAPTURE = HW / "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
CAPTURE_ORDERED_SHA256 = "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c"
CAPTURE_SHA256SUMS_SHA256 = "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e"
CAPTURE_OUTER_SHA256 = "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed"
CHECKPOINT = HW / "system_handoff/incoming/m2041_ep34_quant_binding_inputs/checkpoint_epoch34.pth"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M1597_REVIEW = HW / "reviews/m1597_m1590_ep34_c1_same_ledger_cycle_model_result_hammer_r1_20260901"
M1597_REVIEW_SHA256 = "bfa3414ebb69d4a3022182ef7a4989d738c8370a855dff3ce5232c320623c33f"
M1597_SHA256SUMS_SHA256 = "36dc79f7ca76bb98dfe1126aa05c7158dfc460d33215ee39d6fee4edd98e016c"
M1597_OUTER_SHA256 = "8f53a7fa74a2d0245448e822bc35b040df31b3e7d40d46d8ea739e6856d4df8b"
DEFAULT_OUTPUT = HW / "results/m2043r2_ep34_s40_eight_operator_int8_bridge_r1_20260902"
ACC24_MIN = -8_388_608
ACC24_MAX = 8_388_607
EXPECTED_STATE_KEYS = 921


class BridgeError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise BridgeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise BridgeError("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == digest, label + " SHA drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           BridgeError("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def parse_sha256sums(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "malformed SHA256SUMS row")
        digest, name = fields
        name = name.lstrip("*")
        require(name not in output and not Path(name).is_absolute() and
                ".." not in Path(name).parts, "unsafe or duplicate manifest member")
        output[name] = digest
    return output


def verify_outer(root: Path, manifest_sha: str, outer_sha: str, label: str) -> dict[str, str]:
    regular_exact(root / "SHA256SUMS", manifest_sha, label + " SHA256SUMS")
    regular_exact(root / "SHA256SUMS.seal.sha256", outer_sha, label + " outer seal")
    require((root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8").split() ==
            [manifest_sha, "SHA256SUMS"], label + " outer content drift")
    return parse_sha256sums(root / "SHA256SUMS")


def load_c1_conservation() -> dict[str, Any]:
    members = verify_outer(M1597_REVIEW, M1597_SHA256SUMS_SHA256,
                           M1597_OUTER_SHA256, "M1597 review")
    regular_exact(M1597_REVIEW / "review.json", M1597_REVIEW_SHA256,
                  "M1597 review.json")
    require(members.get("review.json") == M1597_REVIEW_SHA256,
            "M1597 review is not sealed")
    review = strict_json(M1597_REVIEW / "review.json")
    require(review.get("status") ==
            "PASS_M1597_M1590_EP34_C1_RESULT_HAMMER_WITH_CAPACITY_SUPERSESSION",
            "M1597 status drift")
    ledger = review.get("ledger_and_capture", {})
    conservation = review.get("conservation_and_traffic", {})
    require(ledger.get("checkpoint_capture_ordered_ledger_chain_pass") is True and
            conservation.get("all_equalities_pass") is True and
            ledger.get("rows") == 51_840_000,
            "M1597 C1 conservation authority incomplete")
    return {
        "authority": "sealed_M1597_ep34_51_84M_row_result",
        "review_sha256": M1597_REVIEW_SHA256,
        "checkpoint_capture_ordered_ledger_chain_pass": True,
        "all_equalities_pass": True,
        "rows": ledger["rows"],
        **{key: conservation[key] for key in (
            "parent_edges", "dead_reads", "dead_forwards", "dead_writes",
            "dead_elisions")},
        "recomputed_by_m2043r2": False,
    }


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    regular_exact(AUTHORITY, AUTHORITY_SHA256, "authority")
    regular_exact(EXECUTION_CONTRACT, EXECUTION_CONTRACT_SHA256,
                  "M2043r2 execution contract")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    regular_exact(CHECKPOINT, CHECKPOINT_SHA256, "checkpoint")
    authority = strict_json(AUTHORITY)
    require(authority.get("schema") == "m2042_ep34_s40_operator_int8_authority_r1_v1",
            "authority schema drift")
    export_members = verify_outer(EXPORT, EXPORT_SHA256SUMS_SHA256,
                                  EXPORT_OUTER_SHA256, "M2042 export")
    regular_exact(EXPORT / "result.json", EXPORT_RESULT_SHA256, "M2042 result")
    require(export_members.get("result.json") == EXPORT_RESULT_SHA256 and
            export_members.get("RUN_COMPLETE.txt") is not None,
            "M2042 export membership incomplete")
    export = strict_json(EXPORT / "result.json")
    require(export.get("status") == "PASS_M2042_EP34_EIGHT_OPERATOR_INT8_WEIGHT_EXPORT" and
            export.get("authority_sha256") == AUTHORITY_SHA256 and
            export.get("checkpoint_sha256") == CHECKPOINT_SHA256,
            "M2042 result identity drift")
    capture_members = verify_outer(CAPTURE, CAPTURE_SHA256SUMS_SHA256,
                                   CAPTURE_OUTER_SHA256, "M1458 capture")
    regular_exact(CAPTURE / "unified_ordered_records.jsonl", CAPTURE_ORDERED_SHA256,
                  "M1458 ordered JSONL")
    require(capture_members.get("unified_ordered_records.jsonl") ==
            CAPTURE_ORDERED_SHA256, "M1458 ordered member not sealed")
    contract = strict_json(EXECUTION_CONTRACT)
    require(contract.get("schema") ==
            "m2043r2_ep34_s40_operator_bridge_execution_contract_r1_v1" and
            contract.get("population", {}).get("total_calls") == 160 and
            contract.get("integer_accumulator_method", {}).get(
                "hidden_convolution_algorithm_forbidden") is True,
            "M2043r2 execution contract drift")
    return authority, export, capture_members, load_c1_conservation()


def selected_records(capture_members: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in (CAPTURE / "unified_ordered_records.jsonl").read_text(
            encoding="utf-8").splitlines():
        row = json.loads(line)
        c1 = row.get("cohort") == "c1" and row.get("category") == "c1_conv3x3"
        decoder = (row.get("cohort") == "decoder" and
                   row.get("category") == "decoder_convtranspose")
        if not (c1 or decoder):
            continue
        payload = row.get("payload", {})
        require(payload.get("retained") is True, "selected payload not retained")
        for name_key, sha_key in (("compressed_fp32", "compressed_sha256"),
                                  ("support_sign", "support_sign_sha256")):
            name = payload.get(name_key)
            digest = payload.get(sha_key)
            require(type(name) is str and capture_members.get(name) == digest,
                    "payload not exact-bound by capture seal")
        output.append(row)
    require(len(output) == 160, "selected operator call population is not 160")
    require(sum(row["category"] == "c1_conv3x3" for row in output) == 40 and
            sum(row["category"] == "decoder_convtranspose" for row in output) == 120,
            "selected family population drift")
    return output


def load_state() -> OrderedDict[str, Any]:
    import torch
    value = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    require(type(value) is dict and set(value) == {"model_state_dict"},
            "checkpoint root drift")
    state = value["model_state_dict"]
    require(type(state) is OrderedDict and len(state) == EXPECTED_STATE_KEYS,
            "checkpoint state drift")
    return state


def load_payload(record: dict[str, Any], expected_word: int) -> np.ndarray:
    payload = record["payload"]
    compressed_path = CAPTURE / payload["compressed_fp32"]
    regular_exact(compressed_path, payload["compressed_sha256"], "compressed activation")
    compressed = compressed_path.read_bytes()
    raw = zlib.decompress(compressed)
    require(hashlib.sha256(raw).hexdigest() == payload["raw_fp32_sha256"],
            "raw activation SHA drift")
    shape = tuple(record["input"]["shape"])
    require(len(raw) == math.prod(shape) * 4, "raw activation extent drift")
    words = np.frombuffer(raw, dtype="<u4")
    valid = (words == 0) | (words == np.uint32(expected_word))
    require(bool(np.all(valid)), "activation is not exact zero/layer-constant")
    require(int(np.count_nonzero(words == expected_word)) == record["input"]["active"],
            "activation population drift")
    return np.frombuffer(raw, dtype="<f4").reshape(shape)


def probe_coordinates(shape: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    n, o, h, w = shape
    candidates = [
        (0, 0, 0, 0),
        (0, o // 2, h // 2, w // 2),
        (n - 1, o - 1, h - 1, w - 1),
        (n // 2, o // 3, h // 3, w // 3),
        (n // 3, (2 * o) // 3, (2 * h) // 3, (2 * w) // 3),
        (n - 1, 0, h // 2, w // 2),
        (0, o - 1, h // 2, w // 2),
        (n // 2, o // 2, h - 1, w - 1),
    ]
    return list(dict.fromkeys(candidates))


def c1_probe(support: np.ndarray, code: np.ndarray,
             coord: tuple[int, int, int, int]) -> tuple[int, int, int]:
    n, output, oy, ox = coord
    accumulator = 0
    prefix_min = 0
    prefix_max = 0
    for source in range(support.shape[1]):
        for ky in range(3):
            iy = oy + ky - 1
            if iy < 0 or iy >= support.shape[2]:
                continue
            for kx in range(3):
                ix = ox + kx - 1
                if ix < 0 or ix >= support.shape[3]:
                    continue
                if support[n, source, iy, ix]:
                    accumulator += int(code[output, source, ky, kx])
                    prefix_min = min(prefix_min, accumulator)
                    prefix_max = max(prefix_max, accumulator)
    return accumulator, prefix_min, prefix_max


def decoder_probe(support: np.ndarray, code: np.ndarray,
                  coord: tuple[int, int, int, int]) -> tuple[int, int, int]:
    n, output, oy, ox = coord
    accumulator = 0
    prefix_min = 0
    prefix_max = 0
    for source in range(support.shape[1]):
        for ky in range(3):
            iy_numerator = oy + 1 - ky
            if iy_numerator % 2:
                continue
            iy = iy_numerator // 2
            if iy < 0 or iy >= support.shape[2]:
                continue
            for kx in range(3):
                ix_numerator = ox + 1 - kx
                if ix_numerator % 2:
                    continue
                ix = ix_numerator // 2
                if ix < 0 or ix >= support.shape[3]:
                    continue
                if support[n, source, iy, ix]:
                    accumulator += int(code[output, source, ky, kx])
                    prefix_min = min(prefix_min, accumulator)
                    prefix_max = max(prefix_max, accumulator)
    return accumulator, prefix_min, prefix_max


def direct_exact_integer_accumulator(support: Any, canonical_code: Any,
                                     family: str, functional: Any) -> Any:
    """Return an exact integer-domain FP32 dot product with no cuDNN conv.

    ``support`` and ``canonical_code`` contain only exactly representable
    integers.  The M2042 absolute-code bound is at most 200,219, so every
    possible partial sum is strictly below 2**24.  With TF32 disabled, an FP32
    GEMM therefore represents every product and every reordered sum exactly.
    """
    require(support.dtype.is_floating_point and canonical_code.dtype.is_floating_point,
            "direct accumulator operands must be FP32 tensors")
    require(support.dtype == canonical_code.dtype,
            "direct accumulator dtype mismatch")
    if family == "c1_conv3x3":
        working = support
        code = canonical_code
    else:
        height, width = support.shape[-2:]
        working = support.new_zeros(
            (support.shape[0], support.shape[1], 2 * height, 2 * width))
        working[:, :, 0::2, 0::2] = support
        code = canonical_code.flip((2, 3))
    patches = functional.unfold(working, kernel_size=3, dilation=1,
                                padding=1, stride=1)
    flattened_code = code.reshape(code.shape[0], -1)
    require(patches.shape[1] == flattened_code.shape[1],
            "unfold/code reduction dimension mismatch")
    result = torch_matmul(flattened_code, patches)
    height, width = working.shape[-2:]
    return result.reshape(support.shape[0], code.shape[0], height, width)


def torch_matmul(left: Any, right: Any) -> Any:
    """Isolated for source review: exact-integer FP32 GEMM, never int/TF32."""
    import torch
    require(left.dtype == torch.float32 and right.dtype == torch.float32,
            "exact integer-domain GEMM operands are not FP32")
    return torch.matmul(left, right)


def metric_state() -> dict[str, Any]:
    return {"calls": 0, "elements": 0, "absolute_error_sum": 0.0,
            "squared_error_sum": 0.0, "maximum_absolute_error": 0.0,
            "dot": 0.0, "reference_squared_sum": 0.0,
            "candidate_squared_sum": 0.0, "observed_accumulator_minimum": 0,
            "observed_accumulator_maximum": 0, "sampled_prefix_minimum": 0,
            "sampled_prefix_maximum": 0, "sampled_integer_oracle_probes": 0,
            "sampled_integer_oracle_mismatches": 0}


def finalize_metric(value: dict[str, Any]) -> dict[str, Any]:
    elements = value["elements"]
    require(elements > 0, "empty metric population")
    denominator = math.sqrt(value["reference_squared_sum"] *
                            value["candidate_squared_sum"])
    if denominator:
        cosine = value["dot"] / denominator
        cosine_case = "nonzero_denominator"
    else:
        both_zero = (value["reference_squared_sum"] == 0.0 and
                     value["candidate_squared_sum"] == 0.0)
        cosine = 1.0 if both_zero else None
        cosine_case = "both_zero" if both_zero else "one_zero_undefined"
    return {
        **value,
        "mae": value["absolute_error_sum"] / elements,
        "rmse": math.sqrt(value["squared_error_sum"] / elements),
        "cosine_similarity": cosine,
        "cosine_case": cosine_case,
        "acc24_observed_final_fits": (ACC24_MIN <= value["observed_accumulator_minimum"] <=
                                      value["observed_accumulator_maximum"] <= ACC24_MAX),
        "sampled_hardware_order_prefix_fits_acc24": (
            ACC24_MIN <= value["sampled_prefix_minimum"] <=
            value["sampled_prefix_maximum"] <= ACC24_MAX),
        "sampled_prefix_is_not_full_population": True,
    }


def add_tensor_metrics(bucket: dict[str, Any], reference: Any, candidate: Any,
                       integer_accumulator: Any) -> None:
    diff = candidate.to(dtype=reference.dtype) - reference
    elements = reference.numel()
    bucket["calls"] += 1
    bucket["elements"] += int(elements)
    bucket["absolute_error_sum"] += float(diff.double().abs().sum().item())
    bucket["squared_error_sum"] += float(diff.double().square().sum().item())
    bucket["maximum_absolute_error"] = max(
        bucket["maximum_absolute_error"], float(diff.abs().max().item()))
    bucket["dot"] += float((reference.double() * candidate.double()).sum().item())
    bucket["reference_squared_sum"] += float(reference.double().square().sum().item())
    bucket["candidate_squared_sum"] += float(candidate.double().square().sum().item())
    acc_min = int(integer_accumulator.min().item())
    acc_max = int(integer_accumulator.max().item())
    require(float(integer_accumulator.sub(integer_accumulator.round()).abs().max().item()) == 0.0,
            "GPU integer accumulator is not integral")
    require(ACC24_MIN <= acc_min <= acc_max <= ACC24_MAX,
            "observed final accumulator exceeds Acc24")
    bucket["observed_accumulator_minimum"] = min(
        bucket["observed_accumulator_minimum"], acc_min)
    bucket["observed_accumulator_maximum"] = max(
        bucket["observed_accumulator_maximum"], acc_max)


def run_bridge(output: Path, device_text: str, expected_source_sha256: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    require(type(expected_source_sha256) is str and len(expected_source_sha256) == 64 and
            all(character in "0123456789abcdef" for character in expected_source_sha256),
            "expected producer source SHA is not lowercase SHA256")
    require(sha256(SOURCE) == expected_source_sha256, "producer source SHA drift")
    authority, export, capture_members, c1_conservation = verify_inputs()
    records = selected_records(capture_members)
    require(not output.exists(), "output already exists; automatic retry forbidden")
    temporary = output.parent / ("." + output.name + ".tmp")
    require(not temporary.exists(), "stale temporary output exists")
    require(device_text == "cuda" and torch.cuda.is_available(),
            "production bridge requires available CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    state = load_state()
    export_rows = {row["module"]: row for row in export["layers"]}
    family_authorities = authority["operator_families"]
    aggregate = metric_state()
    layer_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    start = time.monotonic()
    temporary.mkdir(parents=True)
    try:
        for family in ("c1_conv3x3", "decoder_convtranspose"):
            family_body = family_authorities[family]
            for ordinal, module in enumerate(family_body["modules"]):
                selected = [row for row in records if row["name"] == module]
                expected_calls = 10 if family == "c1_conv3x3" else 30
                require(len(selected) == expected_calls, "module call population drift")
                export_row = export_rows[module]
                code_path = EXPORT / export_row["canonical_code_file"]
                exponent_path = EXPORT / export_row["scale_exponent_file"]
                regular_exact(code_path, export_row["canonical_code_sha256"], "INT8 code")
                regular_exact(exponent_path, export_row["scale_exponent_sha256"], "scale exponent")
                code = np.load(code_path, allow_pickle=False)
                exponent = np.load(exponent_path, allow_pickle=False)
                require(code.dtype == np.int8 and exponent.dtype == np.int16 and
                        code.shape[0] == exponent.size, "exported code/scale geometry drift")
                weight = state[module + ".weight"]
                require(type(weight) is torch.Tensor and weight.dtype == torch.float32 and
                        weight.device.type == "cpu", "raw weight tensor drift")
                raw_weight = weight.detach().contiguous().to(device)
                code_gpu = torch.from_numpy(code.astype(np.float32)).to(device)
                scale_gpu = torch.from_numpy(np.exp2(exponent.astype(np.float32))).to(device)
                layer_metric = metric_state()
                expected_word = family_body["input_layer_constant_ieee754_words"][ordinal]
                expected_alpha = struct.unpack("<f", struct.pack("<I", expected_word))[0]
                require(expected_alpha == family_body["input_layer_constants"][ordinal],
                        "authority alpha word/decimal drift")
                for call_ordinal, record in enumerate(selected):
                    array = load_payload(record, expected_word)
                    reshaped = array.reshape(array.shape[0] * array.shape[1],
                                             array.shape[2], array.shape[3], array.shape[4])
                    support = (reshaped != 0)
                    activation = torch.from_numpy(reshaped.copy()).to(device)
                    support_gpu = torch.from_numpy(support.astype(np.float32)).to(device)
                    with torch.inference_mode():
                        if family == "c1_conv3x3":
                            reference = functional.conv2d(activation, raw_weight,
                                                          bias=None, stride=1, padding=1)
                        else:
                            reference = functional.conv_transpose2d(
                                activation, raw_weight, bias=None, stride=2,
                                padding=1, output_padding=1)
                        int_acc = direct_exact_integer_accumulator(
                            support_gpu, code_gpu, family, functional)
                        candidate = int_acc * scale_gpu.reshape(1, -1, 1, 1) * expected_alpha
                    expected_shape = tuple(family_body["output_shapes"][ordinal]) if (
                        family == "decoder_convtranspose") else tuple(record["input"]["shape"])
                    expected_flat_shape = (expected_shape[0] * expected_shape[1],
                                           expected_shape[2], expected_shape[3], expected_shape[4])
                    require(tuple(reference.shape) == expected_flat_shape and
                            tuple(candidate.shape) == expected_flat_shape,
                            "operator output shape drift")
                    add_tensor_metrics(layer_metric, reference, candidate, int_acc)
                    add_tensor_metrics(aggregate, reference, candidate, int_acc)
                    int_acc_cpu = int_acc.detach().cpu().numpy()
                    support_bool = support.astype(np.bool_, copy=False)
                    probes = probe_coordinates(tuple(int(x) for x in int_acc_cpu.shape))
                    mismatches = 0
                    prefix_min = 0
                    prefix_max = 0
                    for coord in probes:
                        if family == "c1_conv3x3":
                            expected, low, high = c1_probe(support_bool, code, coord)
                        else:
                            expected, low, high = decoder_probe(support_bool, code, coord)
                        observed = int(round(float(int_acc_cpu[coord])))
                        mismatches += int(expected != observed)
                        prefix_min = min(prefix_min, low)
                        prefix_max = max(prefix_max, high)
                    require(mismatches == 0, "sampled integer oracle mismatch")
                    for bucket in (layer_metric, aggregate):
                        bucket["sampled_integer_oracle_probes"] += len(probes)
                        bucket["sampled_integer_oracle_mismatches"] += mismatches
                        bucket["sampled_prefix_minimum"] = min(
                            bucket["sampled_prefix_minimum"], prefix_min)
                        bucket["sampled_prefix_maximum"] = max(
                            bucket["sampled_prefix_maximum"], prefix_max)
                    call_rows.append({
                        "family": family,
                        "module": module,
                        "module_call_ordinal": call_ordinal,
                        "global_sample_id": record["global_sample_id"],
                        "sample_key": record["sample_key"],
                        "input_raw_fp32_sha256": record["payload"]["raw_fp32_sha256"],
                        "output_elements": int(reference.numel()),
                        "observed_accumulator_minimum": int(int_acc_cpu.min()),
                        "observed_accumulator_maximum": int(int_acc_cpu.max()),
                        "integer_oracle_probes": len(probes),
                        "integer_oracle_mismatches": mismatches,
                    })
                    del array, reshaped, support, activation, support_gpu
                    del reference, int_acc, candidate, int_acc_cpu
                layer_rows.append({
                    "family": family,
                    "family_ordinal": ordinal,
                    "module": module,
                    "input_alpha_ieee754_word": expected_word,
                    "input_alpha": expected_alpha,
                    "static_accumulator_bound": export_row["static_accumulator_bound"],
                    "formal_all_prefix_acc24_bound": {
                        "method": "every_prefix_abs_le_per_output_absolute_code_sum",
                        "complete_for_all_support_patterns_and_issue_orders": True,
                        "maximum_absolute_bound": export_row[
                            "static_accumulator_bound"]["per_output_maximum"],
                        "fits_signed_acc24": True,
                    },
                    "metrics": finalize_metric(layer_metric),
                })
                del raw_weight, code_gpu, scale_gpu
                torch.cuda.empty_cache()
        result = {
            "schema": "m2043r2_ep34_s40_eight_operator_int8_bridge_result_r1_v1",
            "status": "PASS_M2043R2_EP34_S40_EIGHT_OPERATOR_INT8_BRIDGE",
            "producer_source_sha256": expected_source_sha256,
            "authority_sha256": AUTHORITY_SHA256,
            "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
            "m2042_export_sha256sums_sha256": EXPORT_SHA256SUMS_SHA256,
            "m1458_ordered_jsonl_sha256": CAPTURE_ORDERED_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(device),
                "tf32": False,
                "cudnn_deterministic": True,
            },
            "population": {"calls": len(call_rows), "c1_calls": 40,
                           "decoder_calls": 120, "operators": 8},
            "integer_exactness": {
                "full_population_method": "explicit_zero_insert_then_unfold_plus_TF32_disabled_FP32_GEMM",
                "hidden_convolution_algorithm_used_for_integer_path": False,
                "global_static_absolute_prefix_bound": max(
                    row["static_accumulator_bound"]["per_output_maximum"]
                    for row in export["layers"]),
                "binary32_exact_integer_limit": 16_777_216,
                "formal_all_population_prefix_bound_complete": True,
                "proof": "integer products and every reordered partial sum are exact because abs(prefix)<=200219<2^24",
            },
            "c1_support_parent_add_sub_conservation": c1_conservation,
            "convtranspose_axis1_sentinel_inherited": export[
                "convtranspose_axis1_sentinel"],
            "aggregate_metrics": finalize_metric(aggregate),
            "layers": layer_rows,
            "calls": call_rows,
            "elapsed_seconds_diagnostic_only": time.monotonic() - start,
            "claim_boundary": {
                "operator_local_FP32_vs_QDQ_metrics": True,
                "full_output_population": True,
                "full_final_accumulator_range": True,
                "sampled_integer_oracle": True,
                "full_integer_final_population": True,
                "formal_all_population_prefix_bound": True,
                "sampled_observed_hardware_order_prefix": True,
                "full_observed_hardware_order_prefix_population": False,
                "C1_support_parent_add_sub_conservation": True,
                "required_operator_bridge_outputs_complete_under_m2043r2_contract": True,
                "valid825_AEE": False,
                "downstream_ATLIF_equivalence": False,
                "whole_network_hardware_order_equivalence": False,
                "hardware_cycles": False,
                "hardware_speedup": False,
                "system_speedup": False,
                "energy": False,
                "PPA": False,
                "paper_result": False,
            },
        }
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        (temporary / "RUN_COMPLETE.txt").write_text(
            "PASS_M2043R2_EP34_S40_EIGHT_OPERATOR_INT8_BRIDGE\n", encoding="utf-8")
        members = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "SHA256SUMS").write_text(
            "\n".join(sha256(path) + "  " + path.name for path in members) + "\n",
            encoding="utf-8")
        manifest_sha = sha256(temporary / "SHA256SUMS")
        (temporary / "SHA256SUMS.seal.sha256").write_text(
            manifest_sha + "  SHA256SUMS\n", encoding="utf-8")
        result_sha = sha256(temporary / "result.json")
        outer_sha = sha256(temporary / "SHA256SUMS.seal.sha256")
        os.replace(temporary, output)
        published_members = verify_outer(output, manifest_sha, outer_sha,
                                         "published M2043r2")
        require(published_members.get("result.json") == result_sha and
                sha256(output / "result.json") == result_sha,
            "published result readback failed")
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        authority, export, members, conservation = verify_inputs()
        records = selected_records(members)
        print(json.dumps({"status": "PASS_M2043_PREFLIGHT", "records": len(records),
                          "operators": len(export["layers"]),
                          "c1_conservation_rows": conservation["rows"],
                          "checkpoint_sha256": authority["identity"]["checkpoint"]["sha256"]},
                         sort_keys=True))
        return 0
    require(args.output.resolve() == DEFAULT_OUTPUT.resolve(),
            "production output must be the canonical M2043r2 directory")
    require(args.expected_source_sha256 is not None,
            "production requires --expected-source-sha256")
    result = run_bridge(args.output, args.device, args.expected_source_sha256)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "calls": result["population"]["calls"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError as error:
        print("FAIL_M2043: " + str(error), file=sys.stderr)
        raise SystemExit(2)
