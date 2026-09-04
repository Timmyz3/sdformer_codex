#!/usr/bin/env python3
"""Export the frozen ep34 eight-operator dyadic INT8 weight set.

This is a deliberately narrow one-shot producer.  It authenticates the M2042
authority, M2041 input bundle, M1458 capture identity, and the protected
docs/359 digest before loading the selected checkpoint on CPU.  It exports
both canonical [O,I,Ky,Kx] and hardware [I,Ky,Kx,O] code arrays, per-output
base-2 scale exponents, and static signed-Acc24 bounds.

It does not execute an activation replay and therefore cannot claim AEE,
downstream ATLIF equivalence, cycles, speedup, energy, or PPA.
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
import sys
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
AUTHORITY = HW / "contracts/m2042_ep34_s40_operator_int8_authority_r1_20260902.json"
AUTHORITY_SHA256 = "7f4d09b1d7d9bd3ffafb0e03b5d74100ec0082992518306cc8e81b6939c44cd0"
DEFAULT_OUTPUT = HW / "results/m2042_ep34_s40_eight_operator_int8_export_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_STATE_KEYS = 921
ACC24_MIN = -8_388_608
ACC24_MAX = 8_388_607

C1_WEIGHT_SHAS = (
    "e1377479fcdfcb946b5f6d8f0344140f41953224cb999f8506d6f6e860c692c0",
    "f4620a355f6a13bd29cecb05fd3d31d5f3f40f6a1dd874018e3a345790ba32d0",
    "714d4e02223887174665ec4e685c6cc1854535d012a50de2974f7b8537356677",
    "58b96e585075b6da5d9ed0fdeef60a40063d2c80fa6c27894ae8d327f1be687e",
)
DECODER_WEIGHT_SHAS = (
    "cb1a90a4ff33622024b43ee6b15a3409e2567ea1e7b626715f40cf8a4fbfd83b",
    "35a9214e9fbc2e4e271beea74c4f329c12d6c072cda9252eaae350dd404a51cb",
    "75f9921f3cd9786ece78247115dd07bdda425b4f6e068d43936c884c611d3ef7",
    "6a42dabae358d0048aa46c609c9cb633f1e8d0479e4628e4f85c21e00835ea4e",
)


class ExportError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ExportError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ExportError("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
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
                           ExportError("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root must be an object")
    return value


def resolve_bound(path_text: str) -> Path:
    path = (ROOT / path_text).resolve()
    require(path.is_relative_to(ROOT.resolve()), "authority path escapes repository")
    return path


def verify_outer_seal(root: Path, expected_manifest: str,
                      expected_outer: str, label: str) -> None:
    regular_exact(root / "SHA256SUMS", expected_manifest, label + " SHA256SUMS")
    regular_exact(root / "SHA256SUMS.seal.sha256", expected_outer,
                  label + " outer seal")
    require((root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8").split() ==
            [expected_manifest, "SHA256SUMS"], label + " outer seal content drift")


def verify_authority() -> dict[str, Any]:
    regular_exact(AUTHORITY, AUTHORITY_SHA256, "M2042 authority")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    authority = strict_json(AUTHORITY)
    require(authority.get("schema") ==
            "m2042_ep34_s40_operator_int8_authority_r1_v1",
            "authority schema drift")
    require(authority.get("status") ==
            "FROZEN_EP34_OPERATOR_PTQ_AUTHORITY__SOURCE_REVIEW_REQUIRED",
            "authority status drift")
    identity = authority["identity"]
    m2041 = resolve_bound(identity["m2041_input_manifest"]["path"]).parent
    verify_outer_seal(m2041, identity["m2041_sha256sums"]["sha256"],
                      identity["m2041_sha256sums"]["outer_seal_file_sha256"],
                      "M2041")
    regular_exact(resolve_bound(identity["m2041_input_manifest"]["path"]),
                  identity["m2041_input_manifest"]["sha256"], "M2041 manifest")
    for key in ("checkpoint", "selected_config", "standard_valid825_profile"):
        regular_exact(resolve_bound(identity[key]["path"]), identity[key]["sha256"],
                      "M2041 " + key)
    capture = resolve_bound(identity["m1458_capture"]["path"])
    capture_id = identity["m1458_capture"]
    verify_outer_seal(capture, capture_id["sha256sums_sha256"],
                      capture_id["outer_seal_file_sha256"], "M1458")
    regular_exact(capture / "manifest.json", capture_id["manifest_sha256"],
                  "M1458 manifest")
    regular_exact(capture / "unified_ordered_records.jsonl",
                  capture_id["ordered_jsonl_sha256"], "M1458 ordered JSONL")
    quant = authority["weight_quantization"]
    require(quant == {
        "dtype": "signed_int8",
        "code_range": [-127, 127],
        "reserved_code": -128,
        "zero_point": 0,
        "granularity": "per_output_channel",
        "scale_rule": "scale[o] = 2**ceil(log2(max(abs(weight[o]))/127)); zero channel uses exponent 0",
        "division_precision": "float64",
        "rounding": "IEEE_RNE_TIES_TO_EVEN_via_numpy_rint",
        "clipping": "clip_to_closed_interval_minus127_plus127_after_rounding",
        "scale_storage": "signed_int16_base2_exponent",
        "bias": "absent_for_all_eight_operators",
        "negative_128_forbidden": True,
    }, "weight quantization authority is not exact")
    return authority


def tensor_bytes(tensor: Any) -> bytes:
    import torch
    require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and
            tensor.dtype == torch.float32 and tensor.layout == torch.strided and
            tensor.is_contiguous(), "weight is not contiguous CPU float32")
    value = tensor.detach().numpy()
    require(sys.byteorder == "little" and value.dtype.str == "<f4" and
            value.flags.c_contiguous, "weight byte representation drift")
    return value.tobytes(order="C")


def quantize(canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(canonical.dtype == np.float32 and canonical.ndim == 4 and
            bool(np.isfinite(canonical).all()), "canonical weight invalid")
    flat = canonical.reshape(canonical.shape[0], -1)
    absmax = np.max(np.abs(flat), axis=1)
    exponent = np.zeros(absmax.shape, dtype=np.int16)
    nonzero = absmax > 0
    exponent[nonzero] = np.ceil(
        np.log2(absmax[nonzero].astype(np.float64) / 127.0)).astype(np.int16)
    scale = np.exp2(exponent.astype(np.float64))
    rounded = np.rint(flat.astype(np.float64) / scale[:, None])
    preclip = int(np.count_nonzero((rounded < -127.0) | (rounded > 127.0)))
    code = np.clip(rounded, -127, 127).astype(np.int8)
    require(not bool(np.any(code == -128)), "reserved -128 code emitted")
    dequant = code.astype(np.float64) * scale[:, None]
    error = dequant - flat.astype(np.float64)
    return code.reshape(canonical.shape), exponent, {
        "preclip_violations": preclip,
        "weight_mae": float(np.mean(np.abs(error))),
        "weight_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "weight_maximum_absolute_error": float(np.max(np.abs(error))),
        "code_zero_fraction": float(np.count_nonzero(code == 0)) / float(code.size),
        "scale_exponent_minimum": int(exponent.min()),
        "scale_exponent_maximum": int(exponent.max()),
    }


def static_bound(code: np.ndarray, family: str) -> dict[str, Any]:
    magnitude = np.abs(code.astype(np.int16)).astype(np.int64)
    if family == "c1_conv3x3":
        per_output = magnitude.sum(axis=(1, 2, 3), dtype=np.int64)
        mode = "all_3x3_taps"
    else:
        parity_bounds = []
        for py in range(2):
            for px in range(2):
                parity_bounds.append(magnitude[:, :, py::2, px::2].sum(
                    axis=(1, 2, 3), dtype=np.int64))
        per_output = np.maximum.reduce(parity_bounds)
        mode = "maximum_of_four_stride2_polyphase_classes"
    peak = int(per_output.max())
    require(peak <= ACC24_MAX, "static positive magnitude exceeds Acc24")
    return {
        "mode": mode,
        "per_output_minimum": int(per_output.min()),
        "per_output_maximum": peak,
        "acc24_minimum": ACC24_MIN,
        "acc24_maximum": ACC24_MAX,
        "static_bound_fits_acc24": True,
    }


def load_checkpoint(authority: dict[str, Any]) -> OrderedDict[str, Any]:
    import torch
    checkpoint = resolve_bound(authority["identity"]["checkpoint"]["path"])
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(type(value) is dict and set(value) == {"model_state_dict"},
            "checkpoint root drift")
    state = value["model_state_dict"]
    require(type(state) is OrderedDict and len(state) == EXPECTED_STATE_KEYS,
            "checkpoint state dictionary drift")
    return state


def layer_specs(authority: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in ("c1_conv3x3", "decoder_convtranspose"):
        body = authority["operator_families"][family]
        for ordinal, module in enumerate(body["modules"]):
            output.append({
                "family": family,
                "family_ordinal": ordinal,
                "module": module,
                "key": module + ".weight",
                "native_output_axis": body["native_output_axis"],
                "expected_shape": (tuple(body["weight_shape_each"])
                                   if family == "c1_conv3x3"
                                   else tuple(body["native_weight_shapes"][ordinal])),
                "expected_content_sha256": (C1_WEIGHT_SHAS[ordinal]
                                            if family == "c1_conv3x3"
                                            else DECODER_WEIGHT_SHAS[ordinal]),
                "input_alpha": body["input_layer_constants"][ordinal],
            })
    return output


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def produce(output: Path) -> dict[str, Any]:
    authority = verify_authority()
    require(not output.exists(), "output already exists; automatic retry is forbidden")
    temporary = output.parent / ("." + output.name + ".tmp")
    require(not temporary.exists(), "stale temporary output exists")
    temporary.mkdir(parents=True)
    try:
        state = load_checkpoint(authority)
        rows: list[dict[str, Any]] = []
        for global_ordinal, spec in enumerate(layer_specs(authority)):
            require(spec["key"] in state and spec["key"][:-6] + "bias" not in state,
                    "weight missing or bias unexpectedly present: " + spec["key"])
            tensor = state[spec["key"]]
            require(tuple(tensor.shape) == spec["expected_shape"],
                    "weight shape drift: " + spec["key"])
            raw = tensor_bytes(tensor)
            require(hashlib.sha256(raw).hexdigest() == spec["expected_content_sha256"],
                    "weight content SHA drift: " + spec["key"])
            native = tensor.detach().numpy()
            canonical = np.moveaxis(native, spec["native_output_axis"], 0).copy(order="C")
            require(canonical.dtype == np.float32 and canonical.flags.c_contiguous,
                    "canonical weight is not contiguous float32")
            code, exponent, metrics = quantize(canonical)
            hardware = np.transpose(code, (1, 2, 3, 0)).copy(order="C")
            prefix = "{:02d}_{}".format(global_ordinal, spec["family"])
            canonical_name = prefix + "_canonical_o_i_ky_kx.int8.npy"
            hardware_name = prefix + "_hardware_i_ky_kx_o.int8.npy"
            exponent_name = prefix + "_scale_exp2.int16.npy"
            np.save(temporary / canonical_name, code, allow_pickle=False)
            np.save(temporary / hardware_name, hardware, allow_pickle=False)
            np.save(temporary / exponent_name, exponent, allow_pickle=False)
            rows.append({
                "global_ordinal": global_ordinal,
                "family": spec["family"],
                "family_ordinal": spec["family_ordinal"],
                "module": spec["module"],
                "checkpoint_key": spec["key"],
                "native_output_axis": spec["native_output_axis"],
                "native_shape": list(spec["expected_shape"]),
                "canonical_shape": list(code.shape),
                "hardware_shape": list(hardware.shape),
                "input_alpha": spec["input_alpha"],
                "source_weight_sha256": spec["expected_content_sha256"],
                "canonical_code_file": canonical_name,
                "canonical_code_sha256": sha256(temporary / canonical_name),
                "hardware_code_file": hardware_name,
                "hardware_code_sha256": sha256(temporary / hardware_name),
                "scale_exponent_file": exponent_name,
                "scale_exponent_sha256": sha256(temporary / exponent_name),
                "static_accumulator_bound": static_bound(code, spec["family"]),
                **metrics,
            })
        sentinel = np.zeros((2, 3, 1, 1), dtype=np.float32)
        sentinel[0, 0, 0, 0] = 1.0
        sentinel[0, 1, 0, 0] = 2.0
        sentinel[1, 2, 0, 0] = 4.0
        moved = np.moveaxis(sentinel, 1, 0)
        require(moved.shape == (3, 2, 1, 1) and moved[0, 0, 0, 0] == 1.0 and
                moved[1, 0, 0, 0] == 2.0 and moved[2, 1, 0, 0] == 4.0,
                "ConvTranspose axis-1 sentinel failed")
        result = {
            "schema": "m2042_ep34_s40_eight_operator_int8_export_result_r1_v1",
            "status": "PASS_M2042_EP34_EIGHT_OPERATOR_INT8_WEIGHT_EXPORT",
            "authority_sha256": AUTHORITY_SHA256,
            "checkpoint_sha256": authority["identity"]["checkpoint"]["sha256"],
            "capture_ordered_sha256": authority["identity"]["m1458_capture"]["ordered_jsonl_sha256"],
            "population": {"families": 2, "operators": 8,
                           "c1_calls_reserved_for_bridge": 40,
                           "decoder_calls_reserved_for_bridge": 120},
            "convtranspose_axis1_sentinel": "PASS",
            "layers": rows,
            "claim_boundary": {
                "weight_quantization_rule_executed": True,
                "weight_payload_export_complete": True,
                "static_acc24_bound_complete": True,
                "activation_replay_complete": False,
                "operator_integer_bridge_complete": False,
                "valid825_AEE": False,
                "downstream_ATLIF_equivalence": False,
                "hardware_cycles": False,
                "hardware_speedup": False,
                "system_speedup": False,
                "energy": False,
                "PPA": False,
                "paper_result": False,
            },
        }
        write_json(temporary / "result.json", result)
        (temporary / "RUN_COMPLETE.txt").write_text(
            "PASS_M2042_EP34_EIGHT_OPERATOR_INT8_WEIGHT_EXPORT\n", encoding="utf-8")
        members = sorted(path for path in temporary.iterdir() if path.is_file())
        lines = [sha256(path) + "  " + path.name for path in members]
        (temporary / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest_sha = sha256(temporary / "SHA256SUMS")
        (temporary / "SHA256SUMS.seal.sha256").write_text(
            manifest_sha + "  SHA256SUMS\n", encoding="utf-8")
        os.replace(temporary, output)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-authority", action="store_true")
    args = parser.parse_args()
    if args.audit_authority:
        value = verify_authority()
        print(json.dumps({"status": "PASS_M2042_AUTHORITY_PREFLIGHT",
                          "authority_sha256": AUTHORITY_SHA256,
                          "checkpoint_sha256": value["identity"]["checkpoint"]["sha256"]},
                         sort_keys=True))
        return 0
    result = produce(args.output)
    print(json.dumps({"status": result["status"],
                      "output": str(args.output),
                      "operators": result["population"]["operators"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as error:
        print("FAIL_M2042: " + str(error), file=sys.stderr)
        raise SystemExit(2)
