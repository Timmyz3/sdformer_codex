#!/usr/bin/env python3
"""Read-only tensor audit for the frozen M2044 derived checkpoint bundle.

This checker never constructs a model and never touches CUDA.  It loads the
source and derived state dictionaries on CPU, checks all 921 keys, requires
torch.equal for every non-target tensor, and independently reconstructs each
of the eight target tensors from the sealed M2042 INT8 code and dyadic scale
exponent arrays.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE_CHECKPOINT = (
    HW / "system_handoff/incoming/m2041_ep34_quant_binding_inputs/"
    "checkpoint_epoch34.pth"
)
SOURCE_CHECKPOINT_SHA256 = (
    "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
)
M2042 = HW / "results/m2042_ep34_s40_eight_operator_int8_export_r1_20260902"
M2042_MANIFEST_SHA256 = (
    "519b8621a0c16f67ed33c8c624adc6bbfbc1c4a27224b2812542da3d92fc3881"
)
M2042_OUTER_SHA256 = (
    "da977b9effab3accaff229877bc4d9f0e930f82de1c0833be5c872e63aee142b"
)
M2042_RESULT_SHA256 = (
    "455c9fe7036779b890d4b85911cc42dc47bcb62c9fb6f6a6ce9c28a2c833cf29"
)
BUNDLE = (
    HW / "system_handoff/generated/"
    "m2044_ep34_attention_hw_order_qdq8_bundle_r1_20260902"
)
BUNDLE_MANIFEST_SHA256 = (
    "ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c"
)
BUNDLE_OUTER_SHA256 = (
    "32cf8a7f4a7c015bcf0086fd7676bc0b5360710981be7c425e14ae62475d06a2"
)
BUNDLE_JSON_SHA256 = (
    "01e7aadb454e82ce8fb04d25c4dc40f05bedd59cfd03d7e3835cdb2b967c3aee"
)
DERIVED_CHECKPOINT_NAME = "checkpoint_epoch34_m2044_qdq8.pth"
DERIVED_CHECKPOINT_SHA256 = (
    "daec6c188e7045ca3867c16cfcee5b25d2680eb4a7f1933541dfea17f0ac8371"
)
EXPECTED_KEYS = 921
EXPECTED_TARGET_KEYS = (
    "sttmultires_unet.resblocks.0.conv1.0.weight",
    "sttmultires_unet.resblocks.0.conv2.0.weight",
    "sttmultires_unet.resblocks.1.conv1.0.weight",
    "sttmultires_unet.resblocks.1.conv2.0.weight",
    "sttmultires_unet.decoders.0.deconv.0.weight",
    "sttmultires_unet.decoders.1.deconv.0.weight",
    "sttmultires_unet.decoders.2.deconv.0.weight",
    "sttmultires_unet.decoders.3.deconv.0.weight",
)
EXPECTED_BUNDLE_MEMBERS = {
    "RUN_COMPLETE.txt",
    "bundle.json",
    DERIVED_CHECKPOINT_NAME,
    "m2044_ep34_attention_hw_order_qdq8_valid825.yml",
}


class AuditError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise AuditError("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA256 drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AuditError("nonfinite JSON token: " + token)),
    )
    require(type(value) is dict, "JSON root must be an object")
    return value


def verify_sealed_directory(
    directory: Path,
    expected_manifest_sha256: str,
    expected_outer_sha256: str,
) -> dict[str, str]:
    regular_exact(directory / "SHA256SUMS", expected_manifest_sha256,
                  directory.name + "/SHA256SUMS")
    regular_exact(directory / "SHA256SUMS.seal.sha256", expected_outer_sha256,
                  directory.name + "/SHA256SUMS.seal.sha256")
    words = (directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").split()
    require(words == [expected_manifest_sha256, "SHA256SUMS"],
            directory.name + " outer-seal content drift")
    members: dict[str, str] = {}
    for line in (directory / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "malformed SHA256SUMS line")
        digest, name = fields
        member = Path(name)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in members
            and not member.is_absolute()
            and member.parts == (name,),
            "unsafe or duplicate SHA256SUMS member",
        )
        regular_exact(directory / name, digest, directory.name + "/" + name)
        members[name] = digest
    return members


def initial_counts() -> dict[str, int]:
    return {
        "expected_state_keys": EXPECTED_KEYS,
        "expected_target_keys": len(EXPECTED_TARGET_KEYS),
        "expected_non_target_keys": EXPECTED_KEYS - len(EXPECTED_TARGET_KEYS),
        "source_state_keys": 0,
        "derived_state_keys": 0,
        "tensor_keys_checked": 0,
        "non_target_keys_checked": 0,
        "non_target_torch_equal": 0,
        "target_keys_checked": 0,
        "target_source_sha_equal": 0,
        "target_m2042_metadata_equal": 0,
        "target_qdq_torch_equal": 0,
        "target_recorded_sha_equal": 0,
        "non_target_elements_checked": 0,
        "target_elements_checked": 0,
        "mismatches": 0,
    }


def tensor_bytes(tensor: Any) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    return array.tobytes(order="C")


def audit(counts: dict[str, int]) -> dict[str, Any]:
    import torch

    regular_exact(SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256,
                  "source ep34 checkpoint")
    m2042_members = verify_sealed_directory(
        M2042, M2042_MANIFEST_SHA256, M2042_OUTER_SHA256)
    regular_exact(M2042 / "result.json", M2042_RESULT_SHA256,
                  "M2042 result")
    require(m2042_members.get("result.json") == M2042_RESULT_SHA256,
            "M2042 result is not manifest-bound")
    bundle_members = verify_sealed_directory(
        BUNDLE, BUNDLE_MANIFEST_SHA256, BUNDLE_OUTER_SHA256)
    require(set(bundle_members) == EXPECTED_BUNDLE_MEMBERS,
            "M2044 derived bundle member topology drift")
    require(bundle_members.get("bundle.json") == BUNDLE_JSON_SHA256,
            "bundle.json is not exact-bound")
    require(
        bundle_members.get(DERIVED_CHECKPOINT_NAME)
        == DERIVED_CHECKPOINT_SHA256,
        "derived checkpoint is not exact-bound",
    )

    m2042 = strict_json(M2042 / "result.json")
    bundle = strict_json(BUNDLE / "bundle.json")
    require(
        m2042.get("status")
        == "PASS_M2042_EP34_EIGHT_OPERATOR_INT8_WEIGHT_EXPORT"
        and m2042.get("checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256
        and len(m2042.get("layers", [])) == 8,
        "M2042 authority drift",
    )
    require(
        bundle.get("status") == "PASS_M2044_DERIVED_BUNDLE_NO_ACCURACY_CLAIM"
        and bundle.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256
        and bundle.get("m2042_result_sha256") == M2042_RESULT_SHA256
        and bundle.get("derived_checkpoint_file") == DERIVED_CHECKPOINT_NAME
        and bundle.get("derived_checkpoint_sha256")
        == DERIVED_CHECKPOINT_SHA256,
        "M2044 bundle authority drift",
    )

    m2042_rows = {row["checkpoint_key"]: row for row in m2042["layers"]}
    bundle_rows = {
        row["checkpoint_key"]: row for row in bundle["modified_weights"]
    }
    require(
        tuple(m2042_rows) == EXPECTED_TARGET_KEYS
        and tuple(bundle_rows) == EXPECTED_TARGET_KEYS
        and len(m2042_rows) == len(bundle_rows) == 8,
        "target-key population or order drift",
    )

    source_payload = torch.load(
        SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    derived_payload = torch.load(
        BUNDLE / DERIVED_CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=False,
    )
    require(
        type(source_payload) is dict
        and type(derived_payload) is dict
        and set(source_payload) == {"model_state_dict"}
        and set(derived_payload) == {"model_state_dict"},
        "checkpoint root-container drift",
    )
    source_state = source_payload["model_state_dict"]
    derived_state = derived_payload["model_state_dict"]
    require(
        type(source_state) is OrderedDict
        and type(derived_state) is OrderedDict,
        "state dictionary must be OrderedDict",
    )
    counts["source_state_keys"] = len(source_state)
    counts["derived_state_keys"] = len(derived_state)
    require(
        len(source_state) == len(derived_state) == EXPECTED_KEYS
        and tuple(source_state) == tuple(derived_state),
        "state key count/order drift",
    )

    for key in source_state:
        source_value = source_state[key]
        derived_value = derived_state[key]
        require(
            type(source_value) is torch.Tensor
            and type(derived_value) is torch.Tensor,
            "non-tensor state value: " + key,
        )
        require(
            source_value.device.type == derived_value.device.type == "cpu",
            "non-CPU checkpoint tensor: " + key,
        )
        require(
            source_value.dtype == derived_value.dtype
            and tuple(source_value.shape) == tuple(derived_value.shape),
            "tensor dtype/shape drift: " + key,
        )
        counts["tensor_keys_checked"] += 1

        if key not in m2042_rows:
            counts["non_target_keys_checked"] += 1
            counts["non_target_elements_checked"] += source_value.numel()
            if not torch.equal(source_value, derived_value):
                counts["mismatches"] += 1
                raise AuditError("non-target tensor differs: " + key)
            counts["non_target_torch_equal"] += 1
            continue

        counts["target_keys_checked"] += 1
        counts["target_elements_checked"] += source_value.numel()
        frozen = m2042_rows[key]
        recorded = bundle_rows[key]
        source_digest = hashlib.sha256(tensor_bytes(source_value)).hexdigest()
        if source_digest != frozen["source_weight_sha256"]:
            counts["mismatches"] += 1
            raise AuditError("target source tensor SHA drift: " + key)
        counts["target_source_sha_equal"] += 1

        metadata_fields = (
            "family",
            "native_output_axis",
            "canonical_code_sha256",
            "scale_exponent_sha256",
            "source_weight_sha256",
        )
        if any(recorded[field] != frozen[field] for field in metadata_fields):
            counts["mismatches"] += 1
            raise AuditError("target M2042/bundle metadata drift: " + key)
        counts["target_m2042_metadata_equal"] += 1

        code_path = M2042 / frozen["canonical_code_file"]
        exponent_path = M2042 / frozen["scale_exponent_file"]
        require(
            m2042_members.get(code_path.name)
            == frozen["canonical_code_sha256"]
            and m2042_members.get(exponent_path.name)
            == frozen["scale_exponent_sha256"],
            "target quantization arrays are not manifest-bound: " + key,
        )
        code = np.load(code_path, allow_pickle=False)
        exponent = np.load(exponent_path, allow_pickle=False)
        require(
            code.dtype == np.int8
            and exponent.dtype == np.int16
            and tuple(code.shape) == tuple(frozen["canonical_shape"])
            and tuple(exponent.shape) == (code.shape[0],)
            and not bool(np.any(code == -128)),
            "target quantization array dtype/shape/code drift: " + key,
        )
        canonical = (
            code.astype(np.float64)
            * np.exp2(exponent.astype(np.float64))[:, None, None, None]
        )
        output_axis = int(frozen["native_output_axis"])
        require(output_axis in (0, 1), "unsupported native output axis")
        native = (
            canonical
            if output_axis == 0
            else np.moveaxis(canonical, 0, output_axis)
        )
        require(tuple(native.shape) == tuple(frozen["native_shape"]),
                "reconstructed native shape drift: " + key)
        expected = torch.from_numpy(native.copy(order="C")).to(
            dtype=source_value.dtype)
        if not torch.equal(derived_value, expected):
            counts["mismatches"] += 1
            raise AuditError("target QDQ reconstruction differs: " + key)
        counts["target_qdq_torch_equal"] += 1

        derived_digest = hashlib.sha256(
            tensor_bytes(derived_value)).hexdigest()
        if derived_digest != recorded["qdq_weight_sha256"]:
            counts["mismatches"] += 1
            raise AuditError("target recorded QDQ SHA drift: " + key)
        counts["target_recorded_sha_equal"] += 1

    require(
        counts["tensor_keys_checked"] == 921
        and counts["non_target_keys_checked"] == 913
        and counts["non_target_torch_equal"] == 913
        and counts["target_keys_checked"] == 8
        and counts["target_source_sha_equal"] == 8
        and counts["target_m2042_metadata_equal"] == 8
        and counts["target_qdq_torch_equal"] == 8
        and counts["target_recorded_sha_equal"] == 8
        and counts["mismatches"] == 0,
        "final tensor-count conservation failed",
    )
    return {
        "schema": "m2044_ep34_derived_bundle_tensor_audit_r1_v1",
        "status": "PASS_M2044_EP34_DERIVED_BUNDLE_TENSOR_AUDIT",
        "read_only": True,
        "cuda_used": False,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "derived_checkpoint_sha256": DERIVED_CHECKPOINT_SHA256,
        "m2042_result_sha256": M2042_RESULT_SHA256,
        "bundle_json_sha256": BUNDLE_JSON_SHA256,
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
        "counts": counts,
        "claim_boundary": {
            "derived_checkpoint_tensor_identity": True,
            "non_target_tensors_unchanged": True,
            "eight_target_dyadic_INT8_QDQ_reconstruction": True,
            "model_forward_executed": False,
            "valid825_executed": False,
            "accuracy": False,
            "hardware_cycles": False,
            "energy": False,
            "PPA": False,
        },
    }


def main() -> int:
    counts = initial_counts()
    try:
        result = audit(counts)
    except Exception as error:
        failure = {
            "schema": "m2044_ep34_derived_bundle_tensor_audit_r1_v1",
            "status": "FAIL_M2044_EP34_DERIVED_BUNDLE_TENSOR_AUDIT",
            "read_only": True,
            "cuda_used": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "counts": counts,
        }
        print(json.dumps(failure, sort_keys=True, allow_nan=False),
              file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
