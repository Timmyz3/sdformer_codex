#!/usr/bin/env python3
"""Read-only ep34 decoder-weight identity and future-export source.

This source exact-binds the sealed M1510 layer-constant adapter and the
M1512/M1513 capture-content/provenance authorities.  It loads exactly one
frozen checkpoint on CPU, requires the root to contain only
``model_state_dict``, and derives four ConvTranspose weight identities from
their actual little-endian contiguous float32 bytes.  No weight payload is
written.  The named one-shot export path is a future, separately reviewed
action and remains disabled here.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
SOURCE = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1514_ep34_decoder_weight_identity_export_source.py"
CONTRACT = HW / "contracts/m1514_ep34_decoder_weight_identity_export_source_contract_r1_20260831.json"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
M1510_SOURCE = HERE / "build_m1510_ep34_decoder_layer_constant_adapter_source.py"
M1510_SOURCE_SHA256 = "051b61d5cf8a7b164096da229601afb2ca8867d3b878e491bd7279148e5793aa"
M1510_CONTRACT = HW / "contracts/m1510_ep34_decoder_layer_constant_adapter_source_contract_r1_20260831.json"
M1510_CONTRACT_SHA256 = "88203261b26abee15ec57430e46cef7b4225f53fbb67abe9d18fc87c82d1abd7"
M1512 = HW / "reviews/m1512_m1501_m1458_ep34_capture_source_result_independent_hammer_r1_20260831"
M1512_PINS = (
    "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    "2af7a59b6a4df07dc6047c0d48c52b7798b7f0803e31e290b2ad842e6c154b81",
    "ccbcd7bf1b99fd944062a6fb220d7ec719d96da91c190697db125cbd4ad58f7c",
)
M1513 = HW / "reviews/m1513_m1512_m1458_ep34_production_provenance_addendum_r1_20260831"
M1513_PINS = (
    "1eb36a76fac29d5d15607dbb4ee3f9a434c4b0686843acac11f18116b48c7aaa",
    "966ba95baf00f698b6ca1fb8613afbfb78e40d2a70223f0a72bd4a87dcea04fa",
    "dc19cacbbb5ecae7f0327fd17b310be79a3b144937be7f289c25eb6f64794832",
)
RESULT_MANIFEST_SHA256 = "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e"
RESULT_OUTER_SHA256 = "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

MODULES = tuple("sttmultires_unet.decoders.{}.deconv.0".format(index)
                for index in range(4))
WEIGHT_KEYS = tuple(module + ".weight" for module in MODULES)
BIAS_KEYS = tuple(module + ".bias" for module in MODULES)
WEIGHT_SHAPES = (
    (1536, 384, 3, 3),
    (770, 192, 3, 3),
    (386, 96, 3, 3),
    (194, 96, 3, 3),
)
EXPECTED_CONTENT_SHA256 = (
    "cb1a90a4ff33622024b43ee6b15a3409e2567ea1e7b626715f40cf8a4fbfd83b",
    "35a9214e9fbc2e4e271beea74c4f329c12d6c072cda9252eaae350dd404a51cb",
    "75f9921f3cd9786ece78247115dd07bdda425b4f6e068d43936c884c611d3ef7",
    "6a42dabae358d0048aa46c609c9cb633f1e8d0479e4628e4f85c21e00835ea4e",
)
EXPECTED_STATE_KEYS = 921
EXPECTED_TOTAL_ELEMENTS = 7_140_096
EXPECTED_TOTAL_BYTES = 28_560_384
SCHEMA = "m1514_ep34_decoder_weight_identity_export_source_audit_r1_v1"
STATUS = "PASS_M1514_SOURCE_ONLY_DECODER_WEIGHT_IDENTITY__NO_EXPORT"
SOURCE_STATUS = "SOURCE_ONLY__EP34_DECODER_WEIGHT_IDENTITY__PRODUCTION_FALSE"
FUTURE_EXPORT = {
    "one_shot_runner":
        "hw_autoresearch_nts07/system_simulator/scripts/run_m1516_ep34_decoder_weight_export_one_shot.py",
    "output_directory":
        "hw_autoresearch_nts07/system_handoff/exports/m1516_ep34_decoder_weights_r1",
    "exists_or_written_by_m1514": False,
    "production": False,
    "independent_release_required": True,
}
CLAIM_BOUNDARY = {
    "source_only": True,
    "read_only": True,
    "checkpoint_loaded_cpu": True,
    "weight_identity": True,
    "weight_payload_written": False,
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


class M1514Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1514Error(message)


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
        raise M1514Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == digest, label + " SHA drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1514Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    regular_exact(path, digest, name + " after import")
    return module


M1510 = load_exact("m1514_frozen_m1510", M1510_SOURCE, M1510_SOURCE_SHA256)


def verify_sealed_review(root: Path, pins: tuple[str, str, str],
                         expected_status: str) -> dict[str, Any]:
    review_sha, manifest_sha, outer_sha = pins
    regular_exact(root / "review.json", review_sha, root.name + " review")
    regular_exact(root / "SHA256SUMS", manifest_sha, root.name + " manifest")
    regular_exact(root / "SHA256SUMS.seal.sha256", outer_sha, root.name + " outer")
    require((root / "SHA256SUMS.seal.sha256").read_text().split() ==
            [manifest_sha, "SHA256SUMS"], root.name + " outer content drift")
    listed: set[str] = set()
    prefix = root.relative_to(ROOT).as_posix() + "/"
    for line in (root / "SHA256SUMS").read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, root.name + " manifest row malformed")
        digest, name = fields
        name = name.lstrip("*")
        if name.startswith(prefix):
            name = name[len(prefix):]
        relative = Path(name)
        require(name not in listed and not relative.is_absolute() and
                ".." not in relative.parts,
                root.name + " manifest member unsafe/duplicate")
        regular_exact(root / relative, digest, root.name + " member")
        listed.add(name)
    actual = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        require(not any((base_path / name).is_symlink() for name in dirs + files),
                root.name + " contains symlink")
        for name in files:
            relative = (base_path / name).relative_to(root).as_posix()
            if relative not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                actual.add(relative)
    require(listed == actual and "review.json" in listed,
            root.name + " sealed membership drift")
    value = strict_json(root / "review.json")
    require(value.get("status") == expected_status, root.name + " status drift")
    return value


def verify_capture_authorities() -> dict[str, Any]:
    regular_exact(M1510_CONTRACT, M1510_CONTRACT_SHA256, "M1510 contract")
    M1510.validate_source_policy()
    m1512 = verify_sealed_review(
        M1512, M1512_PINS,
        "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT")
    m1513 = verify_sealed_review(
        M1513, M1513_PINS,
        "PASS_M1513_COMPLETE_M1458_EP34_PRODUCTION_PROVENANCE")
    identity = m1512.get("verification", {}).get("identity", {})
    capture = m1513.get("capture_binding", {})
    require(identity.get("checkpoint_sha256") == CHECKPOINT_SHA256 and
            capture.get("checkpoint_sha256") == CHECKPOINT_SHA256,
            "M1512/M1513 checkpoint identity drift")
    require(m1512.get("bindings", {}).get("result_manifest_sha256") ==
            RESULT_MANIFEST_SHA256 and
            m1512.get("bindings", {}).get("result_outer_file_sha256") ==
            RESULT_OUTER_SHA256 and
            m1513.get("bindings", {}).get("result_manifest_sha256") ==
            RESULT_MANIFEST_SHA256 and
            m1513.get("bindings", {}).get("result_outer_file_sha256") ==
            RESULT_OUTER_SHA256,
            "M1512/M1513 capture seal identity drift")
    require(m1513.get("bindings", {}).get("m1512_review_sha256") == M1512_PINS[0] and
            m1513.get("bindings", {}).get("m1512_manifest_sha256") == M1512_PINS[1] and
            m1513.get("bindings", {}).get("m1512_outer_file_sha256") == M1512_PINS[2],
            "M1513 does not exact-bind M1512")
    require(m1512.get("claim_boundary", {}).get("capture_content_validated") is True and
            m1513.get("claim_boundary", {}).get("production_provenance_complete") is True,
            "capture content/provenance authority incomplete")
    return {"m1512_status": m1512["status"], "m1513_status": m1513["status"]}


def tensor_content_identity(tensor: Any, ordinal: int) -> dict[str, Any]:
    import torch
    require(type(ordinal) is int and ordinal in range(4),
            "weight ordinal is not exact integer 0..3")
    require(type(tensor) is torch.Tensor, "decoder weight is not exact Tensor")
    require(tensor.device.type == "cpu", "decoder weight is not on CPU")
    require(tensor.dtype == torch.float32, "decoder weight dtype is not float32")
    require(tensor.layout == torch.strided, "decoder weight layout is not strided")
    require(tuple(tensor.shape) == WEIGHT_SHAPES[ordinal],
            "decoder weight shape drift")
    require(tensor.is_contiguous(), "decoder weight is not contiguous")
    require(sys.byteorder == "little", "host byte order is not little-endian")
    value = tensor.detach().numpy()
    require(value.dtype.str == "<f4" and value.flags.c_contiguous,
            "decoder NumPy view is not little-endian contiguous float32")
    content = value.tobytes(order="C")
    digest = hashlib.sha256(content).hexdigest()
    require(digest == EXPECTED_CONTENT_SHA256[ordinal],
            "decoder weight content SHA drift")
    elements = tensor.numel()
    require(type(elements) is int and elements > 0 and len(content) == elements * 4,
            "decoder weight byte extent drift")
    return {
        "module_ordinal": ordinal,
        "module": MODULES[ordinal],
        "checkpoint_key": WEIGHT_KEYS[ordinal],
        "shape": list(WEIGHT_SHAPES[ordinal]),
        "dtype": "torch.float32",
        "layout": "C_ORDER_CONTIGUOUS",
        "byte_order": "little",
        "elements": elements,
        "content_bytes": len(content),
        "content_sha256": digest,
        "bias": None,
    }


def validate_checkpoint_object(value: Any) -> list[dict[str, Any]]:
    import torch
    require(type(value) is dict and set(value) == {"model_state_dict"},
            "checkpoint root is not exact model_state_dict-only object")
    state = value["model_state_dict"]
    require(type(state) is OrderedDict, "model_state_dict is not exact OrderedDict")
    require(len(state) == EXPECTED_STATE_KEYS,
            "model_state_dict key population drift")
    require(all(type(key) is str and key for key in state),
            "model_state_dict key is not nonempty exact string")

    identities = []
    storage_addresses: set[int] = set()
    for ordinal, key in enumerate(WEIGHT_KEYS):
        suffix = "decoders.{}.deconv.0.weight".format(ordinal)
        aliases = [candidate for candidate in state if candidate.endswith(suffix)]
        require(aliases == [key], "decoder weight missing or duplicate alias")
        bias_suffix = "decoders.{}.deconv.0.bias".format(ordinal)
        require(not any(candidate.endswith(bias_suffix) for candidate in state),
                "decoder bias key must be absent")
        tensor = state[key]
        require(type(tensor) is torch.Tensor, "decoder weight is not exact Tensor")
        address = tensor.untyped_storage().data_ptr()
        require(address not in storage_addresses,
                "decoder weight storage duplicate alias")
        storage_addresses.add(address)
        identities.append(tensor_content_identity(tensor, ordinal))
    require(sum(row["elements"] for row in identities) == EXPECTED_TOTAL_ELEMENTS and
            sum(row["content_bytes"] for row in identities) == EXPECTED_TOTAL_BYTES,
            "decoder aggregate weight extent drift")
    return identities


def audit_checkpoint(checkpoint: Path = CHECKPOINT,
                     expected_checkpoint_sha256: str = CHECKPOINT_SHA256) -> dict[str, Any]:
    import torch
    regular_exact(checkpoint, expected_checkpoint_sha256, "ep34 checkpoint")
    # map_location is deliberately explicit; this source never requests CUDA.
    value = torch.load(checkpoint, map_location=torch.device("cpu"))
    regular_exact(checkpoint, expected_checkpoint_sha256, "ep34 checkpoint after load")
    identities = validate_checkpoint_object(value)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "checkpoint": {
            "path": str(Path(checkpoint)),
            "sha256": expected_checkpoint_sha256,
            "root_keys": ["model_state_dict"],
            "model_state_dict_keys": EXPECTED_STATE_KEYS,
            "torch_load_map_location": "cpu",
        },
        "capture_authorities": verify_capture_authorities(),
        "weights": identities,
        "aggregate": {
            "layers": 4,
            "elements": sum(row["elements"] for row in identities),
            "content_bytes": sum(row["content_bytes"] for row in identities),
        },
        "future_export": dict(FUTURE_EXPORT),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def validate_source_policy() -> dict[str, Any]:
    regular_exact(M1510_SOURCE, M1510_SOURCE_SHA256, "M1510 source")
    regular_exact(M1510_CONTRACT, M1510_CONTRACT_SHA256, "M1510 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    policy = strict_json(CONTRACT)
    require(policy.get("schema") == SCHEMA and
            policy.get("status") == SOURCE_STATUS,
            "M1514 contract schema/status drift")
    require(policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "M1514 source/test identity drift")
    require(policy.get("future_export") == FUTURE_EXPORT and
            policy.get("claim_boundary") == CLAIM_BOUNDARY,
            "M1514 future-export/claim boundary drift")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--audit-checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.source_self_check:
        validate_source_policy()
        print("PASS_M1514_SOURCE_SELF_CHECK__NO_CHECKPOINT_READ_NO_EXPORT")
        return 0
    result = audit_checkpoint(args.checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1514Error as error:
        print("M1514_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
