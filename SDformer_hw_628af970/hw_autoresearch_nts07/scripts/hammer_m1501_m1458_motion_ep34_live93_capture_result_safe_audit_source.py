#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only M1458 result validator with one exact safe-audit schema repair.

M1455's complete result validation is delegated unchanged.  The only adapter
accepts the exact ten-field H9 checkpoint-load audit emitted by M1458, proves
it is the safe zero-mismatch superset, replaces it in a private copy with
M1455's frozen two-field view, and then invokes frozen M1455 validation.
No capture, remote access, GPU, controller, EDA, or performance claim exists.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
M1455_SOURCE = HW / "scripts/hammer_m1455_m1434_motion_ep34_live93_capture_result_source.py"
M1455_SHA256 = "a77ae63153dbe808d98c73e1db05108d9e2152fdb29ae5837beae6fa5ea7991a"
M1458_SOURCE = HW / "scripts/run_m1458_m1434_motion_ep34_live93_production_one_shot.py"
M1458_SHA256 = "e81c20056dd261619f88884f2f097c9b594887927121d9e599a4f89185d33154"
M1489_SOURCE = HW / "scripts/run_m1489_m1485_m1434_export_alias_bootstrap.py"
M1489_SHA256 = "98b167ed763312b33866550f43c44c01657b60046c40849fd4780da30d04a48e"
M1461 = HW / "reviews/m1461_m1458_m1434_motion_ep34_live93_production_runner_source_blind_hammer_r1_20260831"
M1461_PINS = ("43f7a91567325570a30bc27eeda6516839691a5c1efd749185a086d36e2c4d58",
              "6bbb45f9103e069e453ce212b7bdeba4e75e2624b7609df618acfea6d40aae0d",
              "60cba22e1f6de76ba93d3e1a5730314f413b4b81c3558f452d7a911f511c3343")
M1462 = HW / "contracts/m1462_m1458_m1434_motion_ep34_live93_production_launch_release_r1_20260831.json"
M1462_PINS = ("bd56146574ad5919f326dbe87ccb1dca5da9e06c7e6471412aeaa037a6d0c88f",
              "8d7bfe7317d7ef3eec862a7c0ab4e42f42c8c1e26d6cc79da14fc99ec02a545c",
              "38b29ad65d88a1c8e9a668407f4d8c0bd5d9f8914e4157be97b16e70f618da65")
M1463 = HW / "reviews/m1463_m1462_m1458_m1434_motion_ep34_live93_production_final_launch_hammer_r1_20260831"
M1463_PINS = ("50af875678603940ff3789a516ab27aa1b89842f8d1a31b01c7320c442d2dcc4",
              "bed9e82d88c097d4e1fff8f90f84c69a1acbd86044964205f8acaf5d6bac138e",
              "6effc85b4ca3350b907eb12ac083e62f9414aa6ecf30fcafb83ca4a76ad332cf")
M1490 = HW / "reviews/m1490_m1489_m1434_export_alias_bootstrap_blind_hammer_r1_20260831"
M1490_PINS = ("f3f543c0086c032b513019939634af065fa095ca3be57b2c9122e4479ae8560c",
              "eb1f1851e6002cfc3d7e5f0669dcd5a9186fcf3ed57de026a70f09ac2298e016",
              "6a7fe54bd5af287eeac4f988102a08441bf66863035d881420555a919c0b6338")
CANONICAL_RESULT = HW / "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
TEST = HW / "tests/test_hammer_m1501_m1458_motion_ep34_live93_capture_result_safe_audit_source.py"
CONTRACT = HW / "contracts/m1501_m1458_motion_ep34_live93_capture_result_safe_audit_source_contract_r1_20260831.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_SCHEMA = "m1501_m1458_motion_ep34_live93_capture_result_safe_audit_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1458_RESULT_SAFE_AUDIT_ADAPTER__NO_REMOTE_NO_CAPTURE"
PASS_TOKEN = "PASS_M1501_SOURCE_SELF_CHECK__NO_RESULT_READ_NO_REMOTE_NO_GPU_NO_EDA"


class M1501Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1501Error(message)


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
        raise M1501Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == digest, label + " SHA drift")


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    regular_exact(path, digest, name + " after import")
    return module


M1455 = load_exact("m1501_frozen_m1455", M1455_SOURCE, M1455_SHA256)
FROZEN_VALIDATE_MANIFEST = M1455.validate_manifest

AUDIT_KEYS = {
    "checkpoint", "checkpoint_overlay_keys", "model_overlay_keys",
    "missing_count", "unexpected_count", "overlay_missing_count",
    "overlay_unexpected_count", "missing_sample", "unexpected_sample", "remap",
}
ZERO_FIELDS = ("missing_count", "unexpected_count",
               "overlay_missing_count", "overlay_unexpected_count")
OVERLAY_FIELDS = ("checkpoint_overlay_keys", "model_overlay_keys")
CLAIM_BOUNDARY = {
    "source_only": True, "production_result": False, "capture": False,
    "remote": False, "gpu": False, "controller_signal": False,
    "paper_result": False, "cycles": False, "speedup": False,
    "energy": False, "ppa": False, "system_speedup": False, "headline": False,
}


def strict_json(path: Path) -> dict[str, Any]:
    value = M1455.strict_json(path)
    require(type(value) is dict, "JSON root is not object")
    return value


def verify_authority(root: Path, pins: tuple[str, str, str],
                     status: str) -> dict[str, Any]:
    review, manifest, outer = pins
    regular_exact(root / "review.json", review, root.name + " review")
    regular_exact(root / "SHA256SUMS", manifest, root.name + " manifest")
    regular_exact(root / "SHA256SUMS.seal.sha256", outer, root.name + " outer")
    manifest_path = root / "SHA256SUMS"
    require((root / "SHA256SUMS.seal.sha256").read_text().split() ==
            [manifest, "SHA256SUMS"], root.name + " outer content drift")
    prefix = root.relative_to(ROOT).as_posix() + "/"
    listed: set[str] = set()
    for line in manifest_path.read_text().splitlines():
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
        require(not any((base_path / name).is_symlink()
                        for name in dirs + files),
                root.name + " contains symlink")
        for name in files:
            relative = (base_path / name).relative_to(root).as_posix()
            if relative not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                actual.add(relative)
    # Some previously reviewed directories acquired untracked __pycache__
    # files after sealing.  They are never imported here and cannot affect
    # the exact-pinned review/manifest/outer authority.  Every sealed member
    # must still be present, regular, non-symlink and exact.
    require(listed <= actual and "review.json" in listed,
            root.name + " sealed membership drift")
    value = strict_json(root / "review.json")
    require(value.get("status") == status, root.name + " status drift")
    return value


def verify_release() -> dict[str, Any]:
    for path, digest, label in (
        (M1462, M1462_PINS[0], "M1462 release"),
        (Path(str(M1462) + ".sha256"), M1462_PINS[1], "M1462 sidecar"),
        (Path(str(M1462) + ".sha256.seal.sha256"), M1462_PINS[2],
         "M1462 outer"),
    ):
        regular_exact(path, digest, label)
    require(Path(str(M1462) + ".sha256").read_text().split() ==
            [M1462_PINS[0], M1462.name], "M1462 sidecar content")
    require(Path(str(M1462) + ".sha256.seal.sha256").read_text().split() ==
            [M1462_PINS[1], M1462.name + ".sha256"], "M1462 outer content")
    value = strict_json(M1462)
    require(value.get("status") ==
            "AUTHORIZE_ONE_M1458_M1434_EP34_LIVE93_PRODUCTION_ATTEMPT",
            "M1462 status drift")
    return value


def validate_checkpoint_load_audit(manifest: dict[str, Any]) -> None:
    identity = manifest.get("identity")
    require(type(identity) is dict, "manifest identity missing")
    audit = identity.get("checkpoint_load_audit")
    require(type(audit) is dict, "checkpoint_load_audit missing/not object")
    require(set(audit) == AUDIT_KEYS, "checkpoint_load_audit key set drift")
    for key in ZERO_FIELDS:
        require(type(audit[key]) is int and audit[key] == 0,
                "checkpoint_load_audit " + key + " drift")
    for key in OVERLAY_FIELDS:
        require(type(audit[key]) is int and audit[key] == 210,
                "checkpoint_load_audit " + key + " drift")
    for key in ("missing_sample", "unexpected_sample"):
        require(type(audit[key]) is list and audit[key] == [],
                "checkpoint_load_audit " + key + " drift")
    require(type(audit["remap"]) is str and audit["remap"] == "v1",
            "checkpoint_load_audit remap drift")
    selected = identity.get("selection", {}).get("selected", {})
    checkpoint = selected.get("checkpoint", {}).get("absolute_path")
    require(type(checkpoint) is str and checkpoint and
            type(audit["checkpoint"]) is str and audit["checkpoint"] == checkpoint,
            "checkpoint_load_audit checkpoint path drift")


def validate_manifest(manifest: dict[str, Any]) -> None:
    validate_checkpoint_load_audit(manifest)
    normalized = copy.deepcopy(manifest)
    normalized["identity"]["checkpoint_load_audit"] = {
        "missing_count": 0, "unexpected_count": 0}
    FROZEN_VALIDATE_MANIFEST(normalized)


BASE_ATTENTION_KEYS = {
    "q_shape", "k_shape", "q_bits_packed", "k_bits_packed", "gate_q17"}
PROJECTION_KEYS = {
    "projection_bias_acc_int64", "projection_bias_float32",
    "projection_weight_float32", "projection_weight_int8",
    "projection_weight_scale_exp2"}
ATTENTION_KEYS = BASE_ATTENTION_KEYS | PROJECTION_KEYS


def validate_projection_arrays(data: Any, row: dict[str, Any]) -> None:
    import numpy as np
    require(set(data.files) == ATTENTION_KEYS,
            "attention NPZ exact enriched member set drift")
    windows = row["windows_captured"]
    heads = row["heads"]
    spatial = row["spatial_tokens"]
    lanes = row["lanes"]
    temporal = row["temporal_tokens"]
    expected_shape = [2, windows, heads, spatial, lanes]
    q_shape = data["q_shape"]; k_shape = data["k_shape"]
    q_bits = data["q_bits_packed"]; k_bits = data["k_bits_packed"]
    gate = data["gate_q17"]
    require(q_shape.dtype == np.dtype("int32") and
            k_shape.dtype == np.dtype("int32") and
            q_shape.ndim == k_shape.ndim == 1 and
            q_shape.tolist() == k_shape.tolist() == expected_shape,
            "attention q/k shape metadata drift")
    elements = math.prod(expected_shape)
    require(q_bits.dtype == np.dtype("uint8") and
            k_bits.dtype == np.dtype("uint8") and
            q_bits.ndim == k_bits.ndim == 1 and
            q_bits.size == k_bits.size == (elements + 7) // 8,
            "attention packed q/k dtype or extent drift")
    M1455.M1401.M1338._tail_zero(q_bits, elements, "attention q_bits")
    M1455.M1401.M1338._tail_zero(k_bits, elements, "attention k_bits")
    require(gate.dtype == np.dtype("uint16") and
            gate.shape == (windows, heads, temporal) and gate.size > 0 and
            int(gate.max()) <= 256,
            "attention gate dtype/shape/range drift")

    dim = heads * lanes
    weight_float = data["projection_weight_float32"]
    weight_int8 = data["projection_weight_int8"]
    exponent = data["projection_weight_scale_exp2"]
    bias_float = data["projection_bias_float32"]
    bias_acc = data["projection_bias_acc_int64"]
    require(weight_float.dtype == np.dtype("float32") and
            weight_float.shape == (dim, dim),
            "projection float weight dtype/shape drift")
    require(weight_int8.dtype == np.dtype("int8") and
            weight_int8.shape == (dim, dim),
            "projection INT8 weight dtype/shape drift")
    require(exponent.dtype == np.dtype("int16") and exponent.shape == (dim,),
            "projection exponent dtype/shape drift")
    require(bias_float.dtype == np.dtype("float32") and
            bias_float.shape == (dim,),
            "projection float bias dtype/shape drift")
    require(bias_acc.dtype == np.dtype("int64") and bias_acc.shape == (dim,),
            "projection accumulator bias dtype/shape drift")
    require(bool(np.isfinite(weight_float).all()) and
            bool(np.isfinite(bias_float).all()),
            "projection float payload is nonfinite")
    scale = np.exp2(exponent.astype(np.float32))
    require(bool(np.isfinite(scale).all()) and bool((scale > 0).all()),
            "projection dyadic scale is nonfinite/nonpositive")
    absmax = np.abs(weight_float).max(axis=1)
    expected_exp = np.zeros(dim, dtype=np.int16)
    nonzero = absmax > 0
    expected_exp[nonzero] = np.ceil(
        np.log2(absmax[nonzero] / np.float32(127.0))).astype(np.int16)
    require(np.array_equal(exponent, expected_exp),
            "projection exponent is not canonical per-output dyadic scale")
    expected_code = np.rint(weight_float / scale[:, None])
    expected_code = np.clip(expected_code, -127, 127).astype(np.int8)
    require(np.array_equal(weight_int8, expected_code),
            "projection INT8 code/float/scale relation drift")
    restored = weight_int8.astype(np.float32) * scale[:, None]
    require(bool(np.isfinite(restored).all()) and
            bool((np.abs(restored - weight_float)
                  <= scale[:, None] / np.float32(2.0) + 1e-6).all()),
            "projection dyadic quantization error exceeds half-step")
    bias_units = bias_float / (scale / np.float32(128.0))
    require(bool(np.isfinite(bias_units).all()) and
            bool((np.abs(bias_units) <= np.iinfo(np.int64).max).all()),
            "projection bias accumulator conversion overflow/nonfinite")
    expected_bias = np.rint(bias_units).astype(np.int64)
    require(np.array_equal(bias_acc, expected_bias),
            "projection accumulator bias relation drift")


def validate_attention_exact_archive(root: Path) -> int:
    import numpy as np
    seal_rows, _ = M1455.BASE.verify_recursive_seal(root)
    manifest = strict_json(root / "attention_qk/manifest.json")
    records = manifest.get("records")
    require(type(records) is list and len(records) == 480,
            "attention population is not 480")
    try:
        M1455.M1401.M1338.OLD.M1227.audit_attention_population(
            records, samples=40)
    except Exception as error:
        raise M1501Error("attention 40x12 manifest identity drift") from error
    for row in records:
        sample = row.get("sample_id")
        name = row.get("name")
        require(type(sample) is int and not isinstance(sample, bool) and
                type(name) is str and
                name in M1455.M1401.M1338.OLD.M1227.ATTENTION_ALIASES,
                "attention record sample/module identity drift")
        safe_name = name.replace(".", "_").replace("/", "_")
        relative = "attention_qk/sample{}_{}.npz".format(sample, safe_name)
        require(Path(row.get("file", "")).name == Path(relative).name,
                "attention record basename drift")
        payload = M1455.BASE.safe_member(root, relative)
        digest = row.get("sha256")
        require(type(digest) is str and seal_rows.get(relative) == digest and
                sha256(payload) == digest,
                "attention record/manifest/seal/actual SHA drift")
        try:
            with np.load(payload, allow_pickle=False) as data:
                validate_projection_arrays(data, row)
        except M1501Error:
            raise
        except Exception as error:
            raise M1501Error("enriched attention NPZ unreadable") from error
    return len(records)


def validate_result(root: Path = CANONICAL_RESULT) -> dict[str, Any]:
    """Delegate every non-audit check to exact frozen M1455."""
    original = M1455.validate_manifest
    original_attention = M1455.M1401.M1338.validate_attention_exact_archive
    require(original is FROZEN_VALIDATE_MANIFEST,
            "frozen M1455 validate_manifest already patched")
    M1455.validate_manifest = validate_manifest
    M1455.M1401.M1338.validate_attention_exact_archive = (
        validate_attention_exact_archive)
    try:
        result = M1455.validate_result(root)
    finally:
        M1455.validate_manifest = original
        M1455.M1401.M1338.validate_attention_exact_archive = original_attention
    require(result.get("status") ==
            "PASS_M1455_M1434_EP34_LIVE93_CAPTURE_RESULT",
            "M1455 delegated result status drift")
    output = dict(result)
    output["predecessor_status"] = output["status"]
    output["status"] = "PASS_M1501_M1458_EP34_LIVE93_CAPTURE_RESULT"
    output["audit_adapter"] = {
        "safe_superset_keys": sorted(AUDIT_KEYS),
        "all_mismatch_counts_zero": True,
        "overlay_keys": 210, "samples_empty": True, "remap": "v1",
        "all_other_validation_delegated_to_exact_m1455": True,
    }
    output["attention_adapter"] = {
        "records": 480, "exact_base_keys": sorted(BASE_ATTENTION_KEYS),
        "exact_projection_keys": sorted(PROJECTION_KEYS),
        "dtype_shape_finite_quantization_checked": True,
        "manifest_module_identity_checked": True,
        "all_other_validation_delegated_to_exact_m1455": True,
    }
    return output


def validate_source_policy() -> dict[str, Any]:
    regular_exact(M1458_SOURCE, M1458_SHA256, "M1458 runner")
    regular_exact(M1489_SOURCE, M1489_SHA256, "M1489 bootstrap")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    M1455.validate_source_policy()
    m1461 = verify_authority(
        M1461, M1461_PINS,
        "PASS_M1458_RUNNER_SOURCE__FRESH_RELEASE_MAY_BE_AUTHORED")
    release = verify_release()
    m1463 = verify_authority(
        M1463, M1463_PINS,
        "PASS_M1458_M1434_EP34_LIVE93_FINAL_LAUNCH_AUTHORITY")
    m1490 = verify_authority(
        M1490, M1490_PINS,
        "PASS_M1490_M1489_M1434_EXPORT_ALIAS_BOOTSTRAP")
    require(m1461.get("bindings", {}).get("runner_sha256") == M1458_SHA256 and
            release.get("runner_sha256") == M1458_SHA256 and
            m1463.get("bindings", {}).get("runner_sha256") == M1458_SHA256 and
            m1490.get("bindings", {}).get("runner_sha256") == M1489_SHA256,
            "runner/bootstrap authority binding drift")
    policy = strict_json(CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and
            policy.get("status") == SOURCE_STATUS, "source contract status")
    require(policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "source/test identity drift")
    require(policy.get("canonical_result") ==
            str(CANONICAL_RESULT.relative_to(ROOT)) and
            policy.get("result_read_by_author") is False and
            policy.get("claim_boundary") == CLAIM_BOUNDARY,
            "source-only boundary drift")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--validate-result", type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.source_self_check:
        validate_source_policy()
        print(PASS_TOKEN)
        return 0
    print(json.dumps(validate_result(args.validate_result), indent=2,
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1501Error as error:
        print("M1501_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
