#!/opt/conda/envs/sdformerflow/bin/python
"""Source-only E1/E8 closure producer for the selected Motion ep29 model.

The eventual E1 production mode runs exactly two predeclared deployment
semantics (dyadic Shiftmax and integer hardware-order Shiftmax) on valid825.
It never ranks, tunes, retries, or chooses between them.  The already sealed
standard-valid825 row is the only floating reference.

The eventual E8 production mode loads the same checkpoint once, exports every
Conv2d/ConvTranspose2d/Linear weight with deterministic per-output-channel
dyadic INT8 codes, exports bias/BN parameters, proves conservative Acc19/Acc24
bounds, and captures per-layer dynamic ranges over a contract-pinned cohort.

This checked-in source cannot launch either mode.  A fresh different-author
hammer and a successor launch contract must bind every input and output byte.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from copy import deepcopy
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_CONTRACT = HW / "contracts/m1177r2_motion_ep29_e1e8_source_contract_r1_20260830.json"
PROFILE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "profile_nts11_hardware_p0.py"
)
EVALUATOR = ROOT / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
LEASE = HW / "results/gpu_profile_lease.lock"
PROFILE_SHA256 = "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684"
EVALUATOR_SHA256 = "ba40b42c7395fd703c59a183a19b6a4fd38fa08ed75201008f03fd71b82aaef1"
M1175_REVIEW = HW / (
    "reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830/"
    "review.json"
)
M1175_REVIEW_SHA256 = "8b83690b8b1130d2335bb118d35645ae4d172740966ab69c6fcea9bc8b5d307b"
EXPECTED_COHORT = HW / (
    "contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json"
)
EXPECTED_COHORT_SHA256 = "56bc2e9b032a895c9700d5a6e83cc85c9f32e3f1505848264ad9ee5f38c000db"
EXPECTED_COHORT_SIZE = 19092
EXPECTED_COHORT_INNER_SHA256 = "72c8517331bd19c8faecfe123c1bd144f117237ddb1c3e7ce9b3910514badcee"
EXPECTED_COHORT_OUTER_SHA256 = "79b693ba3f86b3809a9a1c42eee5879ca5f3e3b0fb3693852f1c9cb61c8a82f2"
EXPECTED_SOURCE_HAMMER_REVIEW = HW / (
    "reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830/review.json"
)
TEST_SOURCE = HW / "tests/test_run_m1177r2_motion_ep29_e1e8_closure_source.py"
EXPECTED_CHECKPOINT = {
    "epoch": 29,
    "sha256": "2144dfd628cd928bfb768b92d4fa097b720db112c32d930b9f3cd85c6217286a",
    "size_bytes": 225504447,
    "mtime_ns": 1788057827000000000,
}
EXPECTED_CONFIG_SHA256 = "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955"
EXPECTED_STANDARD = {
    "samples": 825,
    "AEE": 1.209876834190253,
    "AAE": 5.406798340046045,
    "AAE_Benchmark": 5.148612399245754,
}
LEGACY_MARKERS = (
    "capture_m511_h67_convtranspose_binary_inputs.py",
    "m511_capture_watcher",
    "run_m511_h67",
)
WEIGHT_TYPES = {"Conv2d", "ConvTranspose2d", "Linear"}
RANGE_TYPES = WEIGHT_TYPES | {"BatchNorm2d", "ATLIFTernaryPSN"}


class ClosureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ClosureError("non-standard JSON token: " + token)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ClosureError("missing {}: {}".format(label, path)) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be a non-symlink regular file: {}".format(label, path))


def repo_path(relative: str, *, missing_leaf: bool = False) -> Path:
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts,
            "unsafe repository-relative path")
    cursor = ROOT
    for index, part in enumerate(value.parts):
        cursor = cursor / part
        if os.path.lexists(cursor):
            require(not cursor.is_symlink(), "symlink path component rejected")
        else:
            require(missing_leaf and index == len(value.parts) - 1,
                    "missing repository path component: " + str(cursor))
    return ROOT / value


def exact_int(value: Any, label: str) -> int:
    require(type(value) is int, label + " must be an exact integer")
    return value


def finite_float(value: Any, label: str) -> float:
    require(type(value) in {int, float} and type(value) is not bool,
            label + " must be numeric")
    result = float(value)
    require(math.isfinite(result), label + " must be finite")
    return result


def signed_bits_for_bounds(lower: int, upper: int) -> int:
    require(type(lower) is int and type(upper) is int and lower <= upper,
            "invalid signed bounds")
    for bits in range(1, 65):
        if lower >= -(1 << (bits - 1)) and upper <= (1 << (bits - 1)) - 1:
            return bits
    raise ClosureError("signed bound exceeds 64 bits")


def tensor_summary(array: Any) -> dict[str, Any]:
    """Pure numpy-compatible dynamic summary used by production and fixtures."""
    import numpy as np
    value = np.asarray(array)
    require(value.size > 0, "empty tensor range population")
    require(bool(np.isfinite(value).all()), "tensor contains NaN/Infinity")
    minimum = float(value.min())
    maximum = float(value.max())
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "minimum": minimum,
        "maximum": maximum,
        "maximum_absolute": max(abs(minimum), abs(maximum)),
        "zero": int(np.count_nonzero(value == 0)),
        "positive": int(np.count_nonzero(value > 0)),
        "negative": int(np.count_nonzero(value < 0)),
    }


def quantize_dyadic_per_output(weight: Any, output_axis: int) -> dict[str, Any]:
    """Canonical narrow-range INT8 with a power-of-two scale per output."""
    import numpy as np
    source = np.asarray(weight, dtype=np.float32)
    require(source.ndim >= 2 and 0 <= output_axis < source.ndim,
            "weight/output-axis geometry mismatch")
    require(bool(np.isfinite(source).all()), "weight has NaN/Infinity")
    moved = np.moveaxis(source, output_axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    absmax = np.max(np.abs(flat), axis=1)
    exponent = np.zeros(absmax.shape, dtype=np.int16)
    nonzero = absmax > 0
    exponent[nonzero] = np.ceil(np.log2(absmax[nonzero] / np.float32(127.0))).astype(np.int16)
    scale = np.exp2(exponent.astype(np.float32))
    code = np.rint(flat.astype(np.float64) / scale[:, None].astype(np.float64))
    preclip = int(np.count_nonzero((code < -127.0) | (code > 127.0)))
    code = np.clip(code, -127, 127).astype(np.int8)
    require(not bool(np.any(code == -128)), "reserved -128 code emitted")
    dequant = code.astype(np.float64) * scale[:, None].astype(np.float64)
    error = dequant - flat.astype(np.float64)
    sum_abs = np.abs(code.astype(np.int16)).sum(axis=1, dtype=np.int64)
    bitmap_bytes = (int(code.size) + 7) // 8
    nonzero_codes = int(np.count_nonzero(code))
    dense_bytes = int(code.size)
    bitmap_value_bytes = bitmap_bytes + nonzero_codes
    output_tile = min(96, int(code.shape[0]))
    tile_dense = output_tile * int(code.shape[1])
    return {
        "code": code.reshape(moved.shape),
        "scale_exp2": exponent,
        "sum_abs_per_output": sum_abs,
        "preclip_violations": preclip,
        "error": {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "maximum_absolute": float(np.max(np.abs(error))),
        },
        "compression": {
            "dense_int8_bytes": dense_bytes,
            "zero_bitmap_plus_nonzero_value_bytes": bitmap_value_bytes,
            "selected_exact_encoding": (
                "zero_bitmap_plus_nonzero_values" if bitmap_value_bytes < dense_bytes else "dense_int8"
            ),
            "selected_bytes": min(dense_bytes, bitmap_value_bytes),
            "zero_fraction": float(dense_bytes - nonzero_codes) / float(dense_bytes),
            "whole_tensor_fits_240kib": min(dense_bytes, bitmap_value_bytes) <= 240 * 1024,
            "dense_first_output_tile_up_to_96_bytes": tile_dense,
            "dense_first_output_tile_up_to_96_fits_240kib": tile_dense <= 240 * 1024,
        },
    }


def fixed_deploy_configs(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Construct exactly two fixed deploy modes; no options are exposed."""
    require(isinstance(source.get("bsa_attention"), dict) and
            source["bsa_attention"].get("enabled") is True,
            "selected config lacks enabled bsa_attention")
    common = {
        "alpha0": 1.0 / 64.0,
        "castling_matrix_aux_weight": 0.0,
        "castling_matrix_aux_end_step": 0,
        "hardware_quant_enabled": True,
        "hardware_mu_pow2_shift": 0,
        "hardware_score_step": 1.0 / 128.0,
        "hardware_score_min": -2.0,
        "hardware_score_max": 2.0,
        "hardware_gate_step": 1.0 / 128.0,
        "hardware_gate_min": 0.0,
        "hardware_gate_max": 2.0,
    }
    result: dict[str, dict[str, Any]] = {}
    for mode, rtl_order in (("dyadic", False), ("hardware_order", True)):
        config = deepcopy(source)
        config["experiment"] = "m1177r2_motion_ep29_" + mode + "_q7q17_deploy"
        config["bsa_attention"].update(common)
        config["bsa_attention"]["hardware_rtl_shiftmax_enabled"] = rtl_order
        config.setdefault("runtime", {})["deployment_contract"] = {
            "scope": ("attention_core_hardware_order_numeric" if rtl_order
                      else "attention_core_dyadic_numeric"),
            "score_quantization": "Q7_step_2^-7",
            "shiftmax": ("Q8_LUT_integer_rowsum_ceil_pow2" if rtl_order
                         else "float_exp2_from_quantized_score"),
            "gate_quantization": "Q1.7_RNE",
            "full_network_fixed_point": False,
            "systemverilog_replay": False,
            "candidate_selection_or_parameter_search": False,
        }
        result[mode] = config
    require(set(result) == {"dyadic", "hardware_order"}, "deploy mode population drift")
    return result


def validate_profile(profile: dict[str, Any], *, expected_config: Path,
                     checkpoint: Path) -> dict[str, Any]:
    require(exact_int(profile.get("samples"), "valid825 samples") == 825,
            "valid825 sample count drift")
    identity = profile.get("artifact_identity")
    require(isinstance(identity, dict), "valid825 artifact identity absent")
    expected_identity = {
        "config_path": str(expected_config.resolve()),
        "config_sha256": sha256(expected_config),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "checkpoint_sha256": sha256(checkpoint),
    }
    require(all(identity.get(key) == value for key, value in expected_identity.items()),
            "valid825 artifact identity mismatch")
    audit = profile.get("checkpoint_load_audit")
    counts = profile.get("module_counts")
    require(isinstance(audit, dict) and audit.get("missing_count") == 0 and
            audit.get("unexpected_count") == 0, "checkpoint load audit failed")
    require(isinstance(counts, dict) and counts.get("ATLIFTernaryPSN") == 105 and
            counts.get("ShiftmaxAttention") == 12, "module topology drift")
    metrics = profile.get("metrics")
    require(isinstance(metrics, dict), "valid825 metrics absent")
    return {key: finite_float(metrics.get(key), key)
            for key in ("AEE", "AAE", "AAE_Benchmark")}


def running_legacy_watchers(proc_root: Path = Path("/proc")) -> list[int]:
    found: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
        if any(marker in command for marker in LEGACY_MARKERS):
            found.append(int(entry.name))
    return sorted(found)


@contextlib.contextmanager
def exclusive_gpu_lease(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ClosureError("shared GPU profile lease is busy") from error
        require(not running_legacy_watchers(), "legacy M511 watcher remains present")
        yield descriptor
        require(not running_legacy_watchers(), "legacy M511 watcher appeared during run")
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def load_source(name: str, path: Path, expected_sha: str) -> Any:
    regular(path, name)
    require(sha256(path) == expected_sha, name + " source SHA drift")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(sha256(path) == expected_sha, name + " changed during import")
    return module


def exact_keys(value: Any, expected: set[str], label: str) -> None:
    require(isinstance(value, dict) and set(value) == expected,
            "{} exact-key schema drift: got={} expected={}".format(
                label, sorted(value) if isinstance(value, dict) else type(value).__name__,
                sorted(expected)))


def verify_sidecar_seal(payload: Path, expected_sha: str, expected_size: int,
                        expected_inner_sha: str, expected_outer_sha: str) -> None:
    regular(payload, "sealed payload")
    require(payload.stat().st_size == expected_size and sha256(payload) == expected_sha,
            "sealed payload byte identity drift")
    inner = payload.with_name(payload.name + ".sha256")
    outer = payload.with_name(payload.name + ".sha256.seal.sha256")
    regular(inner, "payload SHA sidecar")
    regular(outer, "payload outer SHA seal")
    require(inner.read_text(encoding="utf-8").split() == [expected_sha, payload.name],
            "payload SHA sidecar mismatch")
    require(sha256(inner) == expected_inner_sha and sha256(outer) == expected_outer_sha,
            "payload inner/outer seal byte identity drift")
    require(outer.read_text(encoding="utf-8").split() == [sha256(inner), inner.name],
            "payload outer SHA seal mismatch")


def validate_m1175() -> dict[str, Any]:
    regular(M1175_REVIEW, "M1175 review")
    require(sha256(M1175_REVIEW) == M1175_REVIEW_SHA256,
            "exact M1175 review SHA drift")
    review = strict_json(M1175_REVIEW)
    require(review.get("schema") ==
            "m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_v1" and
            review.get("status") == "PASS", "M1175 schema/status drift")
    verified = review.get("verified")
    require(isinstance(verified, dict) and
            verified.get("samples825_and_zero_load_audits") is True and
            verified.get("module_counts_atlif105_attention12") is True and
            verified.get("semantic_mutations_fail_closed") is True,
            "M1175 load/topology/semantic audit admission absent")
    selection = review.get("selection")
    require(isinstance(selection, dict) and
            selection.get("epoch") == 29 and
            selection.get("checkpoint_sha256") == EXPECTED_CHECKPOINT["sha256"] and
            selection.get("checkpoint_size_bytes") == EXPECTED_CHECKPOINT["size_bytes"] and
            selection.get("checkpoint_mtime_ns") == EXPECTED_CHECKPOINT["mtime_ns"] and
            selection.get("configuration_sha256") == EXPECTED_CONFIG_SHA256 and
            selection.get("samples") == 825 and
            selection.get("AEE") == "1.209876834190253" and
            selection.get("AAE") == "5.406798340046045" and
            selection.get("AAE_Benchmark") == "5.148612399245754",
            "M1175 exact ep29 identity/sample admission drift")
    require(review.get("authorization_after_hammer", {}).get(
        "E0_final_checkpoint_and_deployment_identity") == "ADMITTED",
        "M1175 E0 admission absent")
    return review


def verify_hammer_directory(review_path: Path, declared: dict[str, Any]) -> dict[str, Any]:
    exact_keys(declared, {"path", "review_sha256", "manifest_sha256",
                          "outer_sha256"}, "source-hammer receipt")
    require(review_path == EXPECTED_SOURCE_HAMMER_REVIEW and
            declared["path"] == str(review_path.relative_to(ROOT)),
            "source-hammer canonical review path drift")
    directory = review_path.parent
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(review_path, "source-hammer review")
    regular(manifest, "source-hammer manifest")
    regular(outer, "source-hammer outer seal")
    require(sha256(review_path) == declared["review_sha256"] and
            sha256(manifest) == declared["manifest_sha256"] and
            sha256(outer) == declared["outer_sha256"],
            "source-hammer declared byte identity drift")
    require(outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), "SHA256SUMS"], "source-hammer outer seal mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "malformed source-hammer manifest row")
        name = fields[1].lstrip("*")
        require(Path(name).name == name and name not in rows,
                "unsafe/duplicate source-hammer member")
        member = directory / name
        regular(member, "source-hammer member")
        require(sha256(member) == fields[0], "source-hammer member SHA mismatch")
        rows[name] = fields[0]
    require(rows.get("review.json") == sha256(review_path),
            "source-hammer review is not sealed")
    actual = {item.name for item in directory.iterdir() if item.is_file() and
              item.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "source-hammer unsealed/extra member population")
    review = strict_json(review_path)
    require(review.get("schema") ==
            "m1181_m1177r2_motion_ep29_e1e8_source_hammer_review_r1_v1" and
            review.get("status") == "PASS_SOURCE_HAMMER__RELEASE_AUTHORING_ALLOWED" and
            review.get("production_authorized") is False,
            "source-hammer semantic status drift")
    artifacts = review.get("artifacts")
    exact_keys(artifacts, {"source", "contract", "tests"}, "source-hammer artifacts")
    for label, path in (("source", Path(__file__).resolve()),
                        ("contract", SOURCE_CONTRACT), ("tests", TEST_SOURCE)):
        exact_keys(artifacts[label], {"path", "sha256"},
                   "source-hammer artifact " + label)
        require(artifacts[label]["path"] == str(path.relative_to(ROOT)) and
                artifacts[label]["sha256"] == sha256(path),
                "source-hammer exact artifact binding drift: " + label)
    verified = review.get("verified")
    exact_keys(verified, {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"},
               "source-hammer B1-B8 verdict")
    require(all(verified[key] is True for key in sorted(verified)),
            "source-hammer did not close every B1-B8 finding")
    return review


def load_canonical_cohort(*, verify_source_bytes: bool) -> list[dict[str, Any]]:
    verify_sidecar_seal(EXPECTED_COHORT, EXPECTED_COHORT_SHA256,
                        EXPECTED_COHORT_SIZE, EXPECTED_COHORT_INNER_SHA256,
                        EXPECTED_COHORT_OUTER_SHA256)
    manifest = strict_json(EXPECTED_COHORT)
    exact_keys(manifest, {"schema", "status", "date", "authorities",
                          "population", "rows", "claim_boundary"},
               "canonical cohort manifest")
    require(manifest["schema"] ==
            "m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_v1" and
            manifest["status"] ==
            "SEALED_STATIC_MODEL_INDEPENDENT_COHORT__PRODUCTION_REQUIRES_M1177R2_HAMMER_AND_RELEASE",
            "canonical cohort schema/status drift")
    exact_keys(manifest["authorities"], {"m36_c1", "m699_decoder"},
               "canonical cohort authorities")
    expected_authorities = {
        "m36_c1": ("hw_autoresearch_nts07/results/m36_h67_ep35_patch_embed_profile_s10_r1_20260822/sample_workload.csv",
                   5822, "bb45f8b5406e34835f05e1993692d8cba241c748471037d75fcfa1ec2478cffa"),
        "m699_decoder": ("hw_autoresearch_nts07/contracts/m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json",
                         15961, "43d3b024c1a78d8bc2422af3846c9a376a67bedbecb2ff7396a17bc51ec68fc7"),
    }
    for label, (path_text, size, digest) in expected_authorities.items():
        authority = manifest["authorities"][label]
        exact_keys(authority, {"path", "bytes", "sha256", "selection"},
                   "cohort authority " + label)
        require((authority["path"], authority["bytes"], authority["sha256"]) ==
                (path_text, size, digest), "cohort authority identity drift")
        path = repo_path(path_text)
        regular(path, "cohort authority")
        require(path.stat().st_size == size and sha256(path) == digest,
                "cohort authority bytes drift")
    rows = manifest["rows"]
    require(isinstance(rows, list) and len(rows) == 40,
            "canonical cohort row population drift")
    row_keys = {"global_sample_id", "cohort", "sequence", "sample_key", "path",
                "bytes", "sha256", "authority", "authority_sample_id"}
    require([row.get("global_sample_id") for row in rows] == list(range(40)),
            "canonical cohort order/id drift")
    require(len({row.get("path") for row in rows}) == 40 and
            len({row.get("sha256") for row in rows}) == 40,
            "canonical cohort duplicate path/SHA")
    for index, row in enumerate(rows):
        exact_keys(row, row_keys, "canonical cohort row")
        require((row["cohort"], row["authority"]) ==
                (("c1", "m36") if index < 10 else ("decoder", "m699")),
                "canonical cohort label/authority drift")
        require(Path(row["path"]).name == row["sample_key"] and
                type(row["bytes"]) is int and row["bytes"] > 0 and
                isinstance(row["sha256"], str) and len(row["sha256"]) == 64,
                "canonical cohort row identity malformed")
        if verify_source_bytes:
            path = repo_path(row["path"])
            regular(path, "canonical cohort source")
            require(path.stat().st_size == row["bytes"] and
                    sha256(path) == row["sha256"],
                    "canonical cohort source byte drift")
    return rows


def validate_launch(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(contract.get("schema") == "m1177r2_motion_ep29_e1e8_launch_v2",
            "source-only contract is not production authority")
    require(contract.get("status") ==
            "HAMMERED_R2_SOURCE__M1175_BOUND__EXACTLY_ONE_MODE_AUTHORIZED",
            "M1177R2 launch status is not authorized")
    mode = contract.get("mode")
    require(mode in {"e1", "e8"}, "invalid M1177R2 mode")
    common_top = {"schema", "status", "mode", "contract_path", "common",
                  "output", "one_shot", "gpu_ownership"}
    exact_keys(contract, common_top | {mode}, "M1177R2 launch top-level")
    require(("e1" in contract) == (mode == "e1") and
            ("e8" in contract) == (mode == "e8"), "E1/E8 mode mixing")
    exact_keys(contract["output"], {"path"}, "M1177R2 output")
    exact_keys(contract["one_shot"], {"attempt_marker"}, "M1177R2 one-shot")
    regular(DOCS359, "protected docs/359")
    require(sha256(DOCS359) == DOCS359_SHA256, "protected docs/359 drift")
    common = contract["common"]
    exact_keys(common, {"source", "selection", "checkpoint_path", "config_path",
                        "m1175_result_hammer", "m1177r2_source_hammer"},
               "M1177R2 common inputs")
    exact_keys(common["source"], {"path", "sha256"}, "running source binding")
    require(common["source"] == {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve())}, "running source identity drift")
    require(contract["contract_path"] == str(contract_path.relative_to(ROOT)),
            "contract path binding mismatch")
    selection = common["selection"]
    exact_keys(selection, {"epoch", "checkpoint_sha256", "checkpoint_size_bytes",
                           "checkpoint_mtime_ns", "config_sha256",
                           "standard_valid825"}, "selected identity")
    require(selection == {
        "epoch": 29,
        "checkpoint_sha256": EXPECTED_CHECKPOINT["sha256"],
        "checkpoint_size_bytes": EXPECTED_CHECKPOINT["size_bytes"],
        "checkpoint_mtime_ns": EXPECTED_CHECKPOINT["mtime_ns"],
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "standard_valid825": EXPECTED_STANDARD,
    }, "selected identity/standard row drift")
    checkpoint = Path(common["checkpoint_path"])
    config = Path(common["config_path"])
    regular(checkpoint, "ep29 checkpoint")
    regular(config, "ep29 config")
    require(checkpoint.stat().st_size == EXPECTED_CHECKPOINT["size_bytes"] and
            checkpoint.stat().st_mtime_ns == EXPECTED_CHECKPOINT["mtime_ns"] and
            sha256(checkpoint) == EXPECTED_CHECKPOINT["sha256"],
            "ep29 checkpoint identity mismatch")
    require(sha256(config) == EXPECTED_CONFIG_SHA256, "ep29 config identity mismatch")
    exact_keys(common["m1175_result_hammer"], {"path", "sha256"},
               "M1175 binding")
    require(common["m1175_result_hammer"] == {
        "path": str(M1175_REVIEW.relative_to(ROOT)), "sha256": M1175_REVIEW_SHA256},
        "exact M1175 binding drift")
    validate_m1175()
    verify_hammer_directory(EXPECTED_SOURCE_HAMMER_REVIEW,
                            common["m1177r2_source_hammer"])
    regular(PROFILE, "pinned profiler")
    regular(EVALUATOR, "pinned evaluator")
    require(sha256(PROFILE) == PROFILE_SHA256 and
            sha256(EVALUATOR) == EVALUATOR_SHA256,
            "pinned profiler/evaluator SHA drift")
    exact_keys(contract["gpu_ownership"], {"lease_path"}, "GPU ownership")
    require(contract["gpu_ownership"]["lease_path"] == str(LEASE.relative_to(ROOT)),
            "canonical shared GPU lease path drift")
    if mode == "e1":
        exact_keys(contract["e1"], {"fixed_modes", "standard_valid825",
                                    "evaluator"}, "E1-only payload")
        require(contract["e1"] == {
            "fixed_modes": ["dyadic", "hardware_order"],
            "standard_valid825": EXPECTED_STANDARD,
            "evaluator": {"path": str(EVALUATOR.relative_to(ROOT)),
                          "sha256": EVALUATOR_SHA256}}, "E1 policy drift")
    else:
        exact_keys(contract["e8"], {"canonical_cohort_manifest", "profile",
                                    "expected_dynamic_samples"}, "E8-only payload")
        require(contract["e8"] == {
            "canonical_cohort_manifest": {
                "path": str(EXPECTED_COHORT.relative_to(ROOT)),
                "size_bytes": EXPECTED_COHORT_SIZE,
                "sha256": EXPECTED_COHORT_SHA256,
                "inner_sha256": EXPECTED_COHORT_INNER_SHA256,
                "outer_sha256": EXPECTED_COHORT_OUTER_SHA256},
            "profile": {"path": str(PROFILE.relative_to(ROOT)),
                        "sha256": PROFILE_SHA256},
            "expected_dynamic_samples": 40}, "E8 policy drift")
        load_canonical_cohort(verify_source_bytes=False)
    return {"mode": mode, "checkpoint": checkpoint,
            "config": config, "common": common}


def write_double_seal(directory: Path) -> None:
    members = sorted(item.relative_to(directory) for item in directory.rglob("*")
                     if item.is_file() and item.name not in
                     {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha256(directory / item), item.as_posix())
                                for item in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")


def run_e1(contract: dict[str, Any], binding: dict[str, Any], staging: Path) -> None:
    import yaml
    source = yaml.safe_load(binding["config"].read_text(encoding="utf-8")) or {}
    configs = fixed_deploy_configs(source)
    rows: dict[str, Any] = {}
    for mode in ("dyadic", "hardware_order"):
        config_path = staging / (mode + ".yml")
        config_path.write_text(yaml.safe_dump(configs[mode], sort_keys=False,
                                              allow_unicode=True), encoding="utf-8")
        output = staging / mode
        output.mkdir()
        env = os.environ.copy()
        env.update({"SDFORMER_USE_MLFLOW": "0", "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
                    "SDFORMER_SNN_BACKEND": "cupy",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        command = [sys.executable, "-u", str(EVALUATOR), "--config", str(config_path),
                   "--checkpoint", str(binding["checkpoint"]), "--path_results", str(output),
                   "--mode", "valid"]
        log = output / "eval.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        require(sha256(EVALUATOR) == EVALUATOR_SHA256 and
                sha256(PROFILE) == PROFILE_SHA256,
                "pinned evaluator/profiler changed across child execution")
        require(completed.returncode == 0, mode + " valid825 failed")
        profile_path = output / "spike_profile.json"
        regular(profile_path, mode + " valid825 profile")
        metrics = validate_profile(strict_json(profile_path), expected_config=config_path,
                                   checkpoint=binding["checkpoint"])
        rows[mode] = {"metrics": metrics, "profile_sha256": sha256(profile_path),
                      "config_sha256": sha256(config_path),
                      "delta_from_standard": {key: metrics[key] - EXPECTED_STANDARD[key]
                                              for key in metrics}}
    result = {
        "schema": "m1177r2_motion_ep29_e1_deploy_valid825_v1",
        "status": "E1_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED",
        "identity": {"epoch": 29, "checkpoint_sha256": EXPECTED_CHECKPOINT["sha256"],
                     "source_config_sha256": EXPECTED_CONFIG_SHA256},
        "standard_valid825": EXPECTED_STANDARD,
        "candidate_policy": {
            "single_checkpoint": True, "parameter_search": False,
            "validation_selection": False,
            "fixed_modes": ["dyadic", "hardware_order"],
            "scope": "attention-core numeric; not full-network fixed point or SV exact",
        },
        "rows": rows,
        "claim_boundary": {"accuracy": True, "hardware_speedup": False,
                           "system_speedup": False, "rtl_exact": False,
                           "fresh_result_hammer_required": True},
    }
    (staging / "e1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")


def module_tensor_census(module: Any) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for label, value in list(module.named_parameters(recurse=False)) + list(
            module.named_buffers(recurse=False)):
        require(label not in tensors, "duplicate module-local tensor name")
        tensors[label] = {
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    return tensors


def build_model_census(model: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, module in model.named_modules():
        kind = module.__class__.__name__
        if kind not in RANGE_TYPES:
            continue
        require(name and name not in seen, "empty/duplicate hardware layer name")
        seen.add(name)
        tensors = module_tensor_census(module)
        row = {"name": name, "kind": kind, "local_tensors": tensors}
        if kind in WEIGHT_TYPES:
            require("weight" in tensors, "weight-bearing census row lacks weight")
            row["weight_shape"] = tensors["weight"]["shape"]
            row["output_axis"] = output_axis(module)
        if kind == "BatchNorm2d":
            require(set(tensors) == {"weight", "bias", "running_mean",
                                     "running_var", "num_batches_tracked"},
                    "BN census exact local tensor set drift")
            row["channels"] = int(module.num_features)
            row["eps"] = float(module.eps)
        rows.append(row)
    require(rows, "empty exact model hardware census")
    dynamic = [row for row in rows if row["kind"] in RANGE_TYPES]
    weights = [row for row in rows if row["kind"] in WEIGHT_TYPES]
    bn = [row for row in rows if row["kind"] == "BatchNorm2d"]
    require(len({row["name"] for row in dynamic}) == len(dynamic) and
            weights and bn, "model census uniqueness/population drift")
    return {"schema": "m1177r2_exact_loaded_model_census_v1",
            "dynamic": dynamic, "weights": weights, "batch_norm": bn,
            "counts": {"dynamic": len(dynamic), "weights": len(weights),
                       "batch_norm": len(bn)}}


class RangeCapture:
    def __init__(self, torch: Any):
        self.torch = torch
        self.handles: list[Any] = []
        self.rows: list[dict[str, Any]] = []
        self.sample: dict[str, Any] | None = None
        self.sample_start = 0

    def begin(self, sample: dict[str, Any]) -> None:
        require(self.sample is None, "nested dynamic range sample")
        self.sample = dict(sample)
        self.sample_start = len(self.rows)

    def end(self, expected_names: set[str]) -> None:
        require(self.sample is not None, "dynamic range sample not active")
        observed = [row["name"] for row in self.rows[self.sample_start:]]
        counts = Counter(observed)
        require(set(counts) == expected_names and
                all(counts[name] == 1 for name in expected_names),
                "dynamic every-layer-once coverage drift")
        self.sample = None

    def attach(self, model: Any, expected: list[dict[str, Any]]) -> None:
        expected_by_name = {row["name"]: row for row in expected}
        require(len(expected_by_name) == len(expected),
                "dynamic census duplicate names")
        attached: set[str] = set()
        for name, module in model.named_modules():
            kind = module.__class__.__name__
            if name not in expected_by_name:
                continue
            require(kind == expected_by_name[name]["kind"],
                    "dynamic module type drift")
            attached.add(name)
            def hook(instance: Any, inputs: Any, output: Any, *, _name=name, _kind=kind) -> None:
                require(self.sample is not None, "range hook fired outside sample")
                source = next((item for item in inputs if self.torch.is_tensor(item)), None)
                target = output[0] if isinstance(output, (tuple, list)) else output
                require(source is not None and self.torch.is_tensor(target),
                        "range hook tensor contract mismatch")
                source_np = source.detach().float().cpu().numpy()
                target_np = target.detach().float().cpu().numpy()
                row = {"sample": self.sample, "name": _name, "kind": _kind,
                       "input": tensor_summary(source_np), "output": tensor_summary(target_np)}
                if _kind == "BatchNorm2d":
                    require(source_np.ndim == 4, "BN input rank drift")
                    import numpy as np
                    mean = source_np.mean(axis=(0, 2, 3), dtype=np.float64)
                    variance = source_np.var(axis=(0, 2, 3), dtype=np.float64)
                    row["current_batch"] = {
                        "channels": int(mean.size), "mean_min": float(mean.min()),
                        "mean_max": float(mean.max()), "variance_min": float(variance.min()),
                        "variance_max": float(variance.max()),
                    }
                self.rows.append(row)
            self.handles.append(module.register_forward_hook(hook))
        require(attached == set(expected_by_name),
                "dynamic census module attachment omission")

    def close(self) -> None:
        while self.handles:
            self.handles.pop().remove()


def output_axis(module: Any) -> int:
    return 1 if module.__class__.__name__ == "ConvTranspose2d" else 0


def export_static(torch: Any, model: Any, staging: Path,
                  expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    payload = staging / "payloads"
    payload.mkdir()
    rows: list[dict[str, Any]] = []
    for index, (name, module) in enumerate(model.named_modules()):
        kind = module.__class__.__name__
        if kind not in WEIGHT_TYPES:
            continue
        weight = module.weight.detach().float().cpu().contiguous().numpy()
        if kind == "ConvTranspose2d":
            require(int(module.groups) == 1,
                    "grouped ConvTranspose2d output-axis export is not admitted")
        axis = output_axis(module)
        quantized = quantize_dyadic_per_output(weight, axis)
        code = quantized.pop("code")
        exponent = quantized.pop("scale_exp2")
        sum_abs = quantized.pop("sum_abs_per_output")
        safe = "{:03d}_{}".format(index, hashlib.sha256(name.encode()).hexdigest()[:16])
        code_path = payload / (safe + ".weight_i8.bin")
        scale_path = payload / (safe + ".scale_exp2_i16le.bin")
        bias_path = payload / (safe + ".bias_f32le.bin")
        code_path.write_bytes(code.astype(np.int8, copy=False).tobytes(order="C"))
        scale_path.write_bytes(exponent.astype("<i2", copy=False).tobytes(order="C"))
        bias = getattr(module, "bias", None)
        bias_array = (np.zeros(int(code.shape[0]), dtype=np.float32) if bias is None
                      else bias.detach().float().cpu().contiguous().numpy().astype(np.float32))
        bias_path.write_bytes(bias_array.astype("<f4", copy=False).tobytes(order="C"))
        max_sum = int(sum_abs.max(initial=0))
        binary_bits = signed_bits_for_bounds(-max_sum, max_sum)
        int8_input_bound = max_sum * 127
        q7_bits = signed_bits_for_bounds(-int8_input_bound, int8_input_bound)
        rows.append({
            "name": name, "kind": kind, "weight_shape": list(weight.shape),
            "output_axis": axis, "output_channels": int(code.shape[0]),
            "input_terms_per_output": int(code.size // code.shape[0]),
            "weight_source_sha256": hashlib.sha256(weight.astype("<f4", copy=False).tobytes()).hexdigest(),
            "weight_payload": {"path": str(code_path.relative_to(staging)),
                               "bytes": code_path.stat().st_size, "sha256": sha256(code_path)},
            "scale_payload": {"path": str(scale_path.relative_to(staging)),
                              "bytes": scale_path.stat().st_size, "sha256": sha256(scale_path),
                              "format": "signed_int16_power_of_two_exponent"},
            "bias_payload": {"path": str(bias_path.relative_to(staging)),
                             "bytes": bias_path.stat().st_size, "sha256": sha256(bias_path),
                             "format": "float32_le_requires_activation_scale_for_integer_bias"},
            "quantization": quantized,
            "accumulator_bounds": {
                "binary_or_ternary_source_maximum_magnitude": max_sum,
                "binary_or_ternary_required_signed_bits": binary_bits,
                "binary_or_ternary_fits_acc19": binary_bits <= 19,
                "binary_or_ternary_fits_acc24": binary_bits <= 24,
                "signed_int8_source_maximum_magnitude": int8_input_bound,
                "signed_int8_source_required_signed_bits": q7_bits,
                "signed_int8_source_fits_acc19": q7_bits <= 19,
                "signed_int8_source_fits_acc24": q7_bits <= 24,
                "bias_excluded_until_activation_scale_is_bound": True,
            },
        })
    require(rows, "no weight-bearing modules exported")
    observed = {(row["name"], row["kind"], tuple(row["weight_shape"]))
                for row in rows}
    required = {(row["name"], row["kind"], tuple(row["weight_shape"]))
                for row in expected}
    require(len(rows) == len(observed) == len(expected) and observed == required,
            "exact weight layer/type/shape census export drift")
    return rows


def export_bn(model: Any, staging: Path,
              expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    payload = staging / "payloads"
    payload.mkdir(exist_ok=True)
    rows = []
    for index, (name, module) in enumerate(model.named_modules()):
        if module.__class__.__name__ != "BatchNorm2d":
            continue
        arrays = {}
        for label in ("weight", "bias", "running_mean", "running_var"):
            value = getattr(module, label, None)
            require(value is not None, "BN required tensor absent: " + label)
            array = value.detach().float().cpu().numpy().astype("<f4", copy=False)
            require(array.shape == (int(module.num_features),) and
                    bool(np.isfinite(array).all()),
                    "BN tensor channel/finite drift: " + label)
            arrays[label] = array
        require(set(arrays) == {"weight", "bias", "running_mean", "running_var"},
                "BN exact four-tensor set drift")
        require(math.isfinite(float(module.eps)) and float(module.eps) > 0.0,
                "BN epsilon must be positive finite")
        safe = "bn_{:03d}_{}.npz".format(index, hashlib.sha256(name.encode()).hexdigest()[:16])
        path = payload / safe
        np.savez(path, **arrays, eps=np.asarray([float(module.eps)], dtype="<f4"))
        rows.append({"name": name, "channels": int(module.num_features),
                     "eps": float(module.eps), "payload": str(path.relative_to(staging)),
                     "bytes": path.stat().st_size, "sha256": sha256(path),
                     "current_batch_coefficients_derived_from_dynamic_capture": True})
    require(rows, "no BatchNorm2d modules exported")
    observed = {(row["name"], row["channels"], row["eps"]) for row in rows}
    required = {(row["name"], row["channels"], row["eps"]) for row in expected}
    require(len(rows) == len(observed) == len(expected) and observed == required,
            "exact BN name/channel/epsilon census export drift")
    return rows


def run_e8(contract: dict[str, Any], binding: dict[str, Any], staging: Path) -> None:
    profile = load_source("m1177r2_profile", PROFILE, PROFILE_SHA256)
    torch = profile.torch
    import numpy as np
    config, device = profile.load_config(binding["config"])
    require(str(device).startswith("cuda") and torch.cuda.is_available(),
            "E8 production range capture requires CUDA")
    model = profile.build_model(config, binding["checkpoint"], device)
    audit = profile.validate_h9_load_audit(model, config)
    require(audit is not None and audit.get("missing_count") == 0 and
            audit.get("unexpected_count") == 0, "E8 checkpoint load failed")
    require(profile.h9_module_counts(model) ==
            {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "E8 topology count drift")
    bn_policy = config.get("test", {}).get("bn_policy", "running")
    require(bn_policy == "no_running", "E8 requires current-batch/no-running BN")
    profile.configure_batch_norm_evaluation(model, bn_policy)
    census = build_model_census(model)
    (staging / "model_census.json").write_text(
        json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    static = export_static(torch, model, staging, census["weights"])
    bn = export_bn(model, staging, census["batch_norm"])
    cohort = load_canonical_cohort(verify_source_bytes=True)
    capture = RangeCapture(torch)
    capture.attach(model, census["dynamic"])
    expected_dynamic_names = {row["name"] for row in census["dynamic"]}
    try:
        with torch.no_grad():
            for row in cohort:
                path = repo_path(row["path"])
                profile.functional.reset_net(model)
                array = np.load(path, allow_pickle=False)
                require(array.shape == (10, 480, 640) and array.dtype == np.float32,
                        "E8 cohort tensor geometry drift")
                chunk = torch.from_numpy(array.copy()).unsqueeze(0)
                label = torch.zeros((1, 2, 480, 640), dtype=torch.float32)
                mask = torch.ones((1, 480, 640), dtype=torch.float32)
                x, _, _ = profile.preprocess_chunk(config, chunk, label, mask, None, device)
                capture.begin({key: row[key] for key in
                               ("global_sample_id", "cohort", "sequence", "sample_key", "sha256")})
                model(x)
                torch.cuda.synchronize(device)
                capture.end(expected_dynamic_names)
    finally:
        capture.close()
    require(capture.rows, "E8 dynamic range population empty")
    expected_dynamic_rows = len(expected_dynamic_names) * 40
    population = Counter((row["name"], row["sample"]["global_sample_id"])
                         for row in capture.rows)
    require(len(capture.rows) == expected_dynamic_rows and
            len(population) == expected_dynamic_rows and
            all(value == 1 for value in population.values()),
            "expected_dynamic_rows every-layer-by-40 census drift")
    with (staging / "dynamic_ranges.jsonl").open("w", encoding="utf-8") as stream:
        for row in capture.rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (staging / "static_layers.jsonl").open("w", encoding="utf-8") as stream:
        for row in static:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (staging / "e8_result.json").write_text(json.dumps({
        "schema": "m1177r2_motion_ep29_e8_range_compression_v1",
        "status": "E8_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED",
        "identity": {"epoch": 29, "checkpoint_sha256": EXPECTED_CHECKPOINT["sha256"],
                     "config_sha256": EXPECTED_CONFIG_SHA256,
                     "checkpoint_load_audit": audit},
        "population": {"samples": 40, "dynamic_rows": len(capture.rows),
                       "expected_dynamic_rows": expected_dynamic_rows,
                       "weight_layers": len(static), "bn_layers": len(bn)},
        "files": {"dynamic_ranges": "dynamic_ranges.jsonl",
                  "static_layers": "static_layers.jsonl",
                  "model_census": "model_census.json"},
        "batch_norm": bn,
        "proof_scope": {
            "acc19_acc24": "per-layer conservative integer product-sum bounds; bias excluded until activation scale binding",
            "compression": "exact dense-vs-zero-bitmap-plus-values byte fit; no cycle or SRAM-port claim",
            "dynamic": "fixed 40-sample hardware cohort; not valid825 extrema",
        },
        "claim_boundary": {"range_and_export": True, "valid825_extrema": False,
                           "fixed_point_end_to_end": False, "cycles": False,
                           "speedup": False, "energy": False, "ppa": False,
                           "fresh_result_hammer_required": True},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(contract_path.is_relative_to(ROOT), "contract must be inside repository")
    contract = strict_json(contract_path)
    binding = validate_launch(contract, contract_path)
    mode = binding["mode"]
    output = repo_path(contract["output"]["path"], missing_leaf=True)
    attempt = repo_path(contract["one_shot"]["attempt_marker"], missing_leaf=True)
    require(not os.path.lexists(output) and not os.path.lexists(attempt),
            "fresh output and attempt namespaces required")
    with exclusive_gpu_lease(LEASE):
        descriptor = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        os.write(descriptor, ("M1177R2_{}_ATTEMPT_CONSUMED__NO_RETRY\n".format(mode)).encode())
        os.fsync(descriptor)
        os.close(descriptor)
        output.parent.mkdir(parents=True, exist_ok=True)
        if mode == "e1":
            # The evaluator seals the absolute config path in artifact_identity.
            # Therefore E1 must write directly into its fresh canonical namespace;
            # an atomic staging rename would make both profile identities stale.
            # A failed one-shot remains visibly unsealed and FAILED_DO_NOT_CITE.
            output.mkdir()
            try:
                run_e1(contract, binding, output)
                (output / "RUN_COMPLETE.txt").write_text(
                    "PASS_M1177R2_E1__FRESH_RESULT_HAMMER_REQUIRED\n", encoding="utf-8")
                write_double_seal(output)
            except BaseException as error:
                (output / "FAILED_DO_NOT_CITE.json").write_text(json.dumps({
                    "status": "FAIL_CLOSED_UNSEALED_DO_NOT_CITE",
                    "reason": "{}: {}".format(type(error).__name__, error),
                }, indent=2) + "\n", encoding="utf-8")
                raise
        else:
            staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
            try:
                run_e8(contract, binding, staging)
                (staging / "RUN_COMPLETE.txt").write_text(
                    "PASS_M1177R2_E8__FRESH_RESULT_HAMMER_REQUIRED\n", encoding="utf-8")
                write_double_seal(staging)
                os.replace(staging, output)
            except BaseException as error:
                (staging / "FAILED.json").write_text(json.dumps({
                    "status": "FAIL_CLOSED_NO_CANONICAL_RESULT",
                    "reason": "{}: {}".format(type(error).__name__, error),
                }, indent=2) + "\n", encoding="utf-8")
                raise
    print("PASS M1177R2 {} {}".format(mode, output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
