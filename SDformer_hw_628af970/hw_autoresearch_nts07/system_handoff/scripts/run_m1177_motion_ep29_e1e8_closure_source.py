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
SOURCE_CONTRACT = HW / "contracts/m1177_motion_ep29_e1e8_source_contract_r1_20260830.json"
PROFILE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "profile_nts11_hardware_p0.py"
)
EVALUATOR = ROOT / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
LEASE = HW / "results/gpu_profile_lease.lock"
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
        config["experiment"] = "m1177_motion_ep29_" + mode + "_q7q17_deploy"
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


def validate_launch(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(contract.get("schema") == "m1177_motion_ep29_e1e8_launch_v1",
            "source-only contract is not production authority")
    require(contract.get("status") ==
            "HAMMERED_SOURCE__M1175_BOUND__EXACTLY_ONE_MODE_AUTHORIZED",
            "M1177 launch status is not authorized")
    require(contract.get("mode") in {"e1", "e8"}, "invalid M1177 mode")
    regular(DOCS359, "protected docs/359")
    require(sha256(DOCS359) == DOCS359_SHA256, "protected docs/359 drift")
    inputs = contract["inputs"]
    require(inputs["source"]["sha256"] == sha256(Path(__file__).resolve()),
            "running source SHA drift")
    require(contract["contract_path"] == str(contract_path.relative_to(ROOT)),
            "contract path binding mismatch")
    selection = inputs["selection"]
    require(selection == {
        "epoch": 29,
        "checkpoint_sha256": EXPECTED_CHECKPOINT["sha256"],
        "checkpoint_size_bytes": EXPECTED_CHECKPOINT["size_bytes"],
        "checkpoint_mtime_ns": EXPECTED_CHECKPOINT["mtime_ns"],
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "standard_valid825": EXPECTED_STANDARD,
    }, "selected identity/standard row drift")
    checkpoint = Path(inputs["checkpoint_path"])
    config = Path(inputs["config_path"])
    regular(checkpoint, "ep29 checkpoint")
    regular(config, "ep29 config")
    require(checkpoint.stat().st_size == EXPECTED_CHECKPOINT["size_bytes"] and
            checkpoint.stat().st_mtime_ns == EXPECTED_CHECKPOINT["mtime_ns"] and
            sha256(checkpoint) == EXPECTED_CHECKPOINT["sha256"],
            "ep29 checkpoint identity mismatch")
    require(sha256(config) == EXPECTED_CONFIG_SHA256, "ep29 config identity mismatch")
    for label in ("m1175_result_hammer", "m1177_source_hammer"):
        receipt = inputs[label]
        path = repo_path(receipt["path"])
        regular(path, label)
        require(sha256(path) == receipt["sha256"], label + " SHA drift")
    for label, path in (("profile", PROFILE), ("evaluator", EVALUATOR)):
        regular(path, label)
        require(inputs[label]["sha256"] == sha256(path),
                label + " source SHA drift")
    return {"mode": contract["mode"], "checkpoint": checkpoint,
            "config": config, "inputs": inputs}


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
        "schema": "m1177_motion_ep29_e1_deploy_valid825_v1",
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


class RangeCapture:
    def __init__(self, torch: Any):
        self.torch = torch
        self.handles: list[Any] = []
        self.rows: list[dict[str, Any]] = []
        self.sample: dict[str, Any] | None = None

    def attach(self, model: Any) -> None:
        for name, module in model.named_modules():
            kind = module.__class__.__name__
            if kind not in RANGE_TYPES:
                continue
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

    def close(self) -> None:
        while self.handles:
            self.handles.pop().remove()


def output_axis(module: Any) -> int:
    return 1 if module.__class__.__name__ == "ConvTranspose2d" else 0


def export_static(torch: Any, model: Any, staging: Path) -> list[dict[str, Any]]:
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
    return rows


def export_bn(model: Any, staging: Path) -> list[dict[str, Any]]:
    import numpy as np
    payload = staging / "payloads"
    rows = []
    for index, (name, module) in enumerate(model.named_modules()):
        if module.__class__.__name__ != "BatchNorm2d":
            continue
        arrays = {}
        for label in ("weight", "bias", "running_mean", "running_var"):
            value = getattr(module, label, None)
            if value is not None:
                arrays[label] = value.detach().float().cpu().numpy().astype("<f4", copy=False)
        safe = "bn_{:03d}_{}.npz".format(index, hashlib.sha256(name.encode()).hexdigest()[:16])
        path = payload / safe
        np.savez(path, **arrays, eps=np.asarray([float(module.eps)], dtype="<f4"))
        rows.append({"name": name, "channels": int(module.num_features),
                     "eps": float(module.eps), "payload": str(path.relative_to(staging)),
                     "bytes": path.stat().st_size, "sha256": sha256(path),
                     "current_batch_coefficients_derived_from_dynamic_capture": True})
    require(rows, "no BatchNorm2d modules exported")
    return rows


def run_e8(contract: dict[str, Any], binding: dict[str, Any], staging: Path) -> None:
    inputs = binding["inputs"]
    profile = load_source("m1177_profile", PROFILE, inputs["profile"]["sha256"])
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
    static = export_static(torch, model, staging)
    bn = export_bn(model, staging)
    cohort = inputs["cohort"]
    require(len(cohort) == 40 and [row["global_sample_id"] for row in cohort] == list(range(40)),
            "E8 cohort population/order drift")
    capture = RangeCapture(torch)
    capture.attach(model)
    try:
        with torch.no_grad():
            for row in cohort:
                path = repo_path(row["path"])
                regular(path, "E8 cohort source")
                require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                        "E8 cohort source identity drift")
                profile.functional.reset_net(model)
                array = np.load(path, allow_pickle=False)
                require(array.shape == (10, 480, 640) and array.dtype == np.float32,
                        "E8 cohort tensor geometry drift")
                chunk = torch.from_numpy(array.copy()).unsqueeze(0)
                label = torch.zeros((1, 2, 480, 640), dtype=torch.float32)
                mask = torch.ones((1, 480, 640), dtype=torch.float32)
                x, _, _ = profile.preprocess_chunk(config, chunk, label, mask, None, device)
                capture.sample = {key: row[key] for key in
                                  ("global_sample_id", "cohort", "sequence", "sample_key", "sha256")}
                model(x)
                torch.cuda.synchronize(device)
                capture.sample = None
    finally:
        capture.close()
    require(capture.rows, "E8 dynamic range population empty")
    with (staging / "dynamic_ranges.jsonl").open("w", encoding="utf-8") as stream:
        for row in capture.rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (staging / "static_layers.jsonl").open("w", encoding="utf-8") as stream:
        for row in static:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (staging / "e8_result.json").write_text(json.dumps({
        "schema": "m1177_motion_ep29_e8_range_compression_v1",
        "status": "E8_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED",
        "identity": {"epoch": 29, "checkpoint_sha256": EXPECTED_CHECKPOINT["sha256"],
                     "config_sha256": EXPECTED_CONFIG_SHA256,
                     "checkpoint_load_audit": audit},
        "population": {"samples": 40, "dynamic_rows": len(capture.rows),
                       "weight_layers": len(static), "bn_layers": len(bn)},
        "files": {"dynamic_ranges": "dynamic_ranges.jsonl",
                  "static_layers": "static_layers.jsonl"},
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
    with exclusive_gpu_lease(repo_path(contract["gpu_ownership"]["lease_path"])):
        descriptor = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        os.write(descriptor, ("M1177_{}_ATTEMPT_CONSUMED__NO_RETRY\n".format(mode)).encode())
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
                    "PASS_M1177_E1__FRESH_RESULT_HAMMER_REQUIRED\n", encoding="utf-8")
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
                    "PASS_M1177_E8__FRESH_RESULT_HAMMER_REQUIRED\n", encoding="utf-8")
                write_double_seal(staging)
                os.replace(staging, output)
            except BaseException as error:
                (staging / "FAILED.json").write_text(json.dumps({
                    "status": "FAIL_CLOSED_NO_CANONICAL_RESULT",
                    "reason": "{}: {}".format(type(error).__name__, error),
                }, indent=2) + "\n", encoding="utf-8")
                raise
    print("PASS M1177 {} {}".format(mode, output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
