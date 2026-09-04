#!/usr/bin/env python3
"""Stream the exact post-BN2 H67 FFN residual metrics for the frozen S10.

This is a no-training opportunity capture.  Full activation/residual tensors
are never written.  Each completed FFN call is reduced over its channel axis
and immediately emitted as one compressed sample/stage/block metric payload.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
STAGE_BLOCKS = (2, 2, 6, 2)
STAGE_CHANNELS = (96, 192, 384, 768)
STAGE_TOKEN_SHAPES = (
    (10, 1, 120, 160),
    (10, 1, 60, 80),
    (10, 1, 30, 40),
    (10, 1, 15, 20),
)
TAU_GRID = (
    ("zero_exact", 0.0),
    ("2^-16", 2.0 ** -16),
    ("2^-14", 2.0 ** -14),
    ("2^-12", 2.0 ** -12),
    ("2^-10", 2.0 ** -10),
    ("2^-8", 2.0 ** -8),
    ("2^-6", 2.0 ** -6),
)
DENOMINATOR_FLOOR = 2.0 ** -24
EXPECTED_SAMPLES = 10
EXPECTED_BN_MODULES_CHANGED = 78


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path_text.startswith(("neuron_experiments/", "third_party/")):
        return ROOT / path
    return HW / path


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_name(stage, block):
    return ("sttmultires_unet.encoders.swin3d.layers.{}."
            "swin_blocks.{}.mlp").format(stage, block)


def all_targets():
    return tuple((stage, block, target_name(stage, block))
                 for stage, blocks in enumerate(STAGE_BLOCKS)
                 for block in range(blocks))


def read_workload(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == EXPECTED_SAMPLES, "M460 requires exact S10 workload")
    require([int(row["sample_id"]) for row in rows] ==
            list(range(EXPECTED_SAMPLES)), "M460 sample order drift")
    require(all(row["sequence_key"] == "zurich_city_09_a" for row in rows),
            "M460 sequence identity drift")
    return rows


def validate_contract(contract_path):
    contract_path = Path(contract_path).resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m460_h67_g8_ffn_token_residual_s10_capture_contract_v1",
            "M460 contract schema drift")
    require(contract.get("status") ==
            "READY_REMOTE_A800_MANUAL_LAUNCH__PREINPUT_FROZEN",
            "M460 preinput contract is not frozen")
    observed = {}
    for name, record in contract["identity"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = resolve_path(record["path"]).resolve()
        require(path.is_file(), "M460 missing identity {}: {}".format(name, path))
        actual = sha256(path)
        require(actual == record["sha256"],
                "M460 SHA drift {} expected={} observed={}".format(
                    name, record["sha256"], actual))
        observed[name] = {"path": str(path), "sha256": actual}
    own_sha = sha256(Path(__file__).resolve())
    require(own_sha == contract["identity"]["capture_script"]["sha256"],
            "M460 capture script self SHA drift")

    source = Path(observed["swin_source"]["path"]).read_text(
        encoding="utf-8")
    for fragment in (
            "class MS_Spiking_Mlp(Spiking_Mlp):",
            "x = self.sn1(x)", "x = self.fc1(x)", "x= self.bn1(",
            "x = self.sn2(x)", "x = self.fc2(x)", "x = self.bn2(",
            "self.mlp(x.permute(1,0,2,3,4)).permute(1,0,2,3,4)"):
        require(fragment in source, "M460 FFN topology source drift: " + fragment)
    workload = read_workload(Path(observed["sample_workload"]["path"]))
    return contract, observed, workload


def array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode(
        "ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def array_receipt(array):
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": [int(item) for item in value.shape],
        "elements": int(value.size),
        "bytes": int(value.nbytes),
        "logical_sha256": array_sha256(value),
    }


class TorchTokenOps(object):
    """Channel-axis reductions used by the A800 path."""

    def __init__(self, torch):
        self.torch = torch

    def is_tensor(self, value):
        return self.torch.is_tensor(value)

    def shape(self, value):
        return tuple(int(item) for item in value.shape)

    def vector_metrics(self, value):
        torch = self.torch
        source = value.detach().float()
        require(source.ndim == 5, "M460 expects [T,N,H,W,C]")
        element_finite = torch.isfinite(source)
        finite = element_finite.all(dim=-1)
        safe = torch.where(element_finite, source, torch.zeros_like(source))
        safe64 = safe.to(torch.float64)
        result = {
            "l1": safe64.abs().sum(dim=-1),
            "l2_sq": (safe64 * safe64).sum(dim=-1),
            "linf": safe.abs().amax(dim=-1),
            "finite": finite,
            "exact_zero": safe.eq(0).all(dim=-1) & finite,
        }
        return {name: tensor.detach().cpu().contiguous().numpy()
                for name, tensor in result.items()}

    def source_metrics(self, value):
        torch = self.torch
        source = value.detach()
        require(source.ndim == 5, "M460 source expects [T,N,H,W,C]")
        finite = torch.isfinite(source).all(dim=-1)
        nnz = torch.count_nonzero(source, dim=-1).to(torch.int32)
        return {
            "nnz": nnz.cpu().contiguous().numpy(),
            "finite": finite.cpu().contiguous().numpy(),
        }


class NumpyTokenOps(object):
    """CPU dependency-free semantic twin used only by the M460 micro-test."""

    def is_tensor(self, value):
        return isinstance(value, np.ndarray)

    def shape(self, value):
        return tuple(int(item) for item in value.shape)

    def vector_metrics(self, value):
        source = np.asarray(value, dtype=np.float32)
        require(source.ndim == 5, "M460 expects [T,N,H,W,C]")
        element_finite = np.isfinite(source)
        finite = np.all(element_finite, axis=-1)
        safe = np.where(element_finite, source, np.float32(0.0))
        safe64 = safe.astype(np.float64)
        return {
            "l1": np.sum(np.abs(safe64), axis=-1, dtype=np.float64),
            "l2_sq": np.sum(safe64 * safe64, axis=-1, dtype=np.float64),
            "linf": np.max(np.abs(safe), axis=-1).astype(np.float32),
            "finite": finite,
            "exact_zero": np.all(safe == 0, axis=-1) & finite,
        }

    def source_metrics(self, value):
        source = np.asarray(value)
        require(source.ndim == 5, "M460 source expects [T,N,H,W,C]")
        return {
            "nnz": np.count_nonzero(source, axis=-1).astype(np.int32),
            "finite": np.all(np.isfinite(source), axis=-1),
        }


class FFNResidualStreamCapture(object):
    """Five-hook exact-boundary capture for all 12 MS FFNs."""

    def __init__(self, ops, output_dir, enforce_h67_geometry=True):
        self.ops = ops
        self.output_dir = Path(output_dir)
        self.enforce_h67_geometry = bool(enforce_h67_geometry)
        self.handles = []
        self.installed = []
        self.state = {}
        self.calls = {}
        self.current = None
        self.records = []

    def attach(self, model):
        require(not self.handles, "M460 hooks already attached")
        require(getattr(model, "training", False) is False,
                "M460 model must be eval before hook attach")
        named = dict(model.named_modules())
        for stage, block, name in all_targets():
            child_names = {
                "sn1": name + ".sn1",
                "sn2": name + ".sn2",
                "fc2": name + ".fc2",
                "bn1_norm": name + ".bn1.norm_layer",
                "bn2_norm": name + ".bn2.norm_layer",
            }
            require(name in named, "M460 missing full FFN: " + name)
            for role, child_name in child_names.items():
                require(child_name in named,
                        "M460 missing {} child {}".format(role, child_name))
            mlp = named[name]
            require(mlp.__class__.__name__ == "MS_Spiking_Mlp",
                    "M460 target is not MS_Spiking_Mlp: " + name)
            require(str(getattr(mlp, "norm_layer", "")) in
                    ("BN", "BNTT", "tdBN", "IN"),
                    "M460 target lacks post-FC2 normalization: " + name)
            for norm_role in ("bn1_norm", "bn2_norm"):
                norm = named[child_names[norm_role]]
                if self.enforce_h67_geometry:
                    require(getattr(norm, "track_running_stats", None) is False,
                            "M460 requires no_running {}: {}".format(
                                norm_role, name))
                    require(getattr(norm, "running_mean", "sentinel") is None and
                            getattr(norm, "running_var", "sentinel") is None,
                            "M460 dynamic BN buffers remain live: " + name)
            self.handles.append(mlp.register_forward_pre_hook(
                self._make_pre_hook(stage, block, name)))
            self.handles.append(named[child_names["sn1"]].register_forward_hook(
                self._make_source_hook(name, "sn1")))
            self.handles.append(named[child_names["sn2"]].register_forward_hook(
                self._make_source_hook(name, "sn2")))
            self.handles.append(named[child_names["fc2"]].register_forward_hook(
                self._make_fc2_hook(name)))
            self.handles.append(mlp.register_forward_hook(
                self._make_output_hook(stage, block, name)))
            self.installed.append(name)
        require(len(self.installed) == 12 and len(self.handles) == 60,
                "M460 exact 12-FFN/60-hook population drift")

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def begin_sample(self, sample_id, sample_key, sequence_key):
        require(self.current is None and not self.state,
                "M460 previous sample/call remains active")
        require(int(sample_id) == len({record["sample_id"]
                                      for record in self.records}),
                "M460 sample order drift")
        self.current = {
            "sample_id": int(sample_id),
            "sample_key": str(sample_key),
            "sequence_key": str(sequence_key),
        }
        self.calls = {name: 0 for name in self.installed}

    def end_sample(self):
        require(self.current is not None and not self.state,
                "M460 unfinished FFN call at sample end")
        require(all(count == 1 for count in self.calls.values()),
                "M460 FFN coverage drift sample={}".format(
                    self.current["sample_id"]))
        self.current = None
        self.calls = {}

    def _require_tensor(self, value, context):
        require(self.ops.is_tensor(value), "M460 non-tensor " + context)
        require(len(self.ops.shape(value)) == 5,
                "M460 rank drift {} {}".format(context, self.ops.shape(value)))

    def _make_pre_hook(self, stage, block, name):
        def hook(_module, inputs):
            require(self.current is not None, "M460 pre-hook outside sample")
            require(name not in self.state and isinstance(inputs, tuple) and
                    len(inputs) == 1, "M460 duplicate/invalid MLP pre-hook: " + name)
            value = inputs[0]
            self._require_tensor(value, name + " input")
            shape = self.ops.shape(value)
            if self.enforce_h67_geometry:
                require(shape[:-1] == STAGE_TOKEN_SHAPES[stage] and
                        shape[-1] == STAGE_CHANNELS[stage],
                        "M460 H67 FFN input geometry drift {} {}".format(
                            name, shape))
            self.state[name] = {
                "stage": stage,
                "block": block,
                "x": self.ops.vector_metrics(value),
                "shape": shape,
            }
        return hook

    def _make_source_hook(self, name, role):
        def hook(_module, _inputs, output):
            require(name in self.state and role not in self.state[name],
                    "M460 {} hook order/dup drift: {}".format(role, name))
            self._require_tensor(output, name + " " + role)
            self.state[name][role] = self.ops.source_metrics(output)
            self.state[name][role + "_shape"] = self.ops.shape(output)
        return hook

    def _make_fc2_hook(self, name):
        def hook(_module, _inputs, output):
            require(name in self.state and "pre_bn2" not in self.state[name],
                    "M460 fc2 hook order/dup drift: " + name)
            self._require_tensor(output, name + " fc2")
            self.state[name]["pre_bn2"] = self.ops.vector_metrics(output)
            self.state[name]["pre_bn2_shape"] = self.ops.shape(output)
        return hook

    def _make_output_hook(self, stage, block, name):
        def hook(_module, _inputs, output):
            require(name in self.state, "M460 full MLP output without pre-hook")
            call = self.state[name]
            require(all(key in call for key in ("sn1", "sn2", "pre_bn2")),
                    "M460 incomplete internal hook coverage: " + name)
            self._require_tensor(output, name + " post-BN2 residual")
            shape = self.ops.shape(output)
            require(shape == call["shape"] and
                    call["pre_bn2_shape"] == shape,
                    "M460 residual/pre-BN2 shape drift: " + name)
            require(call["sn1_shape"][:-1] == shape[:-1] and
                    call["sn2_shape"][:-1] == shape[:-1],
                    "M460 source token axes drift: " + name)
            channels = shape[-1]
            require(call["sn1_shape"][-1] == channels and
                    call["sn2_shape"][-1] == 4 * channels,
                    "M460 MS FFN expansion geometry drift: " + name)
            f_metrics = self.ops.vector_metrics(output)
            arrays = {
                "x_l1": np.asarray(call["x"]["l1"], dtype=np.float64),
                "x_l2_sq": np.asarray(call["x"]["l2_sq"], dtype=np.float64),
                "x_linf": np.asarray(call["x"]["linf"], dtype=np.float32),
                "sn1_nnz": np.asarray(call["sn1"]["nnz"], dtype=np.int32),
                "sn2_nnz": np.asarray(call["sn2"]["nnz"], dtype=np.int32),
                "pre_bn2_l1": np.asarray(
                    call["pre_bn2"]["l1"], dtype=np.float64),
                "f_exact_zero": np.asarray(
                    f_metrics["exact_zero"], dtype=np.bool_),
                "f_l1": np.asarray(f_metrics["l1"], dtype=np.float64),
                "f_l2_sq": np.asarray(f_metrics["l2_sq"], dtype=np.float64),
                "f_linf": np.asarray(f_metrics["linf"], dtype=np.float32),
            }
            finite = (np.asarray(call["x"]["finite"], dtype=np.bool_) &
                      np.asarray(call["sn1"]["finite"], dtype=np.bool_) &
                      np.asarray(call["sn2"]["finite"], dtype=np.bool_) &
                      np.asarray(call["pre_bn2"]["finite"], dtype=np.bool_) &
                      np.asarray(f_metrics["finite"], dtype=np.bool_))
            arrays["finite"] = finite
            arrays["rho"] = arrays["f_l1"] / np.maximum(
                arrays["x_l1"], DENOMINATOR_FLOOR)
            token_shape = shape[:-1]
            require(all(tuple(value.shape) == token_shape
                        for value in arrays.values()),
                    "M460 reduced token extent mismatch: " + name)
            self._write_call(stage, block, name, channels, arrays,
                             call["pre_bn2_shape"])
            self.calls[name] += 1
            del self.state[name]
        return hook

    def _write_call(self, stage, block, name, channels, arrays,
                    pre_bn2_shape):
        require(self.current is not None, "M460 write outside sample")
        sample_id = self.current["sample_id"]
        filename = "s{:02d}_stage{}_block{}_ffn_metrics.npz".format(
            sample_id, stage, block)
        path = self.output_dir / filename
        require(not path.exists(), "M460 refuses NPZ overwrite: " + str(path))
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as sealed:
            require(set(sealed.files) == set(arrays),
                    "M460 NPZ member population drift")
            for key, expected in arrays.items():
                require(np.array_equal(sealed[key], expected),
                        "M460 NPZ readback mismatch {} {}".format(filename, key))

        source_work = (arrays["sn1_nnz"].astype(np.int64) * (4 * channels) +
                       arrays["sn2_nnz"].astype(np.int64) * channels)
        threshold_rows = []
        for tau_name, tau in TAU_GRID:
            if tau == 0.0:
                strict = arrays["finite"] & arrays["f_exact_zero"]
                equal = strict.copy()
                inclusive = strict.copy()
                rule = "numeric_exact_zero_and_finite"
            else:
                strict = arrays["finite"] & (arrays["rho"] < tau)
                equal = arrays["finite"] & (arrays["rho"] == tau)
                inclusive = strict | equal
                rule = "finite_and_rho_strictly_less_than_tau"
            threshold_rows.append({
                "tau_name": tau_name,
                "tau": float(tau),
                "rule": rule,
                "strict_skip_tokens": int(np.count_nonzero(strict)),
                "equal_boundary_tokens": int(np.count_nonzero(equal)),
                "inclusive_skip_tokens": int(np.count_nonzero(inclusive)),
                "strict_source_work_oracle_saved": int(source_work[strict].sum()),
                "strict_dense_mac_oracle_saved": int(
                    np.count_nonzero(strict) * 8 * channels * channels),
                "strict_selected_f_l1_sum": float(arrays["f_l1"][strict].sum()),
                "strict_selected_f_l2_sq_sum": float(
                    arrays["f_l2_sq"][strict].sum()),
                "strict_selected_f_linf_max": float(
                    arrays["f_linf"][strict].max()) if np.any(strict) else 0.0,
            })
        record = dict(self.current)
        record.update({
            "stage": int(stage),
            "block": int(block),
            "module": name,
            "residual_boundary": "post_bn2_before_parent_sew_add",
            "pre_bn2_is_residual": False,
            "dynamic_bn_policy": "no_running/current-batch",
            "input_shape": [int(item) for item in pre_bn2_shape],
            "token_shape_t_n_h_w": [int(item) for item in arrays["f_l1"].shape],
            "channels": int(channels),
            "tokens": int(arrays["f_l1"].size),
            "finite_tokens": int(np.count_nonzero(arrays["finite"])),
            "npz": filename,
            "npz_sha256": sha256(path),
            "arrays": {key: array_receipt(value)
                       for key, value in sorted(arrays.items())},
            "tau_grid": threshold_rows,
        })
        self.records.append(record)


def dry_run(contract_path):
    contract, observed, workload = validate_contract(contract_path)
    payload = {
        "status": "PASS_M460_STATIC_EXACT_SHA_PREINPUT_DRY_RUN",
        "schema": contract["schema"],
        "identity_inputs": len(observed),
        "samples": len(workload),
        "sequence_keys": sorted(set(row["sequence_key"] for row in workload)),
        "ffn_modules": len(all_targets()),
        "hook_points": 5 * len(all_targets()),
        "residual_boundary": "post_bn2_before_parent_sew_add",
        "gpu_touched": False,
        "output_created": False,
        "training": False,
        "automatic_launch": False,
        "system_speedup": False,
        "headline": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def execute(contract_path, output_dir):
    contract, observed, workload = validate_contract(contract_path)
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "refusing to overwrite M460 output")
    output_dir.mkdir(parents=True)

    import torch
    require(torch.cuda.is_available(), "M460 capture requires CUDA")
    profile = load_module(Path(observed["profile_script"]["path"]),
                          "m460_profile")
    base = load_module(Path(observed["m40_loader"]["path"]),
                       "m460_m40_loader")
    config, device = profile.load_config(Path(observed["config"]["path"]))
    require(torch.device(device).type == "cuda", "M460 requires CUDA device")
    require(config.get("test", {}).get("bn_policy") == "no_running",
            "M460 frozen config BN policy drift")
    sample_keys = tuple(row["sample_key"] for row in workload)
    require(base.read_frozen_sample_keys(
        Path(observed["sample_workload"]["path"])) == sample_keys,
        "M460 M40 frozen loader identity drift")

    dataset = profile.DSECDatasetLite(
        config, file_list="valid", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1))
    observed_keys = tuple(
        "|".join(str(item) for item in row)
        if isinstance(row, (list, tuple)) else str(row)
        for row in dataset.files[:EXPECTED_SAMPLES])
    require(observed_keys == sample_keys,
            "M460 dataset first-ten identity/order drift")
    dataset_receipts = base.dataset_file_receipts(
        config["data"]["path"], sample_keys)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        pin_memory=False, num_workers=0)
    transform = None
    if config["loader"].get("crop") is not None:
        transform = profile.Compose([
            profile.CenterCrop(tuple(config["loader"]["crop"]))])

    model = profile.build_model(
        config, Path(observed["checkpoint"]["path"]), device)
    load_audit = profile.validate_h9_load_audit(model, config)
    require(load_audit.get("missing_count") == 0 and
            load_audit.get("unexpected_count") == 0,
            "M460 checkpoint load mismatch")
    require(not hasattr(model, "_m71_pattern_paft_state"),
            "M460 training-only PAFT state leaked into H67 capture")
    model.eval()
    bn_changed = profile.configure_batch_norm_evaluation(model, "no_running")
    require(bn_changed == EXPECTED_BN_MODULES_CHANGED,
            "M460 no_running BN population drift")

    capture = FFNResidualStreamCapture(
        TorchTokenOps(torch), output_dir, enforce_h67_geometry=True)
    capture.attach(model)
    processed = 0
    try:
        with torch.no_grad():
            for chunk, mask, label in itertools.islice(loader, EXPECTED_SAMPLES):
                row = workload[processed]
                profile.functional.reset_net(model)
                capture.begin_sample(
                    processed, row["sample_key"], row["sequence_key"])
                x, transformed_label, transformed_mask = profile.preprocess_chunk(
                    config, chunk, label, mask, transform, device)
                del transformed_label, transformed_mask
                model(x)
                torch.cuda.synchronize(device)
                capture.end_sample()
                processed += 1
                print("[M460 H67 G8 FFN residual] {}/{}".format(
                    processed, EXPECTED_SAMPLES), flush=True)
    finally:
        capture.detach()
    require(processed == EXPECTED_SAMPLES and len(capture.records) == 120,
            "M460 S10/12-FFN population incomplete")

    sample_csv = output_dir / "samples.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sample_id", "sample_key", "sequence_key"))
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames}
                         for row in workload)
    records_json = output_dir / "per_sample_module_manifest.json"
    records_json.write_text(json.dumps(
        {"schema": "m460_h67_g8_ffn_token_residual_payload_manifest_v1",
         "records": capture.records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    summary = {
        "schema": "m460_h67_g8_ffn_token_residual_s10_capture_v1",
        "status": "PASS_M460_H67_EP35_NO_RUNNING_S10_STREAM_CAPTURE",
        "identity": {
            "contract_path": str(Path(contract_path).resolve()),
            "contract_sha256": sha256(Path(contract_path).resolve()),
            "capture_script_sha256": sha256(Path(__file__).resolve()),
            "inputs": observed,
            "checkpoint_load_audit": load_audit,
            "source_config_bn_policy": "no_running",
            "capture_bn_policy": "no_running/current-batch",
            "bn_modules_changed": bn_changed,
            "model_mode": "eval",
            "dataset_input_files": dataset_receipts,
            "cuda_device_name": torch.cuda.get_device_name(device),
        },
        "population": {
            "samples": processed,
            "sequence_keys": ["zurich_city_09_a"],
            "ffn_modules": len(capture.installed),
            "sample_module_records": len(capture.records),
            "tokens": sum(record["tokens"] for record in capture.records),
            "expected_tokens": 5580000,
        },
        "semantics": {
            "ffn_order": "sn1-drop1(fc p=0)-fc1-bn1-sn2-drop2(p=0)-fc2-bn2",
            "residual": "captured full .mlp output is post-BN2 F(x), immediately before parent ADD sew; parent y=x+F(x)",
            "fc2": "captured separately only as pre-BN2 diagnostic; never substituted for F(x)",
            "bn": "no_running/current-batch over real call population; aggregate density is never substituted for token residual",
            "token_identity": "checkpoint,bn_policy,sample_id,sample_key,sequence_key,stage,block,t,n,h,w; C-order over [T,N,H,W]",
            "full_tensor_dumped": False,
        },
        "threshold_contract": {
            "rho": "||F_token||_1 / max(||x_token||_1, 2^-24)",
            "tau_grid": [{"name": name, "value": value}
                         for name, value in TAU_GRID],
            "tau0": "finite and every post-BN2 F channel numerically exactly zero",
            "positive_tau": "strict count uses finite and rho < tau; equality and inclusive counts are separate",
            "post_compute_oracle": True,
            "executable_precompute_skip": False,
        },
        "sensitivity_support": {
            "captured_injection_terms": [
                "per-token ||F||_1", "per-token ||F||_2^2",
                "per-token ||F||_infinity", "per-token ||x|| norms"],
            "local_identity": "skip emits x instead of x+F, hence local delta_y=-F exactly",
            "tail_bound_form": "||delta network output||_p <= L_tail(stage,block,p) * ||F||_p",
            "tail_lipschitz_certified_here": False,
            "aee_bound_certified_here": False,
        },
        "files": {
            "samples": sample_csv.name,
            "per_sample_module_manifest": records_json.name,
            "npz_payloads": len(capture.records),
        },
        "admission": {
            "checkpoint_bound_s10_capture": True,
            "token_skip_rate": True,
            "local_norm_budget": True,
            "precompute_certificate": False,
            "valid825_accuracy": False,
            "cycle_speedup": False,
            "energy": False,
            "rtl": False,
            "system_speedup": False,
            "headline": False,
            "training": False,
        },
        "claim_boundary": (
            "Frozen H67-ep35/no-running S10 post-compute oracle metrics only. "
            "No Delta-AEE, executable skip, cycle speedup, energy, PPA, "
            "system speedup or headline is admitted."),
    }
    summary_path = output_dir / "m460_h67_g8_ffn_token_residual_s10_capture.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    evidence = sorted(output_dir.glob("*.npz")) + [
        sample_csv, records_json, summary_path]
    manifest = output_dir / "manifest.sha256"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(path), path.name) for path in evidence),
        encoding="utf-8")
    require(sha256(Path(__file__).resolve()) ==
            contract["identity"]["capture_script"]["sha256"],
            "M460 capture script changed during run")
    require(sha256(resolve_path(contract["identity"]["docs359"]["path"])) ==
            contract["identity"]["docs359"]["sha256"],
            "protected docs/359 changed during M460")
    print("PASS M460 {} {}".format(summary_path, sha256(manifest)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require(args.dry_run != (args.output_dir is not None),
            "choose exactly one of --dry-run or --output-dir")
    if args.dry_run:
        dry_run(args.contract)
    else:
        execute(args.contract, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
