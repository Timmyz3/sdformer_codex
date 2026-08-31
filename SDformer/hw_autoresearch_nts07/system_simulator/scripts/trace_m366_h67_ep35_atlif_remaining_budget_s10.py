#!/usr/bin/env python3
"""Stream exact H67-ep35 ATLIF remaining-budget statistics over frozen S10.

The capture never dumps full-resolution tensors.  It quantizes each live PSN
input with the already sealed checkpoint/site scales from the DP-TME vector
manifest, evaluates conservative signed-INT8 suffix bounds, and emits exact
histograms plus a bounded witness reservoir.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"


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
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path_text.startswith("neuron_experiments/"):
        return ROOT / path
    return HW / path


def signed_field(value, index, width):
    mask = (1 << width) - 1
    result = (value >> (index * width)) & mask
    if result & (1 << (width - 1)):
        result -= 1 << width
    return result


def read_hex_lines(path):
    return [int(line.strip(), 16) for line in Path(path).read_text(
        encoding="ascii").splitlines() if line.strip()]


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_contract(contract_path):
    contract_path = Path(contract_path).resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m366_h67_ep35_atlif_remaining_budget_s10_contract_v1",
            "M366 contract schema drift")
    observed = {}
    for name, record in contract["identity"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = resolve_path(record["path"]).resolve()
        require(path.is_file(), "M366 missing identity input {}: {}".format(
            name, path))
        actual = sha256(path)
        require(actual == record["sha256"],
                "M366 SHA drift {} expected={} observed={}".format(
                    name, record["sha256"], actual))
        observed[name] = {"path": str(path), "sha256": actual}
    script_sha = sha256(Path(__file__).resolve())
    require(script_sha == contract["identity"]["capture_script"]["sha256"],
            "M366 self SHA drift")
    return contract, observed


def decode_static_sites(contract, observed):
    manifest_path = Path(observed["dptme_manifest"]["path"])
    manifest = strict_json(manifest_path)
    require(manifest.get("schema") == "checkpoint_atlif_dptme_vectors_v1",
            "DP-TME manifest schema drift")
    require(manifest["summary"]["live_sites"] == 81 and
            manifest["summary"]["live_t10_sites"] == 45 and
            manifest["summary"]["live_t2_sites"] == 36,
            "DP-TME live-site population drift")
    vector_dir = manifest_path.parent
    weight_lines = read_hex_lines(vector_dir / "weight.mem")
    bias_lines = read_hex_lines(vector_dir / "bias.mem")
    threshold_lines = read_hex_lines(vector_dir / "threshold.mem")
    require(len(weight_lines) == 522 and len(bias_lines) == 81 and
            len(threshold_lines) == 81,
            "DP-TME packed stream population drift")

    sites = {}
    cycle_base = 0
    for command_index, command in enumerate(manifest["commands"]):
        name = command["name"]
        temporal = int(command["temporal_steps"])
        require(name not in sites and temporal in (2, 10),
                "duplicate/invalid DP-TME site")
        weight = [[0 for _ in range(temporal)]
                  for _ in range(temporal)]
        for source_time in range(temporal):
            packed = weight_lines[cycle_base + source_time]
            for output_time in range(temporal):
                weight[output_time][source_time] = signed_field(
                    packed, output_time, 8)
        bias = [signed_field(bias_lines[command_index], t, 24)
                for t in range(temporal)]
        threshold = signed_field(
            threshold_lines[command_index], 0, 24)
        order = sorted(
            range(temporal),
            key=lambda source_time: -sum(
                abs(weight[output_time][source_time])
                for output_time in range(temporal)))
        suffix_min = [[0 for _ in range(temporal + 1)]
                      for _ in range(temporal)]
        suffix_max = [[0 for _ in range(temporal + 1)]
                      for _ in range(temporal)]
        for output_time in range(temporal):
            for k in range(temporal - 1, -1, -1):
                value = weight[output_time][order[k]]
                a = value * -128
                b = value * 127
                suffix_min[output_time][k] = (
                    suffix_min[output_time][k + 1] + min(a, b))
                suffix_max[output_time][k] = (
                    suffix_max[output_time][k + 1] + max(a, b))
        sites[name] = {
            "name": name,
            "temporal_steps": temporal,
            "x_scale": float(command["x_scale"]),
            "weight_scale": float(command["weight_scale"]),
            "accumulator_scale": float(command["accumulator_scale"]),
            "weight_q8": weight,
            "bias_q24": bias,
            "threshold_q24": threshold,
            "issue_order": order,
            "suffix_min": suffix_min,
            "suffix_max": suffix_max,
        }
        cycle_base += temporal
    require(cycle_base == len(weight_lines),
            "DP-TME weight stream not fully consumed")
    return manifest, sites


def pure_python_self_test():
    weight = [[3, -2, 5], [-4, 1, 2], [1, 1, -1]]
    x = [4, -3, 2]
    bias = [1, -2, 0]
    threshold = [5, 1, 3]
    order = [2, 0, 1]
    lower = [-8, -8, -8]
    upper = [7, 7, 7]
    final = [bias[t] + sum(weight[t][s] * x[s] for s in range(3))
             for t in range(3)]
    for t in range(3):
        partial = bias[t]
        resolved = None
        for k in range(4):
            rmin = 0
            rmax = 0
            for j in range(k, 3):
                source = order[j]
                a = weight[t][source] * lower[source]
                b = weight[t][source] * upper[source]
                rmin += min(a, b)
                rmax += max(a, b)
            low = partial + rmin
            high = partial + rmax
            require(low <= final[t] <= high,
                    "synthetic remaining-bound violation")
            if low >= threshold[t]:
                resolved = True
                break
            if high < threshold[t]:
                resolved = False
                break
            if k < 3:
                source = order[k]
                partial += weight[t][source] * x[source]
        require(resolved == (final[t] >= threshold[t]),
                "synthetic early decision mismatch")
    return {"vectors": 3, "bound_violations": 0,
            "decision_mismatches": 0}


def dry_run(contract_path):
    contract, observed = validate_contract(contract_path)
    manifest, sites = decode_static_sites(contract, observed)
    result = {
        "status": "PASS_M366_STATIC_EXACT_SHA_AND_PROOF_DRY_RUN",
        "identity_inputs": len(observed),
        "live_sites": len(sites),
        "live_t10_sites": sum(
            site["temporal_steps"] == 10 for site in sites.values()),
        "live_t2_sites": sum(
            site["temporal_steps"] == 2 for site in sites.values()),
        "packed_weight_cycles": sum(
            int(command["temporal_steps"])
            for command in manifest["commands"]),
        "self_test": pure_python_self_test(),
        "gpu_touched": False,
        "output_created": False,
        "headline": False,
        "system_speedup": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def new_counter(site):
    temporal = site["temporal_steps"]
    return {
        "calls": 0,
        "samples": 0,
        "spatial_lanes": 0,
        "output_lanes": 0,
        "input_values": 0,
        "input_nonfinite": 0,
        "signed_q8_range_violations": 0,
        "symmetric_q8_range_violations": 0,
        "unclamped_code_min": None,
        "unclamped_code_max": None,
        "resolved_at_k": [0 for _ in range(temporal + 1)],
        "positive_resolved_at_k": [0 for _ in range(temporal + 1)],
        "zero_resolved_at_k": [0 for _ in range(temporal + 1)],
        "term_total": 0,
        "term_skipped": 0,
        "bound_violations": 0,
        "integer_early_mismatches": 0,
        "integer_vs_float_event_mismatches": 0,
        "unresolved_after_t": 0,
        "tile_resolved_at_k": [0 for _ in range(temporal + 1)],
        "baseline_issue_cycles": 0,
        "lane_compaction_need_by_step": [0 for _ in range(temporal)],
        "lane_compaction_issue_cycles": 0,
    }


def add_counter(destination, source):
    for key in (
            "calls", "samples", "spatial_lanes", "output_lanes",
            "input_values", "input_nonfinite",
            "signed_q8_range_violations",
            "symmetric_q8_range_violations", "term_total",
            "term_skipped", "bound_violations",
            "integer_early_mismatches",
            "integer_vs_float_event_mismatches", "unresolved_after_t",
            "baseline_issue_cycles", "lane_compaction_issue_cycles"):
        destination[key] += int(source[key])
    for key in ("resolved_at_k", "positive_resolved_at_k",
                "zero_resolved_at_k", "tile_resolved_at_k",
                "lane_compaction_need_by_step"):
        require(len(destination[key]) == len(source[key]),
                "counter width drift")
        destination[key] = [a + int(b) for a, b in zip(
            destination[key], source[key])]
    value = source["unclamped_code_min"]
    if value is not None:
        if destination["unclamped_code_min"] is None:
            destination["unclamped_code_min"] = int(value)
        else:
            destination["unclamped_code_min"] = min(
                destination["unclamped_code_min"], int(value))
    value = source["unclamped_code_max"]
    if value is not None:
        if destination["unclamped_code_max"] is None:
            destination["unclamped_code_max"] = int(value)
        else:
            destination["unclamped_code_max"] = max(
                destination["unclamped_code_max"], int(value))


class RemainingBudgetCapture(object):
    def __init__(self, torch, sites, chunk_columns, lane_width,
                 witnesses_per_stratum):
        self.torch = torch
        self.sites = sites
        self.chunk_columns = chunk_columns
        self.lane_width = lane_width
        self.witnesses_per_stratum = witnesses_per_stratum
        self.handles = []
        self.installed_names = set()
        self.current_sample_id = None
        self.current_sample_key = None
        self.current_calls = {}
        self.called_by_sample = []
        self.dead_called = set()
        self.rows = []
        self.aggregate = {name: new_counter(site)
                          for name, site in sites.items()}
        self.witnesses = {name: {"early": [], "never": [], "float_flip": []}
                          for name in sites}

    def attach(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ != "ATLIFTernaryPSN":
                continue
            self.installed_names.add(name)
            self.handles.append(module.register_forward_hook(
                self._make_hook(name)))
        require(len(self.installed_names) == 105,
                "installed ATLIF population drift: {}".format(
                    len(self.installed_names)))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def begin_sample(self, sample_id, sample_key):
        require(self.current_sample_id is None,
                "previous sample not closed")
        self.current_sample_id = int(sample_id)
        self.current_sample_key = str(sample_key)
        self.current_calls = {}

    def end_sample(self):
        require(set(self.current_calls) == set(self.sites),
                "live ATLIF coverage drift sample={} missing={} extra={}".format(
                    self.current_sample_id,
                    sorted(set(self.sites) - set(self.current_calls))[:8],
                    sorted(set(self.current_calls) - set(self.sites))[:8]))
        require(all(value == 1 for value in self.current_calls.values()),
                "live ATLIF called more than once")
        self.called_by_sample.append(sorted(self.current_calls))
        self.current_sample_id = None
        self.current_sample_key = None
        self.current_calls = {}

    def _validate_module_static(self, name, module, site):
        torch = self.torch
        temporal = site["temporal_steps"]
        require(int(module.T) == temporal,
                "ATLIF T drift {}".format(name))
        require(str(module.output_mode) == "binary" and
                str(module.threshold_mode) == "official_atlif",
                "ATLIF output contract drift {}".format(name))
        require(str(module.center_mode) == "zero",
                "ATLIF center contract drift {}".format(name))
        require(int(getattr(module, "temporal_factor_rank", 0)) == 0,
                "M366 dense proof cannot consume factorized ATLIF {}".format(name))
        require(tuple(module.weight.shape) == (temporal, temporal),
                "ATLIF weight shape drift {}".format(name))
        require(int(torch.count_nonzero(module.center.detach()).item()) == 0,
                "ATLIF center nonzero {}".format(name))
        weight_q = torch.round(
            module.weight.detach().float().cpu() /
            site["weight_scale"]).clamp(-128, 127).to(torch.int64).tolist()
        require(weight_q == site["weight_q8"],
                "ATLIF weight code drift {}".format(name))
        bias_q = torch.round(
            module.bias.detach().float().reshape(-1).cpu() /
            site["accumulator_scale"]).clamp(
                -(1 << 23), (1 << 23) - 1).to(torch.int64).tolist()
        require(bias_q == site["bias_q24"],
                "ATLIF bias code drift {}".format(name))
        threshold_q = int(torch.round(
            module.thresh.detach().float().cpu() /
            site["accumulator_scale"]).clamp(
                -(1 << 23), (1 << 23) - 1).item())
        require(threshold_q == site["threshold_q24"],
                "ATLIF threshold code drift {}".format(name))

    def _capture_witnesses(self, name, sample_id, column_base, x_q,
                           resolve_k, full_hidden, fixed_event, float_event):
        torch = self.torch
        temporal = self.sites[name]["temporal_steps"]
        lane_resolve = resolve_k.max(dim=0)[0]
        flip_lane = fixed_event.ne(float_event).any(dim=0)
        masks = {
            "early": lane_resolve.lt(temporal),
            "never": lane_resolve.eq(temporal),
            "float_flip": flip_lane,
        }
        for stratum, mask in masks.items():
            reservoir = self.witnesses[name][stratum]
            remaining = self.witnesses_per_stratum - len(reservoir)
            if remaining <= 0:
                continue
            indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if int(indices.numel()) == 0:
                continue
            indices = indices[:remaining].detach().cpu().tolist()
            for index in indices:
                reservoir.append({
                    "sample_id": int(sample_id),
                    "column": int(column_base + index),
                    "x_q8": [int(value) for value in
                              x_q[:, index].detach().cpu().tolist()],
                    "resolved_at_k_by_output": [int(value) for value in
                        resolve_k[:, index].detach().cpu().tolist()],
                    "final_hidden_q24": [int(value) for value in
                        full_hidden[:, index].detach().cpu().tolist()],
                    "integer_event": [bool(value) for value in
                        fixed_event[:, index].detach().cpu().tolist()],
                    "float_model_event": [bool(value) for value in
                        float_event[:, index].detach().cpu().tolist()],
                })

    def _analyze(self, name, module, x_seq, output):
        torch = self.torch
        site = self.sites[name]
        self._validate_module_static(name, module, site)
        temporal = site["temporal_steps"]
        require(int(x_seq.shape[0]) == temporal and
                tuple(output.shape) == tuple(x_seq.shape),
                "ATLIF input/output shape drift {}".format(name))
        flat = x_seq.detach().reshape(temporal, -1)
        out_flat = output.detach().reshape(temporal, -1)
        columns = int(flat.shape[1])
        row = new_counter(site)
        row["calls"] = 1
        row["samples"] = 1
        row["spatial_lanes"] = columns
        row["output_lanes"] = columns * temporal
        row["input_values"] = columns * temporal
        row["baseline_issue_cycles"] = (
            int(math.ceil(float(columns) / self.lane_width)) * temporal)

        device = flat.device
        weight = torch.tensor(site["weight_q8"], device=device,
                              dtype=torch.float32)
        bias = torch.tensor(site["bias_q24"], device=device,
                            dtype=torch.float32).reshape(temporal, 1)
        threshold = torch.full(
            (temporal, 1), float(site["threshold_q24"]), device=device,
            dtype=torch.float32)
        suffix_min = torch.tensor(site["suffix_min"], device=device,
                                  dtype=torch.float32)
        suffix_max = torch.tensor(site["suffix_max"], device=device,
                                  dtype=torch.float32)
        order = site["issue_order"]

        for begin in range(0, columns, self.chunk_columns):
            end = min(columns, begin + self.chunk_columns)
            source = flat[:, begin:end].float()
            float_event = out_flat[:, begin:end].ne(0)
            finite = torch.isfinite(source)
            nonfinite = int((~finite).sum().item())
            row["input_nonfinite"] += nonfinite
            safe = torch.where(finite, source, torch.zeros_like(source))
            unclamped = torch.round(safe / site["x_scale"])
            row["signed_q8_range_violations"] += int(
                (unclamped.lt(-128) | unclamped.gt(127)).sum().item())
            row["symmetric_q8_range_violations"] += int(
                (unclamped.lt(-127) | unclamped.gt(127)).sum().item())
            local_min = int(unclamped.min().item())
            local_max = int(unclamped.max().item())
            if row["unclamped_code_min"] is None:
                row["unclamped_code_min"] = local_min
                row["unclamped_code_max"] = local_max
            else:
                row["unclamped_code_min"] = min(
                    row["unclamped_code_min"], local_min)
                row["unclamped_code_max"] = max(
                    row["unclamped_code_max"], local_max)
            x_q = unclamped.clamp(-128, 127).float()

            full_hidden = bias.expand(-1, end - begin).clone()
            for source_time in range(temporal):
                full_hidden = full_hidden + (
                    weight[:, source_time:source_time + 1] *
                    x_q[source_time:source_time + 1, :])
            require(float(full_hidden.abs().max().item()) < (1 << 23),
                    "Acc24 final overflow {}".format(name))
            fixed_event = full_hidden.ge(threshold)

            partial = bias.expand(-1, end - begin).clone()
            resolved = torch.zeros_like(fixed_event)
            early_event = torch.zeros_like(fixed_event)
            resolve_k = torch.full_like(
                full_hidden, temporal, dtype=torch.int16)
            for k in range(temporal + 1):
                lower = partial + suffix_min[:, k:k + 1]
                upper = partial + suffix_max[:, k:k + 1]
                row["bound_violations"] += int((
                    full_hidden.lt(lower) | full_hidden.gt(upper)).sum().item())
                positive = lower.ge(threshold)
                zero = upper.lt(threshold)
                new = (~resolved) & (positive | zero)
                positive_new = new & positive
                zero_new = new & zero
                count = int(new.sum().item())
                row["resolved_at_k"][k] += count
                row["positive_resolved_at_k"][k] += int(
                    positive_new.sum().item())
                row["zero_resolved_at_k"][k] += int(zero_new.sum().item())
                resolve_k[new] = k
                early_event[positive_new] = True
                resolved |= new
                if k < temporal:
                    source_time = order[k]
                    partial = partial + (
                        weight[:, source_time:source_time + 1] *
                        x_q[source_time:source_time + 1, :])
                    require(float(partial.abs().max().item()) < (1 << 23),
                            "Acc24 reordered-prefix overflow {}".format(name))

            row["unresolved_after_t"] += int((~resolved).sum().item())
            row["integer_early_mismatches"] += int(
                early_event.ne(fixed_event).sum().item())
            row["integer_vs_float_event_mismatches"] += int(
                fixed_event.ne(float_event).sum().item())
            row["term_total"] += int(resolve_k.numel()) * temporal
            row["term_skipped"] += int(
                (temporal - resolve_k.to(torch.int32)).sum().item())

            lane_resolve = resolve_k.max(dim=0)[0]
            for step in range(temporal):
                row["lane_compaction_need_by_step"][step] += int(
                    lane_resolve.gt(step).sum().item())
            pad = (-int(lane_resolve.numel())) % self.lane_width
            if pad:
                lane_resolve = torch.cat((lane_resolve,
                    torch.zeros(pad, device=device, dtype=lane_resolve.dtype)))
            tile_rank = lane_resolve.reshape(-1, self.lane_width).max(dim=1)[0]
            tile_hist = torch.bincount(
                tile_rank.to(torch.int64), minlength=temporal + 1)
            values = tile_hist.detach().cpu().tolist()
            row["tile_resolved_at_k"] = [a + int(b) for a, b in zip(
                row["tile_resolved_at_k"], values)]
            self._capture_witnesses(
                name, self.current_sample_id, begin, x_q, resolve_k,
                full_hidden, fixed_event, float_event)

        row["lane_compaction_issue_cycles"] = sum(
            int(math.ceil(float(value) / self.lane_width))
            for value in row["lane_compaction_need_by_step"])
        row.update({
            "sample_id": self.current_sample_id,
            "sample_key": self.current_sample_key,
            "name": name,
            "temporal_steps": temporal,
            "input_shape": [int(value) for value in x_seq.shape],
        })
        self.rows.append(row)
        add_counter(self.aggregate[name], row)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            require(self.current_sample_id is not None,
                    "ATLIF hook fired outside sample")
            if name not in self.sites:
                self.dead_called.add(name)
                return
            require(inputs and self.torch.is_tensor(inputs[0]) and
                    self.torch.is_tensor(output),
                    "ATLIF hook tensor contract drift {}".format(name))
            self.current_calls[name] = self.current_calls.get(name, 0) + 1
            self._analyze(name, module, inputs[0], output)
        return hook


def finalize_counters(counters):
    result = {}
    for name, value in counters.items():
        row = dict(value)
        row["term_skip_ratio"] = (
            float(row["term_skipped"]) / row["term_total"]
            if row["term_total"] else 0.0)
        row["lane_compaction_issue_cycle_reduction"] = (
            1.0 - float(row["lane_compaction_issue_cycles"]) /
            row["baseline_issue_cycles"]
            if row["baseline_issue_cycles"] else 0.0)
        result[name] = row
    return result


def execute(contract_path, output_dir):
    contract, observed = validate_contract(contract_path)
    dptme_manifest, sites = decode_static_sites(contract, observed)
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "refusing to overwrite M366 output")
    output_dir.mkdir(parents=True)

    import torch
    require(torch.cuda.is_available(), "M366 requires CUDA")
    profile = load_module(Path(observed["profile_script"]["path"]),
                          "m366_profile")
    base = load_module(Path(observed["m248_base_tracer"]["path"]),
                       "m366_m248_base")
    config, device = profile.load_config(Path(observed["config"]["path"]))
    require(device.type == "cuda", "profile selected non-CUDA device")
    source_bn_policy = config.get("test", {}).get("bn_policy", "running")
    require(source_bn_policy == "no_running",
            "H67 paper config BN policy drift")

    sample_keys = base.read_frozen_sample_keys(
        Path(observed["sample_workload"]["path"]))
    require(len(sample_keys) == contract["runtime"]["samples"],
            "S10 sample population drift")
    dataset = profile.DSECDatasetLite(
        config, file_list="valid", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1))
    observed_keys = tuple(
        "|".join(str(item) for item in row)
        if isinstance(row, (list, tuple)) else str(row)
        for row in dataset.files[:len(sample_keys)])
    require(observed_keys == sample_keys,
            "dataset first-ten identity/order drift")
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
            "H67 checkpoint load mismatch")
    require(not hasattr(model, "_m71_pattern_paft_state"),
            "training-only PAFT state leaked into H67 capture")
    bn_changed = profile.configure_batch_norm_evaluation(model, "no_running")

    capture = RemainingBudgetCapture(
        torch, sites, int(contract["runtime"]["chunk_columns"]),
        int(contract["runtime"]["lane_width"]),
        int(contract["runtime"]["witnesses_per_stratum"]))
    capture.attach(model)
    processed = 0
    try:
        with torch.no_grad():
            for chunk, mask, label in itertools.islice(loader, len(sample_keys)):
                profile.functional.reset_net(model)
                capture.begin_sample(processed, observed_keys[processed])
                x, transformed_label, transformed_mask = profile.preprocess_chunk(
                    config, chunk, label, mask, transform, device)
                del transformed_label, transformed_mask
                model(x)
                capture.end_sample()
                processed += 1
                print("[M366 H67 ep35 ATLIF bound] {}/{}".format(
                    processed, len(sample_keys)), flush=True)
    finally:
        capture.detach()
    require(processed == len(sample_keys), "M366 S10 incomplete")

    aggregate = finalize_counters(capture.aggregate)
    t10_names = sorted(name for name, site in sites.items()
                       if site["temporal_steps"] == 10)
    t2_names = sorted(name for name, site in sites.items()
                      if site["temporal_steps"] == 2)

    def sum_field(names, key):
        return sum(int(aggregate[name][key]) for name in names)

    def combined(names):
        term_total = sum_field(names, "term_total")
        term_skipped = sum_field(names, "term_skipped")
        base_cycles = sum_field(names, "baseline_issue_cycles")
        compact_cycles = sum_field(names, "lane_compaction_issue_cycles")
        return {
            "sites": len(names),
            "calls": sum_field(names, "calls"),
            "spatial_lanes": sum_field(names, "spatial_lanes"),
            "output_lanes": sum_field(names, "output_lanes"),
            "term_total": term_total,
            "term_skipped": term_skipped,
            "term_skip_ratio": (float(term_skipped) / term_total
                                if term_total else 0.0),
            "baseline_issue_cycles": base_cycles,
            "lane_compaction_issue_cycles": compact_cycles,
            "lane_compaction_issue_cycle_reduction": (
                1.0 - float(compact_cycles) / base_cycles
                if base_cycles else 0.0),
            "signed_q8_range_violations": sum_field(
                names, "signed_q8_range_violations"),
            "symmetric_q8_range_violations": sum_field(
                names, "symmetric_q8_range_violations"),
            "input_nonfinite": sum_field(names, "input_nonfinite"),
            "bound_violations": sum_field(names, "bound_violations"),
            "integer_early_mismatches": sum_field(
                names, "integer_early_mismatches"),
            "integer_vs_float_event_mismatches": sum_field(
                names, "integer_vs_float_event_mismatches"),
        }

    t10 = combined(t10_names)
    t2 = combined(t2_names)
    fixed_cycles = int(contract["performance_context"][
        "fixed_compute_reference_cycles"])
    dense_t10_cycles = int(contract["performance_context"][
        "dense_t10_atlif_cycles"])
    reduction = t10["lane_compaction_issue_cycle_reduction"]
    fixed_context_speedup = float(fixed_cycles) / (
        fixed_cycles - dense_t10_cycles * reduction)
    gates = contract["promotion_gates"]
    promotion = {
        "zero_mismatch": t10["integer_early_mismatches"] == 0,
        "zero_bound_violation": t10["bound_violations"] == 0,
        "zero_range_violation": (
            t10["signed_q8_range_violations"] == 0 and
            t10["input_nonfinite"] == 0),
        "term_skip": t10["term_skip_ratio"] >= gates["min_term_skip_ratio"],
        "executable_issue_cycle": (
            reduction >= gates["min_executable_issue_cycle_reduction"]),
        "fixed_context": (
            fixed_context_speedup >= gates["min_fixed_context_speedup"]),
    }
    # S10 opportunity capture can admit the arithmetic/cycle screen, but it
    # does not measure the suffix-table/config traffic or compare/compaction
    # energy.  M360 makes positive *net* energy a mandatory RTL gate, so keep
    # RTL fail-closed until a separately sealed energy audit charges it.
    promotion["metric_gates_pass"] = all(promotion.values())
    promotion["metadata_and_compare_net_energy_positive"] = False
    promotion["all_pass"] = (
        promotion["metric_gates_pass"] and
        promotion["metadata_and_compare_net_energy_positive"])

    manifest = {
        "schema": "m366_h67_ep35_atlif_remaining_budget_s10_capture_v1",
        "status": ("PASS_M366_G12_RTL_PROMOTION" if promotion["all_pass"]
                   else ("PASS_M366_METRIC_GATES__ENERGY_AUDIT_REQUIRED__NO_GO_RTL"
                         if promotion["metric_gates_pass"]
                         else "PASS_M366_CAPTURE__NO_GO_G12_RTL")),
        "identity": {
            "contract_path": str(Path(contract_path).resolve()),
            "contract_sha256": sha256(Path(contract_path).resolve()),
            "capture_script_sha256": sha256(Path(__file__).resolve()),
            "inputs": observed,
            "checkpoint_load_audit": load_audit,
            "source_config_bn_policy": source_bn_policy,
            "capture_bn_policy": "no_running",
            "bn_modules_changed": bn_changed,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "dataset_root": str(Path(config["data"]["path"]).resolve()),
            "dataset_input_files": dataset_receipts,
            "dptme_numeric_contract": dptme_manifest["numeric_contract"],
        },
        "population": {
            "samples": processed,
            "sample_keys": list(sample_keys),
            "installed_atlif_modules": len(capture.installed_names),
            "live_sites": len(sites),
            "live_t10_sites": len(t10_names),
            "live_t2_sites": len(t2_names),
            "dead_called_sites": sorted(capture.dead_called),
        },
        "numeric_contract": {
            "input_scale": "sealed sample0/full-call per-site power-of-two scale from DP-TME manifest",
            "input_code_domain": "signed INT8 [-128,127] with saturation counted as a range violation",
            "weight": "sealed per-site power-of-two signed INT8",
            "bias_threshold_accumulator": "signed Acc24",
            "bounds": "full signed-INT8 per-source interval [-128,127]",
            "issue_order": "per-site descending sum_t(abs(W_q[t,s]))",
            "reference": "quantized integer event; float-model event flips are reported separately",
        },
        "t10_nonattention_main": t10,
        "t2_attention_diagnostic": t2,
        "fixed_compute_projection": {
            "fixed_compute_reference_cycles": fixed_cycles,
            "dense_t10_atlif_cycles": dense_t10_cycles,
            "measured_lane_compaction_issue_cycle_reduction": reduction,
            "conditional_fixed_context_speedup": fixed_context_speedup,
            "system_speedup_admitted": False,
        },
        "promotion_gates": {
            "thresholds": gates,
            "observed": promotion,
            "rtl_decision": ("GO_RTL" if promotion["all_pass"] else
                             ("GO_ENERGY_AUDIT__NO_GO_RTL"
                              if promotion["metric_gates_pass"] else
                              "NO_GO_RTL")),
        },
        "site_aggregate": aggregate,
        "sample_site_rows": capture.rows,
        "witnesses": capture.witnesses,
        "static_site_codes": sites,
        "admission": {
            "s10_exact_integer_bound_audit": True,
            "representative_s10_rate": True,
            "accuracy": False,
            "rtl": False,
            "vcs": False,
            "synopsys_ppa": False,
            "energy": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": (
            "Exact signed-INT8 remaining-budget opportunity over the frozen "
            "H67-ep35/no-running S10 cohort, with a 32-lane within-context "
            "compaction cycle proxy. It is not float/deployment accuracy, RTL, "
            "memory-system cycles, energy, PPA, system speedup or a headline."),
    }
    output_path = output_dir / "m366_h67_ep35_atlif_remaining_budget_s10_capture.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(sha256(Path(__file__).resolve()) ==
            contract["identity"]["capture_script"]["sha256"],
            "M366 capture script changed during run")
    print(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
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
    main()
