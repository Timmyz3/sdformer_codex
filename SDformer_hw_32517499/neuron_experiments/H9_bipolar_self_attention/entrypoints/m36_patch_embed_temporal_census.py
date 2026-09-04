#!/usr/bin/env python3
"""Intrusively census exact Local/Motion activity at patch-embed inputs."""

import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value):
    return int(value.detach().to(device="cpu").item())


def _count(mask):
    return _scalar(torch.count_nonzero(mask))


def census_tensor(tensor):
    value = tensor.detach()
    if value.dim() < 1 or int(value.shape[0]) != 10:
        raise ValueError("M36 target input must have temporal dimension T=10")
    finite = torch.isfinite(value) if value.is_floating_point() else torch.ones_like(
        value, dtype=torch.bool
    )
    local_active = value != 0
    temporal_changed = value[1:] != value[:-1]
    local_by_t = [_count(local_active[index]) for index in range(10)]
    changed_by_edge = [_count(temporal_changed[index]) for index in range(9)]
    temporal_representation_by_t = [local_by_t[0]] + changed_by_edge
    row_select_by_t = [local_by_t[0]] + [
        min(local_by_t[index + 1], changed_by_edge[index])
        for index in range(9)
    ]
    total_elements = int(value.numel())
    elements_per_t = int(value[0].numel())
    local_total = sum(local_by_t)
    motion_total = sum(temporal_representation_by_t)
    selected_total = sum(row_select_by_t)
    if value.is_floating_point():
        near_integer = _count(torch.isclose(
            value, torch.round(value), rtol=0.0, atol=1.0e-6
        ) & finite)
    else:
        near_integer = total_elements
    binary01 = _count(((value == 0) | (value == 1)) & finite)
    ternary = _count(((value == -1) | (value == 0) | (value == 1)) & finite)
    return {
        "dtype": str(value.dtype),
        "device": str(value.device),
        "shape": [int(item) for item in value.shape],
        "temporal_axis": 0,
        "temporal_steps": 10,
        "elements_per_temporal_step": elements_per_t,
        "total_elements": total_elements,
        "finite_elements": _count(finite),
        "binary01_elements": binary01,
        "ternary_elements": ternary,
        "near_integer_elements": near_integer,
        "local_nonzero_by_t": local_by_t,
        "temporal_changed_by_edge": changed_by_edge,
        "motion_initial_plus_delta_by_t": temporal_representation_by_t,
        "row_select_min_local_or_motion_by_t": row_select_by_t,
        "local_nonzero_total": local_total,
        "motion_initial_plus_delta_total": motion_total,
        "row_select_total": selected_total,
        "motion_element_ratio_vs_local": (
            motion_total / float(local_total) if local_total else None
        ),
        "row_select_element_ratio_vs_local": (
            selected_total / float(local_total) if local_total else None
        ),
        "statistics_only": True,
    }


class M36PatchEmbedCensus(object):
    def __init__(self, output_dir, contract, run_identity):
        self.output_dir = Path(output_dir).resolve()
        self.contract = contract
        self.run_identity = run_identity
        self.rows = []
        self.current_sample = None
        self.root_forwards = 0
        self.closed = False
        self.handles = []
        if self.output_dir.exists():
            raise ValueError("refusing to overwrite M36 output")
        self.output_dir.mkdir(parents=True)

    def attach(self, model):
        modules = dict(model.named_modules())
        for target in self.contract["targets"]:
            if target not in modules:
                raise ValueError("M36 target module missing: {}".format(target))

        def root_pre_hook(_module, _inputs):
            if self.current_sample is not None:
                raise RuntimeError("M36 nested root forward")
            self.current_sample = self.root_forwards

        def root_post_hook(_module, _inputs, _output):
            if self.current_sample != self.root_forwards:
                raise RuntimeError("M36 root sample identity drift")
            self.root_forwards += 1
            self.current_sample = None

        self.handles.append(model.register_forward_pre_hook(root_pre_hook))
        self.handles.append(model.register_forward_hook(root_post_hook))
        for target in self.contract["targets"]:
            def hook(_module, inputs, target_name=target):
                if self.current_sample is None or len(inputs) != 1:
                    raise RuntimeError("M36 target call outside root or arity drift")
                row = census_tensor(inputs[0])
                row["sample_id"] = self.current_sample
                row["target"] = target_name
                self.rows.append(row)
            self.handles.append(modules[target].register_forward_pre_hook(hook))

    def close(self, profile_json, sample_workload):
        if self.closed:
            raise RuntimeError("M36 writer already closed")
        self.closed = True
        for handle in self.handles:
            handle.remove()
        expected_samples = int(self.contract["samples"])
        expected_grid = {
            (sample, target)
            for sample in range(expected_samples)
            for target in self.contract["targets"]
        }
        observed_grid = {(row["sample_id"], row["target"]) for row in self.rows}
        failures = []
        if self.root_forwards != expected_samples:
            failures.append("root_forward_count")
        if len(self.rows) != len(expected_grid) or observed_grid != expected_grid:
            failures.append("sample_target_grid")
        profile_json = Path(profile_json).resolve()
        sample_workload = Path(sample_workload).resolve()
        if not profile_json.is_file() or not sample_workload.is_file():
            failures.append("postrun_profile_or_workload_missing")
        rows_path = self.output_dir / "m36_patch_embed_temporal_census.jsonl"
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in sorted(self.rows, key=lambda item: (
                    item["sample_id"], item["target"])):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest = {
            "schema": "m36_patch_embed_temporal_census_manifest_v1",
            "status": "PASS_EXACT_PATCH_EMBED_INPUT_CENSUS_NO_PERFORMANCE_ADMISSION"
                if not failures else "FAIL_M36_CENSUS",
            "failures": failures,
            "records": len(self.rows),
            "root_forwards": self.root_forwards,
            "run_identity": self.run_identity,
            "rows": {"path": str(rows_path), "sha256": sha256(rows_path)},
            "postrun_profile": {
                "path": str(profile_json), "sha256": sha256(profile_json)
            } if profile_json.is_file() else None,
            "sample_workload": {
                "path": str(sample_workload), "sha256": sha256(sample_workload)
            } if sample_workload.is_file() else None,
            "instrumentation": {
                "intrusive": True,
                "statistics_only": True,
                "performance_use_forbidden": True,
            },
            "claim_boundary": self.contract["claim_boundary"],
        }
        manifest_path = self.output_dir / "m36_patch_embed_temporal_census_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path, manifest

    def abort(self, exception):
        if self.closed:
            return
        self.closed = True
        for handle in self.handles:
            handle.remove()
        path = self.output_dir / "m36_patch_embed_temporal_census_abort.json"
        path.write_text(json.dumps({
            "status": "ABORTED",
            "exception_type": type(exception).__name__,
            "exception": str(exception),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
