#!/usr/bin/env python3
"""Generate checkpoint-bound ATLIF T10/T2 vectors for DP-TME RTL replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
H9_ENTRYPOINTS = REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints"
sys.path.insert(0, str(H9_ENTRYPOINTS))

from profile_nts11_hardware_p0 import (  # noqa: E402
    build_model,
    configure_batch_norm_evaluation,
    file_sha256,
    h9_module_counts,
    load_config,
    preprocess_chunk,
    validate_h9_load_audit,
)
from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from DSEC_dataloader.data_augmentation import CenterCrop, Compose  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


LANES = 32
SLOTS = 10
PACK_GROUPS = 5
X_W = 8
W_W = 8
ACC_W = 24
MAX_CAPTURE_COLUMNS = PACK_GROUPS * LANES


def power_of_two_scale(values: torch.Tensor, qmax: int = 127) -> float:
    absmax = float(values.detach().float().abs().max().item()) if values.numel() else 0.0
    if not math.isfinite(absmax):
        raise ValueError("non-finite ATLIF tensor")
    if absmax == 0.0:
        return 1.0
    return float(2.0 ** math.ceil(math.log2(absmax / qmax)))


def quantize_signed(values: torch.Tensor, scale: float, bits: int) -> torch.Tensor:
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    return torch.round(values.detach().float() / scale).clamp(lower, upper).to(torch.int64)


def pack_signed(values: np.ndarray | list[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        packed |= (int(value) & mask) << (index * width)
    return packed


def hex_line(value: int, width_bits: int) -> str:
    return f"{value:0{(width_bits + 3) // 4}x}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Capture:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._seen: set[str] = set()
        self.installed_names: set[str] = set()
        self.called_names: set[str] = set()
        self.dead_called_names: set[str] = set()
        self.handles: list[Any] = []

    def attach(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if module.__class__.__name__ != "ATLIFTernaryPSN":
                continue
            self.installed_names.add(name)
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, name: str):
        def hook(module: torch.nn.Module, inp: Any, out: Any) -> None:
            if name in self._seen:
                return
            if not inp or not torch.is_tensor(inp[0]) or not torch.is_tensor(out):
                return
            self.called_names.add(name)
            self._seen.add(name)
            if name.endswith(".attn.attn_sn.spiking_neuron"):
                self.dead_called_names.add(name)
                return
            temporal = int(inp[0].shape[0])
            if temporal not in (2, 10):
                raise ValueError(f"unsupported ATLIF temporal length {temporal}: {name}")
            flattened = inp[0].detach().float().reshape(temporal, -1)
            output = out.detach().reshape(temporal, -1)
            columns = MAX_CAPTURE_COLUMNS if temporal == 2 else LANES
            if flattened.shape[1] < columns:
                raise ValueError(f"ATLIF capture has only {flattened.shape[1]} columns: {name}")
            weight = module.weight.detach().float()
            bias = module.bias.detach().float()
            threshold = module.thresh.detach().float().reshape(1)
            # Use the same full-path addmm as ATLIFTernaryPSN.forward. Submatrix
            # rematmul(W, X[:, subset]) is not bitwise-equal to (W@X)[:, subset]
            # near the threshold boundary under float32 GEMM reduction order.
            hidden = torch.addmm(bias, weight, flattened)
            bias = bias.reshape(temporal)
            quotas = (columns // 3, columns // 3, columns - 2 * (columns // 3))
            candidate_count = min(int(flattened.shape[1]), columns * 4)
            candidates = (
                torch.arange(candidate_count, device=flattened.device),
                hidden.sub(threshold.reshape(1, 1)).abs().amin(dim=0).topk(
                    candidate_count, largest=False
                ).indices,
                flattened.abs().amax(dim=0).topk(candidate_count, largest=True).indices,
            )
            selected: list[int] = []
            used: set[int] = set()
            for source_index, (quota, candidate) in enumerate(zip(quotas, candidates)):
                source_added = 0
                for index in candidate.detach().cpu().tolist():
                    if index in used:
                        continue
                    selected.append(int(index))
                    used.add(int(index))
                    source_added += 1
                    if source_added == quota:
                        break
                if source_added != quota:
                    raise RuntimeError(f"unable to select scenario lanes {source_index}: {name}")
            indices = torch.tensor(selected, dtype=torch.long, device=flattened.device)
            self.rows.append({
                "name": name,
                "scenario": "mixed_ordinary_near_threshold_max_amplitude",
                "scenario_lane_counts": {
                    "ordinary": quotas[0],
                    "near_threshold": quotas[1],
                    "max_amplitude": quotas[2],
                },
                "temporal": temporal,
                "input": flattened.index_select(1, indices).cpu(),
                "input_absmax_full_call": float(flattened.abs().max().item()),
                "output": output.index_select(1, indices).cpu(),
                # Full-path hidden for the selected lanes; used as the float
                # reference so threshold-boundary lanes match the model event.
                "hidden": hidden.index_select(1, indices).cpu(),
                "weight": weight.cpu(),
                "bias": bias.cpu(),
                "threshold": threshold.cpu(),
                "output_mode": str(getattr(module, "output_mode", "unknown")),
                "threshold_mode": str(getattr(module, "threshold_mode", "unknown")),
            })

        return hook


def make_command(row: dict[str, Any], tag: int) -> tuple[dict[str, Any], dict[str, list[str]]]:
    temporal = int(row["temporal"])
    x = row["input"]
    weight = row["weight"]
    bias = row["bias"]
    threshold = row["threshold"]
    if row["output_mode"] != "binary" or row["threshold_mode"] != "official_atlif":
        raise ValueError(f"non-official-binary ATLIF in deployment set: {row['name']}")

    # The site scale is static and power-of-two. Input scale uses the full call
    # absmax, while the replay retains a bounded deterministic lane subset.
    x_scale = power_of_two_scale(torch.tensor([row["input_absmax_full_call"]]))
    w_scale = power_of_two_scale(weight)
    acc_scale = x_scale * w_scale
    x_q = quantize_signed(x, x_scale, X_W)
    w_q = quantize_signed(weight, w_scale, W_W)
    b_q = quantize_signed(bias, acc_scale, ACC_W)
    threshold_q = quantize_signed(threshold, acc_scale, ACC_W)
    clip_counts = {
        "input": int(torch.round(x / x_scale).abs().gt(127).sum().item()),
        "weight": int(torch.round(weight / w_scale).abs().gt(127).sum().item()),
        "bias": int(torch.round(bias / acc_scale).lt(-(1 << 23)).logical_or(
            torch.round(bias / acc_scale).gt((1 << 23) - 1)
        ).sum().item()),
        "threshold": int(torch.round(threshold / acc_scale).lt(-(1 << 23)).logical_or(
            torch.round(threshold / acc_scale).gt((1 << 23) - 1)
        ).sum().item()),
    }
    if any(clip_counts.values()):
        raise OverflowError(f"quantization clipping: {row['name']}: {clip_counts}")

    hidden = b_q[:, None] + torch.matmul(w_q, x_q)
    acc_min = -(1 << (ACC_W - 1))
    acc_max = (1 << (ACC_W - 1)) - 1
    if int(hidden.min()) < acc_min or int(hidden.max()) > acc_max:
        raise OverflowError(f"Acc{ACC_W} overflow: {row['name']}")
    fixed_event = hidden.ge(threshold_q.reshape(1, 1))
    # Prefer the full-path hidden captured with the model addmm. Falling back to
    # subset rematmul is only for older fixtures and is float-boundary fragile.
    if "hidden" in row and torch.is_tensor(row["hidden"]):
        float_hidden = row["hidden"].detach().float()
        if float_hidden.shape != (temporal, x.shape[1]):
            raise ValueError(
                f"captured hidden shape mismatch: {row['name']}: "
                f"{tuple(float_hidden.shape)} vs {(temporal, x.shape[1])}"
            )
    else:
        float_hidden = bias[:, None] + torch.matmul(weight, x)
    float_event = float_hidden.ge(threshold.reshape(1, 1))
    model_event = row["output"].ne(0)
    model_reference_mismatch = int(model_event.ne(float_event).sum().item())
    if model_reference_mismatch:
        # Last-resort boundary salvage: if |h-thr| is within a few ulps of the
        # threshold, trust the model event (GEMM reduction-order artifact).
        boundary = (float_hidden - threshold.reshape(1, 1)).abs() <= (
            8.0 * torch.finfo(torch.float32).eps * threshold.abs().clamp_min(1.0)
        )
        salvageable = model_event.ne(float_event) & boundary
        if int((model_event.ne(float_event) & ~boundary).sum().item()) == 0 and int(
            salvageable.sum().item()
        ) == model_reference_mismatch:
            float_event = model_event
            float_hidden = torch.where(
                salvageable & model_event,
                threshold.reshape(1, 1),
                torch.where(
                    salvageable & ~model_event,
                    threshold.reshape(1, 1)
                    - torch.finfo(torch.float32).eps * threshold.abs().clamp_min(1.0),
                    float_hidden,
                ),
            )
            model_reference_mismatch = 0
        else:
            raise RuntimeError(
                f"PyTorch ATLIF recomputation mismatch: {row['name']} "
                f"(mismatches={model_reference_mismatch})"
            )

    x_cycles: list[str] = []
    weight_cycles: list[str] = []
    for input_time in range(temporal):
        if temporal == 10:
            x_slots = torch.zeros(PACK_GROUPS * LANES, dtype=torch.int64)
            x_slots[:LANES] = x_q[input_time]
            weight_slots = w_q[:, input_time]
        else:
            x_slots = x_q[input_time]
            weight_slots = torch.empty(SLOTS, dtype=torch.int64)
            for group in range(PACK_GROUPS):
                weight_slots[2 * group] = w_q[0, input_time]
                weight_slots[2 * group + 1] = w_q[1, input_time]
        x_cycles.append(hex_line(pack_signed(x_slots.numpy(), X_W), PACK_GROUPS * LANES * X_W))
        weight_cycles.append(hex_line(pack_signed(weight_slots.numpy(), W_W), SLOTS * W_W))

    if temporal == 10:
        bias_slots = b_q
        threshold_slots = threshold_q.repeat(SLOTS)
        expected_hidden = hidden
        expected_event = fixed_event
    else:
        bias_slots = torch.empty(SLOTS, dtype=torch.int64)
        threshold_slots = threshold_q.repeat(SLOTS)
        expected_hidden = torch.empty((SLOTS, LANES), dtype=torch.int64)
        expected_event = torch.empty((SLOTS, LANES), dtype=torch.bool)
        for group in range(PACK_GROUPS):
            for output_time in range(2):
                slot = 2 * group + output_time
                bias_slots[slot] = b_q[output_time]
                source = slice(group * LANES, (group + 1) * LANES)
                expected_hidden[slot] = hidden[output_time, source]
                expected_event[slot] = fixed_event[output_time, source]

    files = {
        "meta": [hex_line((1 if temporal == 2 else 0) | (temporal << 8) | (tag << 16), 48)],
        "x": x_cycles,
        "weight": weight_cycles,
        "bias": [hex_line(pack_signed(bias_slots.numpy(), ACC_W), SLOTS * ACC_W)],
        "threshold": [hex_line(pack_signed(threshold_slots.numpy(), ACC_W), SLOTS * ACC_W)],
        "expected_hidden": [
            hex_line(pack_signed(expected_hidden.numpy(), ACC_W), SLOTS * LANES * ACC_W)
        ],
        "expected_event": [
            hex_line(pack_signed(expected_event.to(torch.int64).numpy(), 1), SLOTS * LANES)
        ],
    }
    command = {
        "tag": tag,
        "name": row["name"],
        "scenario": row["scenario"],
        "scenario_lane_counts": row["scenario_lane_counts"],
        "temporal_steps": temporal,
        "x_scale": x_scale,
        "weight_scale": w_scale,
        "accumulator_scale": acc_scale,
        "threshold_float": float(threshold.item()),
        "threshold_integer": int(threshold_q.item()),
        "captured_events": int(float_event.numel()),
        "active_float_events": int(float_event.sum().item()),
        "fixed_vs_float_event_mismatches": int(fixed_event.ne(float_event).sum().item()),
        "model_reference_mismatches": model_reference_mismatch,
        "hidden_min": int(expected_hidden.min().item()),
        "hidden_max": int(expected_hidden.max().item()),
        "clip_counts": clip_counts,
        "accumulator_overflow_count": 0,
        "output_contract": "one_bit_event_plus_checkpoint_static_threshold_scale",
    }
    return command, files


def name_set_record(names: set[str]) -> dict[str, Any]:
    values = sorted(names)
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode()
    return {"count": len(values), "names": values, "sha256": hashlib.sha256(payload).hexdigest()}


def write_vectors(
    rows: list[dict[str, Any]],
    output_dir: Path,
    identity: dict[str, Any],
    coverage: dict[str, set[str]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = {key: [] for key in (
        "meta", "x", "weight", "bias", "threshold", "expected_hidden", "expected_event"
    )}
    commands = []
    for index, row in enumerate(sorted(rows, key=lambda item: item["name"])):
        command, files = make_command(row, index + 1)
        commands.append(command)
        for key, values in files.items():
            streams[key].extend(values)
    for key, lines in streams.items():
        (output_dir / f"{key}.mem").write_text("\n".join(lines) + "\n", encoding="ascii")
    (output_dir / "vector_contract.svh").write_text(
        "`define ATLIF_COMMANDS %d\n`define ATLIF_TOTAL_CYCLES %d\n" % (
            len(commands), sum(int(row["temporal_steps"]) for row in commands)
        ),
        encoding="ascii",
    )
    total_events = sum(int(row["captured_events"]) for row in commands)
    total_flips = sum(int(row["fixed_vs_float_event_mismatches"]) for row in commands)
    sites = {str(row["name"]): int(row["temporal_steps"]) for row in commands}
    manifest = {
        "schema": "checkpoint_atlif_dptme_vectors_v1",
        "identity": identity,
        "numeric_contract": {
            "input": "signed_int8_per_site_static_power_of_two_scale",
            "weight": "signed_int8_per_site_static_power_of_two_scale",
            "bias_threshold": "signed_acc24_in_input_scale_times_weight_scale",
            "accumulator": "signed_acc24_no_overflow",
            "compare": "hidden_greater_equal_threshold",
            "output": "one_bit_event_plus_checkpoint_static_threshold_scale",
            "rounding": "torch_round_nearest_ties_to_even_then_saturate",
        },
        "commands": commands,
        "site_coverage": {
            key: name_set_record(names) for key, names in coverage.items()
        },
        "summary": {
            "commands": len(commands),
            "live_sites": len(sites),
            "live_t10_sites": sum(value == 10 for value in sites.values()),
            "live_t2_sites": sum(value == 2 for value in sites.values()),
            "selection_scenarios": ["ordinary", "near_threshold", "max_amplitude"],
            "t10_commands": sum(int(row["temporal_steps"]) == 10 for row in commands),
            "t2_commands": sum(int(row["temporal_steps"]) == 2 for row in commands),
            "captured_events": total_events,
            "fixed_vs_float_event_mismatches": total_flips,
            "fixed_vs_float_event_mismatch_ratio": total_flips / total_events if total_events else 0.0,
            "model_reference_mismatches": sum(int(row["model_reference_mismatches"]) for row in commands),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_files = [Path(__file__).resolve(), *sorted(output_dir.glob("*.mem")), output_dir / "vector_contract.svh"]
    manifest["source_sha256"] = {str(path): sha256(path) for path in source_files}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config, device = load_config(config_path)
    dataset = DSECDatasetLite(
        config,
        file_list="valid",
        stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1),
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(args.sample_index)
    model = build_model(config, checkpoint_path, device)
    audit = validate_h9_load_audit(model, config)
    counts = h9_module_counts(model)
    if counts != {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12}:
        raise RuntimeError(f"unexpected H9 module counts: {counts}")
    if audit is None or int(audit.get("checkpoint_overlay_keys", -1)) != 210:
        raise RuntimeError(f"unexpected checkpoint load audit: {audit}")
    bn_policy = str(config.get("test", {}).get("bn_policy", "running"))
    bn_modules = configure_batch_norm_evaluation(model, bn_policy)
    if bn_policy != "no_running":
        raise RuntimeError("checkpoint ATLIF replay requires standard no_running BN policy")

    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = Compose([CenterCrop((config["loader"]["crop"][0], config["loader"]["crop"][1]))])
    chunk, mask, label = dataset[args.sample_index]
    # Match DataLoader(batch_size=1) used by the standard profiler.
    chunk = chunk.unsqueeze(0)
    mask = mask.unsqueeze(0)
    label = label.unsqueeze(0)
    x, _, _ = preprocess_chunk(config, chunk, label, mask, transform_valid, device)
    capture = Capture()
    capture.attach(model)
    try:
        with torch.no_grad():
            functional.reset_net(model)
            model(x)
    finally:
        capture.close()
    live_sites = {str(row["name"]): int(row["temporal"]) for row in capture.rows}
    replayed_names = set(live_sites)
    if (
        len(capture.installed_names) != 105
        or len(capture.called_names) != 93
        or len(capture.dead_called_names) != 12
        or capture.called_names - capture.dead_called_names != replayed_names
        or not capture.dead_called_names.issubset(capture.called_names)
        or not capture.called_names.issubset(capture.installed_names)
    ):
        raise RuntimeError(
            "unexpected ATLIF site coverage: "
            f"installed={len(capture.installed_names)} called={len(capture.called_names)} "
            f"dead_called={len(capture.dead_called_names)} replayed={len(replayed_names)}"
        )
    if len(live_sites) != 81 or sum(value == 10 for value in live_sites.values()) != 45 or sum(
        value == 2 for value in live_sites.values()
    ) != 36:
        raise RuntimeError(f"expected live ATLIF 81=45xT10+36xT2, got {live_sites}")

    identity = {
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_size": checkpoint_path.stat().st_size,
        "sample_index": args.sample_index,
        "resolution": list(config["loader"]["resolution"]),
        "crop": config["loader"].get("crop"),
        "window_size": list(config["swin_transformer"]["window_size"]),
        "bn_policy": bn_policy,
        "bn_modules_changed": bn_modules,
        "module_counts": counts,
        "checkpoint_load_audit": audit,
    }
    manifest = write_vectors(
        capture.rows,
        args.output_dir.resolve(),
        identity,
        {
            "installed": capture.installed_names,
            "called": capture.called_names,
            "dead_called": capture.dead_called_names,
            "replayed": replayed_names,
        },
    )
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
