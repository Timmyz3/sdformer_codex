#!/usr/bin/env python3
"""Checkpoint-bound structured-group census for H67 FFN and attention weights.

This is a fail-closed opportunity audit.  It measures exact floating-point and
canonical per-output-channel INT8 zeros, plus low-energy paired groups.  It does
not prune a checkpoint, predict accuracy, or admit hardware/system speedup.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKPOINT = HW / (
    "system_handoff/received/h67_ep35_system_trace_handoff_20260821/"
    "h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth")
M41_EXPORTER = HW / (
    "system_simulator/scripts/export_m41_h67_ep35_bottleneck_int8.py")
FFN_LEDGER = HW / (
    "results/motion_ffn_resident_fusion_opportunity_review_r1_20260824/"
    "ffn_pair_ledger.csv")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "m41_exporter": "bc272c5e1449fb745fe200313f25e97c293ad971fa8856d9aad13dfc89785a5e",
    "ffn_ledger": "dcf183e930372253da96c6ce242289e3e6a5e1b0f76a513e095fae4b0d2ae128",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
TOTAL_ENVELOPE_CYCLES = 620302905
FFN_ENVELOPE_CYCLES = 159784111
GROUP_SIZES = (16, 32)
PRUNE_BUDGETS = (0.05, 0.10, 0.25, 0.50)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, label):
    value = float(value)
    require(math.isfinite(value), label + " is not finite")
    return value


def canonical_int8(weight):
    require(weight.ndim == 2 and weight.dtype == np.float32,
            "canonical INT8 input geometry/dtype drift")
    maximum = np.max(np.abs(weight), axis=1).astype(np.float32)
    scale = (maximum / np.float32(127.0)).astype(np.float32)
    scale[maximum == np.float32(0.0)] = np.float32(1.0)
    quantized = np.rint(
        weight.astype(np.float64) / scale[:, None].astype(np.float64))
    return np.clip(quantized, -127.0, 127.0).astype(np.int8)


def load_model():
    spec = importlib.util.spec_from_file_location("m156_m41_exporter",
                                                   M41_EXPORTER)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M41 exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_checkpoint_model(CHECKPOINT)


def load_ffn_ledger():
    with FFN_LEDGER.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 12, "FFN ledger population drift")
    result = {row["pair_id"]: row for row in rows}
    require(len(result) == 12, "duplicate FFN pair id")
    require(sum(int(row["pair_cycles_model"]) for row in rows)
            == FFN_ENVELOPE_CYCLES, "FFN cycle total drift")
    return result


def array(module, name):
    tensor = getattr(module, name).weight.detach().cpu().contiguous()
    result = tensor.numpy().astype(np.float32, copy=False)
    require(bool(np.isfinite(result).all()), name + " weight is non-finite")
    return result


def group_rows(pair_name, stage, block, fc1, fc2, group_size):
    require(fc1.ndim == 2 and fc2.ndim == 2,
            "FFN weight rank drift")
    expanded, channels = fc1.shape
    require(fc2.shape == (channels, expanded), "FFN pair geometry drift")
    require(expanded % group_size == 0, "FFN group tail is not supported")
    q1 = canonical_int8(fc1)
    q2 = canonical_int8(fc2)
    rows = []
    for group, start in enumerate(range(0, expanded, group_size)):
        end = start + group_size
        float_values = (fc1[start:end, :], fc2[:, start:end])
        int_values = (q1[start:end, :], q2[:, start:end])
        energy = math.fsum(float(np.square(value.astype(np.float64)).sum())
                           for value in float_values)
        rows.append({
            "scope": "ffn_pair",
            "module": pair_name,
            "stage": stage,
            "block": block,
            "group_size": group_size,
            "group": group,
            "channel_start": start,
            "channel_end_exclusive": end,
            "float_exact_zero": all(bool(np.all(value == 0.0))
                                    for value in float_values),
            "int8_exact_zero": all(bool(np.all(value == 0))
                                   for value in int_values),
            "weight_energy": finite(energy, "FFN group energy"),
        })
    return rows, {
        "fc1_scalar_float_zero_fraction": float(np.mean(fc1 == 0.0)),
        "fc2_scalar_float_zero_fraction": float(np.mean(fc2 == 0.0)),
        "fc1_scalar_int8_zero_fraction": float(np.mean(q1 == 0)),
        "fc2_scalar_int8_zero_fraction": float(np.mean(q2 == 0)),
    }


def attention_group_rows(name, stage, block, q, k, proj, group_size):
    require(q.ndim == 2 and q.shape == k.shape == proj.shape,
            "attention square weight geometry drift")
    channels = q.shape[0]
    require(q.shape[1] == channels and channels % group_size == 0,
            "attention group tail is not supported")
    qq = canonical_int8(q)
    qk = canonical_int8(k)
    qp = canonical_int8(proj)
    rows = []
    for group, start in enumerate(range(0, channels, group_size)):
        end = start + group_size
        float_values = (q[start:end, :], k[start:end, :],
                        proj[:, start:end])
        int_values = (qq[start:end, :], qk[start:end, :],
                      qp[:, start:end])
        energy = math.fsum(float(np.square(value.astype(np.float64)).sum())
                           for value in float_values)
        rows.append({
            "scope": "attention_qk_proj_shared",
            "module": name,
            "stage": stage,
            "block": block,
            "group_size": group_size,
            "group": group,
            "channel_start": start,
            "channel_end_exclusive": end,
            "float_exact_zero": all(bool(np.all(value == 0.0))
                                    for value in float_values),
            "int8_exact_zero": all(bool(np.all(value == 0))
                                   for value in int_values),
            "weight_energy": finite(energy, "attention group energy"),
        })
    return rows, {
        "q_scalar_float_zero_fraction": float(np.mean(q == 0.0)),
        "k_scalar_float_zero_fraction": float(np.mean(k == 0.0)),
        "proj_scalar_float_zero_fraction": float(np.mean(proj == 0.0)),
        "q_scalar_int8_zero_fraction": float(np.mean(qq == 0)),
        "k_scalar_int8_zero_fraction": float(np.mean(qk == 0)),
        "proj_scalar_int8_zero_fraction": float(np.mean(qp == 0)),
    }


def summarize_exact(rows):
    return {
        "groups": len(rows),
        "float_exact_zero_groups": sum(row["float_exact_zero"] for row in rows),
        "int8_exact_zero_groups": sum(row["int8_exact_zero"] for row in rows),
        "total_weight_energy": finite(
            math.fsum(row["weight_energy"] for row in rows),
            "total group energy"),
    }


def ffn_sensitivities(ffn_rows, ledger):
    results = []
    for group_size in GROUP_SIZES:
        selected_rows = [row for row in ffn_rows
                         if row["group_size"] == group_size]
        by_pair = {}
        for row in selected_rows:
            by_pair.setdefault(row["module"], []).append(row)
        require(set(by_pair) == set(ledger), "FFN group/ledger join drift")
        for budget in PRUNE_BUDGETS:
            removed_cycles = 0.0
            removed_energy = 0.0
            total_energy = 0.0
            removed_groups = 0
            total_groups = 0
            stage2_removed_cycles = 0.0
            stage2_total_cycles = 0
            for name, rows in by_pair.items():
                ordered = sorted(rows, key=lambda row: row["weight_energy"])
                count = max(1, int(math.floor(len(ordered) * budget)))
                chosen = ordered[:count]
                fraction = count / len(ordered)
                cycles = int(ledger[name]["pair_cycles_model"])
                removed_cycles += cycles * fraction
                if int(ledger[name]["stage"]) == 2:
                    stage2_removed_cycles += cycles * fraction
                    stage2_total_cycles += cycles
                removed_energy += math.fsum(row["weight_energy"]
                                            for row in chosen)
                total_energy += math.fsum(row["weight_energy"]
                                          for row in ordered)
                removed_groups += count
                total_groups += len(ordered)
            remaining = TOTAL_ENVELOPE_CYCLES - removed_cycles
            require(remaining > 0.0 and total_energy > 0.0,
                    "degenerate FFN sensitivity")
            results.append({
                "group_size": group_size,
                "requested_prune_budget": budget,
                "actual_removed_group_fraction": removed_groups / total_groups,
                "removed_groups": removed_groups,
                "total_groups": total_groups,
                "paired_weight_energy_removed_fraction":
                    removed_energy / total_energy,
                "ffn_cycle_reduction_sensitivity":
                    removed_cycles / FFN_ENVELOPE_CYCLES,
                "stage2_cycle_reduction_sensitivity":
                    stage2_removed_cycles / stage2_total_cycles,
                "full_compute_envelope_speedup_sensitivity":
                    TOTAL_ENVELOPE_CYCLES / remaining,
                "training_valid825_required": True,
                "hardware_speedup_admitted": False,
                "system_speedup_admitted": False,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite M156 output")
    observed = {
        "checkpoint": sha256(CHECKPOINT),
        "m41_exporter": sha256(M41_EXPORTER),
        "ffn_ledger": sha256(FFN_LEDGER),
        "docs359": sha256(DOCS359),
    }
    require(observed == EXPECTED, "M156 frozen input identity drift")

    model = load_model()
    modules = dict(model.named_modules())
    ledger = load_ffn_ledger()
    ffn_pattern = re.compile(
        r"^sttmultires_unet\.encoders\.swin3d\.layers\.(\d+)\."
        r"swin_blocks\.(\d+)\.mlp$")
    attention_pattern = re.compile(
        r"^sttmultires_unet\.encoders\.swin3d\.layers\.(\d+)\."
        r"swin_blocks\.(\d+)\.attn$")

    ffn_rows = []
    ffn_module_stats = []
    for name in sorted(ledger):
        match = ffn_pattern.match(name)
        require(match is not None and name in modules,
                "missing named FFN pair: " + name)
        stage, block = map(int, match.groups())
        fc1 = array(modules[name], "fc1")
        fc2 = array(modules[name], "fc2")
        for group_size in GROUP_SIZES:
            rows, scalar = group_rows(name, stage, block, fc1, fc2,
                                      group_size)
            ffn_rows.extend(rows)
            ffn_module_stats.append({
                "module": name,
                "stage": stage,
                "block": block,
                "group_size": group_size,
                "channels": int(fc1.shape[1]),
                "expanded_channels": int(fc1.shape[0]),
                **scalar,
                **summarize_exact(rows),
            })

    attention_rows = []
    attention_module_stats = []
    attention_names = sorted(name for name in modules
                             if attention_pattern.match(name))
    require(len(attention_names) == 12, "attention population drift")
    for name in attention_names:
        match = attention_pattern.match(name)
        require(match is not None, "attention name parse drift")
        stage, block = map(int, match.groups())
        module = modules[name]
        q = array(module, "linear_q")
        k = array(module, "linear_k")
        proj = array(module, "proj")
        for group_size in GROUP_SIZES:
            rows, scalar = attention_group_rows(
                name, stage, block, q, k, proj, group_size)
            attention_rows.extend(rows)
            attention_module_stats.append({
                "module": name,
                "stage": stage,
                "block": block,
                "group_size": group_size,
                "channels": int(q.shape[0]),
                **scalar,
                **summarize_exact(rows),
            })

    all_rows = ffn_rows + attention_rows
    result = {
        "schema": "m156_h67_ep35_nonconv_group_sparsity_census_v1",
        "identity": observed,
        "checkpoint_model_type":
            model.__class__.__module__ + "." + model.__class__.__name__,
        "scope": {
            "ffn_pairs": len(ledger),
            "attention_modules": len(attention_names),
            "group_sizes": list(GROUP_SIZES),
            "canonical_int8":
                "symmetric per-output-channel maxabs/127, round-to-even, clip[-127,127]",
        },
        "ffn": {
            "summary_by_module_and_group": ffn_module_stats,
            "exact_summary": summarize_exact(ffn_rows),
            "low_energy_prune_sensitivities":
                ffn_sensitivities(ffn_rows, ledger),
        },
        "attention_qk_proj_shared": {
            "summary_by_module_and_group": attention_module_stats,
            "exact_summary": summarize_exact(attention_rows),
            "cycle_speedup_not_mapped": True,
        },
        "admission": {
            "checkpoint_bound_weight_census": True,
            "existing_exact_structured_skip":
                any(row["int8_exact_zero"] for row in all_rows),
            "trained_structured_mask": False,
            "valid825": False,
            "address_timed_trace": False,
            "hardware_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
        "paper_safe_statement":
            "The frozen H67 ep35 checkpoint was audited for exact paired FFN and shared Q/K/projection 16/32-channel groups. Low-energy removal values are training sensitivities only; no pruning, accuracy, address trace, or hardware speedup is admitted.",
    }

    output.mkdir(parents=True)
    with (output / "m156_h67_ep35_nonconv_group_sparsity_census.json").open(
            "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    fields = [
        "scope", "module", "stage", "block", "group_size", "group",
        "channel_start", "channel_end_exclusive", "float_exact_zero",
        "int8_exact_zero", "weight_energy",
    ]
    with (output / "group_ledger.csv").open(
            "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({
        "status": "PASS_M156_NONCONV_GROUP_CENSUS",
        "ffn": result["ffn"]["exact_summary"],
        "attention": result["attention_qk_proj_shared"]["exact_summary"],
        "existing_exact_structured_skip":
            result["admission"]["existing_exact_structured_skip"],
        "hardware_speedup": False,
        "headline": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
