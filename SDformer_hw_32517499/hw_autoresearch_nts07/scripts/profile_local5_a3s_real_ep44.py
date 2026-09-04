#!/usr/bin/env python3
"""Measure H88/A3S hardware work on the sealed Local5 ep44 Q/K trace.

This is a read-only architecture audit.  It reconstructs the deployed Local5
Q1.7 gates from the Q/K bitmaps, requires an exact match against the archived
source-major descriptors, and only then applies the A3S score offsets.  It does
not run the network, estimate cycles, or modify the production RTL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REQUIRED_ARRAYS = (
    "descriptor_group_offsets",
    "descriptor_source_id",
    "descriptor_q_bitmap",
    "descriptor_k_bitmap",
    "descriptor_incoming_gates",
    "descriptor_valid_mask",
    "source_term_count",
    "source_k_popcount",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bitmap_to_bits(values: np.ndarray, lanes: int = 32) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    shifts = np.arange(lanes, dtype=np.uint64)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def build_local5_geometry(
    *, time_planes: int = 2, spatial_side: int = 15
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = time_planes * spatial_side * spatial_side
    grid = torch.arange(tokens, dtype=torch.long).reshape(
        time_planes, spatial_side, spatial_side
    )
    indices = [grid]
    valids = [torch.ones_like(grid, dtype=torch.bool)]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy = torch.arange(spatial_side).view(1, spatial_side, 1) + dy
        xx = torch.arange(spatial_side).view(1, 1, spatial_side) + dx
        valid = (
            (yy >= 0)
            & (yy < spatial_side)
            & (xx >= 0)
            & (xx < spatial_side)
        )
        yy = yy.clamp(0, spatial_side - 1).expand(
            time_planes, spatial_side, spatial_side
        )
        xx = xx.clamp(0, spatial_side - 1).expand(
            time_planes, spatial_side, spatial_side
        )
        tt = torch.arange(time_planes).view(time_planes, 1, 1).expand_as(yy)
        indices.append(grid[tt, yy, xx])
        valids.append(valid.expand(time_planes, spatial_side, spatial_side))
    return (
        torch.stack(indices, dim=-1).reshape(tokens, 5),
        torch.stack(valids, dim=-1).reshape(tokens, 5),
    )


def destination_to_source(
    gate_codes: torch.Tensor,
    source_index: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose [B,destination,role] gates to [B,source,role]."""

    if gate_codes.ndim != 3 or gate_codes.shape[-1] != 5:
        raise ValueError("gate_codes must be [B,tokens,5]")
    batch, tokens, _ = gate_codes.shape
    if tuple(source_index.shape) != (tokens, 5):
        raise ValueError("source_index shape does not match gate_codes")
    if tuple(valid.shape) != (tokens, 5):
        raise ValueError("valid shape does not match gate_codes")
    incoming = torch.zeros_like(gate_codes)
    incoming_valid = torch.zeros(
        (batch, tokens, 5), dtype=torch.bool, device=gate_codes.device
    )
    for role in range(5):
        role_valid = valid[:, role]
        selected = source_index[role_valid, role]
        if selected.numel():
            counts = torch.bincount(selected, minlength=tokens)
            if bool(counts.gt(1).any()):
                raise ValueError(f"role {role} is not one-to-one at a source")
            incoming[:, selected, role] = gate_codes[:, role_valid, role]
            incoming_valid[:, selected, role] = True
    return incoming, incoming_valid


def source_work(
    incoming_gate_codes: torch.Tensor,
    source_k_popcount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unique gate classes, product terms, and destination updates."""

    if incoming_gate_codes.ndim != 3 or incoming_gate_codes.shape[-1] != 5:
        raise ValueError("incoming gates must be [B,tokens,5]")
    if tuple(source_k_popcount.shape) != tuple(incoming_gate_codes.shape[:2]):
        raise ValueError("source_k_popcount shape mismatch")
    ordered = torch.sort(incoming_gate_codes, dim=-1).values
    first = torch.ones_like(ordered[..., :1], dtype=torch.bool)
    transitions = torch.cat(
        (first, ordered[..., 1:] != ordered[..., :-1]), dim=-1
    )
    unique_nonzero = ((ordered > 0) & transitions).sum(dim=-1)
    nonzero_edges = (incoming_gate_codes > 0).sum(dim=-1)
    product_terms = unique_nonzero * source_k_popcount
    destination_updates = nonzero_edges * source_k_popcount
    return unique_nonzero, product_terms, destination_updates


def valid_mask_codes(incoming_valid: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(
        (1, 2, 4, 8, 16), dtype=torch.int64, device=incoming_valid.device
    )
    return (incoming_valid.to(torch.int64) * weights).sum(dim=-1)


@dataclass
class DeltaTotals:
    groups: int = 0
    descriptors: int = 0
    valid_edges: int = 0
    product_terms: int = 0
    destination_updates: int = 0
    unique_gate_classes: int = 0
    gate_code_changes: int = 0
    direction_ew: int = 0
    direction_pixels: int = 0

    def add(self, other: "DeltaTotals") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def totals_to_dict(value: DeltaTotals, baseline: DeltaTotals) -> dict[str, Any]:
    return {
        **value.__dict__,
        "product_term_ratio_vs_delta0": (
            value.product_terms / baseline.product_terms
            if baseline.product_terms
            else None
        ),
        "destination_update_ratio_vs_delta0": (
            value.destination_updates / baseline.destination_updates
            if baseline.destination_updates
            else None
        ),
        "gate_code_change_fraction": (
            value.gate_code_changes / value.valid_edges if value.valid_edges else None
        ),
        "direction_ew_fraction": (
            value.direction_ew / value.direction_pixels
            if value.direction_pixels
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    hw_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=hw_root
        / "results/local5_ep44_hardware_rebind_20260815_profile100",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root
        / "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
        "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_"
        "hardware_order_q7q17_deploy.yml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=hw_root / "results/local5_a3s_ep44_real_cost_20260821",
    )
    parser.add_argument("--delta-bins", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--batch-groups", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--reference-groups", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    payload_path = args.input_dir / "ordered_term_items.npz"
    manifest_path = args.input_dir / "ordered_term_manifest.json"
    if not payload_path.is_file() or not manifest_path.is_file():
        raise SystemExit("missing ordered Local5 payload or manifest")
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(payload_path) != manifest.get("payload_sha256"):
        raise SystemExit("payload SHA does not match ordered manifest")
    payload = np.load(payload_path, mmap_mode="r")
    missing = [name for name in REQUIRED_ARRAYS if name not in payload.files]
    if missing:
        raise SystemExit(f"missing required arrays: {missing}")
    groups = manifest.get("groups", [])
    group_count = len(groups)
    if args.max_groups:
        group_count = min(group_count, args.max_groups)
    if group_count <= 0:
        raise SystemExit("no groups selected")
    offsets = np.asarray(payload["descriptor_group_offsets"], dtype=np.int64)
    if offsets.shape[0] < group_count + 1:
        raise SystemExit("descriptor offsets do not cover selected groups")
    for group_index in range(group_count):
        begin, end = map(int, offsets[group_index : group_index + 2])
        if end - begin != 450:
            raise SystemExit(f"group {group_index} is not T450")
        source_ids = np.asarray(payload["descriptor_source_id"][begin:end])
        if not np.array_equal(source_ids, np.arange(450, dtype=source_ids.dtype)):
            raise SystemExit(f"group {group_index} descriptor order is not source 0..449")

    repo_root = Path(__file__).resolve().parents[2]
    experiment_root = repo_root / "neuron_experiments/H9_bipolar_self_attention"
    for path in (
        repo_root,
        repo_root / "third_party/SDformerFlow",
        experiment_root / "overlay",
    ):
        sys.path.insert(0, str(path))
    from models.STSwinNet_SNN.bsa_attention import (  # pylint: disable=import-error
        _apply_hardware_gate_quant,
        _apply_hardware_score_quant,
        _binary_axnor_local5_a3s_attention,
        _d3_a3s_offset,
        _d3_axis_field,
        _normalize_consensus_score,
        _rtl_shiftmax_gate_q17,
        config_from_dict,
    )

    raw_config = yaml.safe_load(args.config.read_text())
    cfg = config_from_dict(raw_config["bsa_attention"])
    if not (
        cfg.hardware_quant_enabled
        and cfg.hardware_rtl_shiftmax_enabled
        and cfg.hardware_mask_invalid_candidates
    ):
        raise SystemExit("config is not the frozen Q7/Q1.7 masked RTL Shiftmax path")

    source_index, valid = build_local5_geometry()
    valid_batch = valid.view(1, 1, 450, 5)
    invalid_fill = float(cfg.hardware_score_min)
    deltas = tuple(dict.fromkeys(int(value) for value in args.delta_bins))
    if 0 not in deltas or any(value < 0 for value in deltas):
        raise SystemExit("delta bins must be nonnegative and include zero")

    totals = {delta: DeltaTotals() for delta in deltas}
    stage_totals = {
        stage: {delta: DeltaTotals() for delta in deltas} for stage in range(4)
    }
    baseline_gate_mismatches = 0
    baseline_valid_mismatches = 0
    baseline_term_mismatches = 0
    baseline_kpop_mismatches = 0
    valid_zero_violations = 0
    reference_groups_checked = 0
    reference_gate_mismatches = {delta: 0 for delta in deltas}

    for chunk_start in range(0, group_count, args.batch_groups):
        chunk_end = min(group_count, chunk_start + args.batch_groups)
        begin = int(offsets[chunk_start])
        end = int(offsets[chunk_end])
        chunk_groups = chunk_end - chunk_start
        q_bits = bitmap_to_bits(payload["descriptor_q_bitmap"][begin:end])
        k_bits = bitmap_to_bits(payload["descriptor_k_bitmap"][begin:end])
        q_flat = torch.from_numpy(q_bits.reshape(chunk_groups, 2, 225, 32))
        q_orig = q_flat.permute(1, 0, 2, 3).unsqueeze(2).contiguous()
        k_orig = torch.from_numpy(k_bits.reshape(chunk_groups, 450, 32)).unsqueeze(1)
        archived_gates = torch.from_numpy(
            np.asarray(payload["descriptor_incoming_gates"][begin:end], dtype=np.int64)
            .reshape(chunk_groups, 450, 5)
        )
        archived_valid = torch.from_numpy(
            np.asarray(payload["descriptor_valid_mask"][begin:end], dtype=np.int64)
            .reshape(chunk_groups, 450)
        )
        archived_terms = torch.from_numpy(
            np.asarray(payload["source_term_count"][begin:end], dtype=np.int64)
            .reshape(chunk_groups, 450)
        )
        archived_kpop = torch.from_numpy(
            np.asarray(payload["source_k_popcount"][begin:end], dtype=np.int64)
            .reshape(chunk_groups, 450)
        )
        kpop = k_orig.to(torch.int64).sum(dim=-1)[:, 0]
        baseline_kpop_mismatches += int((kpop != archived_kpop).sum())

        with torch.inference_mode():
            q_event = q_flat.reshape(chunk_groups, 450, 32).unsqueeze(1)
            k_event = k_orig
            k_candidates = k_event[:, :, source_index, :]
            q_candidates = q_event.unsqueeze(-2)
            same_spike = (q_candidates * k_candidates).sum(dim=-1)
            same_silent = ((1.0 - q_candidates) * (1.0 - k_candidates)).sum(dim=-1)
            scores = same_spike + float(cfg.alpha0) * same_silent
            scores = _normalize_consensus_score(
                scores, 32, cfg, active=None
            )
            if float(cfg.matrix_diag_bias) != 0.0:
                scores[..., 0] += float(cfg.matrix_diag_bias)
            directions = _d3_axis_field(q_orig, k_orig)

            baseline_destination = None
            baseline_incoming = None
            chunk_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for delta in deltas:
                shifted = scores
                if delta:
                    shifted = scores + _d3_a3s_offset(scores, directions, delta)
                quantized = _apply_hardware_score_quant(shifted, cfg)
                quantized = quantized.masked_fill(~valid_batch, invalid_fill)
                gate = _rtl_shiftmax_gate_q17(
                    quantized,
                    dim=-1,
                    preserve_mean=bool(cfg.preserve_mean),
                    valid_mask=valid_batch,
                )
                gate = _apply_hardware_gate_quant(gate, cfg)
                destination_codes = torch.round(gate * 128.0).to(torch.int64)[:, 0]
                incoming, incoming_valid = destination_to_source(
                    destination_codes, source_index, valid
                )
                _, terms, updates = source_work(incoming, kpop)
                chunk_outputs[delta] = (incoming, terms, updates)
                if delta == 0:
                    baseline_destination = destination_codes
                    baseline_incoming = incoming
                    baseline_gate_mismatches += int((incoming != archived_gates).sum())
                    baseline_valid_mismatches += int(
                        (valid_mask_codes(incoming_valid) != archived_valid).sum()
                    )
                    baseline_term_mismatches += int((terms != archived_terms).sum())
                    valid_zero_violations += int(
                        ((~valid.view(1, 450, 5)) & destination_codes.ne(0)).sum()
                    )
                remaining_reference = max(
                    0, int(args.reference_groups) - reference_groups_checked
                )
                reference_count = min(chunk_groups, remaining_reference)
                if reference_count:
                    reference_cfg = replace(
                        cfg,
                        mode="local5_a3s",
                        a3s_delta_bins=delta,
                        a3s_delta_warmup_steps=0,
                    )
                    _, _, reference_gate, _ = _binary_axnor_local5_a3s_attention(
                        q_orig[:, :reference_count],
                        k_orig[:reference_count],
                        reference_cfg,
                    )
                    reference_codes = torch.round(reference_gate * 128.0).to(
                        torch.int64
                    )[:, 0]
                    reference_gate_mismatches[delta] += int(
                        (
                            reference_codes
                            != destination_codes[:reference_count]
                        ).sum()
                    )
            if baseline_destination is None or baseline_incoming is None:
                raise AssertionError("delta zero baseline was not evaluated")
            reference_groups_checked += min(
                chunk_groups,
                max(0, int(args.reference_groups) - reference_groups_checked),
            )

        direction_ew_by_group = (
            (directions[:, 0] <= 1).sum(dim=(1, 2)).to(torch.int64)
        )
        for local_group in range(chunk_groups):
            group_index = chunk_start + local_group
            stage = int(groups[group_index]["stage"])
            for delta in deltas:
                incoming, terms, updates = chunk_outputs[delta]
                unique, _, _ = source_work(
                    incoming[local_group : local_group + 1],
                    kpop[local_group : local_group + 1],
                )
                value = DeltaTotals(
                    groups=1,
                    descriptors=450,
                    valid_edges=int(valid.sum()),
                    product_terms=int(terms[local_group].sum()),
                    destination_updates=int(updates[local_group].sum()),
                    unique_gate_classes=int(unique.sum()),
                    gate_code_changes=int(
                        (
                            incoming[local_group]
                            != baseline_incoming[local_group]
                        ).sum()
                    ),
                    direction_ew=int(direction_ew_by_group[local_group]),
                    direction_pixels=225,
                )
                totals[delta].add(value)
                stage_totals[stage][delta].add(value)

        if chunk_end == group_count or chunk_end % 320 == 0:
            print(f"PROGRESS groups={chunk_end}/{group_count}", flush=True)

    checks = {
        "payload_sha_matches_manifest": True,
        "baseline_gate_mismatches": baseline_gate_mismatches,
        "baseline_valid_mismatches": baseline_valid_mismatches,
        "baseline_term_mismatches": baseline_term_mismatches,
        "baseline_kpop_mismatches": baseline_kpop_mismatches,
        "invalid_destination_gate_nonzero": valid_zero_violations,
        "reference_groups_checked": reference_groups_checked,
        "reference_gate_mismatches_by_delta": {
            str(delta): reference_gate_mismatches[delta] for delta in deltas
        },
    }
    exact = all(
        checks[name] == 0
        for name in (
            "baseline_gate_mismatches",
            "baseline_valid_mismatches",
            "baseline_term_mismatches",
            "baseline_kpop_mismatches",
            "invalid_destination_gate_nonzero",
        )
    )
    if not exact:
        raise SystemExit(f"baseline miter failed: {checks}")
    if any(reference_gate_mismatches.values()):
        raise SystemExit(f"A3S reference miter failed: {checks}")

    baseline = totals[0]
    delta_results = {
        str(delta): totals_to_dict(totals[delta], baseline) for delta in deltas
    }
    stage_results = {
        str(stage): {
            str(delta): totals_to_dict(stage_totals[stage][delta], stage_totals[stage][0])
            for delta in deltas
        }
        for stage in range(4)
    }
    positive = [delta for delta in deltas if delta > 0]
    any_hardware_win = any(
        totals[delta].product_terms < baseline.product_terms
        and totals[delta].destination_updates <= baseline.destination_updates
        for delta in positive
    )
    verdict = (
        "CONDITIONAL_HARDWARE_WIN_PENDING_ACCURACY"
        if any_hardware_win
        else "NO_GO_AS_HARDWARE_ACCELERATOR_PENDING_ALGORITHM_RESULT"
    )
    report = {
        "schema": "local5_a3s_ep44_real_hardware_cost_v1",
        "status": verdict,
        "evidence_level": "profile_real_ep44_qk_not_rtl_not_cycle_model",
        "claim_boundary": {
            "workload": "Local5 ep44 post-G0 sampled profile100",
            "groups": group_count,
            "tokens_per_group": 450,
            "lanes": 32,
            "output_dimension": "not modeled",
            "cycles": "not modeled",
            "energy": "not modeled",
            "accuracy": "not measured by this audit",
            "direction_state_bits_per_group": 450,
        },
        "identity": {
            "payload": str(payload_path),
            "payload_sha256": sha256_file(payload_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "checkpoint": manifest.get("checkpoint"),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        },
        "checks": checks,
        "delta_bins": delta_results,
        "by_stage": stage_results,
        "decision": {
            "hardware_win_definition": (
                "strictly fewer source-owned product terms and no more destination updates"
            ),
            "any_positive_delta_hardware_win": any_hardware_win,
            "rtl_admitted": False,
            "paper_claim_admitted": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "stage_delta.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "delta_bins",
                "groups",
                "product_terms",
                "product_term_ratio_vs_delta0",
                "destination_updates",
                "destination_update_ratio_vs_delta0",
                "gate_code_change_fraction",
                "direction_ew_fraction",
            ),
        )
        writer.writeheader()
        for stage in range(4):
            for delta in deltas:
                row = totals_to_dict(
                    stage_totals[stage][delta], stage_totals[stage][0]
                )
                writer.writerow(
                    {name: row.get(name, stage if name == "stage" else delta)
                     for name in writer.fieldnames}
                    | {"stage": stage, "delta_bins": delta}
                )

    lines = [
        "# Local5 A3S real ep44 hardware-cost audit",
        "",
        f"Status: `{verdict}`",
        "",
        "This is a real-Q/K profile audit, not RTL, cycles, energy, or accuracy.",
        "The delta-zero path is admitted only after exact reconstruction of the archived gate, valid-mask, K-popcount, and source-term ledgers.",
        "",
        "| delta bins | product terms | vs delta0 | destination updates | vs delta0 | gate changes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for delta in deltas:
        row = delta_results[str(delta)]
        lines.append(
            f"| {delta} | {row['product_terms']} | "
            f"{row['product_term_ratio_vs_delta0']:.6f} | "
            f"{row['destination_updates']} | "
            f"{row['destination_update_ratio_vs_delta0']:.6f} | "
            f"{row['gate_code_change_fraction']:.6%} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(
        f"PASS status={verdict} groups={group_count} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
