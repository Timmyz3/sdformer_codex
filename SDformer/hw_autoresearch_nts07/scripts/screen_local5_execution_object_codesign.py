#!/usr/bin/env python3
"""Screen exact Local5 execution-object candidates on an ordered profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "results/local5_ep44_hardware_rebind_20260815_profile100"
)
REQUIRED_ARRAYS = {
    "group_offsets",
    "item_destination",
    "item_gate_code",
    "item_lane_id",
    "item_multiplicity",
    "descriptor_group_offsets",
    "source_gate_count",
    "source_k_popcount",
    "source_term_count",
    "source_delivery_count",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_sum(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    values_i64 = np.asarray(values, dtype=np.int64)
    offsets_i64 = np.asarray(offsets, dtype=np.int64)
    prefix = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(values_i64, dtype=np.int64))
    )
    return prefix[offsets_i64[1:]] - prefix[offsets_i64[:-1]]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _histogram(values: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values[mask].tolist()).items())
    }


def analyze_arrays(
    arrays: Mapping[str, np.ndarray], stages: Sequence[int]
) -> dict[str, Any]:
    missing = sorted(REQUIRED_ARRAYS.difference(arrays))
    if missing:
        raise ValueError(f"ordered payload missing arrays: {missing}")

    item_offsets = np.asarray(arrays["group_offsets"], dtype=np.int64)
    descriptor_offsets = np.asarray(
        arrays["descriptor_group_offsets"], dtype=np.int64
    )
    if item_offsets.ndim != 1 or item_offsets.size < 2:
        raise ValueError("group_offsets must be a non-empty prefix sum")
    if descriptor_offsets.shape != item_offsets.shape:
        raise ValueError("item and descriptor group offsets differ")
    group_count = item_offsets.size - 1
    stage_array = np.asarray(stages, dtype=np.int64)
    if stage_array.shape != (group_count,):
        raise ValueError("stage vector does not cover every group")

    item_destination = np.asarray(arrays["item_destination"], dtype=np.int64)
    item_lane = np.asarray(arrays["item_lane_id"], dtype=np.int64)
    item_gate = np.asarray(arrays["item_gate_code"], dtype=np.int64)
    item_multiplicity = np.asarray(
        arrays["item_multiplicity"], dtype=np.int64
    )
    item_vectors = (item_destination, item_lane, item_gate, item_multiplicity)
    if any(vector.shape != item_destination.shape for vector in item_vectors):
        raise ValueError("destination item array shapes differ")
    if int(item_offsets[0]) != 0 or int(item_offsets[-1]) != item_destination.size:
        raise ValueError("item offsets do not cover destination items")
    if item_destination.size and (
        int(item_destination.min()) < 0 or int(item_destination.max()) >= 450
    ):
        raise ValueError("destination id is outside T450")
    if item_lane.size and (int(item_lane.min()) < 0 or int(item_lane.max()) >= 32):
        raise ValueError("lane id is outside L32")

    gate_count = np.asarray(arrays["source_gate_count"], dtype=np.int64)
    k_popcount = np.asarray(arrays["source_k_popcount"], dtype=np.int64)
    source_terms = np.asarray(arrays["source_term_count"], dtype=np.int64)
    source_delivery = np.asarray(
        arrays["source_delivery_count"], dtype=np.int64
    )
    descriptor_vectors = (gate_count, k_popcount, source_terms, source_delivery)
    if any(vector.shape != gate_count.shape for vector in descriptor_vectors):
        raise ValueError("source descriptor array shapes differ")
    if (
        int(descriptor_offsets[0]) != 0
        or int(descriptor_offsets[-1]) != gate_count.size
    ):
        raise ValueError("descriptor offsets do not cover descriptors")

    expected_terms = gate_count * k_popcount
    term_mismatches = int(np.count_nonzero(source_terms != expected_terms))
    if term_mismatches:
        raise ValueError(
            f"source term formula mismatch in {term_mismatches} descriptors"
        )

    active_source = k_popcount > 0
    dual_issue = k_popcount * ((gate_count + 1) // 2)
    ideal_one_gate_issue = k_popcount * (gate_count > 0)
    source_group = grouped_sum(source_terms, descriptor_offsets)
    dual_group = grouped_sum(dual_issue, descriptor_offsets)
    ideal_group = grouped_sum(ideal_one_gate_issue, descriptor_offsets)
    delivery_group = grouped_sum(source_delivery, descriptor_offsets)
    if int(delivery_group.sum()) != int(item_multiplicity.sum()):
        raise ValueError("source/destination delivery is not conserved")

    item_counts = np.diff(item_offsets)
    group_ids = np.repeat(np.arange(group_count, dtype=np.int64), item_counts)
    coefficient_keys = (group_ids * 450 + item_destination) * 32 + item_lane
    unique_keys, inverse = np.unique(coefficient_keys, return_inverse=True)
    coefficient_group_ids = unique_keys // (450 * 32)
    coefficient_group = np.bincount(
        coefficient_group_ids, minlength=group_count
    ).astype(np.int64)
    coefficient_values = np.bincount(
        inverse, weights=item_gate * item_multiplicity
    )
    if coefficient_values.size and np.any(coefficient_values <= 0):
        raise ValueError("coefficient contraction produced a non-positive item")

    group_rows = []
    for group_index in range(group_count):
        serial = int(source_group[group_index])
        dual = int(dual_group[group_index])
        coefficient = int(coefficient_group[group_index])
        group_rows.append(
            {
                "group_index": group_index,
                "stage": int(stage_array[group_index]),
                "source_owned_terms": serial,
                "dual_gate_issue_cycles": dual,
                "ideal_one_gate_issue_cycles": int(ideal_group[group_index]),
                "coefficient_nonzero_terms": coefficient,
                "destination_updates": int(delivery_group[group_index]),
                "dual_faster": dual < serial,
                "coefficient_fewer_terms": coefficient < serial,
            }
        )

    def aggregate(mask: np.ndarray) -> dict[str, Any]:
        source = int(source_group[mask].sum())
        dual = int(dual_group[mask].sum())
        ideal = int(ideal_group[mask].sum())
        coefficient = int(coefficient_group[mask].sum())
        delivery = int(delivery_group[mask].sum())
        lambda_threshold = (
            (coefficient - source) / delivery if delivery else None
        )
        return {
            "groups": int(np.count_nonzero(mask)),
            "source_owned_terms": source,
            "dual_gate_issue_cycles": dual,
            "dual_gate_issue_reduction": 1.0 - _ratio(dual, source),
            "dual_gate_term_phase_speedup": _ratio(source, dual),
            "ideal_one_gate_issue_cycles": ideal,
            "gate_cardinality_shaping_headroom": 1.0 - _ratio(ideal, source),
            "coefficient_nonzero_terms": coefficient,
            "coefficient_over_source_terms": _ratio(coefficient, source),
            "destination_updates": delivery,
            "coefficient_memory_cost_threshold_lambda": lambda_threshold,
            "dual_faster_groups": int(np.count_nonzero(dual_group[mask] < source_group[mask])),
            "dual_equal_groups": int(np.count_nonzero(dual_group[mask] == source_group[mask])),
            "dual_slower_groups": int(np.count_nonzero(dual_group[mask] > source_group[mask])),
            "coefficient_fewer_term_groups": int(
                np.count_nonzero(coefficient_group[mask] < source_group[mask])
            ),
        }

    all_groups = np.ones(group_count, dtype=np.bool_)
    totals = aggregate(all_groups)
    per_stage = {
        str(stage): aggregate(stage_array == stage)
        for stage in sorted(set(stage_array.tolist()))
    }
    totals.update(
        {
            "descriptors": int(gate_count.size),
            "active_source_descriptors": int(np.count_nonzero(active_source)),
            "active_source_gate_cardinality_histogram": _histogram(
                gate_count, active_source
            ),
            "active_source_gate_cardinality_le2_ratio": _ratio(
                int(np.count_nonzero(active_source & (gate_count <= 2))),
                int(np.count_nonzero(active_source)),
            ),
            "active_source_gate_cardinality_gt2": int(
                np.count_nonzero(active_source & (gate_count > 2))
            ),
            "maximum_coefficient": (
                int(coefficient_values.max()) if coefficient_values.size else 0
            ),
            "source_term_formula_mismatches": 0,
            "source_destination_delivery_conserved": True,
        }
    )
    return {"totals": totals, "per_stage": per_stage, "groups": group_rows}


def build_report(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    manifest_path = source_dir / "ordered_term_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = source_dir / manifest["payload_file"]
    payload_sha = sha256(payload_path)
    if payload_sha != manifest.get("payload_sha256"):
        raise ValueError("ordered payload SHA256 mismatch")
    qualification = manifest.get("qualification", {})
    if not qualification.get("qualified"):
        raise ValueError("profile is not qualified")
    if qualification.get("processed_samples") != 100:
        raise ValueError("profile is not the 100-sample cohort")
    if qualification.get("attached_blocks") != 12:
        raise ValueError("profile does not cover all 12 blocks")
    groups = manifest.get("groups", [])
    if len(groups) != qualification.get("captured_groups"):
        raise ValueError("manifest group list does not match qualification")
    stages = [int(group["stage"]) for group in groups]
    with np.load(payload_path, allow_pickle=False) as payload:
        analysis = analyze_arrays(payload, stages)
    if analysis["totals"]["groups"] != 4800:
        raise ValueError("expected the ep44 4,800-group profile")
    return {
        "schema": "local5_execution_object_codesign_screen_v1",
        "status": {
            "dual_gate_hardware_only": "NO_GO_AS_NEW_DATE_OBJECT",
            "gate_cardinality_qat": "CONDITIONAL_PROFILE_ONLY",
            "coefficient_fusion": "CONDITIONAL_MEMORY_DOMINANCE_ONLY",
        },
        "evidence": "[prof]",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_payload": str(payload_path),
        "source_payload_sha256": payload_sha,
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "config_sha256": manifest.get("config_sha256"),
        "analysis": analysis,
        "go_no_go": {
            "dual_gate_hardware_only": [
                "must beat a generic two-wide unique-gate issuer with the same two multipliers",
                "term-phase reduction is not full-pipeline cycle or energy reduction",
            ],
            "gate_cardinality_qat": [
                "valid825 AEE degradation <= 0.5% relative",
                "source-owned product terms reduced >= 25%",
                "real nonempty-group RTL cycles reduced >= 10%",
                "matched-SAIF EDP reduced >= 15%",
                "same legal 1RW accumulator contract and Acc32 zero mismatch",
            ],
            "coefficient_fusion": [
                "score-to-output RTL cycles reduced >= 15%",
                "matched-SAIF energy reduced >= 20%",
                "relation plus partial-accumulator state reduced >= 30%",
                "area increase <= 5% or a strict latency-energy-area Pareto",
                "no hidden full-size Acc32 scratchpad",
            ],
        },
        "claim_boundary": {
            "not_rtl": True,
            "not_cycle_or_energy": True,
            "not_encoder": True,
            "not_asic_ppa": True,
            "does_not_modify_frozen_table": True,
        },
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    totals = report["analysis"]["totals"]
    lines = [
        "# Local5 exact execution-object co-design screen",
        "",
        "## Verdict",
        "",
        "- Dual-gate hardware-only: **NO_GO_AS_NEW_DATE_OBJECT**. It is a generic two-wide unroll of the existing source-owned packet.",
        "- Gate-cardinality QAT: **CONDITIONAL_PROFILE_ONLY**. It must change the trained exact gate-class distribution and beat the same-resource two-wide baseline.",
        "- Shiftmax-to-coefficient fusion: **CONDITIONAL_MEMORY_DOMINANCE_ONLY**. It removes relation/partial-Acc materialization but increases products.",
        "- Evidence: `[prof]`; no RTL, cycle, energy, encoder, or ASIC-PPA claim.",
        "",
        "## Full ep44 profile100 ledger",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| groups / descriptors | {totals['groups']} / {totals['descriptors']} |",
        f"| active source descriptors | {totals['active_source_descriptors']} |",
        f"| current source-owned terms | {totals['source_owned_terms']} |",
        f"| dual-gate issue cycles | {totals['dual_gate_issue_cycles']} |",
        f"| dual-gate term-phase reduction | {100*totals['dual_gate_issue_reduction']:.3f}% |",
        f"| ideal one-gate issue cycles | {totals['ideal_one_gate_issue_cycles']} |",
        f"| QAT gate-cardinality headroom | {100*totals['gate_cardinality_shaping_headroom']:.3f}% |",
        f"| active source C<=2 | {100*totals['active_source_gate_cardinality_le2_ratio']:.3f}% |",
        f"| coefficient nonzero terms | {totals['coefficient_nonzero_terms']} |",
        f"| coefficient/source term ratio | {totals['coefficient_over_source_terms']:.3f}x |",
        f"| destination updates | {totals['destination_updates']} |",
        f"| coefficient break-even lambda | {totals['coefficient_memory_cost_threshold_lambda']:.3f} |",
        f"| observed maximum coefficient | {totals['maximum_coefficient']} |",
        "",
        "The break-even model is `T_source + lambda*destination_updates > T_coefficient`; lambda is the cost of one relation plus partial-Acc update measured in coefficient-product units.",
        "",
        "## Stage ledger",
        "",
        "| Stage | source term | dual issue | reduction | coefficient/source | lambda |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, row in report["analysis"]["per_stage"].items():
        lines.append(
            f"| {stage} | {row['source_owned_terms']} | {row['dual_gate_issue_cycles']} | "
            f"{100*row['dual_gate_issue_reduction']:.3f}% | "
            f"{row['coefficient_over_source_terms']:.3f}x | "
            f"{row['coefficient_memory_cost_threshold_lambda']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Promotion gates",
            "",
            "The hardware-only dual issuer is closed as a DATE contribution. Reopen only as the implementation target of exact gate-cardinality shaping, with a generic two-wide issuer as the named baseline.",
            "",
            "Coefficient fusion advances to RTL only after a matched-port analytical or measured memory model shows that relation plus partial-Acc activity exceeds the reported lambda threshold. Product counts alone reject it.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    md_path = args.output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(md_path, report)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
