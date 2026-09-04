#!/usr/bin/env python3
"""Miter Local5 destination-complete coefficients against source-owned Acc32."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

if __package__:
    from .miter_local5_source_owned_gate_quotient_rtl import (
        GATE_W,
        HEAD_DIM,
        OUT_DIM,
        ROLES,
        SOURCES,
        analyze_group,
        read_memh,
        sha256,
        signed,
        source_for,
        validate_artifact,
    )
else:
    from miter_local5_source_owned_gate_quotient_rtl import (
        GATE_W,
        HEAD_DIM,
        OUT_DIM,
        ROLES,
        SOURCES,
        analyze_group,
        read_memh,
        sha256,
        signed,
        source_for,
        validate_artifact,
    )


def wrap32(value: int) -> int:
    return value & 0xFFFFFFFF


def coefficient_projection(
    *,
    candidate_k: list[int],
    valid_mask: list[int],
    packed_gates: list[int],
    weights: list[list[int]],
) -> dict[str, Any]:
    if not (
        len(candidate_k) == len(valid_mask) == len(packed_gates) == SOURCES
    ):
        raise ValueError("group does not contain 450 destination rows")
    if len(weights) != HEAD_DIM or any(len(row) != OUT_DIM for row in weights):
        raise ValueError("weight shape is not 32x2")

    acc = [[0 for _ in range(OUT_DIM)] for _ in range(SOURCES)]
    coefficient_terms = 0
    active_destinations = 0
    maximum_coefficient = 0
    coefficient_histogram: Counter[int] = Counter()
    invalid_nonzero_gates = 0

    for destination in range(SOURCES):
        destination_active = False
        for lane in range(HEAD_DIM):
            coefficient = 0
            for role in range(ROLES):
                valid = bool((valid_mask[destination] >> role) & 1)
                gate = (packed_gates[destination] >> (role * GATE_W)) & (
                    (1 << GATE_W) - 1
                )
                if valid != (source_for(destination, role) is not None):
                    raise AssertionError(
                        f"topology-valid mismatch destination={destination} role={role}"
                    )
                if not valid:
                    invalid_nonzero_gates += int(gate != 0)
                    continue
                k_bitmap = (
                    candidate_k[destination] >> (role * HEAD_DIM)
                ) & 0xFFFFFFFF
                if (k_bitmap >> lane) & 1:
                    coefficient += gate
            if coefficient == 0:
                continue
            destination_active = True
            coefficient_terms += 1
            maximum_coefficient = max(maximum_coefficient, coefficient)
            coefficient_histogram[coefficient] += 1
            for out_index in range(OUT_DIM):
                acc[destination][out_index] += (
                    coefficient * weights[lane][out_index]
                )
        active_destinations += int(destination_active)

    return {
        "coefficient_terms": coefficient_terms,
        "active_destinations": active_destinations,
        "maximum_coefficient": maximum_coefficient,
        "coefficient_histogram": dict(sorted(coefficient_histogram.items())),
        "invalid_nonzero_gates": invalid_nonzero_gates,
        "acc": acc,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=(
            root
            / "tb_qfit/vectors/local5_ep44_hardware_rebind_20260815_"
            "score_projection100"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root / "results/local5_ep44_destination_coefficient_miter_20260816"
        ),
    )
    args = parser.parse_args()

    manifest_path = args.vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "local5_score_projection_vectors_v1"
        or manifest.get("selection", {}).get("groups") != 100
        or manifest.get("shape", {}).get("out_dim") != OUT_DIM
    ):
        raise ValueError("vector manifest is not the ep44 100-group OUT_DIM=2 cohort")

    source_manifest_path = Path(manifest["source_manifest"])
    if sha256(source_manifest_path) != manifest["source_manifest_sha256"]:
        raise ValueError("source manifest SHA-256 mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = source_manifest.get("checkpoint_sha256")
    if not checkpoint_sha256:
        raise ValueError("source manifest does not bind a checkpoint SHA-256")

    paths = {
        name: validate_artifact(args.vector_dir, manifest, name)
        for name in (
            "input_candidate_k",
            "input_valid",
            "expected_gates",
            "input_weights",
            "expected_terms",
            "expected_updates",
            "expected_acc",
        )
    }
    candidate_k = read_memh(paths["input_candidate_k"])
    valid_mask = read_memh(paths["input_valid"])
    packed_gates = read_memh(paths["expected_gates"])
    weight_values = read_memh(paths["input_weights"])
    expected_terms = read_memh(paths["expected_terms"])
    expected_updates = read_memh(paths["expected_updates"])
    expected_acc = read_memh(paths["expected_acc"])

    totals = Counter()
    stage = Counter()
    group_rows: list[dict[str, Any]] = []
    coefficient_ratios: list[float] = []
    for group, metadata in enumerate(manifest["selection"]["rows"]):
        row_base = group * SOURCES
        weight_base = group * HEAD_DIM * OUT_DIM
        weights = [
            [
                signed(
                    weight_values[weight_base + lane * OUT_DIM + out_index], 8
                )
                for out_index in range(OUT_DIM)
            ]
            for lane in range(HEAD_DIM)
        ]
        inputs = {
            "candidate_k": candidate_k[row_base : row_base + SOURCES],
            "valid_mask": valid_mask[row_base : row_base + SOURCES],
            "packed_gates": packed_gates[row_base : row_base + SOURCES],
            "weights": weights,
        }
        source = analyze_group(**inputs)
        coefficient = coefficient_projection(**inputs)
        source_acc = [value for row in source["acc"] for value in row]
        coefficient_acc = [value for row in coefficient["acc"] for value in row]
        expected = expected_acc[
            group * SOURCES * OUT_DIM : (group + 1) * SOURCES * OUT_DIM
        ]
        source_mismatch = sum(
            wrap32(actual) != wrap32(reference)
            for actual, reference in zip(source_acc, expected, strict=True)
        )
        coefficient_mismatch = sum(
            wrap32(actual) != wrap32(reference)
            for actual, reference in zip(coefficient_acc, expected, strict=True)
        )
        cross_mismatch = sum(
            wrap32(left) != wrap32(right)
            for left, right in zip(source_acc, coefficient_acc, strict=True)
        )
        if source_mismatch or coefficient_mismatch or cross_mismatch:
            raise AssertionError(
                f"group {group} Acc32 mismatch source={source_mismatch} "
                f"coefficient={coefficient_mismatch} cross={cross_mismatch}"
            )
        if source["terms"] != expected_terms[group]:
            raise AssertionError(f"group {group} source term mismatch")
        if source["updates"] != expected_updates[group]:
            raise AssertionError(f"group {group} source update mismatch")
        if coefficient["invalid_nonzero_gates"]:
            raise AssertionError(f"group {group} invalid candidate has nonzero gate")

        coefficient_ratio = (
            coefficient["coefficient_terms"] / source["terms"]
            if source["terms"]
            else 1.0
        )
        coefficient_ratios.append(coefficient_ratio)
        stage_id = int(metadata["stage"])
        totals["source_terms"] += source["terms"]
        totals["source_updates"] += source["updates"]
        totals["coefficient_terms"] += coefficient["coefficient_terms"]
        totals["active_destinations"] += coefficient["active_destinations"]
        totals["acc_values"] += len(expected)
        totals["maximum_coefficient"] = max(
            totals["maximum_coefficient"], coefficient["maximum_coefficient"]
        )
        stage[(stage_id, "source_terms")] += source["terms"]
        stage[(stage_id, "source_updates")] += source["updates"]
        stage[(stage_id, "coefficient_terms")] += coefficient[
            "coefficient_terms"
        ]
        stage[(stage_id, "active_destinations")] += coefficient[
            "active_destinations"
        ]
        group_rows.append(
            {
                "group": group,
                "stage": stage_id,
                "source_terms": source["terms"],
                "source_updates": source["updates"],
                "coefficient_terms": coefficient["coefficient_terms"],
                "active_destinations": coefficient["active_destinations"],
                "coefficient_over_source": coefficient_ratio,
            }
        )

    source_terms = totals["source_terms"]
    source_updates = totals["source_updates"]
    coefficient_terms = totals["coefficient_terms"]
    active_destinations = totals["active_destinations"]
    report = {
        "schema": "local5_destination_coefficient_miter_v1",
        "status": "CONDITIONAL_MEMORY_ONLY_NO_RTL",
        "evidence": "[model]+[numeric] ep44 100-group OUT_DIM=2 true INT8 weights",
        "claim_boundary": (
            "Bit-exact and access-count evidence only; not RTL, cycle, energy, encoder, or PPA."
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": checkpoint_sha256,
        "totals": {
            "groups": 100,
            "acc32_values": totals["acc_values"],
            "acc32_mismatches": 0,
            "source_terms": source_terms,
            "source_updates": source_updates,
            "coefficient_terms": coefficient_terms,
            "active_destinations": active_destinations,
            "coefficient_over_source_terms": (
                coefficient_terms / source_terms if source_terms else 1.0
            ),
            "ideal_update_reduction": (
                1.0 - active_destinations / source_updates
                if source_updates
                else 0.0
            ),
            "maximum_coefficient": totals["maximum_coefficient"],
            "group_ratio_p50": percentile(coefficient_ratios, 0.50),
            "group_ratio_p95": percentile(coefficient_ratios, 0.95),
            "groups_coefficient_fewer_terms": sum(
                row["coefficient_terms"] < row["source_terms"]
                for row in group_rows
            ),
        },
        "per_stage": {
            str(stage_id): {
                metric: stage[(stage_id, metric)]
                for metric in (
                    "source_terms",
                    "source_updates",
                    "coefficient_terms",
                    "active_destinations",
                )
            }
            for stage_id in range(4)
        },
        "promotion_gate": {
            "cycles": "must reduce score-to-output cycles >=15% vs source-owned and destination MFEP+W4",
            "activity": "must reduce total storage bit activity >=30% after wider coefficient products",
            "tails": "all stages non-regressive and p95 group regression <=2%",
            "before_rtl": "matched-port analytic model first; no hidden full-size Acc32",
        },
        "groups": group_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    totals_row = report["totals"]
    markdown = f"""# Local5 destination-complete coefficient miter

Status: `{report['status']}`.

| Metric | Value |
|---|---:|
| Acc32 values / mismatch | {totals_row['acc32_values']} / 0 |
| source-owned terms | {source_terms} |
| coefficient terms | {coefficient_terms} |
| coefficient/source | {totals_row['coefficient_over_source_terms']:.3f}x |
| source partial updates | {source_updates} |
| ideal destination commits | {active_destinations} |
| ideal update reduction | {100*totals_row['ideal_update_reduction']:.3f}% |
| max coefficient | {totals_row['maximum_coefficient']} |
| group ratio p50 / p95 | {totals_row['group_ratio_p50']:.3f}x / {totals_row['group_ratio_p95']:.3f}x |
| groups with fewer coefficient terms | {totals_row['groups_coefficient_fewer_terms']} / 100 |

This is `[model]+[numeric]` evidence with true ep44 INT8 weights. It is not RTL,
cycle, energy, encoder, or PPA evidence. The candidate stays closed until a
matched-port model beats both source-owned and destination MFEP+W4 baselines.
"""
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
