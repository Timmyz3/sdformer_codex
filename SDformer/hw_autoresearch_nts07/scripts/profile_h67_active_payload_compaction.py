#!/usr/bin/env python3
"""Screen packed active-pair Q/K storage against a sparse-write row store."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/h67_exact_metadata_cascade_profile_20260809/report.json"
DEFAULT_OUTPUT = ROOT / "results/h67_active_payload_compaction_screen_20260814"
PAIR_COUNT = 225
PAYLOAD_BITS = 128
MACRO_DEPTH = 256
MACRO_WIDTH = 32
PAYLOAD_MACROS = PAYLOAD_BITS // MACRO_WIDTH


def percentile(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("percentile requires data")
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def build_report(source: dict[str, object]) -> dict[str, object]:
    rows = source["sample0_exact"]["row_details"]["8"]
    active = [int(row["active_pairs"]) for row in rows]
    if len(active) != int(source["sample0_exact"]["rows"]):
        raise ValueError("row count mismatch")

    full_kzero = float(source["full_profile100"]["both_kzero"])
    mean_active = sum(active) / len(active)
    dense_write_bits = len(active) * PAIR_COUNT * PAYLOAD_BITS
    active_write_bits = sum(active) * PAYLOAD_BITS
    macro_bits = PAYLOAD_MACROS * MACRO_DEPTH * MACRO_WIDTH
    maximum = max(active)
    compact_depth = MACRO_DEPTH * math.ceil(maximum / MACRO_DEPTH)
    compact_macro_count = PAYLOAD_MACROS * math.ceil(maximum / MACRO_DEPTH)

    result = {
        "schema": "h67_active_payload_compaction_screen_v1",
        "status": "PASS",
        "evidence": "[prof-sample0] row capacity + [prof] population write activity + [模型] macro/port screen",
        "scope": (
            "TTB8-ZKQI preload/row-store side path with four Q0/Q1/K0/K1 macros; "
            "not the frozen streaming RQTB K-only mainline"
        ),
        "candidate": "pack only non-both-K-zero Q0/Q1/K0/K1 payloads",
        "strong_baseline": "same dense pair addresses with macro write-enable only for active payloads",
        "sample0_rows": {
            "rows": len(active),
            "active_pairs_mean": mean_active,
            "active_pairs_p50": percentile(active, 0.50),
            "active_pairs_p95": percentile(active, 0.95),
            "active_pairs_p99": percentile(active, 0.99),
            "active_pairs_max": maximum,
            "fully_active_rows": sum(value == PAIR_COUNT for value in active),
            "rows_above_192": sum(value > 192 for value in active),
        },
        "write_activity": {
            "sample0_dense_payload_bits": dense_write_bits,
            "sample0_active_payload_bits": active_write_bits,
            "sample0_reduction": 1.0 - active_write_bits / dense_write_bits,
            "profile100_both_kzero_ratio": full_kzero,
            "profile100_active_payload_write_reduction_model": full_kzero,
            "packed_vs_sparse_addressed_incremental_payload_bits": 0,
        },
        "physical_capacity": {
            "payload_width_bits": PAYLOAD_BITS,
            "baseline_macro_count": PAYLOAD_MACROS,
            "candidate_macro_count": compact_macro_count,
            "macro_shape": f"{MACRO_DEPTH}x{MACRO_WIDTH}",
            "baseline_macro_bits": macro_bits,
            "candidate_macro_bits": compact_macro_count * MACRO_DEPTH * MACRO_WIDTH,
            "candidate_required_depth": maximum,
            "candidate_allocated_depth": compact_depth,
            "macro_count_reduction": 1.0 - compact_macro_count / PAYLOAD_MACROS,
        },
        "execution_contract": {
            "ingest_pairs_per_row": PAIR_COUNT,
            "score_issue": "unchanged from existing TTB8-ZKQI active bitmap",
            "gated_k_replay": "requires the same retained active K payload and a second ordered read phase",
            "cycle_speedup_model": 1.0,
        },
        "verdict": "NO_GO_AS_DATE_CONTRIBUTION",
        "reason": (
            "sample0 p95/max active depth is 225, so exact fixed hardware still needs "
            "four 256x32 payload macros. Sparse writes at original pair addresses obtain "
            "the same payload-bit activity as packing, while avoiding packed-address "
            "mapping and dual traversal. The candidate changes neither macro count nor "
            "score/projection issue count."
        ),
    }
    if compact_macro_count != PAYLOAD_MACROS:
        raise AssertionError("screen assumption changed: re-evaluate macro Pareto")
    return result


def render_markdown(report: dict[str, object]) -> str:
    rows = report["sample0_rows"]
    writes = report["write_activity"]
    physical = report["physical_capacity"]
    return f"""# H67 active-payload compaction innovation screen

## Verdict

`{report['verdict']}`. This is a read-only architecture screen, not RTL or PPA.

Scope: {report['scope']}.

## Evidence

- `[prof-sample0]` active pairs/row: mean `{rows['active_pairs_mean']:.2f}`, p50 `{rows['active_pairs_p50']}`, p95 `{rows['active_pairs_p95']}`, p99 `{rows['active_pairs_p99']}`, max `{rows['active_pairs_max']}`.
- `{rows['fully_active_rows']}` of `{rows['rows']}` rows are fully active; `{rows['rows_above_192']}` exceed 192 active pairs.
- `[prof]` profile100 both-K-zero ratio is `{writes['profile100_both_kzero_ratio']:.2%}`. It can reduce payload write activity, but the strong sparse-addressed baseline obtains the same reduction.
- `[模型]` baseline and packed candidate both require `{physical['candidate_macro_count']} x {physical['macro_shape']}` payload macros; macro-count reduction is `{physical['macro_count_reduction']:.2%}`.
- `[模型]` packed vs sparse-addressed incremental payload write bits: `{writes['packed_vs_sparse_addressed_incremental_payload_bits']}`; modeled cycle speedup: `{report['execution_contract']['cycle_speedup_model']:.3f}x`.

## Reason

{report['reason']}

This candidate is not promoted to RTL and does not modify frozen DATE tables.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_report(source)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    rows = report["sample0_rows"]
    print(
        f"PASS p95={rows['active_pairs_p95']} max={rows['active_pairs_max']} "
        f"macro_reduction={report['physical_capacity']['macro_count_reduction']:.2%} "
        f"verdict={report['verdict']}"
    )


if __name__ == "__main__":
    main()
