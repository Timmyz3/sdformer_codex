#!/usr/bin/env python3
"""Best-case go/no-go model for cross-head spatial Acc striping.

The model deliberately favors the candidate. It omits controller replication and
per-head relation-frontier state. If this optimistic lower bound misses the
pre-registered gates, the architecture should not advance to RTL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HEIGHT = 15
WIDTH = 15
TIME_PLANES = 2
HEAD_DIM = 32
OUT_DIM = 32
ACC_W = 32
WEIGHT_W = 8
K_W = 32
ROLLING_SHARED_BITS = 3735
STRIPE_HEIGHTS = (1, 3, 5, 15)
STAGE_HEADS = (3, 6, 12, 24)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model(source: dict) -> dict:
    physical = source["physical_width"]
    if physical["accumulator_payload_bits"] != 460800:
        raise ValueError("unexpected Local5 OUT32 accumulator payload")
    if source["population"]["groups"] != 100:
        raise ValueError("expected the locked 100-group population")
    if source["correctness"]["acc32_mismatch"] != 0:
        raise ValueError("source OUT32 evidence is not bit-exact")

    full_acc_bits = TIME_PLANES * HEIGHT * WIDTH * OUT_DIM * ACC_W
    weight_bits_per_head = HEAD_DIM * OUT_DIM * WEIGHT_W
    baseline_state_bits = full_acc_bits + weight_bits_per_head + ROLLING_SHARED_BITS
    mean_busy_per_head = source["cycles"]["rolling_qsilent"] / source["population"]["groups"]
    weight_values_per_head = HEAD_DIM * OUT_DIM

    rows = []
    for stripe_height in STRIPE_HEIGHTS:
        stripe_count = math.ceil(HEIGHT / stripe_height)
        stripe_acc_bits = (
            TIME_PLANES * stripe_height * WIDTH * OUT_DIM * ACC_W
        )
        active_k_scratch_bits = stripe_height * WIDTH * K_W

        reload_state_bits = (
            stripe_acc_bits
            + weight_bits_per_head
            + active_k_scratch_bits
            + ROLLING_SHARED_BITS
        )
        reload_weight_ratio = stripe_count
        reload_cycles_lower_bound = (
            mean_busy_per_head + stripe_count * weight_values_per_head
        )
        baseline_cycles_with_load = mean_busy_per_head + weight_values_per_head

        for heads in STAGE_HEADS:
            # Keeping two K rows per head is the optimistic minimum needed to
            # avoid rereading the r=1 vertical stencil halo after head switches.
            per_head_k_halo_bits = 2 * WIDTH * K_W
            resident_state_bits = (
                stripe_acc_bits
                + heads * weight_bits_per_head
                + heads * per_head_k_halo_bits
                + active_k_scratch_bits
                + ROLLING_SHARED_BITS
            )
            rows.append(
                {
                    "stripe_height": stripe_height,
                    "stripe_count": stripe_count,
                    "heads": heads,
                    "baseline_state_bits": baseline_state_bits,
                    "reload_per_stripe": {
                        "allocated_state_bits": reload_state_bits,
                        "state_reduction": baseline_state_bits / reload_state_bits,
                        "external_weight_read_ratio": reload_weight_ratio,
                        "cycle_slowdown_lower_bound": (
                            reload_cycles_lower_bound / baseline_cycles_with_load
                        ),
                    },
                    "all_head_weights_resident": {
                        "allocated_state_bits": resident_state_bits,
                        "state_reduction": baseline_state_bits / resident_state_bits,
                        "external_weight_read_ratio": 1.0,
                        # The loop interchange preserves every term/update. It
                        # removes no Acc access and adds no reusable product.
                        "memory_bit_activity_reduction_upper_bound": 0.0,
                    },
                }
            )

    max_head_r1 = next(
        row for row in rows if row["stripe_height"] == 1 and row["heads"] == 24
    )
    reload_r1 = max_head_r1["reload_per_stripe"]
    resident_r1 = max_head_r1["all_head_weights_resident"]
    gates = {
        "acc_payload_reduction_ge_4x": full_acc_bits / (2 * WIDTH * OUT_DIM * ACC_W) >= 4.0,
        "max_head_total_state_reduction_ge_2x": resident_r1["state_reduction"] >= 2.0,
        "memory_bit_activity_reduction_ge_20pct": (
            resident_r1["memory_bit_activity_reduction_upper_bound"] >= 0.20
        ),
        "reload_cycle_loss_le_5pct": reload_r1["cycle_slowdown_lower_bound"] <= 1.05,
    }
    return {
        "schema": "local5_cross_head_spatial_stripe_gonogo_v1",
        "status": "NO_GO_NO_RTL",
        "evidence": "[model]+[optimistic-lower-bound]",
        "candidate": (
            "cross-head spatial stripe accumulator retirement relative to B2v "
            "full-tile vector-resident TCFM5"
        ),
        "baseline": {
            "full_acc_bits": full_acc_bits,
            "one_head_weight_bits": weight_bits_per_head,
            "shared_rolling_bits": ROLLING_SHARED_BITS,
            "modelled_allocated_state_bits": baseline_state_bits,
            "mean_population_busy_cycles_per_head_excluding_weight_load": mean_busy_per_head,
        },
        "best_case_rows": rows,
        "max_head_r1_summary": {
            "heads": 24,
            "stripe_height": 1,
            "acc_payload_reduction": full_acc_bits / (2 * WIDTH * OUT_DIM * ACC_W),
            "resident_all_head_state_reduction": resident_r1["state_reduction"],
            "resident_memory_bit_activity_reduction_upper_bound": resident_r1[
                "memory_bit_activity_reduction_upper_bound"
            ],
            "reload_external_weight_read_ratio": reload_r1["external_weight_read_ratio"],
            "reload_cycle_slowdown_lower_bound": reload_r1[
                "cycle_slowdown_lower_bound"
            ],
        },
        "gates": gates,
        "decision_reasons": [
            "Keeping all 24 head weights and only the minimum two K halo rows reduces total modelled state by less than 2x even though Acc depth falls 15x.",
            "Reloading one head weight tile per stripe raises external weight reads by 15x and is slower even before stripe setup and drain overhead.",
            "The loop interchange preserves all term and Acc updates, so memory bit activity cannot fall by the required 20% without an additional, currently absent execution-object change.",
            "Controller and per-head relation-frontier replication are omitted, making this a candidate-favorable lower bound.",
        ],
        "claim_boundary": [
            "No RTL was added and no frozen performance column changes.",
            "The model does not estimate SRAM macro energy; a shallower Acc may lower per-access energy, but it does not reduce access count.",
            "The current 100-group busy cycles exclude weight load/readback; the reload schedule adds only the unavoidable one-value-per-cycle weight-load lower bound.",
            "The candidate is a loop-interchange/storage-lifetime tradeoff, not a fourth DATE contribution.",
        ],
    }


def markdown(report: dict) -> str:
    s = report["max_head_r1_summary"]
    gates = report["gates"]
    lines = [
        "# Local5 cross-head spatial stripe Acc go/no-go",
        "",
        f"Status: **{report['status']}** (`{report['evidence']}`)",
        "",
        "The model uses a candidate-favorable lower bound against the B2v full-tile vector-resident TCFM5 baseline.",
        "",
        "## Max-head, r=1 result",
        "",
        "| Metric | Result | Gate |",
        "|---|---:|---:|",
        f"| Acc payload reduction | {s['acc_payload_reduction']:.2f}x | >=4x PASS |",
        f"| Total modelled state reduction | {s['resident_all_head_state_reduction']:.3f}x | >=2x {'PASS' if gates['max_head_total_state_reduction_ge_2x'] else 'FAIL'} |",
        f"| Memory bit-activity reduction upper bound | {100*s['resident_memory_bit_activity_reduction_upper_bound']:.1f}% | >=20% {'PASS' if gates['memory_bit_activity_reduction_ge_20pct'] else 'FAIL'} |",
        f"| Reload schedule weight-read ratio | {s['reload_external_weight_read_ratio']:.1f}x | - |",
        f"| Reload schedule cycle slowdown lower bound | {s['reload_cycle_slowdown_lower_bound']:.3f}x | <=1.05x {'PASS' if gates['reload_cycle_loss_le_5pct'] else 'FAIL'} |",
        "",
        "## Decision",
        "",
    ]
    lines.extend(f"- {reason}" for reason in report["decision_reasons"])
    lines.extend(["", "## Boundaries", ""])
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/local5_out32_population_sensitivity_20260814/report.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/local5_cross_head_spatial_stripe_gonogo_20260814"),
    )
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    report = build_model(source)
    report["provenance"] = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(markdown(report))
    print(json.dumps({"status": report["status"], "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
