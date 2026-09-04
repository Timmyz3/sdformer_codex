#!/usr/bin/env python3
"""Summarize the paired VCS performance/bit-cost tradeoff of the M4 state queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ACC_BITS = 96 * 32
IDENTITY_BITS = 2 + 5 + 16 + 32 + 4 + 4 + 1 + 1 + 1 + 32
ENTRY_BITS = ACC_BITS + IDENTITY_BITS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(points: list[tuple[int, Path, dict[str, Any]]]) -> dict[str, Any]:
    if not points:
        raise ValueError("queue DSE needs at least one point")
    points.sort(key=lambda item: item[0])
    if len({depth for depth, _path, _data in points}) != len(points):
        raise ValueError("queue depths must be unique")
    reference_population = points[0][2]["population"]
    local_cycles = int(points[0][2]["overall"]["local_cycles"])
    for depth, _path, data in points:
        if depth <= 0 or data.get("status") != \
                "PASS_M4_STATEFUL_PAIRED_VCS_CORE_CYCLES":
            raise ValueError(f"queue depth {depth} ledger is not admitted")
        if data["population"] != reference_population:
            raise ValueError("queue DSE populations differ")
        if int(data["overall"]["local_cycles"]) != local_cycles:
            raise ValueError("Local-only cycles changed across queue depths")

    best_hybrid = min(int(data["overall"]["hybrid_cycles"])
                      for _depth, _path, data in points)
    rows = []
    for depth, path, data in points:
        hybrid = int(data["overall"]["hybrid_cycles"])
        rows.append({
            "queue_depth": depth,
            "payload_bits": depth * ENTRY_BITS,
            "accumulator_payload_bits": depth * ACC_BITS,
            "identity_payload_bits": depth * IDENTITY_BITS,
            "local_cycles": local_cycles,
            "hybrid_cycles": hybrid,
            "cycle_delta_vs_local": local_cycles - hybrid,
            "speedup_vs_local": local_cycles / hybrid,
            "hybrid_cycle_penalty_vs_best": hybrid - best_hybrid,
            "hybrid_cycle_penalty_vs_best_fraction": hybrid / best_hybrid - 1.0,
            "regression_pairs": data["overall"]["hybrid_regression_pairs"],
            "ledger": str(path.resolve()),
            "ledger_sha256": sha256(path),
        })
    for row in rows:
        row["dominated_by_shallower_equal_cycle_point"] = any(
            other["queue_depth"] < row["queue_depth"]
            and other["hybrid_cycles"] <= row["hybrid_cycles"]
            for other in rows
        )
    throughput_depth = min(
        row["queue_depth"] for row in rows if row["hybrid_cycles"] == best_hybrid
    )
    smallest_depth = rows[0]["queue_depth"]
    return {
        "entry_bits": ENTRY_BITS,
        "accumulator_bits_per_entry": ACC_BITS,
        "identity_bits_per_entry": IDENTITY_BITS,
        "smallest_area_proxy_candidate_depth": smallest_depth,
        "minimum_depth_at_best_cycles": throughput_depth,
        "paper_candidate": "PENDING_DC_AREA_DEPTH1_VS_DEPTH2",
        "points": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point", action="append", nargs=2, metavar=("DEPTH", "LEDGER"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = []
    for raw_depth, raw_path in args.point:
        path = Path(raw_path)
        points.append((
            int(raw_depth), path,
            json.loads(path.read_text(encoding="utf-8")),
        ))
    result = {
        "schema": "m4_state_queue_vcs_dse_v1",
        "status": "PASS_M4_STATE_QUEUE_VCS_DSE_PREMACRO",
        "claim_boundary": (
            "Bit-count area proxy plus paired always-ready VCS core cycles on "
            "the bounded Motion-enriched checkpoint-bitmap subset. No standard-"
            "cell or SRAM macro area/energy is inferred from payload bits; the "
            "depth-1 versus depth-2 candidate decision remains pending DC."
        ),
        **summarize(points),
        "summarizer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "PASS M4 queue DSE: smallest="
        f"{result['smallest_area_proxy_candidate_depth']} best-cycle-depth="
        f"{result['minimum_depth_at_best_cycles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
