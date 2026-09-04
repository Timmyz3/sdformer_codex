#!/usr/bin/env python3
"""Build a paired Local-only/Hybrid ledger from the stateful M4 VCS log."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


SEQUENCE_RE = re.compile(
    r"^M4_STATE_SEQ id=(?P<id>\d+) mode=(?P<mode>\S+) "
    r"cycles=(?P<cycles>\d+) descriptors=(?P<descriptors>\d+) "
    r"outputs=(?P<outputs>\d+) motion_outputs=(?P<motion>\d+)$",
    re.MULTILINE,
)
TOTAL_RE = re.compile(
    r"^PASS_M4_STATEFUL_PERF pairs=(?P<pairs>\d+) "
    r"local_cycles=(?P<local>\d+) hybrid_cycles=(?P<hybrid>\d+)$",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires a non-empty population")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(log_text: str, vector_manifest: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "sequence_id": int(match.group("id")),
            "mode": match.group("mode"),
            "cycles": int(match.group("cycles")),
            "descriptors": int(match.group("descriptors")),
            "outputs": int(match.group("outputs")),
            "motion_outputs": int(match.group("motion")),
        }
        for match in SEQUENCE_RE.finditer(log_text)
    ]
    expected_sequences = int(vector_manifest["population"]["sequences"])
    if len(rows) != expected_sequences:
        raise ValueError(
            f"VCS sequence count mismatch: {len(rows)} != {expected_sequences}"
        )
    if [row["sequence_id"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("VCS sequence IDs are not contiguous")
    if len(rows) % 2:
        raise ValueError("paired VCS population must contain an even sequence count")

    identity_ranges: list[tuple[str, int]] = []
    for label, identity in vector_manifest["identities"].items():
        identity_ranges.append((label, int(identity["selected_sample_groups"])))
    expected_pairs = sum(count for _label, count in identity_ranges)
    if expected_pairs * 2 != len(rows):
        raise ValueError("identity group population does not cover all VCS pairs")

    pairs = []
    cursor = 0
    identity_pair_limits = []
    cumulative = 0
    for label, count in identity_ranges:
        cumulative += count
        identity_pair_limits.append((cumulative, label))
    for pair_index in range(expected_pairs):
        local = rows[cursor]
        hybrid = rows[cursor + 1]
        cursor += 2
        if local["mode"] != "local_only" or hybrid["mode"] != "hybrid_local_motion":
            raise ValueError(f"pair {pair_index} mode ordering is invalid")
        if local["motion_outputs"] != 0 or hybrid["motion_outputs"] <= 0:
            raise ValueError(f"pair {pair_index} lacks Local/Motion separation")
        if local["descriptors"] != hybrid["descriptors"] or \
                local["outputs"] != hybrid["outputs"]:
            raise ValueError(f"pair {pair_index} geometry differs across modes")
        identity = next(
            label for limit, label in identity_pair_limits if pair_index < limit
        )
        pairs.append({
            "pair_index": pair_index,
            "identity": identity,
            "local_sequence_id": local["sequence_id"],
            "hybrid_sequence_id": hybrid["sequence_id"],
            "descriptors": local["descriptors"],
            "outputs": local["outputs"],
            "motion_outputs": hybrid["motion_outputs"],
            "local_cycles": local["cycles"],
            "hybrid_cycles": hybrid["cycles"],
            "speedup_vs_local": local["cycles"] / hybrid["cycles"],
            "cycle_delta": local["cycles"] - hybrid["cycles"],
        })

    total_match = TOTAL_RE.search(log_text)
    if total_match is None:
        raise ValueError("VCS performance total is missing")
    local_cycles = sum(item["local_cycles"] for item in pairs)
    hybrid_cycles = sum(item["hybrid_cycles"] for item in pairs)
    if (
        int(total_match.group("pairs")) != len(pairs)
        or int(total_match.group("local")) != local_cycles
        or int(total_match.group("hybrid")) != hybrid_cycles
    ):
        raise ValueError("VCS performance total does not reconcile with pairs")

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        speeds = [float(item["speedup_vs_local"]) for item in items]
        local_total = sum(int(item["local_cycles"]) for item in items)
        hybrid_total = sum(int(item["hybrid_cycles"]) for item in items)
        return {
            "pairs": len(items),
            "local_cycles": local_total,
            "hybrid_cycles": hybrid_total,
            "cycle_delta": local_total - hybrid_total,
            "aggregate_speedup_vs_local": local_total / hybrid_total,
            "pair_speedup_min": min(speeds),
            "pair_speedup_median": statistics.median(speeds),
            "pair_speedup_p95": percentile(speeds, 0.95),
            "pair_speedup_max": max(speeds),
            "hybrid_regression_pairs": sum(
                item["hybrid_cycles"] > item["local_cycles"] for item in items
            ),
            "equal_cycle_pairs": sum(
                item["hybrid_cycles"] == item["local_cycles"] for item in items
            ),
            "motion_outputs": sum(int(item["motion_outputs"]) for item in items),
        }

    return {
        "overall": aggregate(pairs),
        "per_identity": {
            label: aggregate([item for item in pairs if item["identity"] == label])
            for label, _count in identity_ranges
        },
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-log", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--queue-depth", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.queue_depth <= 0:
        raise ValueError("queue depth must be positive")
    vector_manifest = json.loads(args.vector_manifest.read_text(encoding="utf-8"))
    result = summarize(
        args.simulation_log.read_text(encoding="utf-8", errors="replace"),
        vector_manifest,
    )
    output = {
        "schema": "m4_stateful_vcs_paired_perf_v1",
        "status": "PASS_M4_STATEFUL_PAIRED_VCS_CORE_CYCLES",
        "claim_boundary": (
            "Synopsys VCS cycle-exact paired Local-only/Hybrid execution on the "
            "bounded checkpoint-bitmap subset with deterministic synthetic INT8 "
            "weights, always-ready weight/output interfaces, and a state "
            f"transaction queue of depth {args.queue_depth}. Includes descriptor "
            "handshake, M4 issue/response/control, Local state writes, Motion "
            "synchronous RMW, queue stalls, and final state drain. Excludes target "
            "SRAM macro timing/energy, external memory stalls, unrelated network "
            "operators, and checkpoint-weight Acc32."
        ),
        "queue_depth": args.queue_depth,
        "population": vector_manifest["population"],
        **result,
        "sha256": {
            "simulation_log": sha256(args.simulation_log),
            "vector_manifest": sha256(args.vector_manifest),
            "vector_trace": vector_manifest["sha256"][
                "stateful_real_descriptors.txt"
            ],
            "summarizer": sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    overall = output["overall"]
    print(
        "PASS M4 stateful paired VCS: "
        f"local={overall['local_cycles']} hybrid={overall['hybrid_cycles']} "
        f"speedup={overall['aggregate_speedup_vs_local']:.6f}x "
        f"regressions={overall['hybrid_regression_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
