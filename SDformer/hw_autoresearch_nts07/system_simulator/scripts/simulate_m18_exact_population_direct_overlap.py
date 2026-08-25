#!/usr/bin/env python3
"""Drive M15 finite-credit events from M17 exact full-population C4 costs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulate_m15_finite_credit_retirement import VARIANTS, sha256, simulate_event_stream


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pattern_stream_sha256(patterns: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for pattern in patterns:
        digest.update(
            json.dumps(pattern, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_inputs(
    oracle_dir: Path, reconciliation_path: Path, boundaries_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    # Fail before opening the 44.7 MB prototype file or the ordered NPZ.  The
    # historical adapter is permanently retired: producer P_DONE is not a
    # legal readiness event under dynamic no_running BatchNorm.
    boundaries = json.loads(boundaries_path.read_text(encoding="utf-8"))
    if boundaries.get("status") == "PASS_EXACT_PATH_CERTIFICATES_ALL_BN_BLOCKED_M15_PROHIBITED":
        raise ValueError(
            "M18 path certificates prohibit M15: every historical direct-M4 edge is "
            "behind a no_running BatchNorm global-reduction barrier"
        )
    # The v1 boundary artifact stopped at the nearest producer and incorrectly
    # treated a trace SHA as an immutable hardware version.  M18 v2 proved that
    # all 13 paths cross dynamic BatchNorm, so accepting a legacy v1 artifact
    # here would silently resurrect an invalid overlap result.
    raise ValueError(
        "legacy M18 direct-boundary overlap is invalidated; a future adapter must consume "
        "an exact BN-ready event rather than producer P_DONE"
    )


def build_patterns(
    manifest: dict[str, Any], prototypes: list[dict[str, Any]],
    stream: dict[str, np.ndarray], boundaries: dict[str, Any], *, line: str,
) -> list[dict[str, Any]]:
    if line not in {"local", "hybrid"}:
        raise ValueError("M18 line must be local or hybrid")
    boundary_by_key = {
        (
            row["sample_id"], row["sequence_key"], row["producer"],
            row["producer_call_index"],
        ): (version, row)
        for version, row in enumerate(boundaries["rows"])
    }
    operators = manifest["operators"]
    operator_index = stream["operator_index"]
    population = stream["population_cluster_id"]
    prototype_id = stream["prototype_id"]
    if not (len(operator_index) == len(population) == len(prototype_id) == manifest["ordered_clusters"]):
        raise ValueError("M17 ordered stream cardinality mismatch")
    patterns = []
    for position in range(len(operator_index)):
        operator = operators[int(operator_index[position])]
        prototype = prototypes[int(prototype_id[position])]
        key = (
            operator["sample_id"], operator["sequence_key"], operator["name"],
            operator["operator_call_index"],
        )
        if key not in boundary_by_key:
            raise ValueError("M17 operator lacks one call-qualified M18 boundary")
        version, boundary = boundary_by_key[key]
        if (
            int(prototype["chunks"]) != int(operator["chunks"])
            or int(prototype["fanout"]) != int(operator["fanout"])
            or int(prototype["lane_tiles"]) != int(operator["lane_tiles"])
        ):
            raise ValueError("M17 prototype/operator geometry mismatch")
        lane_field = f"{line}_lane_compute_cycles_by_t"
        descriptor = prototype["descriptor_cycles_by_t"]
        lane_compute = prototype[lane_field]
        if len(descriptor) != operator["temporal_steps"] or len(lane_compute) != len(descriptor):
            raise ValueError("M17 prototype temporal geometry mismatch")
        scheduler_identity = canonical_sha256({
            "operator_scheduler_statistics_sha256": operator[
                "ordered_scheduler_sufficient_statistics_sha256"
            ],
            "population_cluster_id": int(population[position]),
            "prototype_cost_source_sha256": prototype["cost_source_sha256"],
            "line": line,
        })
        patterns.append({
            "sample_id": operator["sample_id"],
            "sequence_key": operator["sequence_key"],
            "producer": operator["name"],
            "edge": boundary["edge"],
            "edge_kind": "direct_m4",
            "admitted_for_overlap": True,
            "producer_call_index": operator["operator_call_index"],
            "edge_call_index": boundary["edge_call_index"],
            "version": version,
            "version_identity_sha256": boundary["immutable_hardware_version_identity_sha256"],
            "sample_cluster_id": position,
            "population_cluster_id": int(population[position]),
            "cost_prototype_id": int(prototype_id[position]),
            "cost_basis": "exact_full_population",
            "scheduler_sufficient_statistics_sha256": scheduler_identity,
            "cost_source_sha256": prototype["cost_source_sha256"],
            "fanout": prototype["fanout"],
            "lane_tiles": prototype["lane_tiles"],
            "chunks": prototype["chunks"],
            "steps": [
                {
                    "temporal_step": timestep,
                    "descriptor_cycles": int(descriptor[timestep]),
                    "lane_compute_cycles": int(lane_compute[timestep]),
                    "contexts": int(prototype["contexts"]),
                }
                for timestep in range(len(descriptor))
            ],
        })
    used_boundaries = {
        (
            item["sample_id"], item["sequence_key"], item["producer"],
            item["producer_call_index"],
        )
        for item in patterns
    }
    if used_boundaries != set(boundary_by_key):
        raise ValueError("M18 direct boundary set and M17 operator set differ")
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--direct-boundaries", type=Path, required=True)
    parser.add_argument("--context-slots", type=int, default=4)
    parser.add_argument("--fifo-depth", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, prototypes, stream, reconciliation, boundaries = load_inputs(
        args.oracle_dir.resolve(), args.reconciliation.resolve(), args.direct_boundaries.resolve()
    )
    results = {}
    for line in ("local", "hybrid"):
        patterns = build_patterns(manifest, prototypes, stream, boundaries, line=line)
        variants = {
            variant: simulate_event_stream(
                patterns, variant=variant, context_slots=args.context_slots,
                fifo_depth=args.fifo_depth,
            )
            for variant in sorted(VARIANTS)
        }
        expected_producer = int(reconciliation["summary"][f"{line}_m4_wall_cycles"])
        expected_atlif = int(boundaries["summary"]["service_cycles_l16"])
        for variant in ("full_context", "lane_cache"):
            if variants[variant]["producer_work_cycles"] != expected_producer:
                raise ValueError(f"{line}/{variant} producer work does not conserve M17 exact wall cycles")
        for variant, result in variants.items():
            if result["atlif_service_cycles"] != expected_atlif:
                raise ValueError(f"{line}/{variant} ATLIF service does not conserve M18 boundaries")
        results[line] = {
            "patterns": len(patterns),
            "pattern_stream_sha256": pattern_stream_sha256(patterns),
            "variants": variants,
        }
    payload = {
        "schema": "m18_exact_population_direct_m4_finite_credit_overlap_v1",
        "status": "PASS_EXACT_POPULATION_DIRECT_M4_FINITE_CREDIT_EVENTS_NOT_SYSTEM_SPEEDUP",
        "configuration": {
            "context_slots": args.context_slots, "fifo_depth": args.fifo_depth,
            "lines": ["local", "hybrid"], "variants": sorted(VARIANTS),
        },
        "results": results,
        "identities": {
            "oracle_manifest_sha256": sha256(args.oracle_dir / "manifest.json"),
            "reconciliation_sha256": sha256(args.reconciliation),
            "direct_boundaries_sha256": sha256(args.direct_boundaries),
            "source_sha256": sha256(Path(__file__).resolve()),
            "m15_source_sha256": sha256(
                Path(__file__).with_name("simulate_m15_finite_credit_retirement.py")
            ),
        },
        "claim_boundary": (
            "Exact sample1 direct-M4 producer plus immediately consuming ATLIF finite-credit "
            "subsystem events. Speedup inside this isolated subsystem is not full-network or "
            "end-to-end speedup; no joins, SRAM/DRAM port timing, RTL, energy, or PPA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS_M18_EXACT_DIRECT_OVERLAP patterns={}".format(results["local"]["patterns"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
