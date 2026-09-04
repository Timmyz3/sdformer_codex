#!/usr/bin/env python3
"""Design-weighted wall-cycle estimator for adjacent C4 Local/Motion clusters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "descriptor_load_cycles", "selected_sources", "selected_product_terms",
    "same_width_dense_issue_cycles",
    "compact_issue_cycles", "chunk_control_cycles", "output_cycles",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_wall() -> Any:
    path = Path(__file__).with_name("analyze_m4_descriptor_resident_wall_cycles.py")
    spec = importlib.util.spec_from_file_location("m14_wall", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import wall model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ledger(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def cluster_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["sample_id"], row["sequence_key"], row["name"], row["operator"],
        row["operator_call_index"], row["weight_group"], row["sample_cluster_id"],
        row["sampling_stratum"], row["source_width"], row["chunks_per_row"],
        row["output_channel_fanout"],
    )


def build_clusters(records: list[dict[str, str]], wall: Any) -> list[dict[str, Any]]:
    bundles = wall.ordered_row_bundles(records)
    grouped: OrderedDict[tuple[str, ...], list[list[int]]] = OrderedDict()
    for bundle in bundles:
        grouped.setdefault(cluster_key(records[bundle[0]]), []).append(bundle)
    clusters = []
    for key, cluster_bundles in grouped.items():
        by_step: dict[int, list[list[int]]] = defaultdict(list)
        for bundle in cluster_bundles:
            row = records[bundle[0]]
            by_step[int(row["temporal_step"])].append(bundle)
        steps = sorted(by_step)
        if steps != list(range(len(steps))):
            raise ValueError(f"non-contiguous cluster timestep: {key}")
        row_ids = None
        for step in steps:
            by_step[step].sort(key=lambda bundle: int(records[bundle[0]]["sample_cluster_lane"]))
            lanes = [int(records[bundle[0]]["sample_cluster_lane"]) for bundle in by_step[step]]
            if lanes != list(range(len(lanes))):
                raise ValueError(f"non-contiguous C4 lanes: {key} step={step}")
            current_ids = [int(records[bundle[0]]["row_id"]) for bundle in by_step[step]]
            if current_ids != list(range(current_ids[0], current_ids[0] + len(current_ids))):
                raise ValueError(f"non-adjacent physical rows in cluster: {key} step={step}")
            if row_ids is None:
                row_ids = current_ids
            elif current_ids != row_ids:
                raise ValueError(f"cluster membership changed across time: {key}")
        first = records[by_step[0][0][0]]
        weight = float(first["cluster_inverse_probability_weight"])
        population = int(first["stratum_population_clusters"])
        samples = int(first["stratum_sample_clusters"])
        if samples <= 0 or not math.isclose(weight, population / samples, rel_tol=1e-12):
            raise ValueError(f"invalid cluster design weight: {key}")
        for bundle in cluster_bundles:
            row = records[bundle[0]]
            fields = (
                float(row["cluster_inverse_probability_weight"]),
                int(row["stratum_population_clusters"]),
                int(row["stratum_sample_clusters"]),
                int(row["sample_cluster_rows"]),
            )
            if fields != (weight, population, samples, len(row_ids or [])):
                raise ValueError(f"cluster metadata drift: {key}")
        clusters.append({
            "key": key,
            "name": key[2],
            "operator": key[3],
            "operator_call_index": int(key[4]),
            "weight_group": int(key[5]),
            "sample_cluster_id": int(key[6]),
            "stratum": key[7],
            "weight": weight,
            "population": population,
            "samples": samples,
            "row_ids": row_ids,
            "steps": [by_step[step] for step in steps],
        })
    return clusters


def cluster_metrics(
    cluster: dict[str, Any], records: list[dict[str, str]], bits: np.ndarray,
    wall: Any, *, issue_width: int, reduce_slots: int, output_lanes: int,
) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    for step_bundles in cluster["steps"]:
        chunks = len(step_bundles[0])
        fanout = int(records[step_bundles[0][0]]["output_channel_fanout"])
        lane_tiles = math.ceil(fanout / output_lanes)
        if any(len(bundle) != chunks for bundle in step_bundles):
            raise ValueError("chunk geometry changed inside C4")
        totals["descriptor_load_cycles"] += sum(len(bundle) for bundle in step_bundles)
        for chunk in range(chunks):
            counts = np.zeros((len(step_bundles), issue_width), dtype=np.int64)
            dense = 0
            for context, bundle in enumerate(step_bundles):
                index = bundle[chunk]
                counts[context] = [
                    int(bits[index, bank::issue_width].sum()) for bank in range(issue_width)
                ]
                dense += math.ceil(int(records[index]["valid_bits"]) / issue_width)
            totals["selected_sources"] += int(counts.sum()) * lane_tiles
            totals["selected_product_terms"] += int(counts.sum()) * fanout
            totals["same_width_dense_issue_cycles"] += dense * lane_tiles
            totals["compact_issue_cycles"] += wall.compact_issue_cycles(counts, reduce_slots) * lane_tiles
            totals["chunk_control_cycles"] += 2 * lane_tiles
        totals["output_cycles"] += len(step_bundles) * lane_tiles
    return {name: totals[name] for name in METRICS}


def finalize_totals(totals: dict[str, float]) -> dict[str, float]:
    m4 = (
        totals["descriptor_load_cycles"] + totals["compact_issue_cycles"]
        + totals["chunk_control_cycles"] + totals["output_cycles"]
    )
    p1 = (
        totals["descriptor_load_cycles"] + totals["selected_sources"]
        + totals["chunk_control_cycles"] + totals["output_cycles"]
    )
    dense = (
        totals["descriptor_load_cycles"] + totals["same_width_dense_issue_cycles"]
        + totals["chunk_control_cycles"] + totals["output_cycles"]
    )
    return {
        **totals,
        "m4_wall_cycles": m4,
        "p1_sparse_wall_cycles": p1,
        "same_width_dense_wall_cycles": dense,
        "speedup_vs_p1_sparse_wall": p1 / m4 if m4 else 1.0,
        "speedup_vs_same_width_dense_wall": dense / m4 if m4 else 1.0,
    }


def analyze_identity(
    tile_dir: Path, ledger: dict[str, dict[str, str]], dependency: dict[str, Any], *, issue_width: int,
    reduce_slots: int, output_lanes: int,
) -> dict[str, Any]:
    wall = load_wall()
    validator = wall.load_validator()
    manifest, records, current, previous = validator.validate(tile_dir)
    if manifest.get("schema") != "dual_line_real_tile_trace_v2":
        raise ValueError("M14 requires the adjacent-C4 v2 tile schema")
    dependency_identity = dependency["identities"]["dependency_manifest"]
    artifact = manifest["run_context"]["artifact_identity"]
    if artifact["checkpoint_sha256"] != dependency_identity["artifact_identity"]["checkpoint_sha256"]:
        raise ValueError("tile/dependency checkpoint mismatch")
    if artifact["config_sha256"] != dependency_identity["artifact_identity"]["config_sha256"]:
        raise ValueError("tile/dependency config mismatch")
    direct_names = {
        row["producers"][0]
        for row in dependency["rows"]
        if row["sample_id"] == 0 and row["category"] == "direct_m4"
        and row["live"] is True and row.get("admitted_for_overlap") is True
    }
    clusters = build_clusters(records, wall)
    current_bits = np.unpackbits(current, axis=1, bitorder="little").astype(bool)
    previous_bits = np.unpackbits(previous, axis=1, bitorder="little").astype(bool)
    use_motion = np.asarray(
        [row["row_use_motion"].lower() == "true" for row in records], dtype=bool
    )[:, None]
    line_bits = {
        "local": current_bits,
        "hybrid": np.where(use_motion, current_bits ^ previous_bits, current_bits),
    }
    result: dict[str, Any] = {}
    for line, bits in line_bits.items():
        overall: defaultdict[str, float] = defaultdict(float)
        per_module: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        cluster_rows = []
        for cluster in clusters:
            raw = cluster_metrics(
                cluster, records, bits, wall, issue_width=issue_width,
                reduce_slots=reduce_slots, output_lanes=output_lanes,
            )
            weight = cluster["weight"]
            for metric, value in raw.items():
                overall[metric] += value * weight
                per_module[cluster["name"]][metric] += value * weight
            per_module[cluster["name"]]["estimated_clusters"] += weight
            cluster_rows.append({
                "name": cluster["name"], "operator_call_index": cluster["operator_call_index"],
                "sample_cluster_id": cluster["sample_cluster_id"], "stratum": cluster["stratum"],
                "weight": weight, "contexts": len(cluster["row_ids"]), **raw,
            })
        modules = {}
        for name, totals in sorted(per_module.items()):
            item = finalize_totals(dict(totals))
            if name not in ledger:
                raise ValueError(f"operator ledger missing sampled module: {name}")
            exact_local = float(ledger[name]["activity_cycles_at_config_lanes"])
            item["ledger_activity_cycles"] = exact_local
            exact_terms = float(ledger[name]["activity_weighted_macs_per_frame"])
            item["ledger_activity_weighted_macs"] = exact_terms
            item["selected_source_relative_error_vs_ledger"] = (
                item["selected_product_terms"] / exact_terms - 1.0 if exact_terms else 0.0
            )
            modules[name] = item
        total = finalize_totals(dict(overall))
        exact_total = sum(float(ledger[name]["activity_cycles_at_config_lanes"]) for name in modules)
        exact_terms_total = sum(
            float(ledger[name]["activity_weighted_macs_per_frame"]) for name in modules
        )
        total["ledger_activity_cycles"] = exact_total
        total["ledger_activity_weighted_macs"] = exact_terms_total
        total["selected_source_relative_error_vs_ledger"] = (
            total["selected_product_terms"] / exact_terms_total - 1.0
            if exact_terms_total else 0.0
        )
        max_module_error = max(
            (abs(item["selected_source_relative_error_vs_ledger"]) for item in modules.values()),
            default=0.0,
        )
        total["max_module_abs_selected_source_error"] = max_module_error
        result[line] = {"totals": total, "modules": modules, "sampled_clusters": cluster_rows}
        missing_direct = sorted(direct_names - set(modules))
        if missing_direct:
            raise ValueError("direct dependency producers missing from C4 trace: " + ",".join(missing_direct))
        direct_estimate = sum(modules[name]["selected_product_terms"] for name in direct_names)
        direct_exact = sum(modules[name]["ledger_activity_weighted_macs"] for name in direct_names)
        direct_errors = {
            name: modules[name]["selected_source_relative_error_vs_ledger"] for name in direct_names
        }
        result[line]["direct_m4_admission"] = {
            "modules": sorted(direct_names),
            "estimated_product_terms": direct_estimate,
            "ledger_product_terms": direct_exact,
            "relative_error": direct_estimate / direct_exact - 1.0,
            "max_module_abs_error": max(abs(value) for value in direct_errors.values()),
            "per_module_relative_error": dict(sorted(direct_errors.items())),
        }
    local_error = abs(result["local"]["totals"]["selected_source_relative_error_vs_ledger"])
    direct_error = abs(result["local"]["direct_m4_admission"]["relative_error"])
    direct_max = result["local"]["direct_m4_admission"]["max_module_abs_error"]
    admitted = local_error <= 0.02 and direct_error <= 0.02 and direct_max <= 0.05
    return {
        "status": (
            "PASS_STRATIFIED_C4_ESTIMATOR_ADMISSION" if admitted
            else "PARTIAL_STRATIFIED_C4_ESTIMATOR_NOT_ADMITTED"
        ),
        "admitted_for_event_timing": admitted,
        "records": len(records),
        "sampled_clusters": len(clusters),
        "manifest_status": manifest["status"],
        "checkpoint_sha256": manifest["run_context"]["artifact_identity"]["checkpoint_sha256"],
        "config_sha256": manifest["run_context"]["artifact_identity"]["config_sha256"],
        "sample_keys": sorted({row["sample_key"] for row in records}),
        "sequence_keys": sorted({row["sequence_key"] for row in records}),
        "lines": result,
        "identities": {
            "manifest_sha256": sha256(tile_dir / "manifest.json"),
            "records_sha256": sha256(tile_dir / "tile_records.csv"),
            "packed_tiles_sha256": sha256(tile_dir / "packed_tiles.npz"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"), required=True)
    parser.add_argument("--operator-ledger", type=Path, required=True)
    parser.add_argument("--dependency-audit", type=Path, required=True)
    parser.add_argument("--issue-width", type=int, default=16)
    parser.add_argument("--reduce-slots", type=int, default=4)
    parser.add_argument("--output-lanes", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = load_ledger(args.operator_ledger)
    dependency = json.loads(args.dependency_audit.read_text(encoding="utf-8"))
    if dependency.get("status") != "PASS_CAUSAL_DEPENDENCY_CLASSIFICATION":
        raise ValueError("dependency audit is not admitted")
    identities = {
        label: analyze_identity(
            Path(directory).resolve(), ledger, dependency, issue_width=args.issue_width,
            reduce_slots=args.reduce_slots, output_lanes=args.output_lanes,
        )
        for label, directory in args.identity
    }
    payload = {
        "schema": "m14_stratified_adjacent_c4_wall_cycle_estimator_v1",
        "status": (
            "PASS_ALL_IDENTITIES_ADMITTED" if all(
                item["admitted_for_event_timing"] for item in identities.values()
            ) else "PARTIAL_ONE_OR_MORE_IDENTITIES_NOT_ADMITTED"
        ),
        "architecture": {
            "issue_width": args.issue_width, "contexts": 4,
            "reduce_slots": args.reduce_slots, "output_lanes": args.output_lanes,
        },
        "identities": identities,
        "sources": {
            "operator_ledger": {"path": str(args.operator_ledger), "sha256": sha256(args.operator_ledger)},
            "dependency_audit": {"path": str(args.dependency_audit), "sha256": sha256(args.dependency_audit)},
            "script_sha256": sha256(Path(__file__).resolve()),
            "wall_source_sha256": sha256(Path(__file__).with_name("analyze_m4_descriptor_resident_wall_cycles.py")),
        },
        "claim_boundary": (
            "Design-weighted adjacent-C4 source-kernel point estimates with exact sampled-cluster "
            "controller cycles. Admission requires <=2% captured-total and <=5% worst-module "
            "Local selected-source error versus the operator ledger. Not full-spatial proof, "
            "cross-sequence confidence, SRAM/DRAM timing, system speedup, energy, or PPA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for label, item in identities.items():
        total = item["lines"]["local"]["totals"]
        direct = item["lines"]["local"]["direct_m4_admission"]
        print(
            f"{label}: {item['status']} clusters={item['sampled_clusters']} "
            f"local_error={total['selected_source_relative_error_vs_ledger']:+.4%} "
            f"direct_error={direct['relative_error']:+.4%} "
            f"direct_max={direct['max_module_abs_error']:.4%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
